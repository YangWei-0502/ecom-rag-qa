"""request-id 中间件 + 结构化访问日志 + 统一异常信封。"""
from __future__ import annotations

import logging
import sys
import time
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .rate_limit import limiter

logger = logging.getLogger("rag.api")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """注入 X-Request-ID；记录方法/路径/耗时/客户端；超限返回 429。"""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        client = request.client.host if request.client else "-"

        allowed, remaining = limiter.check(client)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"error": "请求过于频繁，请稍后再试", "request_id": request_id},
                headers={"X-Request-ID": request_id, "Retry-After": "60"},
            )

        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:  # 兜底，统一成 500 信封
            logger.error("request_id=%s unhandled error: %r", request_id, exc)
            return JSONResponse(
                status_code=500,
                content={"error": "内部服务错误", "request_id": request_id, "detail": str(exc)[:300]},
                headers={"X-Request-ID": request_id},
            )
        cost = (time.perf_counter() - t0) * 1000
        logger.info(
            "request_id=%s client=%s %s %s -> %s %.0fms",
            request_id, client, request.method, request.url.path,
            response.status_code, cost,
        )
        response.headers["X-Request-ID"] = request_id
        return response


def register_exception_handlers(app) -> None:
    from fastapi import Request
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException

    @app.exception_handler(StarletteHTTPException)
    async def http_exc(_: Request, exc: StarletteHTTPException):
        rid = getattr(_.state, "request_id", "-")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": str(exc.detail), "request_id": rid},
            headers={"X-Request-ID": rid},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exc(_: Request, exc: RequestValidationError):
        rid = getattr(_.state, "request_id", "-")
        return JSONResponse(
            status_code=422,
            content={"error": "请求参数校验失败", "request_id": rid,
                     "detail": [e.get("msg") for e in exc.errors()][:5]},
            headers={"X-Request-ID": rid},
        )
