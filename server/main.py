"""FastAPI 应用工厂：uvicorn server.main:app --port 8002。

启动前确保已 init_db / reindex（见 README）。
"""
from fastapi import FastAPI

from app.db import init_db
from server.middleware import RequestContextMiddleware, register_exception_handlers
from server.routers import admin, documents, health, retrieval


def create_app() -> FastAPI:
    app = FastAPI(
        title="电商 RAG 服务",
        version="2.0",
        description="企业级 RAG：摄取生命周期 / 混合检索+重排 / 元数据权限过滤 / 无证据拒答 / 引用溯源",
    )
    init_db()
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(retrieval.router)
    app.include_router(admin.router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    from app import config

    uvicorn.run("server.main:app", host=config.API_HOST, port=config.API_PORT, reload=False)
