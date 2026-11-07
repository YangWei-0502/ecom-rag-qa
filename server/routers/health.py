"""GET /health：DB / 向量库 / 嵌入模型 / 对话模型 状态。"""
from fastapi import APIRouter, Request

from app import knowledge
from app.llm import chat_model_label, embed_model_label, health_check

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request):
    rid = getattr(request.state, "request_id", "-")
    try:
        stats = knowledge.index_stats()
        db_ok = True
    except Exception as exc:  # noqa: BLE001
        db_ok = False
        stats = {"error": str(exc)[:200]}
    llm_ok, llm_msg = health_check()
    return {
        "status": "ok" if db_ok and llm_ok else "degraded",
        "request_id": rid,
        "db_ok": db_ok,
        "llm_ok": llm_ok,
        "llm_message": llm_msg,
        "chat_model": chat_model_label(),
        "embed_model": embed_model_label(),
        "index": stats,
    }
