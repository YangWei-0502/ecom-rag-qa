"""检索 / 问答接口。"""
from fastapi import APIRouter, Request

from app import chat, knowledge
from server.schemas import AskRequest, RetrieveRequest

router = APIRouter(prefix="/api/v1", tags=["retrieval"])


@router.post("/retrieve")
def retrieve(req: RetrieveRequest):
    hits = knowledge.search(
        user_id=req.user_id,
        question=req.question,
        top_k=req.top_k,
        use_hybrid=req.use_hybrid,
        use_rerank=req.use_rerank,
        category=req.category,
    )
    return {"question": req.question, "count": len(hits), "hits": hits}


@router.post("/ask")
def ask(req: AskRequest, request: Request):
    rid = getattr(request.state, "request_id", "-")
    result = chat.answer(
        user_id=req.user_id,
        question=req.question,
        session_id=req.session_id,
    )
    return {
        "answer": result["answer"],
        "refused": result["refused"],
        "sources": result["sources"],
        "request_id": rid,
    }
