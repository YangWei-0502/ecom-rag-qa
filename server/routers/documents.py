"""文档管理：上传 / 列表 / 详情 / 删除 / 重建索引。"""
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app import knowledge
from app.document_loader import DocumentError

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


def _err(request: Request, status: int, msg: str):
    rid = getattr(request.state, "request_id", "-")
    raise HTTPException(status_code=status, detail={"error": msg, "request_id": rid})


@router.post("/upload")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Form(...),
    title: str = Form(""),
    category: str = Form(""),
    tags: str = Form(""),
):
    raw = await file.read()
    try:
        result = knowledge.ingest_file(
            user_id=user_id,
            filename=file.filename or "upload.txt",
            raw=raw,
            title=title or None,
            category=category,
            tags=tags,
        )
    except DocumentError as exc:
        _err(request, 400, str(exc))
    return result


@router.get("")
def list_documents(user_id: str, request: Request):
    docs = knowledge.list_documents(user_id)
    return {"user_id": user_id, "count": len(docs), "documents": docs}


@router.get("/{doc_id}")
def get_document(doc_id: str, user_id: str, request: Request):
    conn = None
    from app.db import get_conn
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM documents WHERE id=? AND user_id=?", (doc_id, user_id)
        ).fetchone()
    finally:
        conn.close()
    if not row:
        _err(request, 404, "文档不存在或无权访问")
    return dict(row)


@router.delete("/{doc_id}")
def delete_document(doc_id: str, user_id: str, request: Request):
    try:
        knowledge.delete_document(user_id, doc_id)
    except RuntimeError as exc:
        _err(request, 500, str(exc))
    return {"deleted": doc_id}


@router.post("/rebuild")
def rebuild_index(request: Request):
    n = knowledge.rebuild_index()
    return {"rebuilt_chunks": n}
