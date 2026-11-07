"""Pydantic 请求/响应模型（v2）。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64)
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(6, ge=1, le=20)
    use_hybrid: bool | None = None
    use_rerank: bool | None = None
    category: str | None = None


class AskRequest(RetrieveRequest):
    session_id: str | None = None


class SourceItem(BaseModel):
    chunk_id: str
    source: str | None = None
    title: str | None = None
    doc_id: str | None = None
    score: float | None = None
    distance: float | None = None
    content: str


class RetrieveResponse(BaseModel):
    question: str
    hits: list[SourceItem]
    count: int


class AskResponse(BaseModel):
    answer: str
    refused: bool
    sources: list[SourceItem]
    request_id: str


class DocumentItem(BaseModel):
    id: str
    user_id: str
    filename: str
    title: str
    category: str
    tags: str
    version: int
    chunk_count: int
    status: str
    error_message: str = ""
    created_at: str


class UploadResponse(BaseModel):
    status: str
    doc_id: str
    added_chunks: int
    message: str


class ErrorResponse(BaseModel):
    error: str
    request_id: str
    detail: str | None = None
