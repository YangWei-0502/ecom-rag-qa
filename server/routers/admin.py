"""管理员：索引统计（SQLite 事实源 vs 向量库一致性）。"""
from fastapi import APIRouter

from app import knowledge

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.get("/stats")
def stats():
    return knowledge.index_stats()
