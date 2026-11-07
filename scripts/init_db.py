"""初始化数据库表结构（幂等）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import init_db  # noqa: E402

if __name__ == "__main__":
    init_db()
    print("电商 RAG 数据库初始化完成（data/rag.db）")
