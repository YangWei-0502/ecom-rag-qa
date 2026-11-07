"""重建向量索引：
- 默认：先载入种子知识文档（幂等），再以 SQLite 为准全量重建 Chroma；
- --full：清空全部文档与向量（含用户上传内容），重新入库种子文档后重建
          （用于调整分块参数后重建；日常切换嵌入模型用默认即可）。

使用场景：首次部署、切换嵌入模型、调整分块参数、向量库损坏。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import knowledge  # noqa: E402
from app.db import db_cursor, get_conn, init_db  # noqa: E402


def full_reset() -> None:
    conn = get_conn()
    ids = [r["id"] for r in conn.execute("SELECT id FROM documents").fetchall()]
    conn.close()
    for doc_id in ids:
        knowledge.delete_document(knowledge.config.DEFAULT_USER_ID, doc_id)
    # 其他用户的文档也一并清掉
    with db_cursor(commit=True) as cur:
        cur.execute("DELETE FROM document_chunks")
        cur.execute("DELETE FROM documents")
    knowledge.rebuild_index()
    print(f"已清空 {len(ids)} 个旧文档")


if __name__ == "__main__":
    init_db()
    if "--full" in sys.argv:
        full_reset()
    n_seed = knowledge.seed_documents()
    print(f"种子文档新入库：{n_seed} 个")
    n_chunks = knowledge.rebuild_index()
    print(f"向量索引重建完成，共 {n_chunks} 个内容块")
