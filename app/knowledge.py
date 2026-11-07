"""知识库服务：文档入库（块级去重）、删除、检索（向量 + 可选 BM25 混合）。"""
from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime
from pathlib import Path

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .db import db_cursor, get_conn
from .document_loader import DocumentError, parse_file
from .llm import get_embeddings


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        length_function=len,
    )


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


def list_documents(user_id: str) -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM documents WHERE user_id=? ORDER BY created_at DESC", (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def count_documents(user_id: str) -> int:
    conn = get_conn()
    try:
        return conn.execute(
            "SELECT COUNT(*) AS c FROM documents WHERE user_id=?", (user_id,)
        ).fetchone()["c"]
    finally:
        conn.close()


def ingest_text(
    user_id: str,
    filename: str,
    text: str,
    title: str | None = None,
    category: str = "",
    tags: str = "",
) -> dict:
    """入库纯文本；返回 {status, added_chunks, doc_id}。

    生命周期：先落 documents(status=processing)，分块/嵌入/入库成功后置 ready；
    任一环节失败置 failed 并记录 error_message，不虚假成功。
    幂等：同 user_id + 同全文 MD5 直接 skipped；块级 MD5 去重重复片段。
    """
    text = text.strip()
    if not text:
        raise DocumentError("文档内容为空")
    file_hash = _md5(text)  # 仅按内容判重，同名内容改名重传也会跳过
    conn = get_conn()
    existed = conn.execute(
        "SELECT id FROM documents WHERE user_id=? AND file_hash=?", (user_id, file_hash)
    ).fetchone()
    conn.close()
    if existed:
        return {"status": "skipped", "added_chunks": 0, "doc_id": existed["id"], "message": "文档已存在，跳过"}

    doc_id = str(uuid.uuid4())
    now = _now()
    title = (title or Path(filename).stem).strip()
    summary = text[:200].replace("\n", " ")
    chunks = get_splitter().split_text(text)
    if not chunks:
        chunks = [text]

    # 先落 processing 记录，失败可被列出/排查
    with db_cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO documents (id, user_id, filename, file_hash, title, category, tags,"
            " summary, version, chunk_count, status, error_message, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (doc_id, user_id, filename, file_hash, title, category, tags,
             summary, 1, 0, "processing", "", now, now),
        )

    # 块级去重（同一用户、同内容的块只入库一次）
    conn = get_conn()
    existing_hashes = {
        r["chunk_hash"]
        for r in conn.execute(
            "SELECT c.chunk_hash FROM document_chunks c "
            "JOIN documents d ON d.id=c.doc_id WHERE d.user_id=?",
            (user_id,),
        ).fetchall()
    }
    conn.close()
    new_chunks = [
        (i, content, _md5(content))
        for i, content in enumerate(chunks)
        if _md5(content) not in existing_hashes
    ]

    prepared = []
    for i, content, chash in new_chunks:
        prepared.append(
            (
                str(uuid.uuid4()),
                content,
                {
                    "user_id": user_id,
                    "doc_id": doc_id,
                    "chunk_id": None,
                    "source": filename,
                    "title": title,
                    "category": category or "",
                    "tags": tags or "",
                    "version": 1,
                    "chunk_index": i,
                },
            )
        )
    for cid, _content, meta in prepared:
        meta["chunk_id"] = cid

    try:
        if prepared:
            get_vector_store().add_texts(
                [c for _, c, _ in prepared],
                metadatas=[m for _, _, m in prepared],
                ids=[cid for cid, _, _ in prepared],
            )
        with db_cursor(commit=True) as cur:
            for cid, content, meta in prepared:
                cur.execute(
                    "INSERT INTO document_chunks (id, doc_id, chunk_index, content, chunk_hash, created_at)"
                    " VALUES (?,?,?,?,?,?)",
                    (cid, doc_id, meta["chunk_index"], content, _md5(content), now),
                )
            cur.execute(
                "UPDATE documents SET status='ready', chunk_count=?, updated_at=? WHERE id=?",
                (len(prepared), now, doc_id),
            )
    except Exception as exc:  # 嵌入超时/向量库故障：落 failed，不写假 ready
        with db_cursor(commit=True) as cur:
            cur.execute(
                "UPDATE documents SET status='failed', error_message=?, updated_at=? WHERE id=?",
                (str(exc)[:500], now, doc_id),
            )
        raise DocumentError(f"文档入库失败：{exc}") from exc

    return {
        "status": "ok",
        "added_chunks": len(new_chunks),
        "doc_id": doc_id,
        "message": f"已入库 {len(new_chunks)} 个内容块（共 {len(chunks)} 块，{len(chunks)-len(new_chunks)} 块重复已跳过）",
    }


def ingest_file(
    user_id: str,
    filename: str,
    raw: bytes,
    title: str | None = None,
    category: str = "",
    tags: str = "",
) -> dict:
    text = parse_file(filename, raw)
    return ingest_text(user_id, filename, text, title=title, category=category, tags=tags)


def delete_document(user_id: str, doc_id: str) -> None:
    """删除文档：校验归属后，保证 SQLite 事实源与 Chroma 向量库一致。

    BM25 语料实时派生自 document_chunks 表，SQLite 删除即自动一致。
    Chroma 删除后用 get(ids=...) 二次校验，残留则重试一次；仍残留则抛错
    （不再静默吞，避免向量库残留导致"已删文档仍被检索命中"）。
    """
    conn = get_conn()
    doc = conn.execute(
        "SELECT id FROM documents WHERE user_id=? AND id=?", (user_id, doc_id)
    ).fetchone()
    if not doc:
        conn.close()
        return
    chunk_ids = [r["id"] for r in conn.execute("SELECT id FROM document_chunks WHERE doc_id=?", (doc_id,)).fetchall()]
    conn.close()

    if chunk_ids:
        store = get_vector_store()
        for attempt in range(2):
            store.delete(ids=chunk_ids)
            left = store.get(ids=chunk_ids)
            if not left or not left.get("ids"):
                break
        else:
            raise RuntimeError(f"向量库删除未完全一致，残留 {len(left.get('ids', []))} 块；请运行 rebuild_index")

    with db_cursor(commit=True) as cur:
        cur.execute("DELETE FROM document_chunks WHERE doc_id=?", (doc_id,))
        cur.execute("DELETE FROM documents WHERE id=?", (doc_id,))


# ---------------- 检索 ----------------

def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text)
    han = re.findall(r"[\u4e00-\u9fff]", text)
    bigrams = [han[i] + han[i + 1] for i in range(len(han) - 1)]
    return words + bigrams


def _user_chunks(user_id: str, category: str | None = None) -> list[dict]:
    conn = get_conn()
    try:
        sql = (
            "SELECT c.id AS chunk_id, c.content, c.chunk_index, d.filename AS source,"
            " d.title AS title, d.id AS doc_id, d.category AS category"
            " FROM document_chunks c JOIN documents d ON d.id=c.doc_id"
            " WHERE d.user_id=?"
        )
        params: list = [user_id]
        if category:
            sql += " AND d.category=?"
            params.append(category)
        sql += " ORDER BY c.rowid"
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _bm25_rank(chunks: list[dict], query: str, k: int) -> list[str]:
    from rank_bm25 import BM25Okapi

    if not chunks:
        return []
    corpus = [_tokenize(c["content"]) for c in chunks]
    bm25 = BM25Okapi(corpus)
    q_tokens = _tokenize(query)
    scores = bm25.get_scores(q_tokens)
    q_set = set(q_tokens)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out = []
    for i in order:
        # 实测（二元分词）：相关查询 BM25 2.6~4.3，离题查询 ≤1.45，取 2.0 为绝对地板
        if scores[i] < config.BM25_MIN_SCORE:
            continue
        matched = len(q_set & set(corpus[i]))
        if matched < config.BM25_MIN_MATCHED_TOKENS:
            continue
        out.append(chunks[i]["chunk_id"])
        if len(out) >= k:
            break
    return out


def search(
    user_id: str,
    question: str,
    top_k: int | None = None,
    use_hybrid: bool | None = None,
    use_rerank: bool | None = None,
    category: str | None = None,
) -> list[dict]:
    """返回命中块：content/source/chunk_id/distance/score/doc_id/title。

    向量检索 + 双阈值 + 可选 BM25(RRF) 融合 + 可选重排序。
    所有查询强制按 user_id 过滤（Chroma metadata filter + SQLite WHERE），
    可选 category 进一步过滤；越权文档不可能进入候选。
    """
    from . import reranker

    top_k = top_k or config.RETRIEVER_TOP_K
    if use_hybrid is None:
        use_hybrid = config.ENABLE_HYBRID_SEARCH
    if use_rerank is None:
        use_rerank = config.ENABLE_RERANK

    if category:
        chroma_filter = {"$and": [{"user_id": user_id}, {"category": category}]}
    else:
        chroma_filter = {"user_id": user_id}

    store = get_vector_store()
    pairs = store.similarity_search_with_score(question, k=top_k * 2, filter=chroma_filter)
    vec_hits = [
        {
            "chunk_id": doc.metadata.get("chunk_id"),
            "content": doc.page_content,
            "source": doc.metadata.get("source"),
            "title": doc.metadata.get("title", ""),
            "doc_id": doc.metadata.get("doc_id"),
            "distance": float(dist),
        }
        for doc, dist in pairs
        if float(dist) <= config.SCORE_DISTANCE_THRESHOLD
    ]
    # 相对阈值：剔除与最优命中差距过大的无关文档，避免上下文被稀释
    if vec_hits and config.SCORE_RELATIVE_SPREAD > 0:
        best = vec_hits[0]["distance"]
        vec_hits = [v for v in vec_hits if v["distance"] <= best + config.SCORE_RELATIVE_SPREAD]

    if not use_hybrid:
        results = [
            {**v, "score": v["distance"]}
            for v in vec_hits[: config.RERANK_CANDIDATES if use_rerank else top_k]
        ]
        return reranker.maybe_rerank(question, results, top_k, use_rerank)

    chunks = _user_chunks(user_id, category=category)
    bm25_ids = _bm25_rank(chunks, question, top_k * 2)
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    # RRF 融合
    rrf: dict[str, float] = {}
    for rank, v in enumerate(vec_hits):
        rrf[v["chunk_id"]] = rrf.get(v["chunk_id"], 0) + 1 / (60 + rank + 1)
    for rank, cid in enumerate(bm25_ids):
        rrf[cid] = rrf.get(cid, 0) + 1 / (60 + rank + 1)
    merged = sorted(rrf, key=lambda c: rrf[c], reverse=True)

    by_chunk = {v["chunk_id"]: v for v in vec_hits}
    results = []
    for cid in merged:
        row = chunk_by_id.get(cid)
        if not row:
            continue
        v = by_chunk.get(cid)
        results.append(
            {
                "chunk_id": cid,
                "content": row["content"],
                "source": row["source"],
                "title": row.get("title", ""),
                "doc_id": row["doc_id"],
                "distance": v["distance"] if v else None,
                "score": round(rrf[cid], 5),
            }
        )
        if len(results) >= config.RERANK_CANDIDATES and use_rerank:
            break
        if not use_rerank and len(results) >= top_k:
            break
    return reranker.maybe_rerank(question, results, top_k, use_rerank)


# ---------------- 索引重建与种子 ----------------

def rebuild_index() -> int:
    """以 SQLite 为准全量重建向量库（切换嵌入模型/向量库损坏时使用）。

    用 delete_collection 清空而非删除目录（Windows 下 SQLite 文件句柄占用会导致删库失败）。
    """
    store = get_vector_store()
    try:
        store.delete_collection()
    except Exception:  # noqa: BLE001  集合不存在时忽略
        pass
    # 删除集合后重新实例化，自动创建同名空集合
    store = get_vector_store()
    chunks = _user_chunks_all()
    if chunks:
        store.add_texts(
            [c["content"] for c in chunks],
            metadatas=[
                {
                    "user_id": c["user_id"],
                    "doc_id": c["doc_id"],
                    "chunk_id": c["chunk_id"],
                    "source": c["source"],
                    "title": c.get("title", ""),
                    "category": c.get("category", ""),
                    "tags": c.get("tags", ""),
                    "version": c.get("version", 1),
                    "chunk_index": c["chunk_index"],
                }
                for c in chunks
            ],
            ids=[c["chunk_id"] for c in chunks],
        )
    return len(chunks)


def _user_chunks_all() -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT c.id AS chunk_id, c.content, c.chunk_index, d.filename AS source,"
            " d.title AS title, d.category AS category, d.tags AS tags, d.version AS version,"
            " d.id AS doc_id, d.user_id"
            " FROM document_chunks c JOIN documents d ON d.id=c.doc_id ORDER BY c.rowid"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def index_stats() -> dict:
    """索引统计：SQLite 事实源 vs 向量库块数 vs 失败文档（管理员用）。"""
    conn = get_conn()
    try:
        total_docs = conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
        total_chunks = conn.execute("SELECT COUNT(*) c FROM document_chunks").fetchone()["c"]
        by_user = [
            dict(r)
            for r in conn.execute(
                "SELECT user_id, COUNT(*) AS docs, SUM(chunk_count) AS chunks"
                " FROM documents GROUP BY user_id"
            ).fetchall()
        ]
        failed = [
            {"id": r["id"], "filename": r["filename"], "error_message": r["error_message"]}
            for r in conn.execute(
                "SELECT id, filename, error_message FROM documents WHERE status='failed'"
            ).fetchall()
        ]
    finally:
        conn.close()
    try:
        vec_count = len(get_vector_store().get().get("ids", []))
    except Exception:  # noqa: BLE001
        vec_count = -1
    return {
        "documents": total_docs,
        "chunks_sqlite": total_chunks,
        "chunks_vector_store": vec_count,
        "consistent": vec_count == total_chunks,
        "by_user": by_user,
        "failed_documents": failed,
    }


def seed_documents() -> int:
    """把 data/knowledge_docs 下的种子文档入库（幂等）。"""
    category_map = {"尺码推荐": "尺码", "洗涤养护": "洗涤", "颜色选择": "颜色"}
    n = 0
    for f in sorted(Path(config.SEED_DOC_DIR).glob("*")):
        if f.suffix.lower() not in (".txt", ".md"):
            continue
        cat = category_map.get(f.stem, "")
        r = ingest_text(
            config.DEFAULT_USER_ID, f.name, f.read_text(encoding="utf-8"),
            category=cat,
        )
        if r["status"] == "ok":
            n += 1
    return n
