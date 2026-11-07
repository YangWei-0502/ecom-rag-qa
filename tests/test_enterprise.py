"""第二轮企业级改造测试：删除一致性、元数据/权限过滤、重排开关、FastAPI。

全部离线：嵌入用字符级词袋伪向量；对话模型用 Fake 替代；不访问 Ollama。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class _FakeEmbeddings:
    """字符级词袋哈希伪向量（共享字符即相似），确定性、不依赖 Ollama。"""

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    @staticmethod
    def _vec(text, dim=128):
        import hashlib

        v = [0.0] * dim
        for ch in text:
            if ch.strip():
                h = int(hashlib.md5(ch.encode("utf-8")).hexdigest(), 16)
                v[h % dim] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]


@pytest.fixture()
def env(monkeypatch, tmp_path):
    import app.config as config
    from app import knowledge
    from app.db import init_db

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "rag.db")
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "ENABLE_HYBRID_SEARCH", False)
    monkeypatch.setattr(config, "ENABLE_RERANK", False)
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(knowledge, "get_embeddings", lambda: _FakeEmbeddings())
    init_db()
    yield knowledge


# ---------- 删除一致性：SQLite + 向量库双写 ----------

def test_delete_consistency(env):
    k = env
    r1 = k.ingest_text("u1", "尺码.txt", "M码适合120斤以内，L码适合140斤。" * 10)
    r2 = k.ingest_text("u1", "洗涤.txt", "针织毛衣冷水手洗，不可漂白，平铺阴干。" * 10)
    doc1, doc2 = r1["doc_id"], r2["doc_id"]

    k.delete_document("u1", doc1)

    # 1) SQLite 事实源：文档与块都没了
    from app.db import get_conn
    conn = get_conn()
    try:
        assert conn.execute("SELECT COUNT(*) c FROM documents WHERE id=?", (doc1,)).fetchone()["c"] == 0
        assert conn.execute("SELECT COUNT(*) c FROM document_chunks WHERE doc_id=?", (doc1,)).fetchone()["c"] == 0
        # 2) 其它文档不受影响
        assert conn.execute("SELECT COUNT(*) c FROM documents WHERE id=?", (doc2,)).fetchone()["c"] == 1
    finally:
        conn.close()

    # 3) 向量库：该文档的 chunk_id 已全部清除（get 返回空）
    store = k.get_vector_store()
    rows = store.get(where={"doc_id": doc1})
    assert not rows.get("ids"), f"向量库残留 {len(rows.get('ids', []))} 块"

    # 4) 检索：该文档内容不再命中；另一文档仍可命中
    assert k.search("u1", "M码适合120斤") == []
    assert k.search("u1", "毛衣冷水手洗")


# ---------- 元数据 / 权限过滤 ----------

def test_metadata_category_filter(env):
    k = env
    k.ingest_text("u1", "尺码.txt", "M码适合120斤以内，L码适合140斤。" * 10, category="尺码")
    k.ingest_text("u1", "洗涤.txt", "针织毛衣冷水手洗，不可漂白。" * 10, category="洗涤")

    # 按 category 过滤只返回该分类
    hits = k.search("u1", "M码 120斤", category="尺码")
    assert hits and all(h["source"] == "尺码.txt" for h in hits)

    # 不相关分类过滤下不应命中该文档
    assert k.search("u1", "M码 120斤", category="洗涤") == []


def test_cross_tenant_forbidden(env):
    k = env
    k.ingest_text("u1", "尺码.txt", "M码适合120斤以内，L码适合140斤。" * 10)
    k.ingest_text("u2", "售后.txt", "七天无理由退换货，不影响二次销售。" * 10)
    # u2 问 u1 专属内容：向量 filter + SQLite 双隔离，必为空
    assert k.search("u2", "L码适合140斤") == []
    assert k.search("u1", "七天无理由退换货") == []


# ---------- 重排开关 ----------

def test_rerank_switch(env):
    k = env
    k.ingest_text("u1", "洗涤.txt", "针织毛衣冷水手洗，不可漂白，平铺阴干。" * 10)
    k.ingest_text("u1", "售后.txt", "七天无理由退换货，商品不影响二次销售。" * 10)

    off = k.search("u1", "毛衣 手洗", use_rerank=False)
    on = k.search("u1", "毛衣 手洗", use_rerank=True)
    assert off and on
    # 重排开启后结果带 rerank_score，且毛衣/洗涤应排到更前
    assert all("rerank_score" in h for h in on)
    assert on[0]["source"] == "洗涤.txt"


# ---------- FastAPI（TestClient，离线） ----------

@pytest.fixture()
def api_client(monkeypatch, tmp_path):
    import app.config as config
    from app import knowledge
    from app.db import init_db

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "rag.db")
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "ENABLE_HYBRID_SEARCH", False)
    monkeypatch.setattr(config, "ENABLE_RERANK", False)
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(knowledge, "get_embeddings", lambda: _FakeEmbeddings())
    init_db()

    # 关闭限流，避免测试互相影响
    import server.middleware as mw
    monkeypatch.setattr(mw.limiter, "limit", 0)

    from fastapi.testclient import TestClient
    from server.main import create_app
    return TestClient(create_app())


def test_api_crud_and_retrieve(api_client):
    # 上传
    files = {"file": ("尺码.txt", "M码适合120斤以内，L码适合140斤。" * 15, "text/plain")}
    data = {"user_id": "api_user", "category": "尺码"}
    r = api_client.post("/api/v1/documents/upload", files=files, data=data)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    doc_id = body["doc_id"]
    assert "X-Request-ID" in r.headers

    # 列表
    r = api_client.get("/api/v1/documents", params={"user_id": "api_user"})
    assert r.status_code == 200 and r.json()["count"] == 1
    assert r.json()["documents"][0]["category"] == "尺码"

    # 检索
    r = api_client.post("/api/v1/retrieve", json={
        "user_id": "api_user", "question": "M码适合多少斤", "top_k": 3})
    assert r.status_code == 200 and r.json()["count"] >= 1

    # 统计
    r = api_client.get("/api/v1/admin/stats")
    assert r.status_code == 200 and r.json()["documents"] == 1

    # 删除 + 不再命中
    r = api_client.delete(f"/api/v1/documents/{doc_id}", params={"user_id": "api_user"})
    assert r.status_code == 200
    r = api_client.post("/api/v1/retrieve", json={"user_id": "api_user", "question": "M码"})
    assert r.json()["hits"] == []


def test_api_validation_and_404(api_client):
    # 缺字段 -> 422
    r = api_client.post("/api/v1/retrieve", json={"question": "无user_id"})
    assert r.status_code == 422

    # 不存在的文档 -> 404
    r = api_client.get("/api/v1/documents/nope", params={"user_id": "x"})
    assert r.status_code == 404

    # 空文件上传 -> 400
    r = api_client.post("/api/v1/documents/upload",
                        files={"file": ("a.txt", b"", "text/plain")},
                        data={"user_id": "u"})
    assert r.status_code == 400


def test_api_ask_with_fake_llm(monkeypatch, api_client):
    import app.chat as chat

    class _FakeMsg:
        content = "（伪造回答）M 码适合 120 斤以内。"

    def fake_build_chain(sources, history):
        class _FakeChain:
            def invoke(self, inp):
                return _FakeMsg()
        return _FakeChain()

    monkeypatch.setattr(chat, "_build_chain", fake_build_chain)
    # 先入库
    api_client.post("/api/v1/documents/upload",
                    files={"file": ("尺码.txt", b"M\xe7\xa0\x81\xe9\x80\x82\xe5\x90\x88120\xe6\x96\xa4\xe4\xbb\xa5\xe5\x86\x85\xe3\x80\x82" * 15, "text/plain")},
                    data={"user_id": "asker"})
    r = api_client.post("/api/v1/ask", json={"user_id": "asker", "question": "M码穿多少斤"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"]
    assert "request_id" in body
