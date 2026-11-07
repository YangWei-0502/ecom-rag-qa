"""知识库块级去重与检索冒烟测试（不访问网络，使用临时 DB 与临时 Chroma）。"""
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
def isolated_env(monkeypatch, tmp_path):
    import app.config as config
    from app import knowledge
    from app.db import init_db

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "rag.db")
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "ENABLE_HYBRID_SEARCH", False)
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(knowledge, "get_embeddings", lambda: _FakeEmbeddings())
    knowledge.get_embeddings.cache_clear() if hasattr(knowledge.get_embeddings, "cache_clear") else None
    init_db()
    yield knowledge


def test_ingest_dedup_and_search(isolated_env):
    knowledge = isolated_env
    text = "针织毛衣需要手洗冷水，不可漂白，平铺晾干。" * 20
    r1 = knowledge.ingest_text("u1", "洗涤养护.txt", text)
    assert r1["status"] == "ok" and r1["added_chunks"] >= 1
    # 同内容再次入库 -> 跳过
    r2 = knowledge.ingest_text("u1", "洗涤养护2.txt", text)
    assert r2["status"] == "skipped"

    docs = knowledge.list_documents("u1")
    assert len(docs) == 1

    hits = knowledge.search("u1", "针织毛衣手洗冷水")
    assert hits and "洗涤" in hits[0]["source"]

    knowledge.delete_document("u1", r1["doc_id"])
    assert knowledge.list_documents("u1") == []
    assert knowledge.search("u1", "毛衣") == []


def test_user_isolation(isolated_env):
    knowledge = isolated_env
    knowledge.ingest_text("u1", "尺码.txt", "M码适合120斤以内，L码适合140斤。" * 5)
    knowledge.ingest_text("u2", "售后.txt", "七天无理由退换货，商品不影响二次销售。" * 5)
    hits1 = knowledge.search("u1", "L码适合140斤")
    hits2 = knowledge.search("u2", "七天无理由退换货")
    assert hits1 and all(h["source"] == "尺码.txt" for h in hits1)
    assert hits2 and all(h["source"] == "售后.txt" for h in hits2)
    # u2 检索 u1 的特征词应被阈值过滤
    assert knowledge.search("u2", "L码适合140斤") == []
