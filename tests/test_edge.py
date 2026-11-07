"""边界与双通道测试：文档解析校验、空库/无命中拒答、块级去重、BM25 混合、模型切换。

全部离线：嵌入用字符级词袋伪向量，对话模型不真正调用网络。
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
def isolated_env(monkeypatch, tmp_path):
    import app.config as config
    from app import knowledge
    from app.db import init_db

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "rag.db")
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "ENABLE_HYBRID_SEARCH", False)
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(knowledge, "get_embeddings", lambda: _FakeEmbeddings())
    init_db()
    yield knowledge


# ---------- 文档解析边界 ----------

def test_parse_rejects_empty_oversize_bad_ext():
    from app.document_loader import DocumentError, parse_file

    with pytest.raises(DocumentError, match="为空"):
        parse_file("a.txt", b"")
    with pytest.raises(DocumentError, match="超过"):
        parse_file("big.txt", b"x" * (10 * 1024 * 1024 + 1))
    with pytest.raises(DocumentError, match="仅支持"):
        parse_file("a.exe", b"MZ")


def test_parse_txt_utf8_and_gbk():
    from app.document_loader import parse_file

    s = "洗涤说明：冷水手洗。"
    assert parse_file("a.txt", s.encode("utf-8")) == s
    assert parse_file("a.md", s.encode("gbk")) == s


# ---------- 空库 / 无命中 ----------

def test_search_empty_kb_returns_empty(isolated_env):
    knowledge = isolated_env
    # 该用户从未入库任何文档
    assert knowledge.search("nobody", "任何问题") == []


def test_search_unrelated_query_rejected_by_threshold(isolated_env):
    knowledge = isolated_env
    knowledge.ingest_text("u1", "尺码.txt", "M码适合120斤以内，L码适合140斤。" * 5)
    # 完全不相关、字符无交集的问法应被距离阈值过滤 -> 拒答，而非编造
    hits = knowledge.search("u1", "阿尔法贝塔伽马德尔塔欧米茄")
    assert hits == []


# ---------- 块级去重 ----------

def test_chunk_level_dedup(isolated_env):
    knowledge = isolated_env
    # 足够长以切成多块；v2 完全复用 v1 前缀 + 新尾，前缀块应被块级 MD5 去重
    body = "精纺羊毛针织面料，透气保暖，建议冷水轻柔手洗，不可漂白，平铺阴干。" * 25
    r1 = knowledge.ingest_text("u1", "v1.txt", body)
    assert r1["status"] == "ok" and r1["added_chunks"] >= 2
    before = knowledge.count_documents("u1")
    extra = " 全新独有片段甲乙丙丁戊己庚辛壬癸。"
    r2 = knowledge.ingest_text("u1", "v2.txt", body + extra)
    assert r2["status"] == "ok"
    total_v2 = len(knowledge.get_splitter().split_text(body + extra))
    # 复用前缀产生了与 v1 完全相同的块 -> 至少一个块被跳过
    assert "重复已跳过" in r2["message"]
    assert r2["added_chunks"] < total_v2
    assert knowledge.count_documents("u1") == before + 1


# ---------- BM25 混合检索（RRF 融合） ----------

def test_hybrid_search_rrf_fusion(monkeypatch, isolated_env):
    knowledge = isolated_env
    import app.config as config

    monkeypatch.setattr(config, "ENABLE_HYBRID_SEARCH", True)
    knowledge.ingest_text("u1", "洗涤.txt", "针织毛衣需要手洗冷水，不可漂白。" * 8)
    knowledge.ingest_text("u1", "售后.txt", "七天无理由退换货，商品不影响二次销售。" * 8)

    hits = knowledge.search("u1", "毛衣 手洗 冷水")
    assert hits, "混合检索应至少命中一个块"
    # RRF 融合结果带 score 字段，且按融合分排序
    assert all("score" in h for h in hits)
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)
    # 命中应被限定在 u1 自己的文档
    assert {h["source"] for h in hits} <= {"洗涤.txt", "售后.txt"}


# ---------- 双通道模型切换（Mock，不联网） ----------

def test_llm_provider_switch(monkeypatch):
    import app.config as config
    from app import llm

    # ollama 路径：构造 ChatOpenAI 指向本地，api_key 占位
    monkeypatch.setattr(config, "MODEL_PROVIDER", "ollama")
    monkeypatch.setattr(config, "OLLAMA_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setattr(config, "OLLAMA_MODEL", "qwen3.5:9b")
    llm.get_chat_model.cache_clear()
    m = llm.get_chat_model()
    assert m.openai_api_base.startswith("http://localhost:11434")
    assert m.model_name == "qwen3.5:9b"
    assert llm.chat_model_label().startswith("Ollama")

    # deepseek 路径：无 key 必须明确报错（而非静默退化）
    monkeypatch.setattr(config, "MODEL_PROVIDER", "deepseek")
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "")
    llm.get_chat_model.cache_clear()
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        llm.get_chat_model()

    # deepseek 路径：有 key 时构造云端客户端（仅构造，不调用）
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "sk-fake-not-used")
    monkeypatch.setattr(config, "DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setattr(config, "DEEPSEEK_MODEL", "deepseek-flash")
    llm.get_chat_model.cache_clear()
    m2 = llm.get_chat_model()
    assert m2.openai_api_base.startswith("https://api.deepseek.com")
    assert llm.chat_model_label().startswith("DeepSeek")

    llm.get_chat_model.cache_clear()


def test_empty_sources_context_is_refusal_marker():
    """检索为空时喂给模型的 context 应为拒答标记，配合 system prompt 让模型拒答。"""
    from app import chat

    assert chat._format_context([]) == "无相关参考资料"
