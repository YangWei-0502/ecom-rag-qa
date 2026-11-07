"""电商 RAG 改造版配置：路径、双模型、检索与上传参数。"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ---- 路径 ----
# RAG_DATA_DIR 允许评测/测试把数据目录指到隔离位置，不污染生产库
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", str(BASE_DIR / "data")))
DB_PATH = DATA_DIR / "rag.db"
CHROMA_DIR = DATA_DIR / "chroma_db"
UPLOAD_DIR = DATA_DIR / "uploads"
SEED_DOC_DIR = BASE_DIR / "data" / "knowledge_docs"
LOG_DIR = BASE_DIR / "logs"
for d in (DATA_DIR, CHROMA_DIR, UPLOAD_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- 模型 ----
# 默认本地 Ollama（免费、隐私不出本机）；如需云端可在 .env 切到 deepseek 并填 DEEPSEEK_API_KEY
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama").strip().lower()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash").strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b").strip()
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:4b").strip()

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
LLM_MAX_HISTORY_MESSAGES = int(os.getenv("LLM_MAX_HISTORY_MESSAGES", "20"))

# ---- 检索 ----
COLLECTION_NAME = "rag"
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "6"))
# 绝对距离上限（Chroma L2，归一化向量约 2-2*cosine）。
# 实测：相关问题 top 距离 0.58~0.95，离题问题 top 距离 1.37~1.46，取 1.2 可干净分隔
SCORE_DISTANCE_THRESHOLD = float(os.getenv("SCORE_DISTANCE_THRESHOLD", "1.2"))
# 相对阈值：在最优命中距离上放宽该值，避免无关文档稀释上下文（0 表示不启用）
SCORE_RELATIVE_SPREAD = float(os.getenv("SCORE_RELATIVE_SPREAD", "0.35"))
# 默认开混合（评测实测：MRR 0.972→1.0，召回/拒答无回归；见 evals/results/latest.md）
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").strip().lower() == "true"
# BM25 通道：候选块至少命中多少个查询词才算有效（单字分词下防虚词误召回）
BM25_MIN_MATCHED_TOKENS = int(os.getenv("BM25_MIN_MATCHED_TOKENS", "2"))
# BM25 绝对地板（二元分词实测：相关 2.6~4.3，离题 ≤1.45）
BM25_MIN_SCORE = float(os.getenv("BM25_MIN_SCORE", "2.0"))

# ---- 重排序（默认开关由 evals/ 实测收益决定，见设计文档 §14） ----
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "false").strip().lower() == "true"
# 送入重排的候选条数（从融合后结果里取前 N 重排，再截断 top_k）
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "10"))
# LLM 重排（Ollama 对小候选集打分；默认关，评测对比后再定）
ENABLE_LLM_RERANK = os.getenv("ENABLE_LLM_RERANK", "false").strip().lower() == "true"
LLM_RERANK_TIMEOUT_SECONDS = int(os.getenv("LLM_RERANK_TIMEOUT_SECONDS", "30"))

# ---- API 限流（进程内滑动窗口，0=关闭） ----
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8002"))

# ---- 分块 ----
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# ---- 上传 ----
ALLOWED_UPLOAD_EXT = {".txt", ".md", ".pdf", ".docx"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "10485760"))  # 10MB

# ---- 多用户 ----
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "local_user")
