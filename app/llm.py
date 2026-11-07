"""模型接入：DeepSeek / Ollama 双模型（OpenAI 兼容），嵌入统一走本地 Ollama。"""
from __future__ import annotations

import functools

from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI

from . import config


@functools.lru_cache(maxsize=1)
def get_chat_model() -> ChatOpenAI:
    if config.MODEL_PROVIDER == "ollama":
        return ChatOpenAI(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            api_key="ollama",
            temperature=config.LLM_TEMPERATURE,
            timeout=config.LLM_TIMEOUT_SECONDS,
        )
    if not config.DEEPSEEK_API_KEY:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，请在 .env 中填写或把 MODEL_PROVIDER 改为 ollama")
    return ChatOpenAI(
        model=config.DEEPSEEK_MODEL,
        base_url=config.DEEPSEEK_BASE_URL,
        api_key=config.DEEPSEEK_API_KEY,
        temperature=config.LLM_TEMPERATURE,
        timeout=config.LLM_TIMEOUT_SECONDS,
    )


@functools.lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    """嵌入统一使用本地 Ollama（免费、无配额、隐私不出本机）。"""
    return OllamaEmbeddings(
        model=config.OLLAMA_EMBED_MODEL,
        base_url=config.OLLAMA_BASE_URL.replace("/v1", ""),
    )


def chat_model_label() -> str:
    return (
        f"Ollama / {config.OLLAMA_MODEL}"
        if config.MODEL_PROVIDER == "ollama"
        else f"DeepSeek / {config.DEEPSEEK_MODEL}"
    )


def embed_model_label() -> str:
    return f"Ollama / {config.OLLAMA_EMBED_MODEL}"


def health_check() -> tuple[bool, str]:
    """检测对话模型与嵌入模型是否可用。"""
    try:
        m = get_chat_model()
        m.invoke("ping")
        chat_ok = True
    except Exception as exc:  # noqa: BLE001
        return False, f"对话模型不可用：{exc}"
    try:
        get_embeddings().embed_query("测试")
        embed_ok = True
    except Exception as exc:  # noqa: BLE001
        if chat_ok:
            return True, f"对话模型已连接；嵌入模型未就绪（请 ollama pull {config.OLLAMA_EMBED_MODEL}）：{exc}"
        return False, f"嵌入模型不可用：{exc}"
    return True, "对话模型与嵌入模型均已连接"
