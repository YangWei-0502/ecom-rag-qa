"""候选重排序：确定性特征重排（零网络、可离线测）+ 可选 Ollama LLM 打分。

设计原则：
- 确定性重排只用候选块自带信号（归一化距离、查询词重叠、标题命中），不发网络请求，
  因此可在离线单测中确定性验证；
- LLM 重排为可选增强：对 <=RERANK_CANDIDATES 条小候选集一次性请模型打分，
  带超时与解析容错，失败时降级为原序（不让重排把系统搞挂）；
- 是否默认开启由 evals/ 实测收益决定，不在代码里拍脑袋。
"""
from __future__ import annotations

import json
import re

from . import config


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text)
    han = re.findall(r"[一-鿿]", text)
    bigrams = [han[i] + han[i + 1] for i in range(len(han) - 1)]
    return words + bigrams


def _feature_scores(question: str, candidates: list[dict]) -> list[float]:
    """为每条候选计算确定性重排分（越大越靠前）。"""
    q_tokens = set(_tokenize(question))
    if not q_tokens:
        q_tokens = {question}
    overlaps: list[float] = []
    dists: list[float] = []
    for c in candidates:
        c_tokens = set(_tokenize(c.get("content", "")))
        overlaps.append(len(q_tokens & c_tokens) / max(len(q_tokens), 1))
        d = c.get("distance")
        dists.append(float(d) if d is not None else None)

    max_overlap = max(overlaps) or 1.0
    out = []
    for c, ov, d in zip(candidates, overlaps, dists):
        # 距离分：L2 越小越好，映射到 [0,1]，阈值内才有效；无距离（BM25 独占命中）给中性分
        if d is None:
            s_dist = 0.4
        else:
            s_dist = max(0.0, 1.0 - d / max(config.SCORE_DISTANCE_THRESHOLD, 0.1))
        s_ov = ov / max_overlap
        # 标题/来源命中加分（商品问答里标题是强信号）
        title_blob = f"{c.get('title', '')} {c.get('source', '')}"
        s_title = 0.15 if q_tokens & set(_tokenize(title_blob)) else 0.0
        out.append(round(0.5 * s_dist + 0.5 * s_ov + s_title, 5))
    return out


def deterministic_rerank(question: str, candidates: list[dict]) -> list[dict]:
    """原地打 rerank_score 并按分重排，返回新列表。"""
    if not candidates:
        return candidates
    scores = _feature_scores(question, candidates)
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    out = []
    for c, s in scored:
        c["rerank_score"] = s
        out.append(c)
    return out


def llm_rerank(question: str, candidates: list[dict]) -> list[dict]:
    """用 Ollama 对话模型对候选打 0-5 相关性分；任何失败/超时都降级为原序。"""
    if not candidates:
        return candidates
    try:
        from .llm import get_chat_model

        block = "\n".join(
            f"[{i}] {c.get('content','')[:300]}" for i, c in enumerate(candidates)
        )
        prompt = (
            f"问题：{question}\n\n以下是候选资料块，请只判断每块与问题的相关性，"
            f"给出 0(完全无关)到 5(直接回答)的整数分。\n{block}\n\n"
            '只输出 JSON 数组，如 [5,3,0]，不要任何解释。'
        )
        text = get_chat_model().invoke(prompt).content or ""
        m = re.search(r"\[[\d,\s]+\]", text)
        scores = json.loads(m.group(0)) if m else None
        if not scores or len(scores) != len(candidates):
            raise ValueError("评分数量不匹配")
        scored = sorted(
            zip(candidates, scores), key=lambda x: x[1], reverse=True
        )
        return [{**c, "rerank_score_llm": float(s)} for c, s in scored]
    except Exception:  # noqa: BLE001  重排失败不阻塞主流程
        return candidates


def maybe_rerank(question: str, candidates: list[dict], top_k: int, enabled: bool) -> list[dict]:
    """按开关执行重排并截断 top_k；enabled=False 时直接截断返回。"""
    if not enabled:
        return candidates[:top_k]
    ranked = deterministic_rerank(question, candidates)
    if config.ENABLE_LLM_RERANK:
        ranked = llm_rerank(question, ranked)
    return ranked[:top_k]
