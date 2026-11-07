"""会话与消息（SQLite）+ RAG 问答（含引用来源、流式输出）。"""
from __future__ import annotations

import json
import uuid
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from . import config, knowledge
from .db import db_cursor, get_conn
from .llm import get_chat_model

SYSTEM_PROMPT = (
    "你是电商店铺的智能客服。请以提供的参考资料为依据回答问题，语气专业、简洁、友好。\n"
    "回答规则（严格遵守）：\n"
    "1. 参考资料中只要存在相关信息（包括数值、尺码、材质、颜色、操作步骤等），"
    "必须先直接、明确地给出资料中的答案，数值原样引用，不得回避或声称“暂无”；\n"
    "2. 顾客用词可能与资料不同（如“毛衣”对应“羊毛/羊绒/针织”，“卫裤”对应“运动裤”），"
    "按材质/品类做合理对应后依据资料回答；\n"
    "3. 回答具体、可操作（水温、洗涤剂、能否机洗、体重范围等），可分点；\n"
    "4. 只有当所有参考资料都与问题完全无关时，才回复“知识库中暂无该信息”，并建议查看商品详情页或联系人工客服。"
)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_session(user_id: str, title: str = "新会话") -> str:
    sid = str(uuid.uuid4())
    now = _now()
    with db_cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO chat_sessions (id, user_id, title, created_at, updated_at) VALUES (?,?,?,?,?)",
            (sid, user_id, title, now, now),
        )
    return sid


def list_sessions(user_id: str) -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM chat_sessions WHERE user_id=? ORDER BY updated_at DESC", (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def rename_session(user_id: str, session_id: str, title: str) -> None:
    with db_cursor(commit=True) as cur:
        cur.execute(
            "UPDATE chat_sessions SET title=?, updated_at=? WHERE id=? AND user_id=?",
            (title, _now(), session_id, user_id),
        )


def delete_session(user_id: str, session_id: str) -> None:
    with db_cursor(commit=True) as cur:
        cur.execute("DELETE FROM chat_messages WHERE session_id=?", (session_id,))
        cur.execute("DELETE FROM chat_sessions WHERE id=? AND user_id=?", (session_id, user_id))


def list_messages(session_id: str) -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT role, content, sources, created_at FROM chat_messages"
            " WHERE session_id=? ORDER BY created_at, rowid",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _add_message(session_id: str, role: str, content: str, sources: list[dict] | None = None) -> None:
    with db_cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO chat_messages (id, session_id, role, content, sources, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (str(uuid.uuid4()), session_id, role, content,
             json.dumps(sources, ensure_ascii=False) if sources else None, _now()),
        )
        cur.execute("UPDATE chat_sessions SET updated_at=? WHERE id=?", (_now(), session_id))


def _format_context(docs: list[dict]) -> str:
    if not docs:
        return "无相关参考资料"
    return "\n\n".join(
        f"[来源 {i + 1}：{d['source']}]\n{d['content']}" for i, d in enumerate(docs)
    )


def retrieve(user_id: str, question: str) -> list[dict]:
    return knowledge.search(user_id, question)


class _AnswerStream:
    """可迭代的回答流；.sources 携带本次检索到的引用来源。"""

    def __init__(self, sources, chain, chain_input, session_id, question):
        self.sources = sources
        self._chain = chain
        self._chain_input = chain_input
        self._session_id = session_id
        self._question = question

    def __iter__(self):
        _add_message(self._session_id, "user", self._question)
        full = ""
        for chunk in self._chain.stream(self._chain_input):
            token = getattr(chunk, "content", "")
            if token:
                full += token
                yield token
        _add_message(self._session_id, "assistant", full or "（模型未返回内容）", self.sources)


def stream_answer(user_id: str, session_id: str, question: str) -> _AnswerStream:
    """返回可迭代回答流（含 .sources）；迭代结束后自动落库问答与来源。"""
    sources = retrieve(user_id, question)
    # 当前问题尚未入库，所以库里消息即历史，取最近 N 条
    history_rows = list_messages(session_id)[-config.LLM_MAX_HISTORY_MESSAGES :]
    history = [
        HumanMessage(r["content"]) if r["role"] == "user" else AIMessage(r["content"])
        for r in history_rows
    ]
    chain = _build_chain(sources, history)
    chain_input = {"context": _format_context(sources), "history": history, "question": question}
    return _AnswerStream(sources, chain, chain_input, session_id, question)


def _build_chain(sources: list[dict], history: list):
    # 参考资料放入用户消息（部分模型对 system 中的长资料利用不充分）
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("user", "参考资料：\n{context}\n\n问题：{question}"),
        ]
    )
    return prompt | get_chat_model()


def answer(user_id: str, question: str, session_id: str | None = None) -> dict:
    """非流式问答（FastAPI 用）：返回 {answer, sources, refused}。

    若传入 session_id 则同 Streamlit 侧栏一样把问答落库；否则仅返回不落库。
    """
    sources = retrieve(user_id, question)
    history = []
    if session_id:
        history_rows = list_messages(session_id)[-config.LLM_MAX_HISTORY_MESSAGES :]
        history = [
            HumanMessage(r["content"]) if r["role"] == "user" else AIMessage(r["content"])
            for r in history_rows
        ]
    chain = _build_chain(sources, history)
    resp = chain.invoke({
        "context": _format_context(sources),
        "history": history,
        "question": question,
    })
    text = resp.content or "（模型未返回内容）"
    if session_id:
        _add_message(session_id, "user", question)
        _add_message(session_id, "assistant", text, sources)
    return {"answer": text, "sources": sources, "refused": not sources}
