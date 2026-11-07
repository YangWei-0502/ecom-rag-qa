"""智能问答页：多会话、流式回答、引用来源。"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import chat, llm  # noqa: E402
from app.db import init_db  # noqa: E402
from app.ui import sidebar_model_status, sidebar_user_box  # noqa: E402

st.set_page_config(page_title="智能问答", page_icon="💬", layout="wide")
init_db()

user_id = sidebar_user_box()
sidebar_model_status()
st.title("💬 智能问答")

# 确保有当前会话
if "rag_session" not in st.session_state or not chat.list_sessions(user_id):
    st.session_state.rag_session = chat.create_session(user_id)

with st.sidebar:
    st.subheader("会话")
    if st.button("✏️ 新建会话", use_container_width=True):
        st.session_state.rag_session = chat.create_session(user_id)
        st.rerun()
    for s in chat.list_sessions(user_id):
        c1, c2 = st.columns([4, 1])
        active = s["id"] == st.session_state.rag_session
        with c1:
            if st.button(s["title"], key=f"s_{s['id']}", type="primary" if active else "secondary",
                         use_container_width=True):
                st.session_state.rag_session = s["id"]
                st.rerun()
        with c2:
            if st.button("🗑", key=f"sd_{s['id']}"):
                chat.delete_session(user_id, s["id"])
                if s["id"] == st.session_state.rag_session:
                    rest = chat.list_sessions(user_id)
                    st.session_state.rag_session = rest[0]["id"] if rest else chat.create_session(user_id)
                st.rerun()

session_id = st.session_state.rag_session
sess = next((s for s in chat.list_sessions(user_id) if s["id"] == session_id), None)
if sess:
    st.caption(f"当前会话：{sess['title']} ｜ 更新于 {sess['updated_at']}")

import json  # noqa: E402

for m in chat.list_messages(session_id):
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m["sources"]:
            try:
                srcs = json.loads(m["sources"])
            except (ValueError, TypeError):
                srcs = []
            if srcs:
                with st.expander(f"📎 引用来源（{len(srcs)}）"):
                    for i, s in enumerate(srcs, 1):
                        st.markdown(f"**[{i}] {s.get('source','?')}**")
                        st.text((s.get("content", "")[:200] + "…") if len(s.get("content", "")) > 200 else s.get("content", ""))

prompt = st.chat_input("请输入商品咨询，例如：针织毛衣如何保养？")
if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    try:
        gen = chat.stream_answer(user_id, session_id, prompt)
        with st.chat_message("assistant"):
            st.write_stream(gen)
            if gen.sources:
                with st.expander(f"📎 引用来源（{len(gen.sources)}）"):
                    for i, s in enumerate(gen.sources, 1):
                        st.markdown(f"**[{i}] {s.get('source','?')}**")
                        st.text(s.get("content", "")[:200])
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        st.error(f"调用失败：{exc}（请检查模型状态后重试）")
