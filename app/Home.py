"""电商智能客服 RAG · 总览页。运行：streamlit run app/Home.py"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import chat, knowledge, llm  # noqa: E402
from app.db import init_db  # noqa: E402
from app.ui import sidebar_model_status, sidebar_user_box  # noqa: E402

st.set_page_config(page_title="电商智能客服", page_icon="🛍️", layout="wide")
init_db()

st.title("🛍️ 电商智能客服 · RAG 知识库系统")
st.markdown(
    "多页面知识库问答系统：上传商品资料后，客服基于知识库回答问题并标注**引用来源**。"
    "对话与知识全部保存在本地 SQLite + Chroma，支持多用户隔离。"
)

user_id = sidebar_user_box()
sidebar_model_status()

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("知识文档数", knowledge.count_documents(user_id))
with c2:
    st.metric("历史会话数", len(chat.list_sessions(user_id)))
with c3:
    st.metric("检索 Top-K", knowledge.config.RETRIEVER_TOP_K)

st.divider()
st.markdown(
    """
    **使用步骤**
    1. 左侧「📚 知识库」上传 .txt / .md / .pdf / .docx 商品资料（尺码、洗涤、售后等）；
    2. 到「💬 智能问答」向客服提问，回答下方会显示引用来源；
    3. 在「🗂️ 会话记录」查看、重命名、导出或删除历史对话。
    """
)
st.caption(f"对话模型：{llm.chat_model_label()} ｜ 嵌入模型：{llm.embed_model_label()}")
