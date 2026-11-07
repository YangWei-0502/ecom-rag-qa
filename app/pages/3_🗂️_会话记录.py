"""会话记录页：查看、重命名、导出 Markdown、删除。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import chat  # noqa: E402
from app.db import init_db  # noqa: E402
from app.ui import sidebar_user_box  # noqa: E402

st.set_page_config(page_title="会话记录", page_icon="🗂️", layout="wide")
init_db()

user_id = sidebar_user_box()
st.title("🗂️ 会话记录")

sessions = chat.list_sessions(user_id)
if not sessions:
    st.info("暂无会话记录。")
    st.stop()

sid = st.selectbox(
    "选择会话",
    options=[s["id"] for s in sessions],
    format_func=lambda i: next(s["title"] for s in sessions if s["id"] == i),
)
sess = next(s for s in sessions if s["id"] == sid)
msgs = chat.list_messages(sid)

c1, c2, c3 = st.columns([2, 1, 1])
new_title = c1.text_input("会话标题", value=sess["title"])
if c2.button("保存标题"):
    chat.rename_session(user_id, sid, new_title)
    st.success("已保存")
if c3.button("删除会话", type="secondary"):
    chat.delete_session(user_id, sid)
    st.rerun()

st.divider()
for m in msgs:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m["sources"]:
            try:
                srcs = json.loads(m["sources"])
            except (ValueError, TypeError):
                srcs = []
            if srcs:
                st.caption("来源：" + "；".join(s.get("source", "?") for s in srcs))

# 导出 Markdown
md = [f"# 会话：{sess['title']}", ""]
for m in msgs:
    who = "我" if m["role"] == "user" else "客服"
    md.append(f"**{who}**：{m['content']}\n")
st.download_button(
    "⬇️ 导出 Markdown",
    data="\n".join(md),
    file_name=f"{sid}.md",
    mime="text/markdown",
)
