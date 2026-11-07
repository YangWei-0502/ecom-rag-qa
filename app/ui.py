"""多页面共享：用户身份与模型状态。"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# 让 app 包可被导入（页面脚本在 app/pages 下，层级不同）
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app import config, llm  # noqa: E402


def current_user() -> str:
    """所有页面共享的用户身份（侧边栏输入，默认本地用户）。"""
    if "user_id" not in st.session_state:
        st.session_state.user_id = config.DEFAULT_USER_ID
    return st.session_state.user_id


def sidebar_user_box() -> str:
    with st.sidebar:
        st.subheader("用户")
        uid = st.text_input("用户 ID（多用户隔离）", value=current_user(), key="user_id_input")
        st.session_state.user_id = uid.strip() or config.DEFAULT_USER_ID
        st.divider()
    return st.session_state.user_id


def sidebar_model_status() -> None:
    with st.sidebar:
        st.subheader("模型状态")
        st.write(f"**对话模型**：{llm.chat_model_label()}")
        st.write(f"**嵌入模型**：{llm.embed_model_label()}（本地）")
        if st.button("检测连接", use_container_width=True):
            with st.spinner("检测中…"):
                ok, msg = llm.health_check()
            (st.success if ok else st.warning)(msg)
        st.caption("切换模型请修改 .env 后重启；切换嵌入模型需在知识库页重建索引。")
