"""知识库管理页：上传、列表、删除、索引重建。"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app import knowledge, llm  # noqa: E402
from app.db import init_db  # noqa: E402
from app.document_loader import DocumentError  # noqa: E402
from app.ui import sidebar_model_status, sidebar_user_box  # noqa: E402

st.set_page_config(page_title="知识库管理", page_icon="📚", layout="wide")
init_db()

user_id = sidebar_user_box()
sidebar_model_status()
st.title("📚 知识库管理")
st.caption(f"嵌入模型：{llm.embed_model_label()} ｜ 支持 .txt / .md / .pdf / .docx，单文件 ≤10MB，块级去重")

tab_upload, tab_docs, tab_tool = st.tabs(["上传文档", "文档列表", "检索测试/索引"])

with tab_upload:
    files = st.file_uploader(
        "可多选上传",
        type=["txt", "md", "pdf", "docx"],
        accept_multiple_files=True,
    )
    if files and st.button("载入知识库", type="primary"):
        for f in files:
            try:
                with st.spinner(f"正在解析并入库 {f.name}…"):
                    r = knowledge.ingest_file(user_id, f.name, f.getvalue())
                icon = "⏭️" if r["status"] == "skipped" else "✅"
                st.write(f"{icon} {f.name}：{r['message']}")
            except DocumentError as exc:
                st.error(f"{f.name}：{exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"{f.name} 入库失败：{exc}")

with tab_docs:
    docs = knowledge.list_documents(user_id)
    if not docs:
        st.info("暂无文档，请先上传。")
    else:
        import pandas as pd

        st.dataframe(
            pd.DataFrame(
                [{"文件名": d["filename"], "内容块": d["chunk_count"], "状态": d["status"],
                  "入库时间": d["created_at"], "doc_id": d["id"]} for d in docs]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.divider()
        for d in docs:
            c1, c2 = st.columns([5, 1])
            c1.write(f"**{d['filename']}**（{d['chunk_count']} 块，{d['created_at']}）")
            if c2.button("删除", key=f"del_{d['id']}"):
                knowledge.delete_document(user_id, d["id"])
                st.rerun()

with tab_tool:
    st.subheader("检索测试")
    q = st.text_input("输入问题（不调用大模型，只看检索命中）")
    if q:
        hits = knowledge.search(user_id, q)
        if not hits:
            st.warning("未命中任何内容（可尝试补充知识文档或降低阈值）")
        for i, h in enumerate(hits, 1):
            st.markdown(f"**[{i}] {h['source']}** ｜ 向量距离 `{h['distance']}`")
            st.text(h["content"][:200])
    st.divider()
    st.subheader("索引重建")
    st.write("切换嵌入模型、向量库损坏或怀疑数据不一致时，以 SQLite 为准全量重建向量库。")
    if st.button("重建向量索引"):
        with st.spinner("重建中…"):
            n = knowledge.rebuild_index()
        st.success(f"重建完成，共 {n} 个内容块")
