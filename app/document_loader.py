"""多格式文档读取：.txt / .md / .pdf / .docx，统一返回纯文本。"""
from __future__ import annotations

import io
from pathlib import Path

from . import config


class DocumentError(Exception):
    pass


def _read_text(raw: bytes) -> str:
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise DocumentError("文本编码无法识别，请另存为 UTF-8 后再上传")


def _read_pdf(raw: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw))
    parts = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(parts).strip()


def _read_docx(raw: bytes) -> str:
    from docx import Document

    document = Document(io.BytesIO(raw))
    return "\n".join(p.text for p in document.paragraphs).strip()


def parse_file(filename: str, raw: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in config.ALLOWED_UPLOAD_EXT:
        raise DocumentError(f"仅支持 {', '.join(sorted(config.ALLOWED_UPLOAD_EXT))} 格式")
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise DocumentError(f"文件超过 {config.MAX_UPLOAD_BYTES // 1024 // 1024}MB 限制")
    if len(raw) == 0:
        raise DocumentError("文件为空")
    if ext in (".txt", ".md"):
        return _read_text(raw)
    if ext == ".pdf":
        return _read_pdf(raw)
    if ext == ".docx":
        return _read_docx(raw)
    raise DocumentError("不支持的格式")
