"""SQLite 存储：用户、文档、文档块、会话与消息（WAL + 外键）。"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from . import config

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    filename     TEXT NOT NULL,
    file_hash    TEXT NOT NULL,
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'ready',
    created_at   TEXT NOT NULL,
    UNIQUE(user_id, file_hash)
);
CREATE TABLE IF NOT EXISTS document_chunks (
    id           TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,
    content      TEXT NOT NULL,
    chunk_hash   TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    UNIQUE(doc_id, chunk_index)
);
CREATE TABLE IF NOT EXISTS chat_sessions (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_messages (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user','assistant')),
    content     TEXT NOT NULL,
    sources     TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON document_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_msgs_session ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON chat_sessions(user_id);
"""


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def db_cursor(commit: bool = False) -> Iterator[sqlite3.Cursor]:
    conn = get_conn()
    try:
        cur = conn.cursor()
        yield cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# 第二轮企业级改造：documents 表新增元数据列（幂等迁移）
_DOC_EXTRA_COLUMNS = {
    "title": "TEXT NOT NULL DEFAULT ''",
    "category": "TEXT NOT NULL DEFAULT ''",
    "tags": "TEXT NOT NULL DEFAULT ''",
    "summary": "TEXT NOT NULL DEFAULT ''",
    "version": "INTEGER NOT NULL DEFAULT 1",
    "error_message": "TEXT NOT NULL DEFAULT ''",
    "updated_at": "TEXT NOT NULL DEFAULT ''",
}


def _migrate_documents(cur: sqlite3.Cursor) -> None:
    existing = {r["name"] for r in cur.execute("PRAGMA table_info(documents)").fetchall()}
    for col, ddl in _DOC_EXTRA_COLUMNS.items():
        if col not in existing:
            cur.execute(f"ALTER TABLE documents ADD COLUMN {col} {ddl}")


def init_db() -> None:
    with db_cursor(commit=True) as cur:
        cur.executescript(SCHEMA_SQL)
        _migrate_documents(cur)
        cur.execute(
            "INSERT OR IGNORE INTO users (id, name, created_at) VALUES (?,?,datetime('now'))",
            (config.DEFAULT_USER_ID, "本地用户"),
        )
