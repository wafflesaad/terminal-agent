"""SQLite connection bootstrap and LangGraph checkpointer setup."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from termagent import config

_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    model         TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0
);
"""


def open_db(path: Path | None = None) -> sqlite3.Connection:
    """Open (creating if needed) the termagent SQLite DB.

    Creates the sessions table if absent; idempotent on re-open.
    Accepts an optional path override so tests can use temp files.
    """
    if path is None:
        data = config.data_dir()
        data.mkdir(parents=True, exist_ok=True)
        path = data / "termagent.db"

    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_CREATE_SESSIONS)
    return conn


def build_checkpointer(conn: sqlite3.Connection) -> SqliteSaver:
    """Build and initialise a SqliteSaver on the shared connection.

    Calls .setup() immediately so the checkpoints/writes tables exist
    before any graph is compiled or any session is deleted.
    """
    saver = SqliteSaver(conn)
    saver.setup()
    return saver
