"""Session metadata: Session dataclass and SessionStore CRUD."""

from __future__ import annotations

import secrets
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    """Millisecond-epoch prefix + short random suffix — lexicographically sortable."""
    return f"{int(time.time() * 1000):013d}-{secrets.token_hex(4)}"


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


@dataclass
class Session:
    id: str
    title: str
    model: str
    created_at: str
    updated_at: str
    message_count: int


def _row_to_session(row: sqlite3.Row) -> Session:
    return Session(
        id=row["id"],
        title=row["title"],
        model=row["model"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        message_count=row["message_count"],
    )


class SessionStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        """Initialise with a shared SQLite connection."""
        self._conn = conn

    def create(self, model: str) -> Session:
        """Create a new session with an empty title and return it."""
        now = _now_iso()
        session = Session(
            id=_new_id(),
            title="",
            model=model,
            created_at=now,
            updated_at=now,
            message_count=0,
        )
        self._conn.execute(
            "INSERT INTO sessions (id, title, model, created_at, updated_at, message_count)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                session.id,
                session.title,
                session.model,
                session.created_at,
                session.updated_at,
                session.message_count,
            ),
        )
        self._conn.commit()
        return session

    def get(self, session_id: str) -> Session | None:
        """Return the session with the given id, or None if not found."""
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return _row_to_session(row) if row else None

    def list(self, limit: int = 20) -> list[Session]:
        """Return up to *limit* sessions ordered by updated_at descending."""
        rows = self._conn.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [_row_to_session(r) for r in rows]

    def set_title(self, session_id: str, title: str) -> None:
        """Update the title and bump updated_at."""
        self._conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, _now_iso(), session_id),
        )
        self._conn.commit()

    def set_model(self, session_id: str, model: str) -> None:
        """Update the model and bump updated_at."""
        self._conn.execute(
            "UPDATE sessions SET model = ?, updated_at = ? WHERE id = ?",
            (model, _now_iso(), session_id),
        )
        self._conn.commit()

    def touch(self, session_id: str, message_count: int) -> None:
        """Bump updated_at and set message_count."""
        self._conn.execute(
            "UPDATE sessions SET message_count = ?, updated_at = ? WHERE id = ?",
            (message_count, _now_iso(), session_id),
        )
        self._conn.commit()

    def delete(self, session_id: str) -> None:
        """Delete the session metadata and all checkpointer rows for this thread."""
        self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        if _table_exists(self._conn, "checkpoints"):
            self._conn.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?", (session_id,)
            )
        if _table_exists(self._conn, "writes"):
            self._conn.execute("DELETE FROM writes WHERE thread_id = ?", (session_id,))
        self._conn.commit()
