"""Tests for termagent.store: open_db, build_checkpointer, SessionStore."""

from __future__ import annotations

import time

import pytest

from termagent.store.db import build_checkpointer, open_db
from termagent.store.sessions import SessionStore


@pytest.fixture()
def db(tmp_path):
    """Temp-file SQLite connection shared by db + checkpointer."""
    conn = open_db(tmp_path / "termagent.db")
    yield conn
    conn.close()


@pytest.fixture()
def store(db):
    return SessionStore(db)


@pytest.fixture()
def checkpointer(db):
    return build_checkpointer(db)


# ---------------------------------------------------------------------------
# DB bootstrap
# ---------------------------------------------------------------------------


def test_open_db_creates_sessions_table(db):
    row = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sessions'"
    ).fetchone()
    assert row is not None


def test_open_db_idempotent(tmp_path):
    path = tmp_path / "termagent.db"
    conn1 = open_db(path)
    conn1.close()
    conn2 = open_db(path)
    conn2.close()


def test_build_checkpointer_creates_checkpoint_tables(db, checkpointer):
    for table in ("checkpoints", "writes"):
        row = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        assert row is not None, f"table '{table}' not found"


# ---------------------------------------------------------------------------
# create / get round-trip
# ---------------------------------------------------------------------------


def test_create_get_roundtrip(store):
    s = store.create("ollama")
    got = store.get(s.id)
    assert got == s


def test_create_defaults(store):
    s = store.create("groq")
    assert s.title == ""
    assert s.message_count == 0
    assert s.created_at == s.updated_at
    assert s.model == "groq"


def test_get_missing_returns_none(store):
    assert store.get("nonexistent-id") is None


# ---------------------------------------------------------------------------
# list ordering
# ---------------------------------------------------------------------------


def test_list_orders_by_updated_at_desc(store):
    a = store.create("ollama")
    time.sleep(0.01)
    b = store.create("ollama")
    time.sleep(0.01)
    store.touch(a.id, 3)  # bump a to top

    result = store.list()
    ids = [s.id for s in result]
    assert ids.index(a.id) < ids.index(b.id)


def test_list_respects_limit(store):
    for _ in range(5):
        store.create("ollama")
    assert len(store.list(limit=3)) == 3


# ---------------------------------------------------------------------------
# mutation methods
# ---------------------------------------------------------------------------


def test_set_title(store):
    s = store.create("ollama")
    time.sleep(0.01)
    store.set_title(s.id, "My session")
    got = store.get(s.id)
    assert got.title == "My session"
    assert got.updated_at > s.updated_at


def test_set_model(store):
    s = store.create("ollama")
    store.set_model(s.id, "groq")
    got = store.get(s.id)
    assert got.model == "groq"


def test_touch_bumps_count_and_updated_at(store):
    s = store.create("ollama")
    time.sleep(0.01)
    store.touch(s.id, 7)
    got = store.get(s.id)
    assert got.message_count == 7
    assert got.updated_at > s.updated_at


# ---------------------------------------------------------------------------
# delete removes metadata + checkpoints
# ---------------------------------------------------------------------------


def test_delete_removes_metadata(store):
    s = store.create("ollama")
    store.delete(s.id)
    assert store.get(s.id) is None


def test_delete_removes_checkpoint_rows(store, checkpointer):
    from langgraph.checkpoint.base import Checkpoint

    s = store.create("ollama")
    thread_id = s.id

    # Write a minimal checkpoint directly via the saver
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": "cp1",
        }
    }
    empty_checkpoint: Checkpoint = {
        "v": 1,
        "id": "cp1",
        "ts": "2026-01-01T00:00:00+00:00",
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    checkpointer.put(config, empty_checkpoint, {}, {})

    # Confirm it was written
    assert (
        checkpointer.get_tuple({"configurable": {"thread_id": thread_id}}) is not None
    )

    store.delete(thread_id)

    assert store.get(thread_id) is None
    assert checkpointer.get_tuple({"configurable": {"thread_id": thread_id}}) is None


def test_delete_without_checkpointer_is_safe(store):
    """delete() must not crash when checkpoints table doesn't exist yet."""
    s = store.create("ollama")
    store.delete(s.id)  # no checkpointer set up — should not raise
    assert store.get(s.id) is None
