# 02 — Persistence & sessions

## Goal

A local SQLite layer that holds two things: LangGraph's checkpoint state (full conversation
state per session) and a small `sessions` metadata table for the `/session` picker.

## Depends on

- 01 (uses `data_dir()` for the DB path).

## Deliverables

- `termagent/store/db.py` — open the DB, build the `SqliteSaver` checkpointer.
- `termagent/store/sessions.py` — `Session` model + `SessionStore`.
- `tests/test_sessions.py`.

## Design

One SQLite file at `data_dir() / "termagent.db"`. Two concerns share it:

1. **Checkpointer (LangGraph-managed).** `SqliteSaver` from `langgraph-checkpoint-sqlite`
   manages its own tables. The session id is used as the LangGraph `thread_id`. The graph
   (spec 06) is compiled with this checkpointer, which gives conversation history and
   time-travel (used by `/undo` in spec 08) for free.
2. **Session metadata (ours).** The checkpointer does not give us a clean "list of sessions
   with titles", so we keep one table:

```sql
CREATE TABLE IF NOT EXISTS sessions (
  id            TEXT PRIMARY KEY,   -- also the LangGraph thread_id
  title         TEXT NOT NULL,      -- derived from the first prompt
  model         TEXT NOT NULL,      -- provider last used in this session
  created_at    TEXT NOT NULL,      -- ISO 8601
  updated_at    TEXT NOT NULL,
  message_count INTEGER NOT NULL DEFAULT 0
);
```

Session id: a short sortable id (e.g. `uuid7`-style or `time-based + short random`). Title:
start as a truncation of the first user prompt (first ~60 chars); a nicer LLM-generated title
can come later but is out of scope here.

Connection note: `SqliteSaver` wants a `sqlite3.Connection` with `check_same_thread=False`.
The REPL is single-threaded, so a single shared connection is fine. Set
`PRAGMA journal_mode=WAL` for resilience.

## Interfaces

```python
@dataclass
class Session:
    id: str
    title: str
    model: str
    created_at: str
    updated_at: str
    message_count: int

class SessionStore:
    def __init__(self, conn: sqlite3.Connection): ...
    def create(self, model: str) -> Session: ...           # new id, default title ""
    def get(self, session_id: str) -> Session | None: ...
    def list(self, limit: int = 20) -> list[Session]: ...   # newest updated_at first
    def set_title(self, session_id: str, title: str) -> None: ...
    def set_model(self, session_id: str, model: str) -> None: ...
    def touch(self, session_id: str, message_count: int) -> None: ...  # bump updated_at
    def delete(self, session_id: str) -> None: ...          # also delete checkpoints for thread

def open_db() -> sqlite3.Connection: ...
def build_checkpointer(conn: sqlite3.Connection) -> SqliteSaver: ...
```

`delete` must remove both the metadata row **and** the checkpointer rows for that `thread_id`
so a deleted session leaves nothing behind.

## Acceptance criteria

- DB and tables are created on first open; reopening is idempotent.
- `create` → `get` → `list` round-trips; `list` orders by `updated_at` descending.
- `delete` removes the metadata row and the checkpoint rows for that thread (verify the
  checkpointer returns no state for the deleted thread).
- Tests use a temp-file DB (not `:memory:`, since two connections/threads may be involved).

## Out of scope / deferred

- LLM-generated titles. Use the truncated-prompt title for now.
- Any agent logic — this spec is storage only.
