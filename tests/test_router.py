"""Tests for termagent.router + command handlers (spec 08)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel as _Base,
)

from termagent.agent.graph import build_graph
from termagent.commands import COMMANDS, REGISTRY
from termagent.config import Settings
from termagent.context import Context
from termagent.router import dispatch, parse
from termagent.store.db import build_checkpointer, open_db
from termagent.store.sessions import SessionStore


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeModel(_Base):
    def bind_tools(self, tools, **kwargs):  # noqa: ANN001
        return self


class FakeConsole:
    """Captures printed lines and serves queued input() answers."""

    def __init__(self, inputs: list[str] | None = None) -> None:
        self.lines: list[str] = []
        self.inputs = list(inputs or [])

    def print(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        self.lines.append(" ".join(str(a) for a in args))

    def input(self, prompt: str = "") -> str:
        return self.inputs.pop(0)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _tool_call(cmd: str, call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"id": call_id, "name": "run_shell", "args": {"command": cmd}}],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path):
    conn = open_db(tmp_path / "termagent.db")
    yield conn
    conn.close()


@pytest.fixture()
def settings():
    return Settings(timeout_seconds=10, max_output_chars=8000, auto_approve=False)


def _make_ctx(db, settings, *, inputs=None, responses=None, cwd="/tmp") -> Context:
    store = SessionStore(db)
    checkpointer = build_checkpointer(db)
    fake = FakeModel(responses=responses or [AIMessage(content="ok")])
    graph = build_graph(lambda: fake, settings, checkpointer)
    session = store.create("test-model")
    return Context(
        settings=settings,
        store=store,
        checkpointer=checkpointer,
        graph=graph,
        session=session,
        provider_name="ollama",
        model=fake,
        cwd=cwd,
        auto_approve=settings.auto_approve,
        console=FakeConsole(inputs=inputs),
    )


def _run_task(ctx: Context, text: str) -> None:
    """Drive one full task through the graph to create checkpoint state."""
    cfg = ctx.thread_config()
    payload = {"messages": [HumanMessage(content=text)], "cwd": ctx.cwd}
    for _ in ctx.graph.stream(payload, config=cfg, stream_mode="values"):
        pass


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line,expected",
    [
        ("/model groq", ("model", "groq")),
        ("/help", ("help", "")),
        ("/MODEL Groq", ("model", "Groq")),
        ("  /new  ", None),  # leading space → not a command
        ("hello world", None),
        ("/", ("", "")),
        ("/rename my long title", ("rename", "my long title")),
    ],
)
def test_parse(line, expected):
    assert parse(line) == expected


def test_registry_names_unique_and_callable():
    names = [c.name for c in REGISTRY]
    assert len(names) == len(set(names))
    assert all(callable(c.handler) for c in REGISTRY)


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def test_unknown_command_prints_hint(db, settings):
    ctx = _make_ctx(db, settings)
    result = dispatch(ctx, "/frobnicate now")
    assert result.action == "continue"
    assert "unknown command" in ctx.console.text


def test_dispatch_prompt_returns_none():
    # parse() of a non-slash line is None; dispatch is only fed slash lines by the REPL.
    assert parse("just a prompt") is None


# ---------------------------------------------------------------------------
# flag toggles
# ---------------------------------------------------------------------------


def test_auto_toggle_warns_on_enable(db, settings):
    ctx = _make_ctx(db, settings)
    dispatch(ctx, "/auto")
    assert ctx.auto_approve is True
    assert "auto-approve" in ctx.console.text.lower()
    dispatch(ctx, "/auto")
    assert ctx.auto_approve is False


def test_verbose_toggle(db, settings):
    ctx = _make_ctx(db, settings)
    dispatch(ctx, "/verbose")
    assert ctx.verbose is True
    dispatch(ctx, "/verbose")
    assert ctx.verbose is False


def test_help_lists_every_command(db, settings):
    ctx = _make_ctx(db, settings)
    dispatch(ctx, "/help")
    text = ctx.console.text
    assert all(f"/{c.name}" in text for c in REGISTRY)


def test_exit_returns_exit(db, settings):
    ctx = _make_ctx(db, settings)
    assert dispatch(ctx, "/exit").action == "exit"
    assert dispatch(ctx, "/quit").action == "exit"


# ---------------------------------------------------------------------------
# retry
# ---------------------------------------------------------------------------


def test_retry_replays_last_prompt(db, settings):
    ctx = _make_ctx(db, settings)
    ctx.last_prompt = "list files"
    result = dispatch(ctx, "/retry")
    assert result.action == "prompt"
    assert result.prompt == "list files"


def test_retry_without_history(db, settings):
    ctx = _make_ctx(db, settings)
    result = dispatch(ctx, "/retry")
    assert result.action == "continue"
    assert "nothing to retry" in ctx.console.text


# ---------------------------------------------------------------------------
# session lifecycle
# ---------------------------------------------------------------------------


def test_new_creates_session(db, settings):
    ctx = _make_ctx(db, settings)
    old_id = ctx.session.id
    dispatch(ctx, "/new")
    assert ctx.session.id != old_id
    assert ctx.store.get(ctx.session.id) is not None


def test_rename_sets_title(db, settings):
    ctx = _make_ctx(db, settings)
    dispatch(ctx, "/rename my session")
    assert ctx.session.title == "my session"
    assert ctx.store.get(ctx.session.id).title == "my session"


def test_session_switch_rebinds_thread(db, settings):
    ctx = _make_ctx(db, settings)
    first_id = ctx.session.id
    dispatch(ctx, "/new")  # create a second session, now current
    second_id = ctx.session.id
    assert first_id != second_id

    # Pick index of the first (older) session from the listing.
    sessions = ctx.store.list()
    idx = next(i for i, s in enumerate(sessions) if s.id == first_id)
    ctx.console.inputs = [str(idx)]
    dispatch(ctx, "/session")

    assert ctx.session.id == first_id
    assert ctx.thread_config()["configurable"]["thread_id"] == first_id


def test_delete_confirms_and_clears_checkpoints(db, settings):
    ctx = _make_ctx(db, settings, inputs=["y"])
    _run_task(ctx, "echo hi")  # writes checkpoint rows for this thread
    thread_id = ctx.session.id

    rows_before = db.execute(
        "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (thread_id,)
    ).fetchone()[0]
    assert rows_before > 0

    dispatch(ctx, "/delete")  # deletes current, then makes a new session

    assert ctx.store.get(thread_id) is None
    assert ctx.session.id != thread_id
    rows_after = db.execute(
        "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (thread_id,)
    ).fetchone()[0]
    assert rows_after == 0


def test_delete_cancelled_keeps_session(db, settings):
    ctx = _make_ctx(db, settings, inputs=["n"])
    sid = ctx.session.id
    dispatch(ctx, "/delete")
    assert ctx.store.get(sid) is not None
    assert ctx.session.id == sid


# ---------------------------------------------------------------------------
# undo
# ---------------------------------------------------------------------------


def test_undo_drops_last_exchange(db, settings):
    ctx = _make_ctx(
        db, settings, responses=[AIMessage(content="hello back")]
    )
    _run_task(ctx, "say hello")

    before = len(ctx.graph.get_state(ctx.thread_config()).values["messages"])
    assert before > 0

    dispatch(ctx, "/undo")

    after = len(ctx.graph.get_state(ctx.thread_config()).values["messages"])
    assert after < before


def test_undo_with_nothing(db, settings):
    ctx = _make_ctx(db, settings)
    dispatch(ctx, "/undo")
    assert "nothing to undo" in ctx.console.text


# ---------------------------------------------------------------------------
# cd
# ---------------------------------------------------------------------------


def test_cd_updates_cwd_and_state(db, settings, tmp_path):
    ctx = _make_ctx(db, settings, cwd=str(tmp_path))
    sub = tmp_path / "sub"
    sub.mkdir()
    dispatch(ctx, "/cd sub")
    assert ctx.cwd == str(sub)
    state = ctx.graph.get_state(ctx.thread_config())
    assert state.values["cwd"] == str(sub)


def test_cd_missing_dir_errors(db, settings, tmp_path):
    ctx = _make_ctx(db, settings, cwd=str(tmp_path))
    before = ctx.cwd
    dispatch(ctx, "/cd does-not-exist")
    assert ctx.cwd == before  # unchanged
    assert "no such file" in ctx.console.text.lower()


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def test_model_no_arg_shows_current(db, settings):
    ctx = _make_ctx(db, settings)
    with patch("termagent.commands.model.available_models", return_value={"ollama": ["m"], "groq": ["g"]}):
        dispatch(ctx, "/model")
    assert "current model" in ctx.console.text
    assert "ollama" in ctx.console.text


def test_model_swap_updates_session_metadata(db, settings):
    ctx = _make_ctx(db, settings)
    new_model = FakeModel(responses=[AIMessage(content="x")])
    with patch("termagent.commands.model.build_model", return_value=new_model) as bm:
        dispatch(ctx, "/model groq")
        bm.assert_called_once()
    assert ctx.provider_name == "groq"
    assert ctx.model is new_model
    assert ctx.store.get(ctx.session.id).model == settings.groq_model


def test_model_invalid_provider(db, settings):
    ctx = _make_ctx(db, settings)
    dispatch(ctx, "/model gpt4")
    assert ctx.provider_name == "ollama"  # unchanged
    assert "unknown provider" in ctx.console.text


# ---------------------------------------------------------------------------
# context / history / export
# ---------------------------------------------------------------------------


def test_context_reports_estimate(db, settings):
    ctx = _make_ctx(db, settings, responses=[AIMessage(content="hello back")])
    _run_task(ctx, "say hello")
    dispatch(ctx, "/context")
    line = ctx.console.text
    assert "tokens" in line
    assert str(settings.context_token_budget) in line


def test_history_prints_messages(db, settings):
    ctx = _make_ctx(db, settings, responses=[AIMessage(content="hello back")])
    _run_task(ctx, "say hello")
    dispatch(ctx, "/history")
    text = ctx.console.text
    assert "say hello" in text
    assert "hello back" in text


def test_export_writes_file(db, settings, tmp_path):
    ctx = _make_ctx(
        db, settings, cwd=str(tmp_path), responses=[AIMessage(content="hello back")]
    )
    _run_task(ctx, "say hello")

    dispatch(ctx, "/export md")
    md = tmp_path / f"termagent-{ctx.session.id}.md"
    assert md.exists()
    assert "hello back" in md.read_text(encoding="utf-8")

    dispatch(ctx, "/export json")
    js = tmp_path / f"termagent-{ctx.session.id}.json"
    assert js.exists()
    assert "say hello" in js.read_text(encoding="utf-8")
