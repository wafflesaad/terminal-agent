"""Tests for termagent.repl: REPL loop, streaming, interrupts, bookkeeping."""

from __future__ import annotations

import io
import sys
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rich.console import Console

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel as _Base,
)

from termagent.agent.graph import build_graph
from termagent.config import Settings
from termagent.repl import Repl
from termagent.store.db import build_checkpointer, open_db
from termagent.store.sessions import SessionStore


# ---------------------------------------------------------------------------
# Helpers shared with test_graph.py — kept local to avoid coupling
# ---------------------------------------------------------------------------


class FakeModel(_Base):
    """FakeMessagesListChatModel with a no-op bind_tools."""

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001
        return self


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
def checkpointer(db):
    return build_checkpointer(db)


@pytest.fixture()
def store(db):
    return SessionStore(db)


@pytest.fixture()
def settings():
    return Settings(timeout_seconds=10, max_output_chars=8000, strict_mode=True, auto_approve=False)


@pytest.fixture()
def console():
    return Console(file=io.StringIO(), highlight=False)


def _make_repl(
    settings: Settings,
    store: SessionStore,
    checkpointer,
    graph,
    console: Console,
    *,
    model_name: str = "test-model",
) -> Repl:
    session = store.create(model_name)
    return Repl(settings, store, checkpointer, graph, session, console=console)


# ---------------------------------------------------------------------------
# Read-only task: no approval prompt, output rendered
# ---------------------------------------------------------------------------


def test_readonly_task_no_confirm(tmp_path, db, checkpointer, store, settings, console):
    """A read-only command (echo) completes with no confirmation prompt."""
    fake = FakeModel(responses=[_tool_call("echo hello"), AIMessage(content="Done.")])
    graph = build_graph(lambda: fake, settings, checkpointer)
    repl = _make_repl(settings, store, checkpointer, graph, console)

    confirm_calls: list = []
    repl.confirm = lambda *a: confirm_calls.append(a) or False  # type: ignore[method-assign]

    repl.handle_task("say hello")

    assert not confirm_calls
    output = console.file.getvalue()  # type: ignore[attr-defined]
    assert "Done." in output


# ---------------------------------------------------------------------------
# Write task: user approves → side-effect happens
# ---------------------------------------------------------------------------


def test_write_task_approve_executes(tmp_path, db, checkpointer, store, settings, console):
    """Approving a write command executes it and produces the side-effect."""
    target = tmp_path / "created.txt"
    cmd = f"{sys.executable} -c \"open(r'{target}', 'w').close()\""

    fake = FakeModel(responses=[_tool_call(cmd), AIMessage(content="File created.")])
    graph = build_graph(lambda: fake, settings, checkpointer)
    repl = _make_repl(settings, store, checkpointer, graph, console)
    repl.confirm = lambda *a: True  # type: ignore[method-assign]

    repl.handle_task("create a file")

    assert target.exists()


# ---------------------------------------------------------------------------
# Write task: user declines → agent is told, loops back
# ---------------------------------------------------------------------------


def test_write_task_decline_loops_agent(tmp_path, db, checkpointer, store, settings, console):
    """Declining a write command emits a decline ToolMessage and the agent recovers."""
    cmd = f'{sys.executable} -c "import sys; sys.exit(0)"'

    fake = FakeModel(
        responses=[_tool_call(cmd), AIMessage(content="Understood, I won't run it.")]
    )
    graph = build_graph(lambda: fake, settings, checkpointer)
    repl = _make_repl(settings, store, checkpointer, graph, console)
    repl.confirm = lambda *a: False  # type: ignore[method-assign]

    repl.handle_task("run something")

    cfg = {"configurable": {"thread_id": repl._session.id}}
    snap = graph.get_state(cfg)
    tool_msgs = [m for m in snap.values["messages"] if isinstance(m, ToolMessage)]
    assert any("user declined" in m.content for m in tool_msgs)
    assert snap.values["messages"][-1].content == "Understood, I won't run it."


# ---------------------------------------------------------------------------
# Ctrl-C mid-task: aborts to prompt, no crash
# ---------------------------------------------------------------------------


def test_ctrl_c_mid_task_aborts_gracefully(store, settings, checkpointer, console):
    """KeyboardInterrupt during a task cancels it and returns to the prompt."""

    class KIGraph:
        def stream(self, *a, **kw):
            raise KeyboardInterrupt

        def get_state(self, cfg):  # pragma: no cover — never reached in this path
            snap = MagicMock()
            snap.next = []
            snap.values = {"messages": [], "cwd": "/tmp"}
            return snap

    session = store.create("test")
    repl = Repl(settings, store, checkpointer, KIGraph(), session, console=console)

    mock_session = MagicMock()
    mock_session.prompt.side_effect = ["ls", EOFError()]
    with patch("prompt_toolkit.PromptSession", return_value=mock_session):
        repl.run()

    output = console.file.getvalue()  # type: ignore[attr-defined]
    assert "cancelled" in output


# ---------------------------------------------------------------------------
# auto_approve: no confirm call, warning printed, side-effect happens
# ---------------------------------------------------------------------------


def test_auto_approve_runs_without_confirm(
    tmp_path, db, checkpointer, store, console
):
    """With auto_approve, the approval panel is skipped but a warning is shown."""
    auto_settings = Settings(timeout_seconds=10, max_output_chars=8000, auto_approve=True)
    target = tmp_path / "auto.txt"
    cmd = f"{sys.executable} -c \"open(r'{target}', 'w').close()\""

    fake = FakeModel(responses=[_tool_call(cmd), AIMessage(content="Done.")])
    graph = build_graph(lambda: fake, auto_settings, checkpointer)
    repl = _make_repl(auto_settings, store, checkpointer, graph, console)

    confirm_calls: list = []
    repl.confirm = lambda *a: confirm_calls.append(a) or True  # type: ignore[method-assign]

    repl.handle_task("create a file")

    assert not confirm_calls
    assert target.exists()
    output = console.file.getvalue()  # type: ignore[attr-defined]
    assert "auto-approve" in output.lower()


def test_auto_approve_blocked_still_refused(
    tmp_path, db, checkpointer, store, console
):
    """auto_approve does not bypass the blocklist: interactive programs are still refused."""
    auto_settings = Settings(timeout_seconds=10, max_output_chars=8000, auto_approve=True)

    fake = FakeModel(responses=[_tool_call("vim notes.txt"), AIMessage(content="Can't do that.")])
    graph = build_graph(lambda: fake, auto_settings, checkpointer)
    repl = _make_repl(auto_settings, store, checkpointer, graph, console)

    repl.handle_task("edit a file")

    cfg = {"configurable": {"thread_id": repl._session.id}}
    snap = graph.get_state(cfg)
    tool_msgs = [m for m in snap.values["messages"] if isinstance(m, ToolMessage)]
    assert any("interactive program" in m.content for m in tool_msgs)


# ---------------------------------------------------------------------------
# Session bookkeeping: touch and title after a task
# ---------------------------------------------------------------------------


def test_session_touch_and_title_after_task(
    tmp_path, db, checkpointer, store, settings, console
):
    """After a task, message_count advances and the title is set from the first prompt."""
    fake = FakeModel(responses=[_tool_call("echo hi"), AIMessage(content="Done.")])
    graph = build_graph(lambda: fake, settings, checkpointer)
    session = store.create("test-model")
    repl = Repl(settings, store, checkpointer, graph, session, console=console)

    assert session.message_count == 0
    assert session.title == ""

    repl.handle_task("say hi")

    updated = store.get(session.id)
    assert updated is not None
    assert updated.message_count > 0
    assert updated.title == "say hi"
    # In-memory session object is also updated.
    assert repl._session.title == "say hi"


def test_title_set_only_once(tmp_path, db, checkpointer, store, settings, console):
    """The session title is taken from the first prompt only."""
    fake = FakeModel(
        responses=[
            _tool_call("echo first"),
            AIMessage(content="First done."),
            _tool_call("echo second"),
            AIMessage(content="Second done."),
        ]
    )
    graph = build_graph(lambda: fake, settings, checkpointer)
    session = store.create("test-model")
    repl = Repl(settings, store, checkpointer, graph, session, console=console)

    repl.handle_task("first task")
    repl.handle_task("second task")

    updated = store.get(session.id)
    assert updated is not None
    assert updated.title == "first task"
