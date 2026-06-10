"""Tests for termagent.agent: StateGraph wiring, interrupt gate, execute paths."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel as _Base,
)

from termagent.agent.graph import build_graph
from termagent.agent.nodes import make_gate_router, route_after_confirm
from termagent.agent.state import AgentState
from termagent.config import Settings
from termagent.store.db import build_checkpointer, open_db


class FakeMessagesListChatModel(_Base):
    """FakeMessagesListChatModel with a no-op bind_tools (returns self)."""

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001
        return self


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
def settings():
    return Settings(
        timeout_seconds=10,
        max_output_chars=8000,
        strict_mode=True,
        auto_approve=False,
    )


def _config(thread_id: str = "test-thread") -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _tool_call_msg(cmd: str, call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"id": call_id, "name": "run_shell", "args": {"command": cmd}}],
    )


# ---------------------------------------------------------------------------
# Router unit tests (no graph needed)
# ---------------------------------------------------------------------------


def test_route_end_when_no_tool_calls(settings):
    from langgraph.graph import END

    router = make_gate_router(settings)
    state: AgentState = {
        "messages": [AIMessage(content="All done!")],
        "cwd": "/tmp",
    }
    assert router(state) == END


def test_route_execute_for_auto_command(settings):
    router = make_gate_router(settings)
    state: AgentState = {
        "messages": [_tool_call_msg("ls -la")],
        "cwd": "/tmp",
    }
    assert router(state) == "execute"


def test_route_confirm_for_mutating_command(settings):
    router = make_gate_router(settings)
    state: AgentState = {
        "messages": [_tool_call_msg("rm -rf /tmp/foo")],
        "cwd": "/tmp",
    }
    assert router(state) == "confirm"


def test_route_execute_for_blocked_command(settings):
    # Blocked commands go to execute so the refusal path lives in one place.
    router = make_gate_router(settings)
    state: AgentState = {
        "messages": [_tool_call_msg("vim file.txt")],
        "cwd": "/tmp",
    }
    assert router(state) == "execute"


def test_route_confirm_for_git_commit(settings):
    router = make_gate_router(settings)
    state: AgentState = {
        "messages": [_tool_call_msg("git commit -m 'msg'")],
        "cwd": "/tmp",
    }
    assert router(state) == "confirm"


def test_route_execute_for_git_status(settings):
    router = make_gate_router(settings)
    state: AgentState = {
        "messages": [_tool_call_msg("git status")],
        "cwd": "/tmp",
    }
    assert router(state) == "execute"


def test_route_after_confirm_declined():
    state: AgentState = {
        "messages": [
            ToolMessage(content="user declined to run `rm x`", tool_call_id="c1")
        ],
        "cwd": "/tmp",
    }
    assert route_after_confirm(state) == "agent"


def test_route_after_confirm_approved():
    state: AgentState = {
        "messages": [AIMessage(content="")],
        "cwd": "/tmp",
    }
    assert route_after_confirm(state) == "execute"


# ---------------------------------------------------------------------------
# Full graph tests
# ---------------------------------------------------------------------------


def test_readonly_auto_path_no_interrupt(tmp_path, checkpointer, settings):
    """A read-only command runs end-to-end without triggering an interrupt."""
    fake = FakeMessagesListChatModel(
        responses=[
            _tool_call_msg("echo hello"),
            AIMessage(content="Done."),
        ]
    )
    graph = build_graph(lambda: fake, settings, checkpointer)
    result = graph.invoke(
        {"messages": [HumanMessage(content="say hello")], "cwd": str(tmp_path)},
        config=_config(),
    )
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert "hello" in tool_msgs[0].content
    assert result["messages"][-1].content == "Done."


def test_write_triggers_interrupt_resume_true_executes(
    tmp_path, checkpointer, settings
):
    """A mutating command triggers an interrupt; resuming True executes it."""
    target = tmp_path / "newfile.txt"
    cmd = f"{sys.executable} -c \"open(r'{target}', 'w').close()\""

    fake = FakeMessagesListChatModel(
        responses=[
            _tool_call_msg(cmd),
            AIMessage(content="File created."),
        ]
    )
    graph = build_graph(lambda: fake, settings, checkpointer)
    cfg = _config("write-thread")

    # First invoke hits the interrupt.
    graph.invoke(
        {"messages": [HumanMessage(content="create a file")], "cwd": str(tmp_path)},
        config=cfg,
    )
    # Should have interrupted — check via get_state
    state_snap = graph.get_state(cfg)
    assert state_snap.next  # still has a pending node (confirm or execute)

    # Resume with approval.
    graph.invoke(Command(resume=True), config=cfg)
    assert target.exists()


def test_write_triggers_interrupt_resume_false_loops(tmp_path, checkpointer, settings):
    """Resuming False emits a decline ToolMessage and loops back to agent."""
    cmd = f'{sys.executable} -c "import sys; sys.exit(0)"'

    fake = FakeMessagesListChatModel(
        responses=[
            _tool_call_msg(cmd),
            AIMessage(content="Understood, I won't run it."),
        ]
    )
    graph = build_graph(lambda: fake, settings, checkpointer)
    cfg = _config("decline-thread")

    graph.invoke(
        {"messages": [HumanMessage(content="run something")], "cwd": str(tmp_path)},
        config=cfg,
    )
    # State should be interrupted at confirm.
    state_snap = graph.get_state(cfg)
    assert state_snap.next

    final = graph.invoke(Command(resume=False), config=cfg)
    tool_msgs = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    # The decline ToolMessage must be present.
    assert any("user declined" in m.content for m in tool_msgs)
    # The agent got another turn and consumed the next scripted response.
    assert final["messages"][-1].content == "Understood, I won't run it."


def test_cd_updates_cwd_no_subprocess(tmp_path, checkpointer, settings):
    """A bare cd command updates cwd in state and spawns no subprocess."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    # Use a relative path to avoid shlex/Windows backslash issues in POSIX mode.
    fake = FakeMessagesListChatModel(
        responses=[
            _tool_call_msg("cd subdir"),
            AIMessage(content="Changed directory."),
        ]
    )
    graph = build_graph(lambda: fake, settings, checkpointer)

    with patch("termagent.agent.nodes.run_command") as mock_run:
        result = graph.invoke(
            {"messages": [HumanMessage(content="go to subdir")], "cwd": str(tmp_path)},
            config=_config("cd-thread"),
        )
        mock_run.assert_not_called()

    assert result["cwd"] == str(subdir)
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert any("cwd is now" in m.content for m in tool_msgs)


def test_blocked_program_refused_no_subprocess(tmp_path, checkpointer, settings):
    """A blocked interactive program gets a refusal ToolMessage; no process is spawned."""
    fake = FakeMessagesListChatModel(
        responses=[
            _tool_call_msg("vim notes.txt"),
            AIMessage(content="I can't run vim."),
        ]
    )
    graph = build_graph(lambda: fake, settings, checkpointer)

    with patch("termagent.agent.nodes.run_command") as mock_run:
        result = graph.invoke(
            {"messages": [HumanMessage(content="edit a file")], "cwd": str(tmp_path)},
            config=_config("blocked-thread"),
        )
        mock_run.assert_not_called()

    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert any("interactive program" in m.content for m in tool_msgs)


def test_persistence_across_restart(tmp_path, db, settings):
    """State (messages + cwd) survives a simulated graph restart with the same thread_id."""
    subdir = tmp_path / "work"
    subdir.mkdir()

    checkpointer1 = build_checkpointer(db)
    fake = FakeMessagesListChatModel(
        responses=[
            _tool_call_msg("cd work"),
            AIMessage(content="Done."),
        ]
    )
    graph1 = build_graph(lambda: fake, settings, checkpointer1)
    thread_id = "persist-thread"
    graph1.invoke(
        {"messages": [HumanMessage(content="change dir")], "cwd": str(tmp_path)},
        config=_config(thread_id),
    )

    # Build a brand-new graph over the same db connection.
    checkpointer2 = build_checkpointer(db)
    fake2 = FakeMessagesListChatModel(responses=[AIMessage(content="Still here.")])
    graph2 = build_graph(lambda: fake2, settings, checkpointer2)

    state_snap = graph2.get_state(_config(thread_id))
    assert state_snap.values["cwd"] == str(subdir)
    # Messages from the first run are rehydrated.
    assert len(state_snap.values["messages"]) > 0
