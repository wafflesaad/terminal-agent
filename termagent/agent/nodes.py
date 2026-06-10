"""Node functions and routers for the termagent StateGraph."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING, Callable

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END
from langgraph.types import interrupt

from termagent.agent.state import AgentState
from termagent.tools.blocklist import check_blocked
from termagent.tools.classify import Decision, classify
from termagent.tools.shell import apply_cd, run_command

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

    from termagent.config import Settings

_SYSTEM_PROMPT = (
    "You are a terminal agent operating a Linux/WSL shell. "
    "You help the user accomplish tasks by running shell commands. "
    "Use the run_shell tool for any command you want to execute. "
    "Commands that write, modify, or delete state require explicit user approval "
    "before they run — you will be notified when a command is pending approval. "
    "Interactive or TTY-dependent programs (vim, top, ssh, less, python REPL, etc.) "
    "are not available and will be refused. "
    "When a command fails, read the output carefully and try a different approach."
)


@tool
def run_shell(command: str) -> str:
    """Run a shell command and return its combined output. Use for any terminal task."""
    # Body is never called — the execute node intercepts before ToolNode runs.
    raise RuntimeError(
        "run_shell body should never execute; intercepted by execute node"
    )


def make_agent_node(get_model: Callable[[], "BaseChatModel"]) -> Callable:
    """Return an agent node that binds run_shell to the current model each turn."""

    def agent(state: "AgentState") -> dict:
        messages = list(state["messages"])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=_SYSTEM_PROMPT)] + messages
        model = get_model().bind_tools([run_shell])
        ai_msg = model.invoke(messages)
        return {"messages": [ai_msg]}

    return agent


def make_gate_router(settings: "Settings") -> Callable:
    """Return a conditional-edge function that routes after the agent node."""

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return END
        # Process only the first tool call per cycle.
        cmd = last.tool_calls[0]["args"].get("command", "")
        # Bare cd is a state update only — no subprocess, no confirmation needed.
        try:
            tokens = shlex.split(cmd)
            if tokens and tokens[0] == "cd" and len(tokens) <= 2:
                return "execute"
        except ValueError:
            pass
        # Blocked commands go to execute so the single refusal path lives there.
        if check_blocked(cmd) is not None:
            return "execute"
        if classify(cmd, strict=settings.strict_mode) == Decision.AUTO:
            return "execute"
        return "confirm"

    return route_after_agent


def confirm_node(state: "AgentState") -> dict:
    """Raise an interrupt for human approval; on resume route to execute or agent."""
    last = state["messages"][-1]
    tool_call = last.tool_calls[0]
    cmd = tool_call["args"].get("command", "")
    tool_call_id = tool_call["id"]

    decision: bool = interrupt(
        {"command": cmd, "cwd": state["cwd"], "reason": "command requires confirmation"}
    )

    if not decision:
        return {
            "messages": [
                ToolMessage(
                    content=f"user declined to run `{cmd}`",
                    tool_call_id=tool_call_id,
                )
            ]
        }
    # Approved — emit nothing; route_after_confirm will send us to execute.
    return {}


def route_after_confirm(state: "AgentState") -> str:
    """Route to execute (approved) or agent (declined, decline ToolMessage already added)."""
    last = state["messages"][-1]
    if isinstance(last, ToolMessage) and last.content.startswith(
        "user declined to run"
    ):
        return "agent"
    return "execute"


def make_execute_node(settings: "Settings") -> Callable:
    """Return an execute node that runs the pending tool call."""

    def execute(state: "AgentState") -> dict:
        # Find the AIMessage with the pending tool call.
        # Walk backwards to find the last AIMessage that has tool_calls.
        ai_msg: AIMessage | None = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                ai_msg = msg
                break

        if ai_msg is None:
            return {}

        tool_call = ai_msg.tool_calls[0]
        cmd: str = tool_call["args"].get("command", "")
        tool_call_id: str = tool_call["id"]
        cwd: str = state["cwd"]

        # 1. Handle bare cd — no subprocess.
        try:
            new_cwd = apply_cd(cmd, cwd)
        except (FileNotFoundError, NotADirectoryError) as exc:
            return {
                "messages": [ToolMessage(content=str(exc), tool_call_id=tool_call_id)]
            }
        if new_cwd is not None:
            return {
                "cwd": new_cwd,
                "messages": [
                    ToolMessage(
                        content=f"cwd is now {new_cwd}",
                        tool_call_id=tool_call_id,
                    )
                ],
            }

        # 2. Refuse blocked interactive programs — no subprocess.
        blocked = check_blocked(cmd)
        if blocked is not None:
            return {
                "messages": [
                    ToolMessage(
                        content=(
                            f"`{blocked}` is an interactive program and cannot be run here. "
                            "Try a non-interactive alternative."
                        ),
                        tool_call_id=tool_call_id,
                    )
                ]
            }

        # 3. Run the command.
        result = run_command(
            cmd,
            cwd,
            timeout_s=settings.timeout_seconds,
            max_output_chars=settings.max_output_chars,
        )

        parts: list[str] = [f"exit code: {result.exit_code}"]
        if result.stdout:
            parts.append(f"stdout:\n{result.stdout}")
        if result.stderr:
            parts.append(f"stderr:\n{result.stderr}")
        if result.timed_out:
            parts.append(f"(command timed out after {settings.timeout_seconds}s)")
        if result.truncated:
            parts.append("(output was truncated to fit context)")

        return {
            "messages": [
                ToolMessage(content="\n".join(parts), tool_call_id=tool_call_id)
            ]
        }

    return execute
