"""Shared helpers for command handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from termagent.agent.context import count_tokens

if TYPE_CHECKING:
    from termagent.context import Context


def state_messages(ctx: "Context") -> list[AnyMessage]:
    """Return the current thread's messages from graph (checkpointer) state."""
    snap = ctx.graph.get_state(ctx.thread_config())
    return list(snap.values.get("messages", []))


def role_of(msg: AnyMessage) -> str:
    """A short human-readable role label for a message."""
    if isinstance(msg, HumanMessage):
        return "user"
    if isinstance(msg, ToolMessage):
        return "tool"
    if isinstance(msg, AIMessage):
        return "assistant"
    return msg.__class__.__name__.replace("Message", "").lower() or "system"


def estimate_tokens(messages: list[AnyMessage]) -> int:
    """Token estimate delegating to the canonical counter in agent.context."""
    return count_tokens(messages)
