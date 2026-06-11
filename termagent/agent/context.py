"""Message trimming and token budgeting for the agent context window (spec 09)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Callable

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


def count_tokens(messages: list[AnyMessage]) -> int:
    """Conservative token estimate: ~(chars / 4) + per-message overhead.

    Counts content chars plus serialized tool_call args for AIMessages.
    Over-counting is intentional — safe direction for budgeting.
    """
    total = 0
    for msg in messages:
        content_chars = len(str(getattr(msg, "content", "")))
        tool_chars = 0
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_chars = sum(
                len(json.dumps(tc.get("args", {}))) for tc in msg.tool_calls
            )
        total += (content_chars + tool_chars) // 4 + 4  # +4 per-message framing
    return total


def _group_into_blocks(messages: list[AnyMessage]) -> list[list[AnyMessage]]:
    """Group messages into atomic blocks that must be kept or dropped together.

    An AIMessage with tool_calls is grouped with all immediately-following
    ToolMessages so no tool-call/result pair is ever orphaned.
    """
    blocks: list[list[AnyMessage]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if isinstance(msg, AIMessage) and msg.tool_calls:
            block: list[AnyMessage] = [msg]
            j = i + 1
            while j < len(messages) and isinstance(messages[j], ToolMessage):
                block.append(messages[j])
                j += 1
            blocks.append(block)
            i = j
        else:
            blocks.append([msg])
            i += 1
    return blocks


def prepare_messages(
    messages: list[AnyMessage],
    budget_tokens: int,
    counter: Callable[[list[AnyMessage]], int],
) -> list[AnyMessage]:
    """Trim oldest turns so the list fits within budget_tokens.

    Always preserves a leading SystemMessage (if present) and at least the most
    recent atomic block. Drops blocks from the oldest end; never splits a
    tool-call AIMessage from its matching ToolMessages.
    """
    if counter(messages) <= budget_tokens:
        return messages

    system: list[AnyMessage] = []
    if messages and isinstance(messages[0], SystemMessage):
        system = [messages[0]]
        rest = messages[1:]
    else:
        rest = list(messages)

    blocks = _group_into_blocks(rest)

    # Drop from oldest end; always keep the last block.
    while len(blocks) > 1:
        candidate = system + [m for block in blocks for m in block]
        if counter(candidate) <= budget_tokens:
            break
        blocks.pop(0)

    return system + [m for block in blocks for m in block]


def summarize_old_turns(
    messages: list[AnyMessage],
    model: "BaseChatModel",
) -> list[AnyMessage]:
    # FUTURE: collapse the oldest dropped turns into a single short summary system
    # message so the agent retains gist without the full transcript.
    # Off by default; do not call yet.
    raise NotImplementedError
