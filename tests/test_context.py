"""Tests for termagent.agent.context — pure, no model, no subprocess."""

from __future__ import annotations

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from termagent.agent.context import (
    count_tokens,
    prepare_messages,
    summarize_old_turns,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool_call_msg(cmd: str, call_id: str = "c1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"id": call_id, "name": "run_shell", "args": {"command": cmd}}],
    )


def _tool_result(call_id: str = "c1", content: str = "ok") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=call_id)


def _one_per_msg(messages):
    """Trivial counter: 1 token per message. Makes budget arithmetic obvious."""
    return len(messages)


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------

class TestCountTokens:
    def test_empty_list(self):
        assert count_tokens([]) == 0

    def test_empty_content_still_has_framing(self):
        msgs = [HumanMessage(content="")]
        result = count_tokens(msgs)
        # framing overhead (+4) with 0 content chars → 0//4 + 4 = 4
        assert result == 4

    def test_content_chars_counted(self):
        # 40 chars / 4 = 10, +4 framing = 14
        msgs = [HumanMessage(content="a" * 40)]
        assert count_tokens(msgs) == 14

    def test_tool_call_args_counted(self):
        plain = [HumanMessage(content="")]
        with_tool = [_tool_call_msg("ls -la")]
        # tool-call message must count more tokens than a plain empty message
        assert count_tokens(with_tool) > count_tokens(plain)

    def test_multiple_messages_additive(self):
        msgs = [HumanMessage(content=""), HumanMessage(content="")]
        single = [HumanMessage(content="")]
        assert count_tokens(msgs) == 2 * count_tokens(single)


# ---------------------------------------------------------------------------
# prepare_messages — fast path
# ---------------------------------------------------------------------------

class TestPrepareMessagesUnderBudget:
    def test_returns_same_list_when_under_budget(self):
        msgs = [SystemMessage(content="sys"), HumanMessage(content="hi")]
        result = prepare_messages(msgs, budget_tokens=10_000, counter=count_tokens)
        assert result == msgs

    def test_returns_same_list_exactly_at_budget(self):
        msgs = [HumanMessage(content="")]
        budget = count_tokens(msgs)
        result = prepare_messages(msgs, budget_tokens=budget, counter=count_tokens)
        assert result == msgs


# ---------------------------------------------------------------------------
# prepare_messages — trimming
# ---------------------------------------------------------------------------

class TestPrepareMessagesTrimming:
    def test_system_message_always_preserved(self):
        sys = SystemMessage(content="you are an agent")
        msgs = [sys] + [HumanMessage(content=f"msg {i}") for i in range(20)]
        # tight budget: 2 tokens → only system + one human can survive
        result = prepare_messages(msgs, budget_tokens=2, counter=_one_per_msg)
        assert result[0] is sys

    def test_most_recent_turn_always_preserved(self):
        last = HumanMessage(content="the very last message")
        msgs = [HumanMessage(content=f"old {i}") for i in range(10)] + [last]
        result = prepare_messages(msgs, budget_tokens=1, counter=_one_per_msg)
        assert result[-1] is last

    def test_result_within_budget(self):
        msgs = [SystemMessage(content="sys")] + [
            HumanMessage(content="x" * 100) for _ in range(30)
        ]
        budget = 500
        result = prepare_messages(msgs, budget_tokens=budget, counter=count_tokens)
        assert count_tokens(result) <= budget

    def test_oldest_messages_dropped_first(self):
        first = HumanMessage(content="first")
        second = HumanMessage(content="second")
        last = HumanMessage(content="last")
        msgs = [first, second, last]
        # budget of 2 → must drop oldest (first)
        result = prepare_messages(msgs, budget_tokens=2, counter=_one_per_msg)
        assert first not in result
        assert last in result

    def test_system_not_trimmed_even_when_large(self):
        sys = SystemMessage(content="s" * 400)
        recent = HumanMessage(content="latest")
        msgs = [sys, HumanMessage(content="old"), recent]
        # tiny budget — system + recent are the floor
        result = prepare_messages(msgs, budget_tokens=1, counter=_one_per_msg)
        assert isinstance(result[0], SystemMessage)
        assert result[-1] is recent


# ---------------------------------------------------------------------------
# prepare_messages — tool-call pairing invariants
# ---------------------------------------------------------------------------

class TestToolCallPairing:
    def test_tool_call_and_result_dropped_together(self):
        """If a tool-call block is dropped, its ToolMessage goes with it."""
        sys = SystemMessage(content="sys")
        old_call = _tool_call_msg("rm -rf /", call_id="old")
        old_result = _tool_result(call_id="old", content="done")
        recent = HumanMessage(content="latest")

        msgs = [sys, old_call, old_result, recent]
        # budget = 2 → drop the old tool-call block, keep sys + recent
        result = prepare_messages(msgs, budget_tokens=2, counter=_one_per_msg)

        ids_in_result = {id(m) for m in result}
        assert id(old_call) not in ids_in_result
        assert id(old_result) not in ids_in_result

    def test_no_orphaned_tool_call(self):
        """Result never contains an AIMessage(tool_calls) without its ToolMessage."""
        sys = SystemMessage(content="sys")
        call1 = _tool_call_msg("ls", call_id="a")
        res1 = _tool_result(call_id="a")
        call2 = _tool_call_msg("pwd", call_id="b")
        res2 = _tool_result(call_id="b")
        recent = HumanMessage(content="done")

        msgs = [sys, call1, res1, call2, res2, recent]
        result = prepare_messages(msgs, budget_tokens=4, counter=_one_per_msg)

        # Verify: every AIMessage with tool_calls has its ToolMessage immediately after
        for i, msg in enumerate(result):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                call_id = msg.tool_calls[0]["id"]
                # There must be a ToolMessage for this call_id somewhere after it
                tail_ids = {
                    getattr(m, "tool_call_id", None) for m in result[i + 1:]
                }
                assert call_id in tail_ids, (
                    f"AIMessage with tool_call id={call_id} has no matching ToolMessage"
                )

    def test_no_leading_orphaned_tool_message(self):
        """Result never starts (after system) with a dangling ToolMessage."""
        sys = SystemMessage(content="sys")
        old_call = _tool_call_msg("echo hi", call_id="x")
        old_result = _tool_result(call_id="x")
        recent = HumanMessage(content="new")

        msgs = [sys, old_call, old_result, recent]
        result = prepare_messages(msgs, budget_tokens=2, counter=_one_per_msg)

        non_system = [m for m in result if not isinstance(m, SystemMessage)]
        if non_system:
            assert not isinstance(non_system[0], ToolMessage), (
                "Result must not begin with a dangling ToolMessage"
            )

    def test_multiple_tool_pairs_partial_trim(self):
        """Exactly the right number of pairs are dropped."""
        sys = SystemMessage(content="sys")
        pairs = []
        for i in range(5):
            cid = f"c{i}"
            pairs += [_tool_call_msg(f"cmd{i}", call_id=cid), _tool_result(cid)]
        recent = HumanMessage(content="last")

        msgs = [sys] + pairs + [recent]
        # budget = 4 → keep sys + last pair + recent (3 messages), drop older ones
        result = prepare_messages(msgs, budget_tokens=4, counter=_one_per_msg)

        # No ToolMessage without its preceding AIMessage(tool_calls)
        seen_call_ids: set[str] = set()
        for msg in result:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                seen_call_ids.add(msg.tool_calls[0]["id"])
            if isinstance(msg, ToolMessage):
                assert msg.tool_call_id in seen_call_ids, (
                    f"ToolMessage {msg.tool_call_id} has no preceding AIMessage"
                )


# ---------------------------------------------------------------------------
# summarize_old_turns stub
# ---------------------------------------------------------------------------

class TestSummarizeStub:
    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            summarize_old_turns([], model=object())  # type: ignore[arg-type]
