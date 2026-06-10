# 09 — Context management

## Goal

Keep long sessions from blowing past the model's context window — especially the smaller local
Ollama model — by trimming old turns to a token budget before each model call. A summarization
step is designed but stubbed off by default.

## Depends on

- 06 (wraps the agent node / pre-model step), 01 (`context_token_budget`).

## Deliverables

- `termagent/agent/context.py` — `prepare_messages`, token estimate, summary stub.
- Wire it into the agent node (spec 06).
- `tests/test_context.py`.

## Design

Run a preparation step on `state["messages"]` right before the model call:

```python
def prepare_messages(messages: list[AnyMessage], budget_tokens: int,
                     counter: Callable[[list[AnyMessage]], int]) -> list[AnyMessage]: ...
```

Behavior:

- Always keep the system message (if any) and the most recent turns.
- Trim from the **oldest** non-system messages until the estimated token count is within budget.
- **Never orphan a tool call.** An `AIMessage` carrying `tool_calls` and its matching
  `ToolMessage` results must be kept or dropped together — dropping one and keeping the other
  produces an invalid sequence that the API/runtime will reject. Trim at turn boundaries, not
  mid-pair. LangChain's `trim_messages` with `start_on="human"`/`include_system=True` handles
  this; otherwise implement boundary-aware trimming yourself.

Token counting:

- If the provider exposes a tokenizer, use it. Otherwise a cheap heuristic (`~len(text)/4`) is
  fine for budgeting — being conservative (over-counting) is the safe direction.

Summarization (stub, default off behind a flag):

```python
def summarize_old_turns(messages, model) -> list[AnyMessage]:
    # FUTURE: collapse the oldest dropped turns into a single short summary system message,
    # so the agent retains gist without the full transcript. Off by default; do not call yet.
    raise NotImplementedError
```

`/context` (spec 08) reports `counter(messages)` vs `budget_tokens` so the user can see headroom.

## Acceptance criteria

- A synthetic over-budget message list is trimmed to within budget, the system message survives,
  and the most recent turn survives.
- Trimming never leaves an `AIMessage(tool_calls=...)` without its `ToolMessage`, and never
  leaves a `ToolMessage` without its preceding tool-call `AIMessage`.
- `prepare_messages` is pure and tested without a model.
- `/context` reflects the estimate before and after a trim.

## Out of scope / deferred

- Actual summarization (kept as a stub). Turn it on only after trimming is proven.
- Per-model exact tokenizers — heuristic is acceptable for v1.
