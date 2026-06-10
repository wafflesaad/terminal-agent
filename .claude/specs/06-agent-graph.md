# 06 — Agent graph

## Goal

The LangGraph `StateGraph` that ties everything together: reason with the model, classify the
proposed command, gate writes behind a human-approval interrupt, execute, observe, loop.

## Depends on

- 02 (checkpointer), 03 (model), 04 (executor + cd + blocklist), 05 (classifier).

## Deliverables

- `termagent/agent/state.py` — `AgentState`.
- `termagent/agent/nodes.py` — node functions.
- `termagent/agent/graph.py` — graph construction + compile.
- `tests/test_graph.py`.

## State

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    cwd: str
    # `pending` is optional; the interrupt payload carries the command, so state can stay lean.
```

`add_messages` is the reducer so `AIMessage` tool calls and `ToolMessage` results accumulate
correctly and survive checkpointing.

## The single tool

Expose one tool to the model:

```python
@tool
def run_shell(command: str) -> str:
    """Run a shell command and return its combined output. Use for any terminal task."""
    # Body is never actually called by LangChain's ToolNode here — execution happens in the
    # `execute` node so classification/confirmation/cd can intercept first. The decorator exists
    # to give the model a clean tool schema. Bind it in the agent node.
```

Bind with `model.bind_tools([run_shell])` inside the agent node so a `/model` swap is picked up.

## Nodes

- **agent** — call the (tool-bound) model on `state["messages"]`; append the returned
  `AIMessage`. The system prompt tells the model it operates a Linux/WSL shell, that writes are
  user-gated, and that interactive programs are unavailable.
- **execute** — for the pending tool call(s): first `apply_cd` (spec 04); if it returns a path,
  update `cwd` and emit a `ToolMessage` like `"cwd is now <path>"` (no subprocess). Else
  `check_blocked` (spec 04) → if blocked, emit the refusal `ToolMessage`. Else `run_command`
  with `cwd=state["cwd"]`, cap output, and emit a `ToolMessage` with exit code + capped output.
- **confirm** — call LangGraph `interrupt({...})` with the command, its `cwd`, and the reason it
  was flagged. The REPL (spec 07) surfaces this and resumes with `True`/`False`. On `True` →
  route to `execute`. On `False` → emit a `ToolMessage` ("user declined to run `<cmd>`") and
  route back to `agent` so it can choose a different approach.

Keep it simple: process **one** tool call per cycle. If the model emits several, handle the
first and let the loop come back for the rest (avoids a half-confirmed batch).

## Edges

```
START → agent
agent → route_after_agent (conditional):
    no tool calls            → END
    tool call present        → gate
gate → route_gate (conditional):   # gate is a thin router, not a node with side effects
    classify(cmd, strict) == AUTO        → execute
    classify(cmd, strict) == CONFIRM     → confirm
    check_blocked(cmd) is not None       → execute   # execute emits the refusal, loops to agent
confirm → (resume True) execute | (resume False) agent
execute → agent
```

Implementation note: you can fold `gate` into the conditional function on the `agent` edge —
it just inspects the last message's tool call and returns the next node name. Blocked commands
are routed to `execute` so the single refusal path lives in one place.

## Compile

```python
graph = builder.compile(checkpointer=checkpointer)
# Invoke/stream with config={"configurable": {"thread_id": session_id}}
```

Use the dynamic `interrupt()` inside the confirm node (current LangGraph), not the older
`interrupt_before=` compile flag — it carries a payload to the REPL and resumes cleanly via
`Command(resume=...)`.

## Acceptance criteria

- A prompt needing a read-only command runs end-to-end with no confirmation and returns output.
- A prompt needing a write triggers an interrupt; resuming `True` executes, `False` loops back
  with a decline `ToolMessage` and the model gets another turn.
- A `cd` tool call updates `cwd` in state and runs no subprocess; a later command runs in the
  new directory.
- A blocked program yields the refusal `ToolMessage` and never spawns a process.
- State persists across a simulated restart (same `thread_id` rehydrates messages + cwd).
- Tests use a fake/stub chat model that emits scripted tool calls (no real LLM in CI).

## Out of scope / deferred

- Context trimming/summary — that wraps the agent node in spec 09.
- Multi-tool-call batching.
