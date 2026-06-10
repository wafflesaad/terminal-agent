# 07 — REPL & rendering

## Goal

The interactive loop the user actually touches: read a line, run the agent, stream output,
and — critically — present the confirmation prompt and resume the graph on the answer.

## Depends on

- 06 (drives the compiled graph), 03 (current model), 02 (active session), 01 (settings).

## Deliverables

- `termagent/repl.py`.
- `termagent/__main__.py` updated to start the REPL (create a session, build the model+graph).
- `tests/test_repl.py` (logic-level; UI rendering can be light).

## Design

Startup:

1. Load settings (spec 01). If `default_provider == "groq"`, `ensure_groq_key`.
2. Open DB + checkpointer (spec 02); create a new session; build the model + graph (spec 03/06).
3. Print a small banner: app name, active model, session id, `/help` hint.

Loop (per line):

- Input via `prompt_toolkit` `PromptSession` with persistent history
  (`~/.local/share/termagent/history`), multi-line off by default.
- A line starting with `/` goes to the router (spec 08) and never touches the model.
- Otherwise it's a task. Wrap the user text as a `HumanMessage` and run the graph with
  `config={"configurable": {"thread_id": session.id}}`.

Running the graph + handling interrupts:

- Prefer `graph.stream(..., stream_mode="values")` (or `"messages"` for token streaming) so the
  user sees progress. Show a `rich` spinner/status while the agent node is thinking.
- When the stream yields an interrupt (LangGraph surfaces `__interrupt__`), pause and render the
  **approval panel**:
  - the exact command, the `cwd` it would run in, and the reason (write/remove/unknown).
  - prompt `Run this? [y/N]` — default No.
  - resume with `graph.stream(Command(resume=ans), config=...)`.
- Render `ToolMessage` output in a muted/collapsed style; render the final `AIMessage` as the
  answer. If `verbose` (spec 08) is on, also show the model's tool-call reasoning.

After each completed task, `SessionStore.touch(session.id, message_count)` and, if the session
title is still the placeholder, set it from the first user prompt (spec 02).

`auto_approve` (default off): when on, the approval panel is replaced by an auto-yes **with a
visible warning line** so it's never silent. It does not bypass the spec 04 blocklist or
timeouts.

Signals:

- `Ctrl-C` during a running task cancels that task and returns to the prompt (does not exit).
- `Ctrl-D` / `/exit` exits cleanly (close DB connection).

## Interfaces

```python
class Repl:
    def __init__(self, settings, store, checkpointer): ...
    def run(self) -> None: ...                 # the main loop
    def handle_task(self, text: str) -> None:  # run graph, manage interrupts
    def confirm(self, command: str, cwd: str, reason: str) -> bool: ...  # the y/N panel
```

## Acceptance criteria

- Typing a read-only task shows output with no prompt; typing a write task shows the approval
  panel and only runs on `y`.
- Declining returns to the prompt with the agent having been told it was declined.
- `Ctrl-C` mid-task aborts to the prompt without crashing or exiting.
- With `auto_approve` on, a warning is printed and the command runs without a prompt — but a
  blocked program is still refused.
- The session's `updated_at`/`message_count` advance after a task.

## Out of scope / deferred

- Slash command handlers themselves (spec 08) — the REPL only routes `/` lines to the router.
- Fancy TUI (panes, mouse). Single scrolling transcript is the target.
