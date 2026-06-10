# 08 — Router & slash commands

## Goal

Parse `/`-prefixed lines and dispatch to command handlers. Implement the full command set,
including `/model` (provider swap), `/session` (switch), and `/undo` (checkpointer time-travel).

## Depends on

- 07 (the REPL owns the loop and the shared context), 02 (sessions), 03 (model build),
  06 (graph / checkpointer for `/undo`, `/context`).

## Deliverables

- `termagent/router.py` — parse + dispatch.
- `termagent/commands/` — one module/handler per command.
- `tests/test_router.py`.

## Design

```python
def parse(line: str) -> tuple[str, str] | None:
    # "/model groq" -> ("model", "groq"); "hello world" -> None (it's a prompt)
```

A registry maps name → handler. Handlers receive a shared `Context` (settings, store, current
session, current provider name, the compiled graph + checkpointer, REPL flags like
`verbose`/`auto_approve`) and the argument string, and may mutate context (e.g. rebind model,
switch session).

Unknown `/command` → print "unknown command, try /help" (do **not** fall through to the model).

### Command set

| Command | Behavior |
|---|---|
| `/help` | List commands and one-line descriptions. |
| `/model [ollama\|groq]` | No arg: show current + `/model list`. With arg: rebuild via `build_model` (spec 03), prompting for the Groq key if missing. Update `sessions.model`. |
| `/session` | List recent sessions (spec 02 `list`) with index, title, model, updated_at; let the user pick; rebind `thread_id` to the chosen session. State rehydrates from the checkpointer automatically. |
| `/new` | Create a new session + thread_id without restarting. |
| `/cd <path>` | Set the agent's `cwd` in graph state for the current thread (resolve + validate like spec 04 `apply_cd`). |
| `/undo` | Use the checkpointer history (`get_state_history`) to rewind to the previous checkpoint for this thread, dropping the last exchange. |
| `/retry` | Re-send the last user prompt (useful after `/model`). |
| `/auto` | Toggle `auto_approve`; print the warning when turning it on. |
| `/history` | Print the current session's messages from state. |
| `/rename <title>` | `SessionStore.set_title`. |
| `/delete [id]` | Delete a session (spec 02 `delete`, which also clears checkpoints). Confirm first. Default: current session, then `/new`. |
| `/export [md\|json]` | Dump the current session's transcript to a file in cwd. |
| `/context` | Show the estimated token use vs `context_token_budget` (spec 09). |
| `/verbose` | Toggle showing the agent's reasoning + tool calls. |
| `/exit` | Quit cleanly. |

### Notes on the tricky ones

- `/session` and `/undo` both lean entirely on the checkpointer — there is no separate "load
  messages" step; rebinding the `thread_id` (or rolling back its history) is the whole job.
- `/model` is the one command that may trigger the Groq key prompt mid-session.
- `/cd` writes into graph state via `graph.update_state(config, {"cwd": new})`.
- `/delete` is destructive — gate it behind its own y/N (consistent with the app's whole ethos),
  separate from the shell confirmation gate.

## Acceptance criteria

- `parse` distinguishes commands from prompts; unknown commands don't reach the model.
- `/model groq` with no stored key prompts once, swaps the model, and updates session metadata.
- `/session` switches threads and the next prompt continues the older conversation.
- `/undo` removes the last exchange (verify message count drops and state matches the prior
  checkpoint).
- `/delete` requires confirmation and leaves no checkpoint rows for the thread.
- Handlers are unit-testable with a fake Context (no real model needed for parsing/dispatch).

## Out of scope / deferred

- Custom user-defined commands. Fixed set for now.
