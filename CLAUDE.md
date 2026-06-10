# CLAUDE.md — termagent

A terminal AI agent. The user types a natural-language task; the agent achieves it by
running shell commands in a ReAct loop. Built on LangGraph. Runs on Linux / WSL.

> Keep this file tight. It is read at the start of every session. Put enduring facts here;
> put feature detail in `docs/specs/`.

## What we are building

- A REPL: type a prompt, the agent reasons and runs shell commands until the task is done.
- Agentic loop is a LangGraph `StateGraph` (reason → pick command → maybe confirm → run → observe).
- Two model backends, switchable at runtime: local Ollama and cloud Groq.
- Local SQLite for session state + chat history. Each launch starts a new session;
  `/session` switches to an older one.
- Slash commands for control (`/model`, `/session`, …). See spec 08.

## Non-negotiable safety invariants

These are the whole point of the app. Never weaken them to make a task pass.

1. **Confirmation gate.** Any command that writes or removes state requires explicit user
   `y/n` approval before it runs. This is enforced in the graph via a human-approval
   interrupt — never inside the model's reasoning, never by trusting the model to self-report.
2. **Allowlist that fails closed.** Classification is allowlist-based: a known read-only
   command runs automatically; *everything else*, including anything unrecognized, routes to
   confirmation. Unknown == confirm. When in doubt, confirm.
3. **Interactive programs are refused, not run.** TTY-dependent programs (`vim`, `top`, `ssh`,
   `less`, …) are blocked at the executor with a helpful message. They are never spawned.
4. **Every execution is bounded.** Timeout on every command; captured output is capped before
   it re-enters context.
5. `auto_approve` exists but defaults **off** and prints a warning when enabled. It never
   becomes the default and never silences invariant 3 or the timeout.

If a change would relax any of these, stop and ask the user first.

## Locked design decisions

- Command classification: allowlist, fail-closed (spec 05).
- Groq API key: prompted on first run (or first time Groq is selected with no key) and saved
  to the local config with `0600` perms. Never hard-code or log it (spec 01).
- Working directory: tracked in agent state and passed as `cwd=` to every `subprocess.run`.
  `cd` is special-cased — it updates state, it is **not** executed as a subprocess (spec 04).
- Context: trim oldest turns to a token budget; a summarization step is stubbed but off by
  default (spec 09).

## Tech stack

- Python 3.11+
- `langgraph`, `langchain-core`
- `langchain-ollama` (ChatOllama), `langchain-groq` (ChatGroq)
- `langgraph-checkpoint-sqlite` (SqliteSaver), stdlib `sqlite3`
- `prompt_toolkit` (input line), `rich` (rendering)
- `pydantic` (config + state), `tomllib`/`tomli-w` (config file)
- stdlib `subprocess`, `shlex`, `os` for execution
- `pytest` for tests

## Project structure

```
termagent/
  __main__.py        # entry point → starts the REPL
  config.py          # settings model, load/save, first-run key prompt   (spec 01)
  store/
    db.py            # sqlite + SqliteSaver setup                         (spec 02)
    sessions.py      # sessions metadata table + SessionStore             (spec 02)
  providers/
    base.py          # provider protocol                                 (spec 03)
    ollama.py
    groq.py
    registry.py      # build_model(name, config)                         (spec 03)
  tools/
    shell.py         # run_command, cwd/cd handling, timeout, output cap  (spec 04)
    blocklist.py     # interactive-program refusal                       (spec 04)
    classify.py      # allowlist fail-closed classifier                  (spec 05)
  agent/
    state.py         # AgentState TypedDict                              (spec 06)
    nodes.py         # agent / confirm / execute nodes                   (spec 06)
    graph.py         # StateGraph wiring + edges + interrupt             (spec 06)
    context.py       # message trimming + summary stub                   (spec 09)
  repl.py            # prompt loop, streaming, approval UI                (spec 07)
  router.py          # slash vs prompt dispatch                          (spec 08)
  commands/          # one handler per slash command                     (spec 08)
tests/
docs/specs/          # numbered implementation specs
```

## Implementation order

Build in this order; each spec lists what it depends on. Land one spec per PR, with tests,
before starting the next.

1. `docs/specs/01-foundation-and-config.md`
2. `docs/specs/02-persistence-and-sessions.md`
3. `docs/specs/03-model-providers.md`
4. `docs/specs/04-shell-executor.md`
5. `docs/specs/05-command-classifier.md`
6. `docs/specs/06-agent-graph.md`
7. `docs/specs/07-repl-and-rendering.md`
8. `docs/specs/08-router-and-slash-commands.md`
9. `docs/specs/09-context-management.md`

## Conventions

- Type-hint everything. Public functions get docstrings stating the contract.
- Pure functions where possible — `classify` and the output-capping logic must be pure and
  unit-tested in isolation (no subprocess, no I/O).
- No secrets in code, logs, or this file. The Groq key lives only in the config file.
- Errors from shell commands are data, not exceptions — capture exit code + stderr and feed
  them back to the agent so it can recover.
- Small, reviewable commits scoped to one spec.

## Dev commands

```bash
python -m termagent           # run the REPL
pytest                        # run tests
pytest tests/test_classify.py # the safety-critical unit tests
ruff check . && ruff format . # lint + format
```

## Do not

- Do not bypass or auto-approve the confirmation gate to make a task succeed.
- Do not run `cd` as a subprocess. Update `cwd` in state.
- Do not let the model decide whether a command is safe — that is the classifier's job.
- Do not spawn interactive/TTY programs.
- Do not print, log, or commit the Groq API key.
