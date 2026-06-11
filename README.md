# termagent

A terminal AI agent. Type a natural-language task; the agent reasons and runs shell commands until the task is done.

Built on [LangGraph](https://github.com/langchain-ai/langgraph) with two switchable model backends — a local [Ollama](https://ollama.com) instance or the [Groq](https://console.groq.com) cloud API. All session history is persisted in a local SQLite database.

> **Platform:** Linux / WSL. Not tested on macOS or native Windows.

---

## How it works

termagent runs a **ReAct loop** implemented as a LangGraph `StateGraph`:

```
user prompt
     │
     ▼
 ┌─────────┐     no tool call      ┌─────┐
 │  agent  │ ───────────────────►  │ END │
 └─────────┘                       └─────┘
     │ tool call
     ▼
 ┌──────────────┐   read-only (AUTO)   ┌─────────┐
 │ gate router  │ ──────────────────►  │ execute │ ◄─┐
 └──────────────┘                      └─────────┘   │
     │ mutating / unknown (CONFIRM)          │        │
     ▼                                      └────────►│
 ┌─────────┐  approve   ┌─────────┐         (loop back to agent)
 │ confirm │ ─────────► │ execute │
 │  (y/N)  │            └─────────┘
 └─────────┘
     │ decline
     └──────────► agent (tries again)
```

**Command classification** is allowlist-based and fail-closed: anything not on the known read-only list goes to the confirmation gate. Unknown always means confirm, never auto-run.

**Session state** is checkpointed to SQLite via `langgraph-checkpoint-sqlite`, so every session is resumable and `/undo` can replay history.

---

## Requirements

- Python 3.11+
- Linux or WSL
- One of:
  - **Ollama** running locally (`http://localhost:11434`) with a model pulled
  - A **Groq API key** (free tier available at [console.groq.com](https://console.groq.com))

---

## Installation

```bash
# Clone
git clone https://github.com/wafflesaad/terminal-agent.git
cd terminal-agent

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install
pip install -e .

# Install dev dependencies (tests + linter)
pip install -e ".[dev]"
```

---

## Configuration

Config is stored at `~/.config/termagent/config.toml` (XDG-compliant) with `0600` permissions. It is created with defaults on first run.

| Key | Default | Description |
|---|---|---|
| `default_provider` | `"ollama"` | Which backend to use (`"ollama"` or `"groq"`) |
| `ollama_model` | `"qwen3.5:9b"` | Ollama model tag |
| `ollama_host` | `"http://localhost:11434"` | Ollama API base URL |
| `groq_model` | `"meta-llama/llama-4-scout-17b-16e-instruct"` | Groq model ID |
| `groq_api_key` | *(prompted on first use)* | Stored in config file, never logged |
| `timeout_seconds` | `60` | Per-command execution timeout |
| `max_output_chars` | `8000` | Output cap before re-entering context |
| `context_token_budget` | `6000` | Oldest messages trimmed above this token count |
| `auto_approve` | `false` | Skip confirmation gate (off by default, prints a warning when on) |

### Ollama setup

```bash
# Install Ollama, then pull a model
ollama pull qwen3.5:9b
```

### Groq setup

The key is prompted automatically the first time you select the Groq provider. You can also set or update it at any time:

```bash
# Inside termagent
/groq-key
```

---

## Running

```bash
# Via the installed script
termagent

# Or directly
python -m termagent
```

Type any natural-language task at the prompt. The agent will reason, propose shell commands, and ask for confirmation on anything that writes or removes state.

---

## Safety model

Five invariants are enforced at the framework level and are never weakened:

1. **Confirmation gate** — any command that writes or removes state requires explicit `y/N` approval via a LangGraph interrupt, not model self-reporting.
2. **Allowlist, fail-closed** — the classifier permits only known read-only commands automatically. Everything else, including anything unrecognised, goes to confirmation.
3. **Interactive programs refused** — TTY-dependent programs are blocked at the executor before any subprocess is spawned.

   Blocked: `vim`, `vi`, `nano`, `emacs`, `less`, `more`, `top`, `htop`, `man`, `ssh`, `tmux`, `screen`, `watch`, `tail`, `python`, `python3`, `node`, `irb`, `psql`, `mysql`, `sqlite3`

4. **Every execution is bounded** — each command runs with a configurable timeout; captured output is capped before re-entering context.
5. **`auto_approve` defaults off** — it exists for scripting/testing but prints a warning when enabled and never silences the blocklist or timeout.

---

## Slash commands

| Command | Description |
|---|---|
| `/help` | List all commands |
| `/model [ollama\|groq]` | Show or swap the active provider |
| `/session` | List and switch to a past session |
| `/new` | Start a fresh session |
| `/cd <path>` | Change the agent's working directory |
| `/undo` | Drop the last exchange |
| `/retry` | Re-send the last prompt |
| `/history` | Print the session transcript |
| `/rename <title>` | Rename the current session |
| `/delete [id]` | Delete a session (confirms first) |
| `/export [md\|json]` | Write the transcript to a file |
| `/context` | Show token usage vs the budget |
| `/verbose` | Toggle reasoning/tool-call display |
| `/auto` | Toggle auto-approve (off by default) |
| `/groq-key` | Update the saved Groq API key |
| `/groq-key-remove` | Remove the saved Groq API key |
| `/exit` or `/quit` | Quit termagent |

---

## Development

```bash
pytest                        # run all tests
pytest tests/test_classify.py # safety-critical classifier tests
ruff check . && ruff format . # lint + format
```

Session data is stored at `~/.local/share/termagent/` (XDG-compliant).

---

## Project structure

```
termagent/
  __main__.py        # entry point
  config.py          # settings, XDG paths, key management
  context.py         # shared live state (model, session, flags)
  repl.py            # prompt loop, streaming, confirmation UI
  router.py          # slash vs prompt dispatch
  commands/          # one handler per slash command
  store/
    db.py            # SQLite + LangGraph checkpointer setup
    sessions.py      # session metadata table + CRUD
  providers/
    ollama.py / groq.py / registry.py
  tools/
    shell.py         # run_command, cd handling, timeout, output cap
    blocklist.py     # interactive-program refusal
    classify.py      # allowlist fail-closed classifier
  agent/
    state.py         # AgentState TypedDict
    nodes.py         # agent / confirm / execute nodes
    graph.py         # StateGraph wiring + edges + interrupt
    context.py       # message trimming + token budgeting
tests/
```
