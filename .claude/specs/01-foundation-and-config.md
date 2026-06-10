# 01 — Foundation & config

## Goal

Scaffold the package and the configuration layer, including the first-run Groq key prompt.
After this spec, `python -m termagent` runs, loads (or creates) config, and prints a banner.

## Depends on

Nothing. This is the first spec.

## Deliverables

- `pyproject.toml` with dependencies and a `termagent` package.
- `termagent/__main__.py` — entry point (for now: load config, print banner, exit).
- `termagent/config.py` — settings model, load/save, first-run prompt.
- `tests/test_config.py`.

## Design

Config lives at an XDG path, created on first run:

- Config file: `${XDG_CONFIG_HOME:-~/.config}/termagent/config.toml`
- Data dir (used later by spec 02): `${XDG_DATA_HOME:-~/.local/share}/termagent/`

Settings model (pydantic `BaseModel`):

```python
class Settings(BaseModel):
    default_provider: Literal["ollama", "groq"] = "ollama"
    ollama_model: str = "qwen3.5:9b"
    ollama_host: str = "http://localhost:11434"
    groq_model: str = "llama-3.3-70b-versatile"
    groq_api_key: str | None = None       # filled on first run; never logged
    timeout_seconds: int = 60
    max_output_chars: int = 8000
    strict_mode: bool = True              # classifier strictness (spec 05)
    auto_approve: bool = False            # spec 07/08; warns when enabled
    context_token_budget: int = 6000      # spec 09
```

Load/save:

- `load() -> Settings`: read TOML if present, else return defaults and write the file.
- `save(settings)`: write TOML. After writing, `chmod 0600` because it can hold the key.

First-run key prompt (`ensure_groq_key`):

- Called when the active provider is `groq` and `groq_api_key` is empty.
- Prompt with `getpass.getpass("Groq API key: ")` so it is not echoed. Persist via `save()`.
- Do not prompt for Ollama (no key needed). Do not prompt at import time — only when Groq is
  actually about to be used (startup if `default_provider == "groq"`, or on `/model groq`).

## Interfaces

```python
def config_path() -> Path: ...
def data_dir() -> Path: ...
def load() -> Settings: ...
def save(settings: Settings) -> None: ...     # writes TOML, then chmod 0600
def ensure_groq_key(settings: Settings) -> str: ...   # prompts + saves if missing
```

## Acceptance criteria

- Fresh machine: first run creates the config dir + file with defaults; file mode is `0600`.
- `ensure_groq_key` prompts once, saves, and does not prompt again on the next run.
- The key never appears in stdout, the banner, or any log line.
- `tests/test_config.py` covers: defaults written when absent; round-trip load/save; key prompt
  triggers only when missing and provider is groq (mock `getpass`).

## Out of scope / deferred

- Actually building models (spec 03) or the DB (spec 02). This spec only reads/writes settings.
