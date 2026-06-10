# 03 — Model providers

## Goal

A thin abstraction over the two backends so the agent graph never knows which is active.
Switching providers swaps one object; the graph code is unchanged.

## Depends on

- 01 (reads model names, host, key from `Settings`; calls `ensure_groq_key`).

## Deliverables

- `termagent/providers/base.py` — provider protocol.
- `termagent/providers/ollama.py`, `termagent/providers/groq.py`.
- `termagent/providers/registry.py` — `build_model`, availability checks.
- `tests/test_providers.py`.

## Design

Both `ChatOllama` (from `langchain-ollama`) and `ChatGroq` (from `langchain-groq`) implement
the LangChain `BaseChatModel` interface, including `.bind_tools()` and normalized tool-call
output. So a provider's only job is to construct a configured `BaseChatModel`. Tool binding
happens later in the agent node (spec 06), not here.

```python
class Provider(Protocol):
    name: str
    def build(self) -> BaseChatModel: ...
    def check(self) -> None: ...   # raise a clear error if unusable
```

- `OllamaProvider(model, host)` → `ChatOllama(model=..., base_url=host)`.
  `check()` pings the host (`GET /api/tags`) and verifies the model tag is present; if missing,
  raise an error telling the user to `ollama pull <model>`.
- `GroqProvider(model, api_key)` → `ChatGroq(model=..., api_key=...)`.
  `check()` verifies a key is present (it should be, via `ensure_groq_key`).

Both target models must support tool calling (Qwen 3.5 locally; a tool-capable Groq model such
as `llama-3.3-70b-versatile`). Note this requirement in errors if a non-tool model is set.

Registry / factory:

```python
def build_model(provider_name: str, settings: Settings) -> BaseChatModel:
    # groq → ensure_groq_key(settings) first, then build + check
    # ollama → build + check
```

The REPL/router (specs 07/08) hold the "current model" and rebuild via `build_model` on
`/model`. Because history is stored as provider-agnostic messages (spec 02/06), switching
mid-session is safe.

## Interfaces

```python
def build_model(provider_name: Literal["ollama", "groq"], settings: Settings) -> BaseChatModel: ...
def available_models(settings: Settings) -> dict[str, list[str]]: ...  # for `/model list`
```

## Acceptance criteria

- `build_model("ollama", settings)` returns a `ChatOllama`; a missing model tag raises a
  message naming the `ollama pull` command.
- `build_model("groq", settings)` calls `ensure_groq_key` when the key is missing, then returns
  a `ChatGroq`.
- Tests mock the Ollama HTTP check and the Groq client; no real network calls in CI.

## Out of scope / deferred

- Streaming wiring (spec 07 decides whether to call `.stream()` vs `.invoke()`).
- Tool binding (spec 06).
