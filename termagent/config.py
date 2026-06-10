"""Configuration: settings model, XDG paths, load/save, first-run Groq key prompt."""

from __future__ import annotations

import getpass
import os
import tomllib
from pathlib import Path
from typing import Literal

import tomli_w
from pydantic import BaseModel


class Settings(BaseModel):
    default_provider: Literal["ollama", "groq"] = "ollama"
    ollama_model: str = "qwen3.5:9b"
    ollama_host: str = "http://localhost:11434"
    groq_model: str = "llama-3.3-70b-versatile"
    groq_api_key: str | None = None
    timeout_seconds: int = 60
    max_output_chars: int = 8000
    strict_mode: bool = True
    auto_approve: bool = False
    context_token_budget: int = 6000


def config_path() -> Path:
    """Return path to the config file, honouring XDG_CONFIG_HOME."""
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "termagent" / "config.toml"


def data_dir() -> Path:
    """Return the data directory, honouring XDG_DATA_HOME."""
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "termagent"


def load() -> Settings:
    """Load settings from the config file; create with defaults if absent."""
    path = config_path()
    if path.exists():
        with path.open("rb") as fh:
            data = tomllib.load(fh)
        return Settings(**data)
    settings = Settings()
    save(settings)
    return settings


def save(settings: Settings) -> None:
    """Write settings to the config file and set 0600 permissions."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {k: v for k, v in settings.model_dump().items() if v is not None}
    with path.open("wb") as fh:
        tomli_w.dump(data, fh)
    os.chmod(path, 0o600)


def ensure_groq_key(settings: Settings) -> str:
    """Return the Groq API key, prompting and persisting it if not yet set."""
    if settings.groq_api_key:
        return settings.groq_api_key
    key = getpass.getpass("Groq API key: ")
    settings.groq_api_key = key
    save(settings)
    return key
