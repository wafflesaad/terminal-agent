"""Ollama provider: constructs ChatOllama and validates the local Ollama server."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from termagent.providers.base import ProviderError

_CHECK_TIMEOUT = 5  # seconds for /api/tags ping


class OllamaProvider:
    """Builds a ChatOllama and verifies the model tag is present on the local server."""

    name = "ollama"

    def __init__(self, model: str, host: str) -> None:
        self.model = model
        self.host = host.rstrip("/")

    def build(self) -> BaseChatModel:
        """Return a ChatOllama configured with this provider's model and host."""
        return ChatOllama(model=self.model, base_url=self.host)

    def check(self) -> None:
        """Raise ProviderError if Ollama is unreachable or the model tag is missing."""
        url = f"{self.host}/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=_CHECK_TIMEOUT) as resp:
                data = json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise ProviderError(
                f"Cannot reach Ollama at {self.host}: {exc.reason}. "
                "Is Ollama running?"
            ) from exc

        tags: list[str] = [m.get("name", "") for m in data.get("models", [])]
        if not _tag_matches(self.model, tags):
            raise ProviderError(
                f"Model '{self.model}' is not available in Ollama at {self.host}. "
                f"Run:  ollama pull {self.model}\n"
                "Note: the model must support tool calling (e.g. qwen3.5:9b)."
            )


def _tag_matches(configured: str, available: list[str]) -> bool:
    """Return True if configured name matches any available tag (lenient :latest handling)."""
    if configured in available:
        return True
    # If configured has no tag suffix, also match "<configured>:latest"
    if ":" not in configured:
        return f"{configured}:latest" in available
    return False
