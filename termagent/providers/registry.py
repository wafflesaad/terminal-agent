"""Registry / factory: build_model and available_models."""

from __future__ import annotations

import json
import urllib.request
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel

from termagent.config import Settings, ensure_groq_key
from termagent.providers.base import ProviderError
from termagent.providers.groq import GroqProvider
from termagent.providers.ollama import OllamaProvider

_TAGS_TIMEOUT = 5


def build_model(
    provider_name: Literal["ollama", "groq"], settings: Settings
) -> BaseChatModel:
    """Construct, validate, and return a BaseChatModel for the named provider.

    Raises ProviderError if the provider is misconfigured or unavailable.
    """
    if provider_name == "ollama":
        provider = OllamaProvider(settings.ollama_model, settings.ollama_host)
        provider.check()
        return provider.build()
    if provider_name == "groq":
        key = ensure_groq_key(settings)
        provider = GroqProvider(settings.groq_model, key)
        provider.check()
        return provider.build()
    raise ProviderError(
        f"Unknown provider '{provider_name}'. Valid options: 'ollama', 'groq'."
    )


def available_models(settings: Settings) -> dict[str, list[str]]:
    """Return a mapping of provider name → list of available model tags.

    Ollama tags are fetched live; on failure an empty list is returned (never raises).
    Groq returns the configured model name only (no live listing endpoint).
    """
    ollama_tags: list[str] = []
    try:
        url = f"{settings.ollama_host.rstrip('/')}/api/tags"
        with urllib.request.urlopen(url, timeout=_TAGS_TIMEOUT) as resp:
            data = json.loads(resp.read())
        ollama_tags = [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        pass

    return {
        "ollama": ollama_tags,
        "groq": [settings.groq_model],
    }
