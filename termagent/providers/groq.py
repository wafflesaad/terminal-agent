"""Groq provider: constructs ChatGroq with an API key."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq

from termagent.providers.base import ProviderError


class GroqProvider:
    """Builds a ChatGroq configured with the given model and API key."""

    name = "groq"

    def __init__(self, model: str, api_key: str | None) -> None:
        self.model = model
        self.api_key = api_key

    def build(self) -> BaseChatModel:
        """Return a ChatGroq configured with this provider's model and key."""
        return ChatGroq(model=self.model, api_key=self.api_key)

    def check(self) -> None:
        """Raise ProviderError if no API key is present."""
        if not self.api_key:
            raise ProviderError(
                "Groq API key is missing. Run termagent and select the groq provider "
                "to be prompted for a key, or set groq_api_key in your config file.\n"
                "Note: the model must support tool calling (e.g. llama-3.3-70b-versatile)."
            )
