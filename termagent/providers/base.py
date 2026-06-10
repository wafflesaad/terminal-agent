"""Provider protocol and shared error type."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from langchain_core.language_models.chat_models import BaseChatModel


class ProviderError(RuntimeError):
    """Raised when a provider cannot be used (missing model, no key, unreachable host)."""


@runtime_checkable
class Provider(Protocol):
    name: str

    def build(self) -> BaseChatModel:
        """Return a configured BaseChatModel instance."""
        ...

    def check(self) -> None:
        """Raise ProviderError with a human-readable fix if the provider is unusable."""
        ...
