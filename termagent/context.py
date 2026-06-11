"""Shared mutable REPL context handed to slash-command handlers.

One instance is created at startup and lives for the whole process. The REPL reads
its live fields each turn (so a `/model` swap or `/verbose` toggle takes effect on the
next task) and slash-command handlers mutate it in place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from rich.console import Console

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

    from termagent.config import Settings
    from termagent.store.sessions import Session, SessionStore


@dataclass
class Context:
    """The live shared state every handler receives and may mutate."""

    settings: "Settings"
    store: "SessionStore"
    checkpointer: "BaseCheckpointSaver"
    graph: "CompiledStateGraph"
    session: "Session"
    provider_name: str
    model: "Optional[BaseChatModel]" = None
    cwd: str = field(default_factory=os.getcwd)
    verbose: bool = False
    auto_approve: bool = False
    last_prompt: "Optional[str]" = None
    console: Console = field(default_factory=Console)

    def thread_config(self) -> dict:
        """The LangGraph config addressing this session's checkpoint thread."""
        return {"configurable": {"thread_id": self.session.id}}

    def model_label(self) -> str:
        """The model tag for the current provider (used for session metadata)."""
        return (
            self.settings.ollama_model
            if self.provider_name == "ollama"
            else self.settings.groq_model
        )
