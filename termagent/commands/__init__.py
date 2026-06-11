"""Slash-command registry: maps command names to handlers and descriptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from termagent.commands.fs import cd_cmd
from termagent.commands.meta import (
    auto_cmd,
    context_cmd,
    exit_cmd,
    help_cmd,
    history_cmd,
    retry_cmd,
    verbose_cmd,
)
from termagent.commands.model import groq_key_cmd, groq_key_remove_cmd, model_cmd
from termagent.commands.session import (
    delete_cmd,
    export_cmd,
    new_cmd,
    rename_cmd,
    session_cmd,
    undo_cmd,
)
from termagent.context import Context
from termagent.router import Result

Handler = Callable[[Context, str], Result]


@dataclass(frozen=True)
class Command:
    name: str
    handler: Handler
    summary: str
    usage: str


REGISTRY: list[Command] = [
    Command("help", help_cmd, "List commands and what they do.", "/help"),
    Command("model", model_cmd, "Show or swap the active provider.", "/model [ollama|groq]"),
    Command("session", session_cmd, "List and switch to a past session.", "/session"),
    Command("new", new_cmd, "Start a fresh session.", "/new"),
    Command("cd", cd_cmd, "Change the agent's working directory.", "/cd <path>"),
    Command("undo", undo_cmd, "Drop the last exchange.", "/undo"),
    Command("retry", retry_cmd, "Re-send the last prompt.", "/retry"),
    Command("auto", auto_cmd, "Toggle auto-approve (off by default).", "/auto"),
    Command("history", history_cmd, "Print the session transcript.", "/history"),
    Command("rename", rename_cmd, "Rename the current session.", "/rename <title>"),
    Command("delete", delete_cmd, "Delete a session (confirms first).", "/delete [id]"),
    Command("export", export_cmd, "Write the transcript to a file.", "/export [md|json]"),
    Command("context", context_cmd, "Show token usage vs the budget.", "/context"),
    Command("verbose", verbose_cmd, "Toggle reasoning/tool-call display.", "/verbose"),
    Command("groq-key", groq_key_cmd, "Update the saved Groq API key.", "/groq-key"),
    Command("groq-key-remove", groq_key_remove_cmd, "Remove the saved Groq API key.", "/groq-key-remove"),
    Command("exit", exit_cmd, "Quit termagent.", "/exit"),
]

COMMANDS: dict[str, Command] = {c.name: c for c in REGISTRY}
# /quit is an alias for /exit.
COMMANDS["quit"] = COMMANDS["exit"]

__all__ = ["Command", "COMMANDS", "REGISTRY", "Handler"]
