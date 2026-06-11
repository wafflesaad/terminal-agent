"""Parse `/`-prefixed lines and dispatch to command handlers.

`parse` is pure and decides command-vs-prompt. `dispatch` looks the name up in the
command registry (spec 08) and runs the handler, or prints a hint for an unknown
command — it never falls through to the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from termagent.context import Context


@dataclass
class Result:
    """What a handler asks the REPL loop to do next.

    - ``continue`` (default): return to the prompt.
    - ``exit``: quit the loop cleanly.
    - ``prompt``: run ``prompt`` as if the user had typed it (used by ``/retry``).
    """

    action: Literal["continue", "exit", "prompt"] = "continue"
    prompt: Optional[str] = None


def parse(line: str) -> Optional[tuple[str, str]]:
    """Split a command line into (name, arg), or return None if it is a prompt.

    "/model groq" -> ("model", "groq"); "/help" -> ("help", "");
    "hello world" -> None (it is a prompt, not a command).
    """
    if not line.startswith("/"):
        return None
    body = line[1:].strip()
    if not body:
        return ("", "")
    parts = body.split(maxsplit=1)
    name = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    return (name, arg)


def dispatch(ctx: "Context", line: str) -> Result:
    """Parse *line* and run the matching handler; unknown commands print a hint."""
    parsed = parse(line)
    if parsed is None:
        # Caller is responsible for only routing slash lines here.
        return Result()

    name, arg = parsed
    from termagent.commands import COMMANDS

    command = COMMANDS.get(name)
    if command is None:
        ctx.console.print(f"[red]unknown command[/red] /{name} — try /help")
        return Result()
    return command.handler(ctx, arg)
