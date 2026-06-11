"""Filesystem commands: /cd."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

from termagent.router import Result
from termagent.tools.shell import apply_cd

if TYPE_CHECKING:
    from termagent.context import Context


def cd_cmd(ctx: "Context", arg: str) -> Result:
    """Set the agent's cwd in graph state for the current thread.

    Resolves + validates the path exactly like the executor's apply_cd (spec 04),
    then writes it into both graph state and the live context.
    """
    path = arg.strip()
    command = f"cd {shlex.quote(path)}" if path else "cd"
    try:
        new_cwd = apply_cd(command, ctx.cwd)
    except (FileNotFoundError, NotADirectoryError) as exc:
        ctx.console.print(f"[red]{exc}[/red]")
        return Result()

    if new_cwd is None:
        ctx.console.print("[red]invalid path[/red]")
        return Result()

    ctx.graph.update_state(ctx.thread_config(), {"cwd": new_cwd})
    ctx.cwd = new_cwd
    ctx.console.print(f"cwd is now {new_cwd}")
    return Result()
