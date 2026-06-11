"""Session lifecycle commands: session, new, rename, delete, undo, export."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, RemoveMessage

from termagent.commands._util import role_of, state_messages
from termagent.router import Result

if TYPE_CHECKING:
    from termagent.context import Context


def session_cmd(ctx: "Context", arg: str) -> Result:
    """List recent sessions and switch to the chosen one by rebinding the thread."""
    sessions = ctx.store.list()
    if not sessions:
        ctx.console.print("[dim]no sessions[/dim]")
        return Result()

    for i, sess in enumerate(sessions):
        marker = "*" if sess.id == ctx.session.id else " "
        title = sess.title or "[dim](untitled)[/dim]"
        ctx.console.print(
            f"{marker} [cyan]{i}[/cyan]  {title}  "
            f"[dim]{sess.model}  {sess.updated_at}[/dim]"
        )

    choice = ctx.console.input("switch to # (blank to cancel): ").strip()
    if not choice:
        return Result()
    try:
        index = int(choice)
        chosen = sessions[index]
    except (ValueError, IndexError):
        ctx.console.print("[red]no such session[/red]")
        return Result()

    ctx.session = chosen
    # State rehydrates automatically: the next task uses the new thread_id.
    ctx.console.print(f"[green]switched to[/green] {chosen.title or chosen.id}")
    return Result()


def new_cmd(ctx: "Context", arg: str) -> Result:
    """Create a new session + thread_id without restarting."""
    ctx.session = ctx.store.create(ctx.model_label())
    ctx.console.print(f"[green]new session[/green] {ctx.session.id}")
    return Result()


def rename_cmd(ctx: "Context", arg: str) -> Result:
    """Set the current session's title."""
    title = arg.strip()
    if not title:
        ctx.console.print("usage: /rename <title>")
        return Result()
    ctx.store.set_title(ctx.session.id, title)
    ctx.session.title = title
    ctx.console.print(f"[green]renamed to[/green] {title}")
    return Result()


def delete_cmd(ctx: "Context", arg: str) -> Result:
    """Delete a session (and its checkpoints) behind its own y/N gate."""
    target_id = arg.strip() or ctx.session.id
    if ctx.store.get(target_id) is None:
        ctx.console.print("[red]no such session[/red]")
        return Result()

    answer = ctx.console.input(f"delete session {target_id}? [y/N] ").strip().lower()
    if answer != "y":
        ctx.console.print("[dim]cancelled[/dim]")
        return Result()

    deleting_current = target_id == ctx.session.id
    ctx.store.delete(target_id)
    ctx.console.print(f"[green]deleted[/green] {target_id}")
    if deleting_current:
        ctx.session = ctx.store.create(ctx.model_label())
        ctx.console.print(f"[green]new session[/green] {ctx.session.id}")
    return Result()


def undo_cmd(ctx: "Context", arg: str) -> Result:
    """Rewind the thread by dropping the last exchange (last user turn onward).

    Uses the checkpointer-backed state: the add_messages reducer honours
    RemoveMessage, so removing the tail messages rolls the thread back a turn.
    """
    messages = state_messages(ctx)
    last_human = next(
        (i for i in range(len(messages) - 1, -1, -1)
         if isinstance(messages[i], HumanMessage)),
        None,
    )
    if last_human is None:
        ctx.console.print("[dim]nothing to undo[/dim]")
        return Result()

    dropped = messages[last_human:]
    ctx.graph.update_state(
        ctx.thread_config(),
        {"messages": [RemoveMessage(id=m.id) for m in dropped]},
    )
    ctx.console.print(f"[green]undid last exchange[/green] ({len(dropped)} messages)")
    return Result()


def export_cmd(ctx: "Context", arg: str) -> Result:
    """Dump the current session's transcript to a file in cwd (md or json)."""
    from pathlib import Path

    fmt = (arg.strip().lower() or "md")
    if fmt not in ("md", "json"):
        ctx.console.print("usage: /export [md|json]")
        return Result()

    messages = state_messages(ctx)
    path = Path(ctx.cwd) / f"termagent-{ctx.session.id}.{fmt}"

    if fmt == "json":
        payload = [
            {"role": role_of(m), "content": str(getattr(m, "content", ""))}
            for m in messages
        ]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        lines = [f"# {ctx.session.title or ctx.session.id}", ""]
        for m in messages:
            content = str(getattr(m, "content", "")).strip()
            if not content:
                continue
            lines.append(f"**{role_of(m)}:** {content}")
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")

    ctx.console.print(f"[green]exported to[/green] {path}")
    return Result()
