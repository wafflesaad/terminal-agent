"""Meta / control commands: help, exit, retry, auto, verbose, context, history."""

from __future__ import annotations

from typing import TYPE_CHECKING

from termagent.commands._util import estimate_tokens, role_of, state_messages
from termagent.router import Result

if TYPE_CHECKING:
    from termagent.context import Context


def help_cmd(ctx: "Context", arg: str) -> Result:
    """List commands and one-line descriptions."""
    from termagent.commands import REGISTRY

    ctx.console.print("[bold]Commands[/bold]")
    width = max(len(c.usage) for c in REGISTRY)
    for command in REGISTRY:
        ctx.console.print(f"  [cyan]{command.usage:<{width}}[/cyan]  {command.summary}")
    return Result()


def exit_cmd(ctx: "Context", arg: str) -> Result:
    """Quit cleanly."""
    return Result(action="exit")


def retry_cmd(ctx: "Context", arg: str) -> Result:
    """Re-send the last user prompt (useful after /model)."""
    if not ctx.last_prompt:
        ctx.console.print("[dim]nothing to retry[/dim]")
        return Result()
    return Result(action="prompt", prompt=ctx.last_prompt)


def auto_cmd(ctx: "Context", arg: str) -> Result:
    """Toggle auto_approve; warn when turning it on."""
    ctx.auto_approve = not ctx.auto_approve
    if ctx.auto_approve:
        ctx.console.print(
            "[yellow]⚠ auto-approve ON — commands run without confirmation. "
            "Blocked interactive programs are still refused.[/yellow]"
        )
    else:
        ctx.console.print("[green]auto-approve OFF[/green]")
    return Result()


def verbose_cmd(ctx: "Context", arg: str) -> Result:
    """Toggle showing the agent's reasoning + tool calls."""
    ctx.verbose = not ctx.verbose
    ctx.console.print(f"verbose {'ON' if ctx.verbose else 'OFF'}")
    return Result()


def context_cmd(ctx: "Context", arg: str) -> Result:
    """Show estimated token use vs context_token_budget (spec 09)."""
    budget = ctx.settings.context_token_budget
    used = estimate_tokens(state_messages(ctx))
    pct = (used / budget * 100) if budget else 0
    ctx.console.print(
        f"context: ~{used} / {budget} tokens ({pct:.0f}%) [dim](estimated)[/dim]"
    )
    return Result()


def history_cmd(ctx: "Context", arg: str) -> Result:
    """Print the current session's messages from state."""
    messages = state_messages(ctx)
    if not messages:
        ctx.console.print("[dim]no messages yet[/dim]")
        return Result()
    for msg in messages:
        content = str(getattr(msg, "content", "")).strip()
        if not content:
            continue
        ctx.console.print(f"[bold]{role_of(msg)}:[/bold] {content}")
    return Result()
