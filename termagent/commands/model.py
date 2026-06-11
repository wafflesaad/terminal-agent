"""The /model, /groq-key, and /groq-key-remove commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from termagent import config
from termagent.providers.base import ProviderError
from termagent.providers.registry import available_models, build_model
from termagent.router import Result

if TYPE_CHECKING:
    from termagent.context import Context

_VALID = ("ollama", "groq")


def model_cmd(ctx: "Context", arg: str) -> Result:
    """No arg: show current + available. With arg: rebuild and rebind the model.

    Rebuilding via build_model (spec 03) prompts for the Groq key if it is missing.
    On success the session's stored model tag is updated.
    """
    name = arg.strip().lower()

    if not name:
        ctx.console.print(f"current model: [cyan]{ctx.provider_name}[/cyan] / {ctx.model_label()}")
        models = available_models(ctx.settings)
        for provider, tags in models.items():
            listed = ", ".join(tags) if tags else "[dim](unavailable)[/dim]"
            ctx.console.print(f"  {provider}: {listed}")
        return Result()

    if name not in _VALID:
        ctx.console.print(f"[red]unknown provider[/red] '{name}' — choose ollama or groq")
        return Result()

    try:
        model = build_model(name, ctx.settings)  # may prompt for the Groq key
    except ProviderError as exc:
        ctx.console.print(f"[red]could not switch:[/red] {exc}")
        return Result()

    ctx.model = model
    ctx.provider_name = name
    label = ctx.model_label()
    ctx.store.set_model(ctx.session.id, label)
    ctx.session.model = label
    ctx.console.print(f"[green]model is now[/green] {name} / {label}")
    return Result()


def groq_key_cmd(ctx: "Context", arg: str) -> Result:
    """Prompt for a new Groq API key and persist it to the config file."""
    try:
        key = input("New Groq API key: ").strip()
    except EOFError:
        return Result()
    if not key:
        ctx.console.print("[yellow]no key entered — unchanged[/yellow]")
        return Result()
    ctx.settings.groq_api_key = key
    config.save(ctx.settings)
    ctx.console.print("[green]Groq API key updated[/green]")
    return Result()


def groq_key_remove_cmd(ctx: "Context", arg: str) -> Result:
    """Clear the saved Groq API key from the config file."""
    if not ctx.settings.groq_api_key:
        ctx.console.print("[yellow]no Groq key is set[/yellow]")
        return Result()
    try:
        confirm = input("Remove saved Groq key? [y/N] ").strip().lower()
    except EOFError:
        return Result()
    if confirm != "y":
        ctx.console.print("cancelled")
        return Result()
    ctx.settings.groq_api_key = None
    config.save(ctx.settings)
    ctx.console.print("[green]Groq API key removed[/green]")
    return Result()
