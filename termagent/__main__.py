"""Entry point: wire up settings, DB, model, graph, and start the REPL."""

from __future__ import annotations

import os

from termagent import config
from termagent.agent.graph import build_graph
from termagent.context import Context
from termagent.providers.base import ProviderError
from termagent.providers.registry import build_model
from termagent.repl import Repl
from termagent.store.db import build_checkpointer, open_db
from termagent.store.sessions import SessionStore


def _model_label(settings: config.Settings) -> str:
    return (
        settings.ollama_model
        if settings.default_provider == "ollama"
        else settings.groq_model
    )


def main() -> int:
    settings = config.load()

    if settings.auto_approve:
        print("WARNING: auto_approve is enabled — confirmation gate is bypassed.")

    conn = open_db()
    try:
        checkpointer = build_checkpointer(conn)
        store = SessionStore(conn)

        provider_name = settings.default_provider
        model = None
        while model is None:
            try:
                if provider_name == "groq":
                    config.ensure_groq_key(settings)
                model = build_model(provider_name, settings)
            except ProviderError as exc:
                other = "groq" if provider_name == "ollama" else "ollama"
                print(f"Provider error: {exc}")
                try:
                    choice = input(f"Switch to {other}? [y/N] ").strip().lower()
                except EOFError:
                    return 1
                if choice != "y":
                    return 1
                provider_name = other

        # The graph reads ctx.model each turn so /model can swap it live.
        graph = build_graph(lambda: ctx.model, settings, checkpointer)
        session = store.create(_model_label(settings))

        ctx = Context(
            settings=settings,
            store=store,
            checkpointer=checkpointer,
            graph=graph,
            session=session,
            provider_name=provider_name,
            model=model,
            cwd=os.getcwd(),
            auto_approve=settings.auto_approve,
        )

        Repl(settings, store, checkpointer, graph, session, context=ctx).run()
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
