"""Entry point: load config, handle first-run key prompt, print banner."""

from __future__ import annotations

from termagent import config


def _banner(settings: config.Settings) -> str:
    model = (
        settings.ollama_model
        if settings.default_provider == "ollama"
        else settings.groq_model
    )
    return f"termagent  [{settings.default_provider}]  {model}"


def main() -> int:
    settings = config.load()

    if settings.default_provider == "groq":
        config.ensure_groq_key(settings)

    if settings.auto_approve:
        print("WARNING: auto_approve is enabled — confirmation gate is bypassed.")

    print(_banner(settings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
