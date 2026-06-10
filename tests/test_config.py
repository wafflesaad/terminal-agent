"""Tests for termagent.config: defaults, round-trip, 0600 perms, key-prompt gating."""

from __future__ import annotations

import os
import stat
from unittest.mock import patch

import pytest

from termagent import config
from termagent.config import Settings


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    """Redirect XDG_CONFIG_HOME and XDG_DATA_HOME to a temp directory."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))


# ---------------------------------------------------------------------------
# Defaults written when absent
# ---------------------------------------------------------------------------


def test_first_load_creates_file():
    path = config.config_path()
    assert not path.exists()
    settings = config.load()
    assert path.exists()
    assert settings == Settings()


def test_first_load_writes_defaults():
    settings = config.load()
    assert settings.default_provider == "ollama"
    assert settings.ollama_model == "qwen3.5:9b"
    assert settings.groq_api_key is None
    assert settings.auto_approve is False


# ---------------------------------------------------------------------------
# File permissions
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions not enforced on Windows")
def test_config_file_mode_0600():
    config.load()
    path = config.config_path()
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600


# ---------------------------------------------------------------------------
# Round-trip load/save
# ---------------------------------------------------------------------------


def test_round_trip_with_key():
    settings = Settings(default_provider="groq", groq_api_key="gsk_test123")
    config.save(settings)
    loaded = config.load()
    assert loaded.default_provider == "groq"
    assert loaded.groq_api_key == "gsk_test123"


def test_round_trip_preserves_all_fields():
    original = Settings(
        ollama_model="llama3:8b",
        timeout_seconds=30,
        max_output_chars=4000,
        strict_mode=False,
        context_token_budget=3000,
    )
    config.save(original)
    loaded = config.load()
    assert loaded == original


# ---------------------------------------------------------------------------
# ensure_groq_key: prompt gating
# ---------------------------------------------------------------------------


def test_ensure_groq_key_prompts_when_missing():
    settings = config.load()
    assert settings.groq_api_key is None

    with patch(
        "termagent.config.getpass.getpass", return_value="gsk_newkey"
    ) as mock_gp:
        key = config.ensure_groq_key(settings)

    mock_gp.assert_called_once()
    assert key == "gsk_newkey"
    assert settings.groq_api_key == "gsk_newkey"


def test_ensure_groq_key_persists_and_no_second_prompt():
    settings = config.load()

    with patch("termagent.config.getpass.getpass", return_value="gsk_saved"):
        config.ensure_groq_key(settings)

    # Reload from disk and ensure no prompt on second call
    reloaded = config.load()
    with patch("termagent.config.getpass.getpass") as mock_gp:
        key = config.ensure_groq_key(reloaded)

    mock_gp.assert_not_called()
    assert key == "gsk_saved"


def test_ensure_groq_key_skips_prompt_when_key_present():
    settings = Settings(groq_api_key="gsk_existing")
    config.save(settings)

    loaded = config.load()
    with patch("termagent.config.getpass.getpass") as mock_gp:
        key = config.ensure_groq_key(loaded)

    mock_gp.assert_not_called()
    assert key == "gsk_existing"
