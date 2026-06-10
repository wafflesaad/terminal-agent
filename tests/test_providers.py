"""Tests for termagent.providers: build_model, check(), available_models."""

from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from termagent import config
from termagent.config import Settings
from termagent.providers.base import ProviderError
from termagent.providers.registry import available_models, build_model


@pytest.fixture(autouse=True)
def isolate_config(tmp_path, monkeypatch):
    """Redirect XDG dirs to tmp so ensure_groq_key persistence is sandboxed."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_tags_response(*model_names: str):
    """Return a mock urlopen context manager yielding an /api/tags JSON body."""
    body = json.dumps({"models": [{"name": n} for n in model_names]}).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# OllamaProvider — happy path
# ---------------------------------------------------------------------------


def test_ollama_build_model_returns_chat_ollama():
    settings = Settings(ollama_model="qwen3.5:9b", ollama_host="http://localhost:11434")

    with patch("termagent.providers.ollama.urllib.request.urlopen") as mock_open, patch(
        "termagent.providers.ollama.ChatOllama"
    ) as mock_cls:
        mock_open.return_value = _fake_tags_response("qwen3.5:9b")
        fake_instance = MagicMock()
        mock_cls.return_value = fake_instance

        model = build_model("ollama", settings)

    mock_open.assert_called_once()
    mock_cls.assert_called_once_with(model="qwen3.5:9b", base_url="http://localhost:11434")
    assert model is fake_instance


# ---------------------------------------------------------------------------
# OllamaProvider — missing tag
# ---------------------------------------------------------------------------


def test_ollama_missing_tag_raises_provider_error():
    settings = Settings(ollama_model="qwen3.5:9b", ollama_host="http://localhost:11434")

    with patch("termagent.providers.ollama.urllib.request.urlopen") as mock_open:
        mock_open.return_value = _fake_tags_response("llama3:latest")
        with pytest.raises(ProviderError, match="ollama pull qwen3.5:9b"):
            build_model("ollama", settings)


# ---------------------------------------------------------------------------
# OllamaProvider — unreachable host
# ---------------------------------------------------------------------------


def test_ollama_unreachable_raises_provider_error():
    settings = Settings(ollama_model="qwen3.5:9b", ollama_host="http://localhost:11434")

    with patch("termagent.providers.ollama.urllib.request.urlopen") as mock_open:
        mock_open.side_effect = urllib.error.URLError("Connection refused")
        with pytest.raises(ProviderError, match="Cannot reach Ollama"):
            build_model("ollama", settings)


# ---------------------------------------------------------------------------
# OllamaProvider — lenient :latest tag matching
# ---------------------------------------------------------------------------


def test_ollama_lenient_latest_match():
    """Configured 'llama3' should match available 'llama3:latest'."""
    settings = Settings(ollama_model="llama3", ollama_host="http://localhost:11434")

    with patch("termagent.providers.ollama.urllib.request.urlopen") as mock_open, patch(
        "termagent.providers.ollama.ChatOllama"
    ) as mock_cls:
        mock_open.return_value = _fake_tags_response("llama3:latest")
        mock_cls.return_value = MagicMock()

        build_model("ollama", settings)  # should not raise

    mock_cls.assert_called_once_with(model="llama3", base_url="http://localhost:11434")


# ---------------------------------------------------------------------------
# GroqProvider — happy path (key already set)
# ---------------------------------------------------------------------------


def test_groq_build_model_returns_chat_groq():
    settings = Settings(groq_model="llama-3.3-70b-versatile", groq_api_key="gsk_test")

    with patch("termagent.providers.groq.ChatGroq") as mock_cls:
        fake_instance = MagicMock()
        mock_cls.return_value = fake_instance

        model = build_model("groq", settings)

    mock_cls.assert_called_once_with(model="llama-3.3-70b-versatile", api_key="gsk_test")
    assert model is fake_instance


# ---------------------------------------------------------------------------
# GroqProvider — key missing, prompts and persists
# ---------------------------------------------------------------------------


def test_groq_prompts_when_key_missing():
    settings = config.load()
    assert settings.groq_api_key is None

    with patch("termagent.config.getpass.getpass", return_value="gsk_new") as mock_gp, patch(
        "termagent.providers.groq.ChatGroq"
    ) as mock_cls:
        mock_cls.return_value = MagicMock()
        model = build_model("groq", settings)

    mock_gp.assert_called_once()
    assert model is mock_cls.return_value
    assert settings.groq_api_key == "gsk_new"


# ---------------------------------------------------------------------------
# Unknown provider
# ---------------------------------------------------------------------------


def test_unknown_provider_raises_provider_error():
    settings = Settings()
    with pytest.raises(ProviderError, match="Unknown provider"):
        build_model("unknown", settings)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# available_models
# ---------------------------------------------------------------------------


def test_available_models_ollama_returns_tags():
    settings = Settings(ollama_host="http://localhost:11434", groq_model="llama-3.3-70b-versatile")

    with patch("termagent.providers.registry.urllib.request.urlopen") as mock_open:
        mock_open.return_value = _fake_tags_response("qwen3.5:9b", "llama3:latest")
        result = available_models(settings)

    assert result["ollama"] == ["qwen3.5:9b", "llama3:latest"]
    assert result["groq"] == ["llama-3.3-70b-versatile"]


def test_available_models_ollama_empty_on_failure():
    settings = Settings(ollama_host="http://localhost:11434", groq_model="llama-3.3-70b-versatile")

    with patch("termagent.providers.registry.urllib.request.urlopen") as mock_open:
        mock_open.side_effect = urllib.error.URLError("refused")
        result = available_models(settings)

    assert result["ollama"] == []
    assert result["groq"] == ["llama-3.3-70b-versatile"]


def test_available_models_groq_returns_configured_model():
    settings = Settings(groq_model="llama-3.3-70b-versatile", ollama_host="http://localhost:11434")

    with patch("termagent.providers.registry.urllib.request.urlopen") as mock_open:
        mock_open.return_value = _fake_tags_response()
        result = available_models(settings)

    assert "llama-3.3-70b-versatile" in result["groq"]
