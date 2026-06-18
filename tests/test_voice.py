"""Tests for voice dictation helpers."""

from __future__ import annotations

import pytest

from wattle import voice
from wattle.auth import AuthCredential


def test_voice_config_uses_voice_api_key_env() -> None:
    config = voice.resolve_voice_dictation_config(
        {
            "VOICE_DICTATION_API_KEY": " voice-key ",
            "VOICE_DICTATION_MODEL": " custom-transcribe ",
        }
    )

    assert config.api_key == "voice-key"
    assert config.model == "custom-transcribe"


def test_voice_config_falls_back_to_openai_api_key_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_api_key_credential(vendor: str) -> AuthCredential:
        assert vendor == "openai"
        return AuthCredential(
            kind="api_key",
            bearer_token="auth-file-key",
            source="test",
        )

    monkeypatch.setattr(voice, "get_api_key_credential", fake_get_api_key_credential)

    config = voice.resolve_voice_dictation_config({})

    assert config.api_key == "auth-file-key"
    assert config.model == voice.DEFAULT_VOICE_DICTATION_MODEL


def test_voice_config_rejects_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_api_key_credential(_vendor: str) -> AuthCredential:
        raise KeyError("missing")

    monkeypatch.setattr(voice, "get_api_key_credential", fake_get_api_key_credential)

    with pytest.raises(voice.VoiceDictationError, match="VOICE_DICTATION_API_KEY"):
        voice.resolve_voice_dictation_config({})
