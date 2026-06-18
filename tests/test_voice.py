"""Tests for voice dictation helpers."""

from __future__ import annotations

import pytest

from wattle import voice


def test_voice_config_uses_voice_api_key_env() -> None:
    config = voice.resolve_voice_dictation_config(
        {
            "WATTLE_VOICE_DICTATION_API_KEY": " voice-key ",
            "VOICE_DICTATION_MODEL": " custom-transcribe ",
        }
    )

    assert config.api_key == "voice-key"
    assert config.model == "custom-transcribe"


def test_voice_config_rejects_missing_api_key() -> None:
    with pytest.raises(voice.VoiceDictationError, match="WATTLE_VOICE_DICTATION_API_KEY"):
        voice.resolve_voice_dictation_config({})


def test_openai_voice_auth_error_mentions_dictation_key() -> None:
    class FakeOpenAIAuthError(Exception):
        status_code = 401
        code = "invalid_api_key"

    message = voice._openai_voice_auth_error_message(FakeOpenAIAuthError("bad key"))

    assert message is not None
    assert "OpenAI voice dictation API key is not working" in message
    assert "WATTLE_VOICE_DICTATION_API_KEY" in message
