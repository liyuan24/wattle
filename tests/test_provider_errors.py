from __future__ import annotations

import pytest

from wattle.provider_errors import is_context_length_error


@pytest.mark.parametrize(
    "message",
    [
        "context_length_exceeded",
        "This model's maximum context length is 128000 tokens.",
        "Request failed: context length exceeded.",
        "prompt is too long: 220000 tokens > 200000 maximum",
        "input tokens exceeds the model limit",
        "input tokens exceed context window",
        "Your input is too long for this model.",
        "too many input tokens in request",
        "Codex error: request exceeds the context window.",
        "The request is larger than the context window.",
    ],
)
def test_is_context_length_error_matches_common_provider_messages(message: str) -> None:
    assert is_context_length_error(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "rate_limit_exceeded",
        "maximum output tokens reached",
        "tool call failed with context unavailable",
        "invalid API key",
        "connection window closed before response completed",
        "",
    ],
)
def test_is_context_length_error_rejects_unrelated_errors(message: str) -> None:
    assert is_context_length_error(message) is False


def test_is_context_length_error_reads_provider_payload_fields() -> None:
    error = {
        "error": {
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
            "message": "The request is too large.",
        }
    }

    assert is_context_length_error(error) is True


def test_is_context_length_error_reads_exception_attributes() -> None:
    class ProviderError(Exception):
        code = None

        def __init__(self) -> None:
            super().__init__("bad request")
            self.body = {"error": {"message": "Input tokens exceed context window"}}

    assert is_context_length_error(ProviderError()) is True


def test_is_context_length_error_reads_exception_chain() -> None:
    cause = RuntimeError("prompt is too long for this model")

    try:
        raise ValueError("provider request failed") from cause
    except ValueError as exc:
        assert is_context_length_error(exc) is True
