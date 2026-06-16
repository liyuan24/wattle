from __future__ import annotations

import pytest

from wattle.provider_errors import is_context_length_error, normalize_provider_error
from wattle.providers import (
    ProviderAuthError,
    ProviderBillingError,
    ProviderInvalidRequestError,
    ProviderQuotaError,
    ProviderRequestTooLargeError,
    TransientProviderError,
)


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


class _ProviderException(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: object | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}


class _ResponseException(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        body: object | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.body = body
        self.response = type(
            "Response",
            (),
            {"status_code": status_code, "headers": headers or {}},
        )()


def test_normalize_openai_429_rate_limit_payload_is_retryable() -> None:
    err = _ProviderException(
        "rate limit",
        status_code=429,
        body={"error": {"code": "rate_limit_exceeded", "message": "try again in 2s"}},
    )

    normalized = normalize_provider_error(err, provider="openai")

    assert isinstance(normalized, TransientProviderError)
    assert normalized.status_code == 429
    assert normalized.retry_after == 2
    assert normalized.provider == "openai"


def test_normalize_openai_429_quota_payload_is_non_retryable() -> None:
    err = _ProviderException(
        "quota",
        status_code=429,
        body={"error": {"code": "insufficient_quota", "message": "quota exceeded"}},
    )

    normalized = normalize_provider_error(err)

    assert isinstance(normalized, ProviderQuotaError)
    assert normalized.status_code == 429


def test_normalize_openai_503_overload_is_retryable() -> None:
    err = _ProviderException(
        "overload",
        status_code=503,
        body={"error": {"code": "server_is_overloaded", "message": "busy"}},
    )

    normalized = normalize_provider_error(err)

    assert isinstance(normalized, TransientProviderError)
    assert normalized.code == "server_is_overloaded"


def test_normalize_nested_server_error_event_is_retryable() -> None:
    err = {
        "type": "error",
        "error": {
            "type": "server_error",
            "code": "server_error",
            "message": "An error occurred while processing your request.",
        },
        "sequence_number": 6,
    }

    normalized = normalize_provider_error(err, provider="openai_codex")

    assert isinstance(normalized, TransientProviderError)
    assert normalized.code == "server_error"
    assert normalized.provider == "openai_codex"
    assert "processing your request" in str(normalized)


def test_normalize_anthropic_529_overloaded_is_retryable() -> None:
    err = _ProviderException(
        "overloaded",
        status_code=529,
        body={
            "type": "error",
            "request_id": "req_123",
            "error": {"type": "overloaded_error", "message": "overloaded"},
        },
    )

    normalized = normalize_provider_error(err, provider="anthropic")

    assert isinstance(normalized, TransientProviderError)
    assert normalized.status_code == 529
    assert normalized.request_id == "req_123"


def test_normalize_anthropic_429_rate_limit_uses_retry_after() -> None:
    err = _ProviderException(
        "rate limit",
        status_code=429,
        body={"error": {"type": "rate_limit_error", "message": "rate limit"}},
        headers={"retry-after": "6"},
    )

    normalized = normalize_provider_error(err)

    assert isinstance(normalized, TransientProviderError)
    assert normalized.retry_after == 6


def test_normalize_anthropic_billing_and_request_too_large_are_non_retryable() -> None:
    billing = normalize_provider_error(
        _ProviderException(
            "billing",
            status_code=402,
            body={"error": {"type": "billing_error", "message": "billing required"}},
        )
    )
    large = normalize_provider_error(
        _ProviderException(
            "too large",
            status_code=413,
            body={"error": {"type": "request_too_large", "message": "too large"}},
        )
    )

    assert isinstance(billing, ProviderBillingError)
    assert isinstance(large, ProviderRequestTooLargeError)


def test_normalize_context_payload_preserves_compaction_signal() -> None:
    err = _ProviderException(
        "bad request",
        status_code=400,
        body={"error": {"code": "context_length_exceeded", "message": "too long"}},
    )

    normalized = normalize_provider_error(err)

    assert isinstance(normalized, RuntimeError)
    assert is_context_length_error(normalized)


def test_normalize_unknown_status_fallbacks_are_conservative() -> None:
    retryable = normalize_provider_error(_ProviderException("bad gateway", status_code=502))
    non_retryable = normalize_provider_error(_ProviderException("bad request", status_code=400))
    auth = normalize_provider_error(_ResponseException("auth", status_code=401))

    assert isinstance(retryable, TransientProviderError)
    assert isinstance(non_retryable, ProviderInvalidRequestError)
    assert isinstance(auth, ProviderAuthError)
