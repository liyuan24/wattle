"""Provider error classifiers shared by higher-level request handling."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from wattle.providers.base import (
    ProviderAuthError,
    ProviderBillingError,
    ProviderError,
    ProviderInvalidRequestError,
    ProviderPermissionError,
    ProviderPolicyError,
    ProviderQuotaError,
    ProviderRequestTooLargeError,
    TransientProviderError,
)

_CONTEXT_LENGTH_CODES = {
    "context length exceeded",
    "context window exceeded",
    "max context length exceeded",
}
_CONTEXT_LENGTH_PATTERNS = (
    re.compile(r"\b(?:maximum|max)\s+context\s+length\b"),
    re.compile(r"\bcontext\s+length\s+(?:exceeded|exceeds)\b"),
    re.compile(r"\bcontext\s+window\s+(?:exceeded|exceeds|overflow)\b"),
    re.compile(r"\b(?:exceeded|exceeds|exceeding)\s+(?:the\s+)?context\s+window\b"),
    re.compile(r"\b(?:larger|longer)\s+than\s+(?:the\s+)?context\s+window\b"),
    re.compile(r"\bprompt\s+is\s+too\s+long\b"),
    re.compile(r"\binput\s+(?:is\s+)?too\s+long\b"),
    re.compile(r"\binput\s+tokens?\s+(?:exceeded|exceeds|exceed)\b"),
    re.compile(r"\btoo\s+many\s+input\s+tokens?\b"),
)

_RETRYABLE_STATUS_CODES = {408, 500, 502, 503, 504, 529}
_RETRYABLE_CODES = {
    "api_error",
    "internal_server_error",
    "overloaded_error",
    "rate_limit_error",
    "rate_limit_exceeded",
    "server_error",
    "server_is_overloaded",
    "slow_down",
    "timeout_error",
}
_QUOTA_CODES = {"insufficient_quota", "usage_limit_reached", "quota_exceeded"}
_AUTH_CODES = {"authentication_error", "invalid_api_key"}
_PERMISSION_CODES = {"permission_error", "permission_denied"}
_BILLING_CODES = {"billing_error"}
_POLICY_CODES = {"cyber_policy", "policy_error", "safety_error"}
_INVALID_REQUEST_CODES = {
    "bad_request",
    "invalid_prompt",
    "invalid_request",
    "invalid_request_error",
    "not_found_error",
    "unprocessable_entity",
    "unprocessable_entity_error",
}
_REQUEST_TOO_LARGE_CODES = {"request_too_large"}
_QUOTA_WORD_RE = re.compile(
    r"\b(?:quota|billing|bill|spend|usage limit|plan limit|insufficient_quota)\b", re.I
)
_RATE_LIMIT_WORD_RE = re.compile(r"\b(?:rate limit|rate_limit|too many requests)\b", re.I)
_RETRY_AFTER_RE = re.compile(r"(?i)try again in\s*(\d+(?:\.\d+)?)\s*(ms|s|seconds?)")


@dataclass(frozen=True, slots=True)
class ProviderErrorPayload:
    message: str | None = None
    code: str | None = None
    type: str | None = None
    status_code: int | None = None
    headers: Mapping[str, Any] | None = None
    request_id: str | None = None
    raw: object | None = None


def is_context_length_error(error: object) -> bool:
    """Return whether a provider error indicates context-window overflow."""

    for value in _error_values(error):
        normalized = _normalize(value)
        if not normalized:
            continue
        if normalized in _CONTEXT_LENGTH_CODES:
            return True
        if any(pattern.search(normalized) for pattern in _CONTEXT_LENGTH_PATTERNS):
            return True
    return False


def normalize_provider_error(error: object, *, provider: str | None = None) -> BaseException:
    """Map a provider/SDK error shape to Wattle's normalized provider errors."""

    if isinstance(error, (TransientProviderError, ProviderError)):
        return error

    payload = extract_error_payload(error)
    status_code = status_code_from_error(error)
    if status_code is None and payload is not None:
        status_code = payload.status_code
    headers = headers_from_error(error) or (payload.headers if payload is not None else None)
    retry_after = retry_after_from_headers(headers)
    code = error_code(payload) or error_code(error)
    message = error_message(payload) or error_message(error) or str(error) or type(error).__name__
    request_id = request_id_from_headers(headers) or (payload.request_id if payload else None)
    retry_after = retry_after if retry_after is not None else retry_after_from_message(message)

    metadata = {
        "status_code": status_code,
        "code": code,
        "retry_after": retry_after,
        "request_id": request_id,
        "provider": provider,
    }

    if is_context_length_error(error) or is_context_length_error(payload):
        return RuntimeError(f"context_length_exceeded: {message}")

    if _is_auth_error(code, status_code):
        return ProviderAuthError(message, **metadata)
    if _is_permission_error(code, status_code):
        return ProviderPermissionError(message, **metadata)
    if _is_billing_error(code, status_code):
        return ProviderBillingError(message, **metadata)
    if _is_policy_error(code):
        return ProviderPolicyError(message, **metadata)
    if _is_request_too_large_error(code, status_code):
        return ProviderRequestTooLargeError(message, **metadata)
    if _is_quota_error(code, message) and not _is_rate_limit_error(code, message):
        return ProviderQuotaError(message, **metadata)

    if _is_retryable_error(error, code, message, status_code):
        return TransientProviderError(message, **metadata)

    if _is_invalid_request_error(code, status_code):
        return ProviderInvalidRequestError(message, **metadata)

    return error if isinstance(error, BaseException) else RuntimeError(message)


def raise_normalized_provider_error(error: object, *, provider: str | None = None) -> None:
    normalized = normalize_provider_error(error, provider=provider)
    if isinstance(normalized, BaseException):
        if isinstance(error, BaseException):
            raise normalized from error
        raise normalized
    raise RuntimeError(str(normalized))


def extract_error_payload(error: object) -> ProviderErrorPayload | None:
    return _extract_error_payload(error, set())


def error_code(payload_or_error: object) -> str | None:
    if isinstance(payload_or_error, ProviderErrorPayload):
        if payload_or_error.code or payload_or_error.type:
            return payload_or_error.code or payload_or_error.type
        if isinstance(payload_or_error.raw, Mapping):
            return error_code(payload_or_error.raw)
    for value in _walk_payload_values(payload_or_error):
        if isinstance(value, ProviderErrorPayload):
            if value.code or value.type:
                return value.code or value.type
            if isinstance(value.raw, Mapping):
                nested_code = error_code(value.raw)
                if nested_code:
                    return nested_code
        elif isinstance(value, Mapping):
            for key in ("code", "type"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate:
                    return candidate
    return None


def error_message(payload_or_error: object) -> str | None:
    if isinstance(payload_or_error, ProviderErrorPayload):
        return payload_or_error.message
    if isinstance(payload_or_error, BaseException):
        text = str(payload_or_error)
        if text:
            return text
    for value in _walk_payload_values(payload_or_error):
        if isinstance(value, ProviderErrorPayload) and value.message:
            return value.message
        if isinstance(value, Mapping):
            candidate = value.get("message")
            if isinstance(candidate, str) and candidate:
                return candidate
    return None


def status_code_from_error(error: object) -> int | None:
    if isinstance(error, ProviderErrorPayload):
        return error.status_code
    for attr in ("status_code", "status", "code"):
        value = getattr(error, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int):
        return value
    payload = extract_error_payload(error)
    return payload.status_code if payload is not None else None


def headers_from_error(error: object) -> Mapping[str, Any] | None:
    if isinstance(error, ProviderErrorPayload):
        return error.headers
    headers = getattr(error, "headers", None)
    if isinstance(headers, Mapping):
        return headers
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if isinstance(headers, Mapping):
        return headers
    return None


def retry_after_from_headers(headers: object) -> float | None:
    if headers is None:
        return None
    get = getattr(headers, "get", None)
    if not callable(get):
        return None
    value = get("retry-after") or get("Retry-After")
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def retry_after_from_message(message: str | None) -> float | None:
    if not message:
        return None
    match = _RETRY_AFTER_RE.search(message)
    if match is None:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    return value / 1000 if unit == "ms" else value


def request_id_from_headers(headers: object) -> str | None:
    if headers is None:
        return None
    get = getattr(headers, "get", None)
    if not callable(get):
        return None
    for name in ("request-id", "x-request-id", "x-oai-request-id"):
        value = get(name) or get(name.title())
        if value is not None:
            return str(value)
    return None


def _extract_error_payload(error: object, seen: set[int]) -> ProviderErrorPayload | None:
    if error is None:
        return None
    error_id = id(error)
    if error_id in seen:
        return None
    seen.add(error_id)

    if isinstance(error, ProviderErrorPayload):
        return error
    if isinstance(error, Mapping):
        return _payload_from_mapping(error)
    if isinstance(error, bytes):
        return _payload_from_text(error.decode("utf-8", errors="replace"))
    if isinstance(error, str):
        return _payload_from_text(error)

    for attr in ("body", "response"):
        try:
            value = getattr(error, attr)
        except Exception:
            continue
        payload = _extract_error_payload(value, seen)
        if payload is not None:
            return payload

    for attr in ("text", "content"):
        try:
            value = getattr(error, attr)
        except Exception:
            continue
        payload = _extract_error_payload(value, seen)
        if payload is not None:
            return payload

    if isinstance(error, BaseException):
        if error.__cause__ is not None:
            payload = _extract_error_payload(error.__cause__, seen)
            if payload is not None:
                return payload
        if error.__context__ is not None:
            payload = _extract_error_payload(error.__context__, seen)
            if payload is not None:
                return payload
    return None


def _payload_from_text(text: str) -> ProviderErrorPayload | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return _payload_from_mapping(parsed) if isinstance(parsed, Mapping) else None


def _payload_from_mapping(mapping: Mapping[str, object]) -> ProviderErrorPayload | None:
    raw_error = mapping.get("error")
    request_id = _string_value(mapping, "request_id")
    status_code = _int_value(mapping, "status") or _int_value(mapping, "status_code")
    headers = mapping.get("headers") if isinstance(mapping.get("headers"), Mapping) else None

    if not isinstance(raw_error, Mapping) and any(
        isinstance(mapping.get(key), Mapping) for key in ("raw", "body")
    ):
        for key in ("raw", "body"):
            nested_raw = mapping.get(key)
            if isinstance(nested_raw, Mapping):
                nested = _payload_from_mapping(nested_raw)
                if nested is not None:
                    return ProviderErrorPayload(
                        message=_string_value(mapping, "message") or nested.message,
                        code=_string_value(mapping, "code") or nested.code,
                        type=_string_value(mapping, "type") or nested.type,
                        status_code=status_code or nested.status_code,
                        headers=headers or nested.headers,
                        request_id=request_id or nested.request_id,
                        raw=nested_raw,
                    )

    if isinstance(raw_error, Mapping):
        nested = _payload_from_mapping(raw_error)
        if nested is not None:
            return ProviderErrorPayload(
                message=nested.message,
                code=nested.code,
                type=nested.type,
                status_code=nested.status_code or status_code,
                headers=nested.headers or headers,
                request_id=nested.request_id or request_id,
                raw=mapping,
            )
    if any(key in mapping for key in ("message", "code", "type")):
        return ProviderErrorPayload(
            message=_string_value(mapping, "message"),
            code=_string_value(mapping, "code"),
            type=_string_value(mapping, "type"),
            status_code=status_code,
            headers=headers,
            request_id=request_id,
            raw=mapping,
        )
    return None


def _walk_payload_values(value: object) -> Iterator[object]:
    payload = extract_error_payload(value)
    if payload is not None:
        yield payload
        if isinstance(payload.raw, Mapping):
            yield payload.raw
    if isinstance(value, Mapping):
        yield value
        raw_error = value.get("error")
        if isinstance(raw_error, Mapping):
            yield raw_error


def _is_retryable_error(
    error: object,
    code: str | None,
    message: str,
    status_code: int | None,
) -> bool:
    if _is_quota_error(code, message) and not _is_rate_limit_error(code, message):
        return False
    class_name = type(error).__name__.lower()
    if any(term in class_name for term in ("connection", "timeout", "internalserver")):
        return True
    if "ratelimit" in class_name or "rate_limit" in class_name:
        return not _QUOTA_WORD_RE.search(message)
    if status_code == 429:
        return _is_rate_limit_error(code, message)
    if status_code in _RETRYABLE_STATUS_CODES:
        return True
    return code in _RETRYABLE_CODES and not _is_quota_error(code, message)


def _is_rate_limit_error(code: str | None, message: str) -> bool:
    if code in {"rate_limit_error", "rate_limit_exceeded"}:
        return not _QUOTA_WORD_RE.search(message)
    return bool(_RATE_LIMIT_WORD_RE.search(message) and not _QUOTA_WORD_RE.search(message))


def _is_quota_error(code: str | None, message: str) -> bool:
    return code in _QUOTA_CODES or bool(_QUOTA_WORD_RE.search(message))


def _is_auth_error(code: str | None, status_code: int | None) -> bool:
    return status_code == 401 or code in _AUTH_CODES


def _is_permission_error(code: str | None, status_code: int | None) -> bool:
    return status_code == 403 or code in _PERMISSION_CODES


def _is_billing_error(code: str | None, status_code: int | None) -> bool:
    return status_code == 402 or code in _BILLING_CODES


def _is_policy_error(code: str | None) -> bool:
    return code in _POLICY_CODES


def _is_request_too_large_error(code: str | None, status_code: int | None) -> bool:
    return status_code == 413 or code in _REQUEST_TOO_LARGE_CODES


def _is_invalid_request_error(code: str | None, status_code: int | None) -> bool:
    return status_code in {400, 404, 422} or code in _INVALID_REQUEST_CODES


def _string_value(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping.get(key)
    return value if isinstance(value, str) and value else None


def _int_value(mapping: Mapping[str, object], key: str) -> int | None:
    value = mapping.get(key)
    return value if isinstance(value, int) else None


def _error_values(error: object) -> Iterator[str]:
    seen: set[int] = set()
    yield from _walk_error_values(error, seen)


def _walk_error_values(error: object, seen: set[int]) -> Iterator[str]:
    if error is None:
        return

    error_id = id(error)
    if error_id in seen:
        return
    seen.add(error_id)

    if isinstance(error, str):
        yield error
        return

    if isinstance(error, bytes):
        yield error.decode("utf-8", errors="replace")
        return

    if isinstance(error, Mapping):
        for key in ("code", "type", "message", "error", "detail"):
            if key in error:
                yield from _walk_error_values(error[key], seen)
        return

    if isinstance(error, BaseException):
        yield str(error)
        if error.__cause__ is not None:
            yield from _walk_error_values(error.__cause__, seen)
        if error.__context__ is not None:
            yield from _walk_error_values(error.__context__, seen)

    for attr in ("code", "type", "message", "body", "response"):
        try:
            value = getattr(error, attr)
        except Exception:
            continue
        yield from _walk_error_values(value, seen)


def _normalize(value: str) -> str:
    return re.sub(r"[\s_-]+", " ", value).strip().lower()
