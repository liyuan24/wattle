"""Provider error classifiers shared by higher-level request handling."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping

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


def is_context_length_error(error: object) -> bool:
    """Return whether a provider error indicates context-window overflow.

    The classifier accepts raw provider messages, exceptions, and common SDK
    error payloads. It intentionally does not mutate or wrap the error; callers
    can decide whether to compact, retry, or surface the original failure.
    """

    for value in _error_values(error):
        normalized = _normalize(value)
        if not normalized:
            continue
        if normalized in _CONTEXT_LENGTH_CODES:
            return True
        if any(pattern.search(normalized) for pattern in _CONTEXT_LENGTH_PATTERNS):
            return True
    return False


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
