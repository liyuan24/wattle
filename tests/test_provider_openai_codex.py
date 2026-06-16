"""Tests for the ChatGPT/Codex OAuth Responses provider."""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

import anyio

from wattle.providers import (
    CompletionRequest,
    CompletionResponse,
    IncompleteStreamError,
    Message,
    OpenAICodexResponsesProvider,
    StreamComplete,
    StreamEvent,
    TextBlock,
    TextDelta,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseDelta,
    TransientProviderError,
)


def _complete(provider, request):
    return anyio.run(provider.acomplete, request)


async def _collect_stream(provider, request):
    return [event async for event in provider.astream(request)]


def _stream(provider, request):
    return anyio.run(_collect_stream, provider, request)


def _jwt(payload: dict[str, Any]) -> str:
    def encode(data: dict[str, Any]) -> str:
        raw = json.dumps(data).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{encode({'alg': 'none'})}.{encode(payload)}.sig"


def _token() -> str:
    return _jwt(
        {
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct_123",
            }
        }
    )


class _FakeSSE:
    def __init__(
        self,
        events: list[dict[str, Any]],
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._payload = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode(
            "utf-8"
        )
        self._offset = 0
        self.headers = headers or {}

    def __enter__(self) -> _FakeSSE:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._payload):
            return b""
        if size < 0:
            size = len(self._payload) - self._offset
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def readline(self) -> bytes:
        if self._offset >= len(self._payload):
            return b""
        end = self._payload.find(b"\n", self._offset)
        if end == -1:
            end = len(self._payload) - 1
        chunk = self._payload[self._offset : end + 1]
        self._offset += len(chunk)
        return chunk


class _FakeJsonResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _FakeJsonResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _size: int = -1) -> bytes:
        return self._payload


class _LineOnlySSE(_FakeSSE):
    def read(self, size: int = -1) -> bytes:
        raise AssertionError("SSE parser should not wait for fixed-size reads")


class _DelayedSSE:
    def __init__(self, lines: list[tuple[float, bytes]]) -> None:
        self._lines = list(lines)
        self.headers: dict[str, str] = {}

    def __enter__(self) -> _DelayedSSE:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def readline(self) -> bytes:
        if not self._lines:
            return b""
        delay, line = self._lines.pop(0)
        if delay:
            time.sleep(delay)
        return line


class _TimeoutSSE:
    headers: dict[str, str] = {}

    def __enter__(self) -> _TimeoutSSE:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def readline(self) -> bytes:
        raise TimeoutError("timed out")


def test_codex_provider_builds_chatgpt_backend_request() -> None:
    captured: list[urllib.request.Request] = []

    def urlopen(req: urllib.request.Request) -> _FakeSSE:
        captured.append(req)
        return _FakeSSE(
            [
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "status": "completed",
                        "output": [],
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 2,
                            "input_tokens_details": {"cached_tokens": 1},
                        },
                    },
                }
            ]
        )

    provider = OpenAICodexResponsesProvider(
        bearer_token=_token(),
        urlopen=urlopen,
        session_id="session_123",
        thread_id="thread_123",
    )
    response = _complete(
        provider,
        CompletionRequest(
            model="gpt-5.5",
            max_tokens=512,
            system="sys",
            messages=[Message(role="user", content=[TextBlock(text="hello")])],
            tools=[
                {
                    "name": "read",
                    "description": "Read a file.",
                    "input_schema": {"type": "object"},
                }
            ],
        ),
    )
    assert response.usage == {"input_tokens": 1, "output_tokens": 2, "cached_tokens": 1}

    req = captured[0]
    assert req.full_url == "https://chatgpt.com/backend-api/codex/responses"
    headers = dict(req.header_items())
    assert headers["Authorization"] == f"Bearer {_token()}"
    assert headers["Chatgpt-account-id"] == "acct_123"
    assert headers["Originator"] == "wattle"
    assert headers["Openai-beta"] == "responses=experimental"
    assert headers["Accept"] == "text/event-stream"
    assert headers["Session-id"] == "session_123"
    assert headers["Thread-id"] == "thread_123"
    assert headers["X-client-request-id"] == "thread_123"

    body = json.loads(req.data.decode("utf-8"))  # type: ignore[union-attr]
    assert body["model"] == "gpt-5.5"
    assert body["store"] is False
    assert body["stream"] is True
    assert body["instructions"] == "sys"
    assert body["prompt_cache_key"] == "thread_123"
    assert "max_output_tokens" not in body
    assert "previous_response_id" not in body
    assert body["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]
    assert body["tools"] == [
        {
            "type": "function",
            "name": "read",
            "description": "Read a file.",
            "parameters": {"type": "object"},
            "strict": None,
        }
    ]


def test_codex_provider_passes_stream_idle_timeout_to_urlopen() -> None:
    captured_timeout: list[float | None] = []

    def urlopen(req: urllib.request.Request, *, timeout: float | None = None) -> _FakeSSE:
        captured_timeout.append(timeout)
        return _FakeSSE(
            [
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "status": "completed",
                        "output": [],
                        "usage": {},
                    },
                }
            ]
        )

    provider = OpenAICodexResponsesProvider(
        bearer_token=_token(),
        urlopen=urlopen,
        stream_idle_timeout_seconds=12.5,
    )

    _complete(provider, _request())

    assert captured_timeout == [12.5]


def test_codex_provider_reads_rate_limit_headers_into_usage() -> None:
    provider = OpenAICodexResponsesProvider(
        bearer_token=_token(),
        urlopen=lambda _req: _FakeSSE(
            [
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "status": "completed",
                        "output": [],
                        "usage": {"input_tokens": 3, "output_tokens": 4},
                    },
                }
            ],
            headers={
                "x-codex-primary-used-percent": "40",
                "x-codex-primary-window-minutes": "300",
                "x-codex-secondary-used-percent": "94",
                "x-codex-secondary-window-minutes": "10080",
            },
        ),
    )

    response = _complete(
        provider,
        CompletionRequest(
            model="gpt-5.5",
            max_tokens=512,
            messages=[Message(role="user", content=[TextBlock(text="hello")])],
        ),
    )

    assert response.usage == {
        "input_tokens": 3,
        "output_tokens": 4,
        "quota_5h_remaining_percent": 60,
        "quota_1w_remaining_percent": 6,
    }


def test_codex_provider_reads_streamed_rate_limit_event_into_usage() -> None:
    provider = OpenAICodexResponsesProvider(
        bearer_token=_token(),
        urlopen=lambda _req: _FakeSSE(
            [
                {
                    "type": "codex.rate_limits",
                    "rate_limits": {
                        "primary": {
                            "used_percent": 28.4,
                            "window_minutes": 300,
                            "reset_at": 1_700_000_000,
                        },
                        "secondary": {
                            "used_percent": 10.1,
                            "window_minutes": 10080,
                            "reset_at": 1_700_000_001,
                        },
                    },
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "status": "completed",
                        "output": [],
                        "usage": {"input_tokens": 3, "output_tokens": 4},
                    },
                },
            ],
        ),
    )

    response = _complete(
        provider,
        CompletionRequest(
            model="gpt-5.5",
            max_tokens=512,
            messages=[Message(role="user", content=[TextBlock(text="hello")])],
        ),
    )

    assert response.usage["quota_5h_remaining_percent"] == 72
    assert response.usage["quota_1w_remaining_percent"] == 90


def test_codex_provider_fetches_quota_usage_from_chatgpt_backend() -> None:
    captured: list[urllib.request.Request] = []

    def urlopen(req: urllib.request.Request) -> _FakeJsonResponse:
        captured.append(req)
        return _FakeJsonResponse(
            {
                "plan_type": "pro",
                "rate_limit": {
                    "allowed": True,
                    "limit_reached": False,
                    "primary_window": {
                        "used_percent": 28,
                        "limit_window_seconds": 18_000,
                        "reset_after_seconds": 10,
                        "reset_at": 1_700_000_000,
                    },
                    "secondary_window": {
                        "used_percent": 7,
                        "limit_window_seconds": 604_800,
                        "reset_after_seconds": 20,
                        "reset_at": 1_700_000_001,
                    },
                },
            }
        )

    provider = OpenAICodexResponsesProvider(
        bearer_token=_token(),
        urlopen=urlopen,
    )

    assert provider.fetch_quota_usage() == {
        "quota_5h_remaining_percent": 72,
        "quota_1w_remaining_percent": 93,
    }

    req = captured[0]
    assert req.full_url == "https://chatgpt.com/backend-api/wham/usage"
    assert req.get_method() == "GET"
    headers = dict(req.header_items())
    assert headers["Authorization"] == f"Bearer {_token()}"
    assert headers["Chatgpt-account-id"] == "acct_123"
    assert headers["Originator"] == "wattle"


def test_codex_provider_is_stateless_and_resends_full_history() -> None:
    captured: list[urllib.request.Request] = []

    def urlopen(req: urllib.request.Request) -> _FakeSSE:
        captured.append(req)
        return _FakeSSE(
            [
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "status": "completed",
                        "output": [],
                        "usage": {"input_tokens": 10, "output_tokens": 20},
                    },
                }
            ]
        )

    provider = OpenAICodexResponsesProvider(
        bearer_token=_token(),
        urlopen=urlopen,
        session_id="session_123",
        thread_id="thread_123",
    )
    user_msg = Message(role="user", content=[TextBlock(text="run a tool")])

    first = _complete(
        provider,
        CompletionRequest(
            model="gpt-5.5",
            max_tokens=512,
            messages=[user_msg],
        ),
    )

    first_body = json.loads(captured[0].data.decode("utf-8"))  # type: ignore[union-attr]
    assert first_body["store"] is False
    assert first_body["prompt_cache_key"] == "thread_123"
    assert "previous_response_id" not in first_body
    assert first_body["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "run a tool"}],
        }
    ]
    assert first.usage == {"input_tokens": 10, "output_tokens": 20}

    assistant_msg = Message(
        role="assistant",
        content=[ToolUseBlock(id="call_1", name="bash", input={"cmd": "ls"})],
    )
    user_tool_result = Message(
        role="user",
        content=[ToolResultBlock(tool_use_id="call_1", content="ok")],
    )

    second = _complete(
        provider,
        CompletionRequest(
            model="gpt-5.5",
            max_tokens=512,
            messages=[user_msg, assistant_msg, user_tool_result],
        ),
    )

    second_body = json.loads(captured[1].data.decode("utf-8"))  # type: ignore[union-attr]
    assert second_body["store"] is False
    assert second_body["prompt_cache_key"] == "thread_123"
    assert "previous_response_id" not in second_body
    assert second_body["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "run a tool"}],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "bash",
            "arguments": '{"cmd": "ls"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "ok",
        },
    ]
    assert second.usage == {"input_tokens": 10, "output_tokens": 20}

    first_headers = dict(captured[0].header_items())
    second_headers = dict(captured[1].header_items())
    assert first_headers["Session-id"] == second_headers["Session-id"] == "session_123"
    assert first_headers["Thread-id"] == second_headers["Thread-id"] == "thread_123"
    assert (
        first_headers["X-client-request-id"]
        == second_headers["X-client-request-id"]
        == "thread_123"
    )


def test_codex_provider_fork_gets_distinct_cache_identity() -> None:
    parent = OpenAICodexResponsesProvider(
        bearer_token=_token(),
        session_id="session_parent",
        thread_id="thread_parent",
    )

    child = parent.fork()

    assert child.session_id != parent.session_id
    assert child.thread_id != parent.thread_id


def test_codex_provider_streams_text_and_tool_calls() -> None:
    events = [
        {"type": "response.output_text.delta", "delta": "hi"},
        {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "call_id": "call_1",
                "name": "read",
            },
        },
        {"type": "response.function_call_arguments.delta", "delta": '{"path":'},
        {"type": "response.function_call_arguments.delta", "delta": '"x"}'},
        {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "hi"}],
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "read",
                        "arguments": '{"path":"x"}',
                    },
                ],
                "usage": {"input_tokens": 3, "output_tokens": 4},
            },
        },
    ]
    provider = OpenAICodexResponsesProvider(
        bearer_token=_token(),
        urlopen=lambda _req: _FakeSSE(events),
    )

    emitted = list(
        _stream(
            provider,
            CompletionRequest(
                model="gpt-5.5",
                max_tokens=512,
                messages=[Message(role="user", content=[TextBlock(text="hello")])],
            ),
        )
    )

    assert isinstance(emitted[0], TextDelta)
    assert emitted[0].text == "hi"
    assert isinstance(emitted[1], ToolUseDelta)
    assert emitted[1].id == "call_1"
    assert emitted[1].name == "read"
    assert isinstance(emitted[2], ToolUseDelta)
    assert emitted[2].partial_json == '{"path":'
    assert isinstance(emitted[3], ToolUseDelta)
    assert emitted[3].partial_json == '"x"}'
    assert isinstance(emitted[4], StreamComplete)
    response = emitted[4].response
    assert response.stop_reason == "tool_use"
    assert response.usage == {"input_tokens": 3, "output_tokens": 4}
    assert response.content == [
        TextBlock(text="hi"),
        ToolUseBlock(id="call_1", name="read", input={"path": "x"}),
    ]


def test_codex_provider_reads_sse_incrementally_by_line() -> None:
    provider = OpenAICodexResponsesProvider(
        bearer_token=_token(),
        urlopen=lambda _req: _LineOnlySSE(
            [
                {"type": "response.output_text.delta", "delta": "hi"},
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "status": "completed",
                        "output": [],
                        "usage": {"input_tokens": 3, "output_tokens": 4},
                    },
                },
            ]
        ),
    )

    emitted = list(
        _stream(
            provider,
            CompletionRequest(
                model="gpt-5.5",
                max_tokens=512,
                messages=[Message(role="user", content=[TextBlock(text="hello")])],
            ),
        )
    )

    assert emitted == [
        TextDelta(text="hi"),
        StreamComplete(
            response=CompletionResponse(
                content=[TextBlock(text="hi")],
                stop_reason="end_turn",
                usage={"input_tokens": 3, "output_tokens": 4},
            )
        ),
    ]


def test_codex_provider_yields_stream_events_before_completion() -> None:
    delta = json.dumps({"type": "response.output_text.delta", "delta": "hi"}).encode()
    completed = json.dumps(
        {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 3, "output_tokens": 4},
            },
        }
    ).encode()
    provider = OpenAICodexResponsesProvider(
        bearer_token=_token(),
        urlopen=lambda _req: _DelayedSSE(
            [
                (0.0, b"data: " + delta + b"\n"),
                (0.0, b"\n"),
                (1.0, b"data: " + completed + b"\n"),
                (0.0, b"\n"),
            ]
        ),
    )

    async def first_event() -> tuple[StreamEvent, float]:
        started_at = time.monotonic()
        stream = provider.astream(_request())
        try:
            event = await anext(stream)
            return event, time.monotonic() - started_at
        finally:
            await stream.aclose()

    event, elapsed = anyio.run(first_event)

    assert event == TextDelta(text="hi")
    assert elapsed < 0.5


def test_codex_provider_raises_retryable_error_for_incomplete_stream() -> None:
    provider = OpenAICodexResponsesProvider(
        bearer_token=_token(),
        urlopen=lambda _req: _FakeSSE([{"type": "response.output_text.delta", "delta": "partial"}]),
    )

    try:
        list(
            _stream(
                provider,
                CompletionRequest(
                    model="gpt-5.5",
                    max_tokens=512,
                    messages=[Message(role="user", content=[TextBlock(text="hello")])],
                ),
            )
        )
    except IncompleteStreamError as exc:
        assert "without a completion event" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected incomplete stream to raise")


def test_codex_provider_raises_retryable_error_for_socket_idle_timeout() -> None:
    provider = OpenAICodexResponsesProvider(
        bearer_token=_token(),
        urlopen=lambda _req: _TimeoutSSE(),
        stream_idle_timeout_seconds=0.2,
    )

    try:
        _stream(provider, _request())
    except TransientProviderError as exc:
        assert exc.provider == "openai_codex"
        assert "idle for 0.2s" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected socket timeout to be retryable")


def _request() -> CompletionRequest:
    return CompletionRequest(
        model="gpt-5.5",
        max_tokens=512,
        messages=[Message(role="user", content=[TextBlock(text="hello")])],
    )


def _provider_for_events(events: list[dict[str, Any]]) -> OpenAICodexResponsesProvider:
    return OpenAICodexResponsesProvider(
        bearer_token=_token(),
        urlopen=lambda _req: _FakeSSE(events),
    )


def _response_failed_error(code: str, message: str | None = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code}
    if message is not None:
        error["message"] = message
    return {"type": "response.failed", "response": {"error": error}}


def test_codex_provider_response_failed_overload_is_retryable() -> None:
    provider = _provider_for_events([_response_failed_error("server_is_overloaded", "server busy")])

    try:
        _stream(provider, _request())
    except TransientProviderError as exc:
        assert "server busy" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected overload to be retryable")


def test_codex_provider_response_failed_slow_down_is_retryable() -> None:
    provider = _provider_for_events([_response_failed_error("slow_down", "slow down")])

    try:
        _stream(provider, _request())
    except TransientProviderError as exc:
        assert "slow down" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected slow_down to be retryable")


def test_codex_provider_response_failed_rate_limit_parses_retry_delay() -> None:
    provider = _provider_for_events(
        [
            _response_failed_error(
                "rate_limit_exceeded",
                "Rate limit exceeded, please try again in 2s.",
            )
        ]
    )

    try:
        _stream(provider, _request())
    except TransientProviderError as exc:
        assert exc.retry_after == 2
        assert "Rate limit exceeded" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected rate_limit_exceeded to be retryable")


def test_codex_provider_error_event_server_error_is_retryable() -> None:
    provider = _provider_for_events(
        [
            {
                "type": "error",
                "error": {
                    "type": "server_error",
                    "code": "server_error",
                    "message": (
                        "An error occurred while processing your request. "
                        "Please include the request ID req_123."
                    ),
                    "param": None,
                },
                "sequence_number": 6,
            }
        ]
    )

    try:
        _stream(provider, _request())
    except TransientProviderError as exc:
        assert exc.provider == "openai_codex"
        assert exc.code == "server_error"
        assert "processing your request" in str(exc)
        assert "sequence_number" not in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected streamed server_error to be retryable")


def test_codex_provider_response_failed_context_length_keeps_recovery_signal() -> None:
    provider = _provider_for_events(
        [_response_failed_error("context_length_exceeded", "too much context")]
    )

    try:
        _stream(provider, _request())
    except RuntimeError as exc:
        assert not isinstance(exc, TransientProviderError)
        assert "context_length_exceeded" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected context_length_exceeded to raise")


def test_codex_provider_response_failed_policy_is_not_retryable() -> None:
    provider = _provider_for_events([_response_failed_error("cyber_policy", "blocked")])

    try:
        _stream(provider, _request())
    except RuntimeError as exc:
        assert not isinstance(exc, TransientProviderError)
        assert "blocked" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected policy error to remain non-retryable")


def test_codex_provider_response_incomplete_unknown_reason_is_retryable() -> None:
    provider = _provider_for_events(
        [
            {
                "type": "response.incomplete",
                "response": {
                    "id": "resp_1",
                    "status": "incomplete",
                    "incomplete_details": {"reason": "content_filter"},
                    "output": [],
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            }
        ]
    )

    try:
        _stream(provider, _request())
    except IncompleteStreamError as exc:
        assert "content_filter" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected non-token incomplete response to raise")


def test_codex_provider_http_429_rate_limit_with_retry_after_is_retryable() -> None:
    provider = OpenAICodexResponsesProvider(
        bearer_token=_token(),
        urlopen=_http_error_urlopen(
            429,
            {"error": {"code": "rate_limit_exceeded", "message": "try again in 3s"}},
            headers={"retry-after": "5"},
        ),
    )

    try:
        _stream(provider, _request())
    except TransientProviderError as exc:
        assert exc.status_code == 429
        assert exc.retry_after == 5
    else:  # pragma: no cover
        raise AssertionError("expected retryable HTTP 429 rate limit")


def test_codex_provider_http_429_usage_limit_is_not_retryable() -> None:
    provider = OpenAICodexResponsesProvider(
        bearer_token=_token(),
        urlopen=_http_error_urlopen(
            429,
            {"error": {"type": "usage_limit_reached", "message": "limit reached"}},
        ),
    )

    try:
        _stream(provider, _request())
    except RuntimeError as exc:
        assert not isinstance(exc, TransientProviderError)
        assert "limit reached" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected usage limit to remain non-retryable")


def _http_error_urlopen(
    status: int,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
) -> Callable[[urllib.request.Request], _FakeSSE]:
    def urlopen(_req: urllib.request.Request) -> _FakeSSE:
        raise urllib.error.HTTPError(
            url="https://chatgpt.com/backend-api/codex/responses",
            code=status,
            msg="error",
            hdrs=headers or {},
            fp=io.BytesIO(json.dumps(payload).encode("utf-8")),
        )

    return urlopen


def test_codex_provider_raises_retryable_error_for_http_503() -> None:
    def urlopen(_req: urllib.request.Request) -> _FakeSSE:
        raise urllib.error.HTTPError(
            url="https://chatgpt.com/backend-api/codex/responses",
            code=503,
            msg="Service Unavailable",
            hdrs={"retry-after": "2"},
            fp=io.BytesIO(b"upstream connect error or disconnect/reset before headers"),
        )

    provider = OpenAICodexResponsesProvider(bearer_token=_token(), urlopen=urlopen)

    try:
        _stream(
            provider,
            CompletionRequest(
                model="gpt-5.5",
                max_tokens=512,
                messages=[Message(role="user", content=[TextBlock(text="hello")])],
            ),
        )
    except TransientProviderError as exc:
        assert exc.status_code == 503
        assert exc.retry_after == 2
        assert "HTTP 503" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected 503 to be retryable")


def test_codex_provider_does_not_retry_auth_http_errors() -> None:
    def urlopen(_req: urllib.request.Request) -> _FakeSSE:
        raise urllib.error.HTTPError(
            url="https://chatgpt.com/backend-api/codex/responses",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=io.BytesIO(b"bad token"),
        )

    provider = OpenAICodexResponsesProvider(bearer_token=_token(), urlopen=urlopen)

    try:
        _stream(
            provider,
            CompletionRequest(
                model="gpt-5.5",
                max_tokens=512,
                messages=[Message(role="user", content=[TextBlock(text="hello")])],
            ),
        )
    except RuntimeError as exc:
        assert not isinstance(exc, TransientProviderError)
        assert "HTTP 401" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected 401 to remain non-retryable")


def test_codex_provider_uses_streamed_text_when_final_output_is_empty() -> None:
    provider = OpenAICodexResponsesProvider(
        bearer_token=_token(),
        urlopen=lambda _req: _FakeSSE(
            [
                {"type": "response.output_text.delta", "delta": "OK"},
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "status": "completed",
                        "output": [],
                        "usage": {"input_tokens": 3, "output_tokens": 4},
                    },
                },
            ]
        ),
    )

    response = _complete(
        provider,
        CompletionRequest(
            model="gpt-5.5",
            max_tokens=512,
            messages=[Message(role="user", content=[TextBlock(text="hello")])],
        ),
    )

    assert response.stop_reason == "end_turn"
    assert response.content == [TextBlock(text="OK")]
