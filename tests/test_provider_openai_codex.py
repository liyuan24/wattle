"""Tests for the ChatGPT/Codex OAuth Responses provider."""

from __future__ import annotations

import base64
import json
import urllib.request
from typing import Any

from wattle.providers import (
    CompletionRequest,
    CompletionResponse,
    Message,
    OpenAICodexResponsesProvider,
    StreamComplete,
    TextBlock,
    TextDelta,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseDelta,
)


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
        self._payload = "".join(
            f"data: {json.dumps(event)}\n\n" for event in events
        ).encode("utf-8")
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


class _LineOnlySSE(_FakeSSE):
    def read(self, size: int = -1) -> bytes:
        raise AssertionError("SSE parser should not wait for fixed-size reads")


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
    response = provider.complete(
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
        )
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

    response = provider.complete(
        CompletionRequest(
            model="gpt-5.5",
            max_tokens=512,
            messages=[Message(role="user", content=[TextBlock(text="hello")])],
        )
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

    response = provider.complete(
        CompletionRequest(
            model="gpt-5.5",
            max_tokens=512,
            messages=[Message(role="user", content=[TextBlock(text="hello")])],
        )
    )

    assert response.usage["quota_5h_remaining_percent"] == 72
    assert response.usage["quota_1w_remaining_percent"] == 90


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

    first = provider.complete(
        CompletionRequest(
            model="gpt-5.5",
            max_tokens=512,
            messages=[user_msg],
        )
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

    second = provider.complete(
        CompletionRequest(
            model="gpt-5.5",
            max_tokens=512,
            messages=[user_msg, assistant_msg, user_tool_result],
        )
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
        }
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
        provider.stream(
            CompletionRequest(
                model="gpt-5.5",
                max_tokens=512,
                messages=[Message(role="user", content=[TextBlock(text="hello")])],
            )
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
        provider.stream(
            CompletionRequest(
                model="gpt-5.5",
                max_tokens=512,
                messages=[Message(role="user", content=[TextBlock(text="hello")])],
            )
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

    response = provider.complete(
        CompletionRequest(
            model="gpt-5.5",
            max_tokens=512,
            messages=[Message(role="user", content=[TextBlock(text="hello")])],
        )
    )

    assert response.stop_reason == "end_turn"
    assert response.content == [TextBlock(text="OK")]
