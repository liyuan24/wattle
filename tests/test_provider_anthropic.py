"""Tests for `AnthropicProvider`.

These tests never make real API calls; the SDK client is mocked so we only
exercise the request/response translation logic.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from wattle.providers import (
    AnthropicProvider,
    CompletionRequest,
    ImageBlock,
    Message,
    RedactedThinkingBlock,
    StreamComplete,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseDelta,
)


def _canned_response(
    *,
    content: list[object] | None = None,
    stop_reason: str = "end_turn",
    input_tokens: int = 1,
    output_tokens: int = 2,
    cache_creation_input_tokens: int | None = None,
    cache_read_input_tokens: int | None = None,
) -> SimpleNamespace:
    """Build a minimal stand-in for an `anthropic.types.Message` response."""
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    if cache_creation_input_tokens is not None:
        usage.cache_creation_input_tokens = cache_creation_input_tokens
    if cache_read_input_tokens is not None:
        usage.cache_read_input_tokens = cache_read_input_tokens
    return SimpleNamespace(
        content=content if content is not None else [],
        stop_reason=stop_reason,
        usage=usage,
    )


def _make_provider(response: SimpleNamespace) -> tuple[AnthropicProvider, MagicMock]:
    client = MagicMock()
    client.messages.create.return_value = response
    return AnthropicProvider(async_client=client), client


def test_request_translation_full_round_trip() -> None:
    """All input fields are translated to the SDK kwargs shape Anthropic wants."""
    tool_spec = {
        "name": "get_weather",
        "description": "Look up the weather.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
    request = CompletionRequest(
        model="claude-opus-4-7",
        max_tokens=512,
        system="You are a helpful weather agent.",
        tools=[tool_spec],
        messages=[
            Message(role="user", content=[TextBlock(text="What's it like in Paris?")]),
            Message(
                role="assistant",
                content=[
                    ToolUseBlock(
                        id="call_42",
                        name="get_weather",
                        input={"city": "Paris"},
                    ),
                ],
            ),
            Message(
                role="user",
                content=[
                    ToolResultBlock(
                        tool_use_id="call_42",
                        content="sunny, 22C",
                        is_error=False,
                    ),
                ],
            ),
        ],
    )
    provider, client = _make_provider(_canned_response())

    provider.complete(request)

    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-opus-4-7"
    assert kwargs["max_tokens"] == 512
    assert kwargs["cache_control"] == {"type": "ephemeral"}
    assert kwargs["system"] == [
        {
            "type": "text",
            "text": "You are a helpful weather agent.",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert kwargs["tools"] == [tool_spec]
    assert kwargs["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "What's it like in Paris?"}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_42",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_42",
                    "content": "sunny, 22C",
                    "is_error": False,
                }
            ],
        },
    ]


def test_user_image_block_translates_to_anthropic_image(tmp_path) -> None:
    image = tmp_path / "shot.png"
    image.write_bytes(b"image-bytes")
    request = CompletionRequest(
        model="claude-opus-4-7",
        max_tokens=512,
        messages=[
            Message(
                role="user",
                content=[
                    TextBlock(text="What is shown?"),
                    ImageBlock(
                        path=str(image),
                        media_type="image/png",
                        filename="shot.png",
                        size_bytes=11,
                    ),
                ],
            )
        ],
    )
    provider, client = _make_provider(_canned_response())

    provider.complete(request)

    assert client.messages.create.call_args.kwargs["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is shown?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "aW1hZ2UtYnl0ZXM=",
                    },
                },
            ],
        }
    ]


def test_system_none_is_omitted_from_kwargs() -> None:
    request = CompletionRequest(
        model="claude-opus-4-7",
        max_tokens=64,
        system=None,
        messages=[Message(role="user", content=[TextBlock(text="hi")])],
    )
    provider, client = _make_provider(_canned_response())

    provider.complete(request)

    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["cache_control"] == {"type": "ephemeral"}
    assert "system" not in kwargs


def test_response_translation_text_and_tool_use() -> None:
    """Mixed response content is mapped back to TextBlock / ToolUseBlock."""
    api_text = SimpleNamespace(type="text", text="Looking it up.")
    api_tool_use = SimpleNamespace(
        type="tool_use",
        id="call_99",
        name="get_weather",
        input={"city": "Tokyo"},
    )
    response = _canned_response(
        content=[api_text, api_tool_use],
        stop_reason="tool_use",
        input_tokens=11,
        output_tokens=7,
        cache_creation_input_tokens=5,
        cache_read_input_tokens=13,
    )
    provider, _client = _make_provider(response)

    request = CompletionRequest(
        model="claude-opus-4-7",
        max_tokens=64,
        messages=[Message(role="user", content=[TextBlock(text="weather?")])],
    )
    result = provider.complete(request)

    assert len(result.content) == 2
    text_block, tool_block = result.content
    assert isinstance(text_block, TextBlock)
    assert text_block.text == "Looking it up."
    assert isinstance(tool_block, ToolUseBlock)
    assert tool_block.id == "call_99"
    assert tool_block.name == "get_weather"
    assert tool_block.input == {"city": "Tokyo"}

    assert result.stop_reason == "tool_use"
    assert result.usage == {
        "input_tokens": 11,
        "output_tokens": 7,
        "cache_creation_input_tokens": 5,
        "cache_read_input_tokens": 13,
        "cached_tokens": 13,
    }


@pytest.mark.parametrize(
    "stop_reason",
    ["tool_use", "end_turn", "max_tokens", "stop_sequence"],
)
def test_stop_reason_passthrough(stop_reason: str) -> None:
    response = _canned_response(stop_reason=stop_reason)
    provider, _client = _make_provider(response)

    request = CompletionRequest(
        model="claude-opus-4-7",
        max_tokens=64,
        messages=[Message(role="user", content=[TextBlock(text="hi")])],
    )
    result = provider.complete(request)

    assert result.stop_reason == stop_reason


# ---------------------------------------------------------------------------
# Extended thinking
# ---------------------------------------------------------------------------


def test_thinking_block_round_trip_in_outgoing_messages() -> None:
    """ThinkingBlock serializes to the Anthropic `thinking` wire shape with
    its signature preserved, and an unsigned ThinkingBlock omits `signature`."""
    provider, client = _make_provider(_canned_response())

    request = CompletionRequest(
        model="claude-opus-4-7",
        max_tokens=64,
        thinking=True,
        budget=4096,
        messages=[
            Message(role="user", content=[TextBlock(text="think hard")]),
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(
                        thinking="Step 1: consider X.",
                        signature="sig_abc",
                    ),
                    ThinkingBlock(thinking="No signature here."),
                    TextBlock(text="Here's my answer."),
                ],
            ),
        ],
    )
    provider.complete(request)

    kwargs = client.messages.create.call_args.kwargs
    assistant_msg = kwargs["messages"][1]
    assert assistant_msg["content"] == [
        {
            "type": "thinking",
            "thinking": "Step 1: consider X.",
            "signature": "sig_abc",
        },
        {"type": "thinking", "thinking": "No signature here."},
        {"type": "text", "text": "Here's my answer."},
    ]


def test_manual_thinking_propagates_to_kwargs() -> None:
    provider, client = _make_provider(_canned_response())

    provider.complete(
        CompletionRequest(
            model="claude-opus-4-7",
            max_tokens=64,
            thinking=True,
            budget=10_000,
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        )
    )

    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 10_000}
    assert "output_config" not in kwargs


def test_thinking_false_omits_thinking_kwarg() -> None:
    provider, client = _make_provider(_canned_response())

    provider.complete(
        CompletionRequest(
            model="claude-opus-4-7",
            max_tokens=64,
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        )
    )

    kwargs = client.messages.create.call_args.kwargs
    assert "thinking" not in kwargs
    assert "output_config" not in kwargs


def test_adaptive_thinking_with_effort_emits_output_config() -> None:
    provider, client = _make_provider(_canned_response())

    provider.complete(
        CompletionRequest(
            model="claude-opus-4-7",
            max_tokens=64,
            thinking=True,
            effort="high",
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        )
    )

    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "high"}


def test_adaptive_thinking_no_effort_omits_output_config() -> None:
    provider, client = _make_provider(_canned_response())

    provider.complete(
        CompletionRequest(
            model="claude-opus-4-7",
            max_tokens=64,
            thinking=True,
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        )
    )

    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert "output_config" not in kwargs


def test_manual_thinking_ignores_effort() -> None:
    provider, client = _make_provider(_canned_response())

    provider.complete(
        CompletionRequest(
            model="claude-opus-4-7",
            max_tokens=64,
            thinking=True,
            budget=4096,
            effort="medium",
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        )
    )

    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert "output_config" not in kwargs


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_adaptive_effort_literals_pass_through(effort: str) -> None:
    provider, client = _make_provider(_canned_response())

    provider.complete(
        CompletionRequest(
            model="claude-opus-4-7",
            max_tokens=64,
            thinking=True,
            effort=effort,  # type: ignore[arg-type]
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        )
    )

    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": effort}


def test_thinking_block_response_translation() -> None:
    """Incoming `thinking` blocks become ThinkingBlocks with signature."""
    api_thinking = SimpleNamespace(
        type="thinking",
        thinking="Let me reason about this.",
        signature="sig_xyz",
    )
    api_text = SimpleNamespace(type="text", text="The answer is 42.")
    response = _canned_response(content=[api_thinking, api_text])
    provider, _client = _make_provider(response)

    result = provider.complete(
        CompletionRequest(
            model="claude-opus-4-7",
            max_tokens=64,
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        )
    )

    assert len(result.content) == 2
    thinking_block = result.content[0]
    assert isinstance(thinking_block, ThinkingBlock)
    assert thinking_block.thinking == "Let me reason about this."
    assert thinking_block.signature == "sig_xyz"
    assert isinstance(result.content[1], TextBlock)


def test_redacted_thinking_round_trip() -> None:
    """RedactedThinkingBlock survives the request/response translation as a
    distinct type carrying the opaque `data` payload."""
    # Outgoing serialization.
    provider, client = _make_provider(_canned_response())
    provider.complete(
        CompletionRequest(
            model="claude-opus-4-7",
            max_tokens=64,
            messages=[
                Message(role="user", content=[TextBlock(text="hi")]),
                Message(
                    role="assistant",
                    content=[RedactedThinkingBlock(data="enc_payload")],
                ),
            ],
        )
    )
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["messages"][1]["content"] == [
        {"type": "redacted_thinking", "data": "enc_payload"}
    ]

    # Incoming deserialization.
    api_redacted = SimpleNamespace(type="redacted_thinking", data="enc_in")
    response = _canned_response(content=[api_redacted])
    provider2, _ = _make_provider(response)
    result = provider2.complete(
        CompletionRequest(
            model="claude-opus-4-7",
            max_tokens=64,
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        )
    )
    assert len(result.content) == 1
    block = result.content[0]
    assert isinstance(block, RedactedThinkingBlock)
    assert block.data == "enc_in"


def test_thinking_with_tool_use_full_multi_turn_round_trip() -> None:
    """A realistic extended-thinking + tool-use turn keeps the thinking
    signature in place for the next request — the API contract."""
    provider, client = _make_provider(_canned_response())

    request = CompletionRequest(
        model="claude-opus-4-7",
        max_tokens=512,
        thinking=True,
        budget=4096,
        tools=[
            {
                "name": "get_weather",
                "description": "weather",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
        messages=[
            Message(role="user", content=[TextBlock(text="What about Paris?")]),
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(
                        thinking="I should call get_weather.",
                        signature="sig_chain_1",
                    ),
                    ToolUseBlock(
                        id="call_42",
                        name="get_weather",
                        input={"city": "Paris"},
                    ),
                ],
            ),
            Message(
                role="user",
                content=[
                    ToolResultBlock(tool_use_id="call_42", content="22C"),
                ],
            ),
        ],
    )
    provider.complete(request)

    kwargs = client.messages.create.call_args.kwargs
    assistant_blocks = kwargs["messages"][1]["content"]
    # ThinkingBlock must come before the ToolUseBlock, with signature intact.
    assert assistant_blocks[0]["type"] == "thinking"
    assert assistant_blocks[0]["signature"] == "sig_chain_1"
    assert assistant_blocks[1]["type"] == "tool_use"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class _FakeAnthropicStream:
    """Minimal stand-in for the SDK's `MessageStream` context manager.

    Iterates a scripted list of duck-typed event objects and exposes
    `get_final_message()` returning a hand-built `Message` shape.
    """

    def __init__(
        self,
        events: list[SimpleNamespace],
        final_message: SimpleNamespace,
    ) -> None:
        self._events = events
        self._final_message = final_message

    def __enter__(self) -> _FakeAnthropicStream:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def __iter__(self):
        return iter(self._events)

    async def __aenter__(self) -> _FakeAnthropicStream:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def __aiter__(self):
        for event in self._events:
            yield event

    def get_final_message(self) -> SimpleNamespace:
        return self._final_message


def _start(content_block: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(type="content_block_start", content_block=content_block)


def _delta(delta_obj: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(type="content_block_delta", delta=delta_obj)


def _stream_provider(
    events: list[SimpleNamespace],
    final_message: SimpleNamespace,
) -> tuple[AnthropicProvider, MagicMock]:
    client = MagicMock()
    fake_stream = _FakeAnthropicStream(events, final_message)
    client.messages.stream.return_value = fake_stream
    return AnthropicProvider(async_client=client), client


def test_stream_text_only_emits_text_deltas_then_complete() -> None:
    text_block = SimpleNamespace(type="text", text="")
    events = [
        _start(text_block),
        _delta(SimpleNamespace(type="text_delta", text="Hello")),
        _delta(SimpleNamespace(type="text_delta", text=", world.")),
    ]
    final = _canned_response(
        content=[SimpleNamespace(type="text", text="Hello, world.")],
        stop_reason="end_turn",
    )
    provider, _client = _stream_provider(events, final)

    request = CompletionRequest(
        model="claude-opus-4-7",
        max_tokens=64,
        messages=[Message(role="user", content=[TextBlock(text="hi")])],
    )

    emitted = list(provider.stream(request))

    text_deltas = [e for e in emitted if isinstance(e, TextDelta)]
    assert [d.text for d in text_deltas] == ["Hello", ", world."]

    completes = [e for e in emitted if isinstance(e, StreamComplete)]
    assert len(completes) == 1
    response = completes[0].response
    assert response.stop_reason == "end_turn"
    assert isinstance(response.content[0], TextBlock)
    assert response.content[0].text == "Hello, world."


def test_stream_tool_use_emits_start_then_input_json_continuations() -> None:
    text_block = SimpleNamespace(type="text", text="")
    tool_block = SimpleNamespace(type="tool_use", id="call_42", name="bash")
    final_text = SimpleNamespace(type="text", text="ok")
    final_tool = SimpleNamespace(
        type="tool_use",
        id="call_42",
        name="bash",
        input={"cmd": "ls"},
    )
    events = [
        _start(text_block),
        _delta(SimpleNamespace(type="text_delta", text="ok")),
        _start(tool_block),
        _delta(SimpleNamespace(type="input_json_delta", partial_json='{"cmd"')),
        _delta(SimpleNamespace(type="input_json_delta", partial_json=': "ls"}')),
    ]
    final = _canned_response(
        content=[final_text, final_tool],
        stop_reason="tool_use",
    )
    provider, _client = _stream_provider(events, final)

    request = CompletionRequest(
        model="claude-opus-4-7",
        max_tokens=64,
        messages=[Message(role="user", content=[TextBlock(text="run ls")])],
    )

    emitted = list(provider.stream(request))

    tool_deltas = [e for e in emitted if isinstance(e, ToolUseDelta)]
    assert len(tool_deltas) == 3
    # First tool delta: start (name set, no partial_json).
    assert tool_deltas[0].id == "call_42"
    assert tool_deltas[0].name == "bash"
    assert tool_deltas[0].partial_json is None
    # Continuations carry the same id, name=None, and a partial_json fragment.
    for d in tool_deltas[1:]:
        assert d.id == "call_42"
        assert d.name is None
    assert "".join(d.partial_json or "" for d in tool_deltas[1:]) == '{"cmd": "ls"}'

    completes = [e for e in emitted if isinstance(e, StreamComplete)]
    assert len(completes) == 1
    response = completes[0].response
    assert response.stop_reason == "tool_use"
    tool_block_out = response.content[1]
    assert isinstance(tool_block_out, ToolUseBlock)
    assert tool_block_out.input == {"cmd": "ls"}


def test_stream_thinking_delta_emitted_for_extended_thinking() -> None:
    thinking_block = SimpleNamespace(type="thinking", thinking="")
    text_block = SimpleNamespace(type="text", text="")
    events = [
        _start(thinking_block),
        _delta(SimpleNamespace(type="thinking_delta", thinking="Step 1.")),
        _delta(SimpleNamespace(type="thinking_delta", thinking=" Step 2.")),
        _start(text_block),
        _delta(SimpleNamespace(type="text_delta", text="answer")),
    ]
    final = _canned_response(
        content=[
            SimpleNamespace(
                type="thinking",
                thinking="Step 1. Step 2.",
                signature="sig",
            ),
            SimpleNamespace(type="text", text="answer"),
        ],
        stop_reason="end_turn",
    )
    provider, _client = _stream_provider(events, final)

    request = CompletionRequest(
        model="claude-opus-4-7",
        max_tokens=64,
        thinking=True,
        budget=4096,
        messages=[Message(role="user", content=[TextBlock(text="think")])],
    )

    emitted = list(provider.stream(request))
    thinking_deltas = [e for e in emitted if isinstance(e, ThinkingDelta)]
    assert [d.thinking for d in thinking_deltas] == ["Step 1.", " Step 2."]

    completes = [e for e in emitted if isinstance(e, StreamComplete)]
    final_resp = completes[0].response
    assert isinstance(final_resp.content[0], ThinkingBlock)
    assert final_resp.content[0].signature == "sig"


def test_stream_redacted_thinking_yields_no_event_but_appears_in_response() -> None:
    redacted = SimpleNamespace(type="redacted_thinking", data="enc_blob")
    text_block = SimpleNamespace(type="text", text="")
    events = [
        _start(redacted),
        _start(text_block),
        _delta(SimpleNamespace(type="text_delta", text="hi")),
    ]
    final = _canned_response(
        content=[
            SimpleNamespace(type="redacted_thinking", data="enc_blob"),
            SimpleNamespace(type="text", text="hi"),
        ],
        stop_reason="end_turn",
    )
    provider, _client = _stream_provider(events, final)

    emitted = list(
        provider.stream(
            CompletionRequest(
                model="claude-opus-4-7",
                max_tokens=64,
                messages=[Message(role="user", content=[TextBlock(text="x")])],
            )
        )
    )

    # No incremental event for the redacted block; only the text delta plus complete.
    incremental = [e for e in emitted if not isinstance(e, StreamComplete)]
    assert all(isinstance(e, TextDelta) for e in incremental)

    final_resp = next(e for e in emitted if isinstance(e, StreamComplete)).response
    assert isinstance(final_resp.content[0], RedactedThinkingBlock)
    assert final_resp.content[0].data == "enc_blob"
