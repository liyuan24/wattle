"""Translation tests for the OpenAI Chat Completions provider plugin.

All tests use `MagicMock`-based fakes; no network calls are made. The plugin
is exercised in two directions:

  * Request side: assert on the kwargs handed to
    `mock_client.chat.completions.create`.
  * Response side: feed the mocked client a hand-built response and assert
    on the resulting `CompletionResponse`.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import anyio
import pytest

from wattle.providers import (
    CompletionRequest,
    ImageBlock,
    Message,
    OpenAICompletionsProvider,
    ProviderQuotaError,
    StreamComplete,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
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


def _fake_response(
    *,
    content: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 7,
    completion_tokens: int = 11,
    cached_tokens: int = 0,
    reasoning_content: str | None = None,
) -> SimpleNamespace:
    """Build a duck-typed Chat Completions response."""
    message = SimpleNamespace(
        content=content, tool_calls=tool_calls, reasoning_content=reasoning_content
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_client(response: SimpleNamespace) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


# ---------------------------------------------------------------------------
# Request translation
# ---------------------------------------------------------------------------


def test_request_translation_full_shape() -> None:
    """A request with system, user text, assistant text+tool_use, tool result,
    and one tool spec produces the expected wire kwargs."""
    client = _make_client(_fake_response(content="ok", finish_reason="stop"))
    provider = OpenAICompletionsProvider(async_client=client)

    request = CompletionRequest(
        model="gpt-4o-mini",
        max_tokens=512,
        system="You are helpful.",
        messages=[
            Message(role="user", content=[TextBlock(text="hi there")]),
            Message(
                role="assistant",
                content=[
                    TextBlock(text="let me check"),
                    ToolUseBlock(id="call_1", name="bash", input={"cmd": "ls"}),
                ],
            ),
            Message(
                role="user",
                content=[
                    ToolResultBlock(tool_use_id="call_1", content="file.txt"),
                ],
            ),
        ],
        tools=[
            {
                "name": "bash",
                "description": "Run a shell command.",
                "input_schema": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
            }
        ],
    )

    _complete(provider, request)

    client.chat.completions.create.assert_called_once()
    kwargs = client.chat.completions.create.call_args.kwargs

    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["max_tokens"] == 512

    # Tools wrapped in OpenAI's function-tool envelope.
    assert kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
            },
        }
    ]

    # Wire messages: system, user, assistant (with tool_calls), tool.
    messages = kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "You are helpful."}
    assert messages[1] == {"role": "user", "content": "hi there"}

    assistant = messages[2]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "let me check"
    assert assistant["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "bash",
                "arguments": json.dumps({"cmd": "ls"}),
            },
        }
    ]

    tool_msg = messages[3]
    assert tool_msg == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "file.txt",
    }
    assert len(messages) == 4


def test_user_image_block_translates_to_chat_image_part(tmp_path) -> None:
    image = tmp_path / "shot.png"
    image.write_bytes(b"image-bytes")
    client = _make_client(_fake_response(content="ok"))
    provider = OpenAICompletionsProvider(async_client=client)

    _complete(
        provider,
        CompletionRequest(
            model="gpt-4o-mini",
            max_tokens=64,
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
        ),
    )

    assert client.chat.completions.create.call_args.kwargs["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is shown?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,aW1hZ2UtYnl0ZXM="},
                },
            ],
        }
    ]


def test_user_text_blocks_are_separated_when_flattened() -> None:
    """Chat Completions has one content string per user message, so multiple
    Wattle text blocks need an explicit separator when flattened."""
    client = _make_client(_fake_response(content="ok"))
    provider = OpenAICompletionsProvider(async_client=client)

    _complete(
        provider,
        CompletionRequest(
            model="gpt-4o-mini",
            max_tokens=64,
            messages=[
                Message(
                    role="user",
                    content=[
                        TextBlock(text="[queued user message 1 of 2]\nfirst"),
                        TextBlock(text="[queued user message 2 of 2]\nsecond"),
                    ],
                )
            ],
        ),
    )

    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert messages == [
        {
            "role": "user",
            "content": (
                "[queued user message 1 of 2]\nfirst\n\n[queued user message 2 of 2]\nsecond"
            ),
        }
    ]


def test_request_no_system_omits_system_message() -> None:
    client = _make_client(_fake_response(content="ok"))
    provider = OpenAICompletionsProvider(async_client=client)

    request = CompletionRequest(
        model="gpt-4o-mini",
        max_tokens=64,
        system=None,
        messages=[Message(role="user", content=[TextBlock(text="hello")])],
    )
    _complete(provider, request)

    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert all(m["role"] != "system" for m in messages)
    assert messages == [{"role": "user", "content": "hello"}]


def test_request_no_tools_omits_tools_kwarg() -> None:
    client = _make_client(_fake_response(content="ok"))
    provider = OpenAICompletionsProvider(async_client=client)

    _complete(
        provider,
        CompletionRequest(
            model="gpt-4o-mini",
            max_tokens=64,
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        ),
    )

    assert "tools" not in client.chat.completions.create.call_args.kwargs


def test_assistant_tool_only_message_has_null_content() -> None:
    """An assistant turn with only tool_use blocks (no text) must set
    `content=None` per the OpenAI convention."""
    client = _make_client(_fake_response(content="ok"))
    provider = OpenAICompletionsProvider(async_client=client)

    _complete(
        provider,
        CompletionRequest(
            model="gpt-4o-mini",
            max_tokens=64,
            messages=[
                Message(role="user", content=[TextBlock(text="go")]),
                Message(
                    role="assistant",
                    content=[ToolUseBlock(id="c1", name="bash", input={})],
                ),
                Message(
                    role="user",
                    content=[ToolResultBlock(tool_use_id="c1", content="done")],
                ),
            ],
        ),
    )

    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assistant = next(m for m in messages if m["role"] == "assistant")
    assert assistant["content"] is None
    assert "tool_calls" in assistant


def test_multiple_tool_results_lift_to_separate_wire_messages() -> None:
    client = _make_client(_fake_response(content="ok"))
    provider = OpenAICompletionsProvider(async_client=client)

    _complete(
        provider,
        CompletionRequest(
            model="gpt-4o-mini",
            max_tokens=64,
            messages=[
                Message(role="user", content=[TextBlock(text="run two")]),
                Message(
                    role="assistant",
                    content=[
                        ToolUseBlock(id="c1", name="bash", input={"cmd": "a"}),
                        ToolUseBlock(id="c2", name="bash", input={"cmd": "b"}),
                    ],
                ),
                Message(
                    role="user",
                    content=[
                        ToolResultBlock(tool_use_id="c1", content="A"),
                        ToolResultBlock(tool_use_id="c2", content="B"),
                    ],
                ),
            ],
        ),
    )

    messages = client.chat.completions.create.call_args.kwargs["messages"]
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert tool_msgs == [
        {"role": "tool", "tool_call_id": "c1", "content": "A"},
        {"role": "tool", "tool_call_id": "c2", "content": "B"},
    ]


def test_user_text_can_follow_tool_results_in_same_wattle_message() -> None:
    client = _make_client(_fake_response(content="ok"))
    provider = OpenAICompletionsProvider(async_client=client)

    _complete(
        provider,
        CompletionRequest(
            model="gpt-4o-mini",
            max_tokens=64,
            messages=[
                Message(role="user", content=[TextBlock(text="run it")]),
                Message(
                    role="assistant",
                    content=[ToolUseBlock(id="c1", name="bash", input={})],
                ),
                Message(
                    role="user",
                    content=[
                        ToolResultBlock(tool_use_id="c1", content="done"),
                        TextBlock(text="also explain what happened"),
                    ],
                ),
            ],
        ),
    )

    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[-2:] == [
        {"role": "tool", "tool_call_id": "c1", "content": "done"},
        {"role": "user", "content": "also explain what happened"},
    ]


def test_is_error_prepends_error_marker() -> None:
    """`is_error=True` is conveyed via a `[error] ` prefix on the wire content."""
    client = _make_client(_fake_response(content="ok"))
    provider = OpenAICompletionsProvider(async_client=client)

    _complete(
        provider,
        CompletionRequest(
            model="gpt-4o-mini",
            max_tokens=64,
            messages=[
                Message(role="user", content=[TextBlock(text="boom")]),
                Message(
                    role="assistant",
                    content=[ToolUseBlock(id="c1", name="boom", input={})],
                ),
                Message(
                    role="user",
                    content=[
                        ToolResultBlock(
                            tool_use_id="c1",
                            content="ValueError: kaboom",
                            is_error=True,
                        )
                    ],
                ),
            ],
        ),
    )

    messages = client.chat.completions.create.call_args.kwargs["messages"]
    tool_msg = next(m for m in messages if m["role"] == "tool")
    assert tool_msg["content"] == "[error] ValueError: kaboom"


# ---------------------------------------------------------------------------
# Response translation
# ---------------------------------------------------------------------------


def test_response_translation_text_and_tool_calls() -> None:
    tool_call_a = SimpleNamespace(
        id="call_a",
        function=SimpleNamespace(
            name="bash",
            arguments=json.dumps({"cmd": "ls"}),
        ),
    )
    tool_call_b = SimpleNamespace(
        id="call_b",
        function=SimpleNamespace(
            name="bash",
            arguments=json.dumps({"cmd": "pwd"}),
        ),
    )
    response = _fake_response(
        content="working on it",
        tool_calls=[tool_call_a, tool_call_b],
        finish_reason="tool_calls",
        prompt_tokens=12,
        completion_tokens=4,
    )
    provider = OpenAICompletionsProvider(async_client=_make_client(response))

    result = _complete(
        provider,
        CompletionRequest(
            model="gpt-4o-mini",
            max_tokens=128,
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        ),
    )

    assert len(result.content) == 3
    assert isinstance(result.content[0], TextBlock)
    assert result.content[0].text == "working on it"

    first_use = result.content[1]
    assert isinstance(first_use, ToolUseBlock)
    assert first_use.id == "call_a"
    assert first_use.name == "bash"
    assert first_use.input == {"cmd": "ls"}

    second_use = result.content[2]
    assert isinstance(second_use, ToolUseBlock)
    assert second_use.id == "call_b"
    assert second_use.input == {"cmd": "pwd"}

    assert result.stop_reason == "tool_use"
    assert result.usage == {"input_tokens": 12, "output_tokens": 4}


def test_usage_includes_cached_tokens_when_reported() -> None:
    response = _fake_response(
        content="all done",
        prompt_tokens=100,
        completion_tokens=5,
        cached_tokens=64,
    )
    provider = OpenAICompletionsProvider(async_client=_make_client(response))

    result = _complete(
        provider,
        CompletionRequest(
            model="gpt-4o-mini",
            max_tokens=64,
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        ),
    )

    assert result.usage == {
        "input_tokens": 100,
        "output_tokens": 5,
        "cached_tokens": 64,
    }


def test_response_with_only_text_no_tool_calls() -> None:
    response = _fake_response(content="all done", tool_calls=None, finish_reason="stop")
    provider = OpenAICompletionsProvider(async_client=_make_client(response))

    result = _complete(
        provider,
        CompletionRequest(
            model="gpt-4o-mini",
            max_tokens=64,
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        ),
    )
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextBlock)
    assert result.content[0].text == "all done"


def test_response_with_only_tool_calls_no_text() -> None:
    tool_call = SimpleNamespace(
        id="call_x",
        function=SimpleNamespace(name="bash", arguments=json.dumps({"cmd": "ls"})),
    )
    response = _fake_response(content=None, tool_calls=[tool_call], finish_reason="tool_calls")
    provider = OpenAICompletionsProvider(async_client=_make_client(response))

    result = _complete(
        provider,
        CompletionRequest(
            model="gpt-4o-mini",
            max_tokens=64,
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        ),
    )
    assert len(result.content) == 1
    assert isinstance(result.content[0], ToolUseBlock)


# ---------------------------------------------------------------------------
# Finish reason mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "finish_reason,expected",
    [
        ("stop", "end_turn"),
        ("tool_calls", "tool_use"),
        ("length", "max_tokens"),
        ("content_filter", "stop_sequence"),
    ],
)
def test_finish_reason_mapping(finish_reason: str, expected: str) -> None:
    response = _fake_response(content="x", finish_reason=finish_reason)
    provider = OpenAICompletionsProvider(async_client=_make_client(response))

    result = _complete(
        provider,
        CompletionRequest(
            model="gpt-4o-mini",
            max_tokens=64,
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        ),
    )
    assert result.stop_reason == expected


# ---------------------------------------------------------------------------
# Thinking handling: reasoning_content round-trips when providers expose it
# ---------------------------------------------------------------------------


def test_thinking_blocks_in_input_are_serialized_as_reasoning_content() -> None:
    """OpenAI-compatible providers such as DeepSeek/Kimi require prior
    assistant reasoning to be replayed as reasoning_content for tool calls."""
    client = _make_client(_fake_response(content="ok"))
    provider = OpenAICompletionsProvider(async_client=client)

    _complete(
        provider,
        CompletionRequest(
            model="gpt-4o-mini",
            max_tokens=64,
            messages=[
                Message(role="user", content=[TextBlock(text="hi")]),
                Message(
                    role="assistant",
                    content=[
                        ThinkingBlock(thinking="reasoning", signature="sig"),
                        TextBlock(text="here you go"),
                        ToolUseBlock(id="c1", name="bash", input={"cmd": "ls"}),
                    ],
                ),
                Message(
                    role="user",
                    content=[ToolResultBlock(tool_use_id="c1", content="done")],
                ),
            ],
        ),
    )

    messages = client.chat.completions.create.call_args.kwargs["messages"]
    serialized = json.dumps(messages)
    assert "thinking" not in serialized

    # Assistant message still carries text and tool_calls; ThinkingBlock text
    # moves to the nonstandard OpenAI-compatible reasoning_content field.
    assistant = next(m for m in messages if m["role"] == "assistant")
    assert assistant["content"] == "here you go"
    assert assistant["reasoning_content"] == "reasoning"
    assert "tool_calls" in assistant


def test_assistant_message_with_only_thinking_blocks_is_skipped() -> None:
    """An assistant turn made entirely of ThinkingBlocks has nothing left
    after stripping; it must not appear on the wire as a malformed
    `{"role": "assistant", "content": None}` message."""
    client = _make_client(_fake_response(content="ok"))
    provider = OpenAICompletionsProvider(async_client=client)

    _complete(
        provider,
        CompletionRequest(
            model="gpt-4o-mini",
            max_tokens=64,
            messages=[
                Message(role="user", content=[TextBlock(text="hi")]),
                Message(
                    role="assistant",
                    content=[ThinkingBlock(thinking="hidden", signature="s")],
                ),
                Message(role="user", content=[TextBlock(text="still there?")]),
            ],
        ),
    )

    messages = client.chat.completions.create.call_args.kwargs["messages"]
    # No assistant wire message at all — the only blocks were thinking blocks.
    assert all(m["role"] != "assistant" for m in messages)
    # The surrounding user messages are still intact.
    assert messages == [
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "still there?"},
    ]


def test_response_reasoning_content_becomes_thinking_block() -> None:
    response = _fake_response(
        content="all done",
        finish_reason="stop",
        reasoning_content="provider reasoning",
    )
    provider = OpenAICompletionsProvider(async_client=_make_client(response))

    result = _complete(
        provider,
        CompletionRequest(
            model="gpt-4o-mini",
            max_tokens=64,
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        ),
    )

    assert isinstance(result.content[0], ThinkingBlock)
    assert result.content[0].thinking == "provider reasoning"
    assert isinstance(result.content[1], TextBlock)
    assert result.content[1].text == "all done"


def test_response_without_reasoning_content_has_no_thinking_block() -> None:
    response = _fake_response(content="all done", finish_reason="stop")
    provider = OpenAICompletionsProvider(async_client=_make_client(response))

    result = _complete(
        provider,
        CompletionRequest(
            model="gpt-4o-mini",
            max_tokens=64,
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        ),
    )

    assert not any(isinstance(b, ThinkingBlock) for b in result.content)


def test_thinking_false_omits_reasoning_effort() -> None:
    """With the default `thinking=False`, no `reasoning_effort` kwarg is
    emitted regardless of any other field state."""
    client = _make_client(_fake_response(content="ok"))
    provider = OpenAICompletionsProvider(async_client=client)

    _complete(
        provider,
        CompletionRequest(
            model="gpt-4o-mini",
            max_tokens=64,
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        ),
    )

    kwargs = client.chat.completions.create.call_args.kwargs
    assert "reasoning_effort" not in kwargs


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh"])
def test_effort_passes_through(effort: str) -> None:
    """Each non-`"max"` effort value is forwarded as-is on the wire."""
    client = _make_client(_fake_response(content="ok"))
    provider = OpenAICompletionsProvider(async_client=client)

    _complete(
        provider,
        CompletionRequest(
            model="gpt-4o-mini",
            max_tokens=64,
            thinking=True,
            effort=effort,  # type: ignore[arg-type]
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        ),
    )

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["reasoning_effort"] == effort


def test_effort_max_clamps_to_xhigh() -> None:
    """`"max"` is Anthropic-only; OpenAI providers clamp it to `"xhigh"`."""
    client = _make_client(_fake_response(content="ok"))
    provider = OpenAICompletionsProvider(async_client=client)

    _complete(
        provider,
        CompletionRequest(
            model="gpt-4o-mini",
            max_tokens=64,
            thinking=True,
            effort="max",
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        ),
    )

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["reasoning_effort"] == "xhigh"


@pytest.mark.parametrize(
    "budget,expected",
    [
        (1024, "low"),
        (4096, "medium"),
        (16384, "high"),
        (50000, "xhigh"),
    ],
)
def test_budget_buckets_to_effort(budget: int, expected: str) -> None:
    """A bare `budget` (no `effort`) is bucketed into an effort level."""
    client = _make_client(_fake_response(content="ok"))
    provider = OpenAICompletionsProvider(async_client=client)

    _complete(
        provider,
        CompletionRequest(
            model="gpt-4o-mini",
            max_tokens=64,
            thinking=True,
            budget=budget,
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        ),
    )

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["reasoning_effort"] == expected


def test_effort_takes_precedence_over_budget() -> None:
    """When both are set, `effort` wins; `budget` is not consulted."""
    client = _make_client(_fake_response(content="ok"))
    provider = OpenAICompletionsProvider(async_client=client)

    _complete(
        provider,
        CompletionRequest(
            model="gpt-4o-mini",
            max_tokens=64,
            thinking=True,
            effort="high",
            budget=1024,
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        ),
    )

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["reasoning_effort"] == "high"


def test_thinking_true_no_effort_no_budget_omits_reasoning_effort() -> None:
    """`thinking=True` with neither `effort` nor `budget` lets the API
    default apply — no `reasoning_effort` kwarg is emitted."""
    client = _make_client(_fake_response(content="ok"))
    provider = OpenAICompletionsProvider(async_client=client)

    _complete(
        provider,
        CompletionRequest(
            model="gpt-4o-mini",
            max_tokens=64,
            thinking=True,
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        ),
    )

    kwargs = client.chat.completions.create.call_args.kwargs
    assert "reasoning_effort" not in kwargs


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def _text_chunk(
    text: str,
    finish_reason: str | None = None,
    reasoning_content: str | None = None,
) -> SimpleNamespace:
    delta = SimpleNamespace(content=text, tool_calls=None, reasoning_content=reasoning_content)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


def _tool_chunk(
    *,
    index: int,
    id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    function = SimpleNamespace(name=name, arguments=arguments)
    tool_call = SimpleNamespace(index=index, id=id, function=function)
    delta = SimpleNamespace(content=None, tool_calls=[tool_call], reasoning_content=None)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


def _final_usage_chunk(
    prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0
) -> SimpleNamespace:
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
    )
    return SimpleNamespace(choices=[], usage=usage)


def _stream_client(chunks: list[SimpleNamespace]) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = iter(chunks)
    return client


def test_stream_text_emits_deltas_then_complete_with_usage() -> None:
    chunks = [
        _text_chunk("Hel", reasoning_content="plan "),
        _text_chunk("lo", reasoning_content="done"),
        _text_chunk("!", finish_reason="stop"),
        _final_usage_chunk(prompt_tokens=10, completion_tokens=4, cached_tokens=6),
    ]
    client = _stream_client(chunks)
    provider = OpenAICompletionsProvider(async_client=client)

    emitted = list(
        _stream(
            provider,
            CompletionRequest(
                model="gpt-4o-mini",
                max_tokens=64,
                messages=[Message(role="user", content=[TextBlock(text="hi")])],
            ),
        )
    )

    thinking_deltas = [e for e in emitted if isinstance(e, ThinkingDelta)]
    assert [d.thinking for d in thinking_deltas] == ["plan ", "done"]
    text_deltas = [e for e in emitted if isinstance(e, TextDelta)]
    assert [d.text for d in text_deltas] == ["Hel", "lo", "!"]

    # `stream=True` and `include_usage` were passed through.
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}

    complete = next(e for e in emitted if isinstance(e, StreamComplete))
    response = complete.response
    assert response.stop_reason == "end_turn"
    assert isinstance(response.content[0], ThinkingBlock)
    assert response.content[0].thinking == "plan done"
    assert isinstance(response.content[1], TextBlock)
    assert response.content[1].text == "Hello!"
    assert response.usage == {"input_tokens": 10, "output_tokens": 4, "cached_tokens": 6}


def test_stream_tool_calls_emit_start_then_argument_fragments() -> None:
    # Realistic ordering: id+name arrive once, arguments stream in fragments.
    chunks = [
        _tool_chunk(index=0, id="call_a", name="bash"),
        _tool_chunk(index=0, arguments='{"cmd"'),
        _tool_chunk(index=0, arguments=': "ls"}'),
        _tool_chunk(index=0, finish_reason="tool_calls"),
        _final_usage_chunk(prompt_tokens=5, completion_tokens=8),
    ]
    client = _stream_client(chunks)
    provider = OpenAICompletionsProvider(async_client=client)

    emitted = list(
        _stream(
            provider,
            CompletionRequest(
                model="gpt-4o-mini",
                max_tokens=64,
                messages=[Message(role="user", content=[TextBlock(text="run ls")])],
            ),
        )
    )

    tool_deltas = [e for e in emitted if isinstance(e, ToolUseDelta)]
    assert len(tool_deltas) == 3
    assert tool_deltas[0].id == "call_a"
    assert tool_deltas[0].name == "bash"
    assert tool_deltas[0].partial_json is None
    for d in tool_deltas[1:]:
        assert d.id == "call_a"
        assert d.name is None
    assert "".join(d.partial_json or "" for d in tool_deltas[1:]) == '{"cmd": "ls"}'

    response = next(e for e in emitted if isinstance(e, StreamComplete)).response
    assert response.stop_reason == "tool_use"
    assert len(response.content) == 1
    block = response.content[0]
    assert isinstance(block, ToolUseBlock)
    assert block.id == "call_a"
    assert block.name == "bash"
    assert block.input == {"cmd": "ls"}


def test_stream_parallel_tool_calls_track_per_index_independently() -> None:
    chunks = [
        _tool_chunk(index=0, id="call_a", name="bash"),
        _tool_chunk(index=1, id="call_b", name="bash"),
        _tool_chunk(index=0, arguments='{"cmd": "ls"}'),
        _tool_chunk(index=1, arguments='{"cmd": "pwd"}'),
        _tool_chunk(index=0, finish_reason="tool_calls"),
        _final_usage_chunk(prompt_tokens=3, completion_tokens=12),
    ]
    client = _stream_client(chunks)
    provider = OpenAICompletionsProvider(async_client=client)

    emitted = list(
        _stream(
            provider,
            CompletionRequest(
                model="gpt-4o-mini",
                max_tokens=64,
                messages=[Message(role="user", content=[TextBlock(text="parallel")])],
            ),
        )
    )

    tool_deltas = [e for e in emitted if isinstance(e, ToolUseDelta)]
    # Two starts (one per index) + two argument continuations.
    starts = [d for d in tool_deltas if d.partial_json is None]
    assert {d.id for d in starts} == {"call_a", "call_b"}
    assert all(d.name == "bash" for d in starts)

    response = next(e for e in emitted if isinstance(e, StreamComplete)).response
    assert response.stop_reason == "tool_use"
    blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]
    assert {b.id for b in blocks} == {"call_a", "call_b"}
    inputs = {b.id: b.input for b in blocks}
    assert inputs["call_a"] == {"cmd": "ls"}
    assert inputs["call_b"] == {"cmd": "pwd"}


def test_stream_finish_reason_length_maps_to_max_tokens() -> None:
    chunks = [
        _text_chunk("partial"),
        _text_chunk("", finish_reason="length"),
        _final_usage_chunk(prompt_tokens=2, completion_tokens=2),
    ]
    client = _stream_client(chunks)
    provider = OpenAICompletionsProvider(async_client=client)

    emitted = list(
        _stream(
            provider,
            CompletionRequest(
                model="gpt-4o-mini",
                max_tokens=2,
                messages=[Message(role="user", content=[TextBlock(text="hi")])],
            ),
        )
    )

    response = next(e for e in emitted if isinstance(e, StreamComplete)).response
    assert response.stop_reason == "max_tokens"


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


def test_complete_normalizes_retryable_chat_completion_errors() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = _ProviderException(
        "rate limit",
        status_code=429,
        body={"error": {"code": "rate_limit_exceeded", "message": "try again in 4s"}},
    )
    provider = OpenAICompletionsProvider(async_client=client)

    with pytest.raises(TransientProviderError) as exc_info:
        _complete(
            provider,
            CompletionRequest(
                model="gpt-4o-mini",
                max_tokens=64,
                messages=[Message(role="user", content=[TextBlock(text="hi")])],
            ),
        )

    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after == 4
    assert exc_info.value.provider == "openai_completions"


def test_complete_normalizes_non_retryable_chat_completion_quota_errors() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = _ProviderException(
        "quota",
        status_code=429,
        body={"error": {"code": "insufficient_quota", "message": "quota exceeded"}},
    )
    provider = OpenAICompletionsProvider(async_client=client)

    with pytest.raises(ProviderQuotaError):
        _complete(
            provider,
            CompletionRequest(
                model="gpt-4o-mini",
                max_tokens=64,
                messages=[Message(role="user", content=[TextBlock(text="hi")])],
            ),
        )


async def _raising_stream(error: BaseException):
    yield _text_chunk("before error")
    raise error


def test_stream_normalizes_chat_completion_iteration_errors() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _raising_stream(
        _ProviderException(
            "overloaded",
            status_code=503,
            body={"error": {"code": "server_is_overloaded", "message": "busy"}},
        )
    )
    provider = OpenAICompletionsProvider(async_client=client)

    with pytest.raises(TransientProviderError) as exc_info:
        _stream(
            provider,
            CompletionRequest(
                model="gpt-4o-mini",
                max_tokens=64,
                messages=[Message(role="user", content=[TextBlock(text="hi")])],
            ),
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.code == "server_is_overloaded"
