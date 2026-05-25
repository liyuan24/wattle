"""Tests for `OpenAIResponsesProvider`.

These tests never make real API calls; the SDK client is mocked so we only
exercise the request/response translation logic against the OpenAI
Responses API wire shape.
"""

from __future__ import annotations

import anyio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from wattle.providers import (
    CompletionRequest,
    ImageBlock,
    Message,
    OpenAIResponsesProvider,
    StreamComplete,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseDelta,
)


def _complete(provider, request):
    return anyio.run(provider.acomplete, request)


async def _collect_stream(provider, request):
    return [event async for event in provider.astream(request)]


def _stream(provider, request):
    return anyio.run(_collect_stream, provider, request)


def _canned_response(
    *,
    response_id: str = "resp_test",
    output: list[object] | None = None,
    status: str = "completed",
    incomplete_details: object | None = None,
    input_tokens: int = 1,
    output_tokens: int = 2,
    cached_tokens: int = 0,
) -> SimpleNamespace:
    """Build a minimal stand-in for an `openai.types.responses.Response`."""
    return SimpleNamespace(
        id=response_id,
        output=output if output is not None else [],
        status=status,
        incomplete_details=incomplete_details,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        ),
    )


def _make_provider(
    response: SimpleNamespace,
) -> tuple[OpenAIResponsesProvider, MagicMock]:
    client = MagicMock()
    client.responses.create.return_value = response
    return OpenAIResponsesProvider(async_client=client), client


# ---------------------------------------------------------------------------
# Request translation — first-call shape
# ---------------------------------------------------------------------------


def test_first_call_translation_full_round_trip() -> None:
    """All input fields are translated to the SDK kwargs shape Responses wants.

    The first call on a fresh provider has no `previous_response_id` and
    sends the full conversation as `input`.
    """
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
        model="gpt-5",
        max_tokens=512,
        system="You are a helpful weather agent.",
        tools=[tool_spec],
        messages=[
            Message(role="user", content=[TextBlock(text="What's it like in Paris?")]),
            Message(
                role="assistant",
                content=[
                    TextBlock(text="Looking it up."),
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

    _complete(provider, request)

    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["model"] == "gpt-5"
    assert kwargs["max_output_tokens"] == 512
    assert kwargs["instructions"] == "You are a helpful weather agent."
    assert kwargs["store"] is True
    # Fresh provider: no chain to attach to yet.
    assert "previous_response_id" not in kwargs

    assert kwargs["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Look up the weather.",
            "parameters": tool_spec["input_schema"],
        }
    ]

    assert kwargs["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What's it like in Paris?"}
            ],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "Looking it up."}
            ],
        },
        {
            "type": "function_call",
            "call_id": "call_42",
            "name": "get_weather",
            "arguments": json.dumps({"city": "Paris"}),
        },
        {
            "type": "function_call_output",
            "call_id": "call_42",
            "output": "sunny, 22C",
        },
    ]


def test_user_image_block_translates_to_input_image(tmp_path) -> None:
    image = tmp_path / "shot.png"
    image.write_bytes(b"image-bytes")
    request = CompletionRequest(
        model="gpt-5",
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

    _complete(provider, request)

    assert client.responses.create.call_args.kwargs["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "What is shown?"}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,aW1hZ2UtYnl0ZXM=",
                }
            ],
        },
    ]


def test_system_none_omits_instructions() -> None:
    request = CompletionRequest(
        model="gpt-5",
        max_tokens=64,
        system=None,
        messages=[Message(role="user", content=[TextBlock(text="hi")])],
    )
    provider, client = _make_provider(_canned_response())

    _complete(provider, request)

    kwargs = client.responses.create.call_args.kwargs
    assert "instructions" not in kwargs


def test_empty_tools_omits_tools_kwarg() -> None:
    request = CompletionRequest(
        model="gpt-5",
        max_tokens=64,
        messages=[Message(role="user", content=[TextBlock(text="hi")])],
    )
    provider, client = _make_provider(_canned_response())

    _complete(provider, request)

    kwargs = client.responses.create.call_args.kwargs
    assert "tools" not in kwargs


def test_tool_result_is_error_prefixes_output() -> None:
    """is_error=True surfaces as a `[error] ` prefix on the output string."""
    request = CompletionRequest(
        model="gpt-5",
        max_tokens=64,
        messages=[
            Message(
                role="user",
                content=[
                    ToolResultBlock(
                        tool_use_id="call_42",
                        content="boom",
                        is_error=True,
                    )
                ],
            )
        ],
    )
    provider, client = _make_provider(_canned_response())

    _complete(provider, request)

    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call_42",
            "output": "[error] boom",
        }
    ]


# ---------------------------------------------------------------------------
# Stateful chaining — the heart of the Responses-native style
# ---------------------------------------------------------------------------


def test_stateful_chain_only_sends_delta_after_first_call() -> None:
    """Subsequent calls pass `previous_response_id` and only the new delta.

    Simulates the loop's mutation pattern between calls:
      - call 1: messages = [user]
      - call 2: messages = [user, assistant_with_tool_use, user_tool_result]
      - call 3: messages = [..., assistant_with_tool_use, user_tool_result]
    """
    client = MagicMock()
    provider = OpenAIResponsesProvider(async_client=client)

    # --- Call 1 -----------------------------------------------------------
    user_msg = Message(role="user", content=[TextBlock(text="run a tool")])
    client.responses.create.return_value = _canned_response(response_id="resp_1")

    _complete(provider,
        CompletionRequest(
            model="gpt-5",
            max_tokens=64,
            messages=[user_msg],
        )
    )

    kw1 = client.responses.create.call_args.kwargs
    assert "previous_response_id" not in kw1
    assert kw1["store"] is True
    assert kw1["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "run a tool"}],
        }
    ]
    # Cursor invariant: 1 input message + 1 assistant we anticipate the loop
    # appending = 2.
    assert provider._seen_messages_count == 1 + 1
    assert provider._previous_response_id == "resp_1"

    # --- Call 2 -----------------------------------------------------------
    # Loop appends the assistant turn (which contained a tool_use) and the
    # user-tool-result message.
    assistant_msg = Message(
        role="assistant",
        content=[ToolUseBlock(id="call_1", name="bash", input={"cmd": "ls"})],
    )
    user_tool_result = Message(
        role="user",
        content=[ToolResultBlock(tool_use_id="call_1", content="ok")],
    )
    client.responses.create.return_value = _canned_response(response_id="resp_2")

    _complete(provider,
        CompletionRequest(
            model="gpt-5",
            max_tokens=64,
            messages=[user_msg, assistant_msg, user_tool_result],
        )
    )

    kw2 = client.responses.create.call_args.kwargs
    assert kw2["previous_response_id"] == "resp_1"
    assert kw2["store"] is True
    # Delta is ONLY the tool_result message, not the user_msg or assistant_msg.
    assert kw2["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "ok",
        }
    ]
    # Cursor: 3 messages in the request + 1 anticipated assistant = 4.
    assert provider._seen_messages_count == 3 + 1
    assert provider._previous_response_id == "resp_2"

    # --- Call 3 -----------------------------------------------------------
    # Another tool round-trip.
    assistant_msg_2 = Message(
        role="assistant",
        content=[ToolUseBlock(id="call_2", name="bash", input={"cmd": "pwd"})],
    )
    user_tool_result_2 = Message(
        role="user",
        content=[ToolResultBlock(tool_use_id="call_2", content="/tmp")],
    )
    client.responses.create.return_value = _canned_response(response_id="resp_3")

    _complete(provider,
        CompletionRequest(
            model="gpt-5",
            max_tokens=64,
            messages=[
                user_msg,
                assistant_msg,
                user_tool_result,
                assistant_msg_2,
                user_tool_result_2,
            ],
        )
    )

    kw3 = client.responses.create.call_args.kwargs
    assert kw3["previous_response_id"] == "resp_2"
    assert kw3["store"] is True
    assert kw3["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call_2",
            "output": "/tmp",
        }
    ]
    assert provider._seen_messages_count == 5 + 1
    assert provider._previous_response_id == "resp_3"


def test_state_is_per_instance() -> None:
    """A fresh provider has no leaked state from a prior provider's chain."""
    client_a = MagicMock()
    client_a.responses.create.return_value = _canned_response(response_id="resp_A")
    provider_a = OpenAIResponsesProvider(async_client=client_a)
    _complete(provider_a, 
        CompletionRequest(
            model="gpt-5",
            max_tokens=64,
            messages=[Message(role="user", content=[TextBlock(text="hello")])],
        )
    )
    assert provider_a._previous_response_id == "resp_A"
    assert provider_a._seen_messages_count == 2

    # New provider: no chain state inherited.
    client_b = MagicMock()
    client_b.responses.create.return_value = _canned_response(response_id="resp_B")
    provider_b = OpenAIResponsesProvider(async_client=client_b)
    assert provider_b._previous_response_id is None
    assert provider_b._seen_messages_count == 0

    _complete(provider_b, 
        CompletionRequest(
            model="gpt-5",
            max_tokens=64,
            messages=[Message(role="user", content=[TextBlock(text="other run")])],
        )
    )
    kwargs_b = client_b.responses.create.call_args.kwargs
    assert "previous_response_id" not in kwargs_b


# ---------------------------------------------------------------------------
# Response translation
# ---------------------------------------------------------------------------


def test_response_translation_text_and_tool_use() -> None:
    """Mixed response output is mapped back to TextBlock / ToolUseBlock."""
    text_part = SimpleNamespace(type="output_text", text="Looking it up.")
    message_item = SimpleNamespace(type="message", content=[text_part])
    function_call_item = SimpleNamespace(
        type="function_call",
        call_id="call_99",
        name="get_weather",
        arguments=json.dumps({"city": "Tokyo"}),
    )
    response = _canned_response(
        output=[message_item, function_call_item],
        status="completed",
        input_tokens=11,
        output_tokens=7,
    )
    provider, _client = _make_provider(response)

    request = CompletionRequest(
        model="gpt-5",
        max_tokens=64,
        messages=[Message(role="user", content=[TextBlock(text="weather?")])],
    )
    result = _complete(provider, request)

    assert len(result.content) == 2
    text_block, tool_block = result.content
    assert isinstance(text_block, TextBlock)
    assert text_block.text == "Looking it up."
    assert isinstance(tool_block, ToolUseBlock)
    assert tool_block.id == "call_99"
    assert tool_block.name == "get_weather"
    assert tool_block.input == {"city": "Tokyo"}

    assert result.usage == {"input_tokens": 11, "output_tokens": 7}


def test_usage_includes_cached_tokens_when_reported() -> None:
    response = _canned_response(input_tokens=100, output_tokens=5, cached_tokens=64)
    provider, _client = _make_provider(response)

    result = _complete(provider,
        CompletionRequest(
            model="gpt-5",
            max_tokens=64,
            messages=[Message(role="user", content=[TextBlock(text="hello")])],
        )
    )

    assert result.usage == {
        "input_tokens": 100,
        "output_tokens": 5,
        "cached_tokens": 64,
    }


def test_stop_reason_tool_use_when_function_call_present() -> None:
    function_call_item = SimpleNamespace(
        type="function_call",
        call_id="call_1",
        name="t",
        arguments="{}",
    )
    response = _canned_response(
        output=[function_call_item],
        status="completed",
    )
    provider, _client = _make_provider(response)

    request = CompletionRequest(
        model="gpt-5",
        max_tokens=64,
        messages=[Message(role="user", content=[TextBlock(text="hi")])],
    )
    result = _complete(provider, request)

    assert result.stop_reason == "tool_use"


def test_stop_reason_max_tokens_when_incomplete_due_to_output_limit() -> None:
    text_part = SimpleNamespace(type="output_text", text="partial...")
    message_item = SimpleNamespace(type="message", content=[text_part])
    response = _canned_response(
        output=[message_item],
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
    )
    provider, _client = _make_provider(response)

    request = CompletionRequest(
        model="gpt-5",
        max_tokens=64,
        messages=[Message(role="user", content=[TextBlock(text="hi")])],
    )
    result = _complete(provider, request)

    assert result.stop_reason == "max_tokens"


def test_stop_reason_end_turn_for_ordinary_completion() -> None:
    text_part = SimpleNamespace(type="output_text", text="all done")
    message_item = SimpleNamespace(type="message", content=[text_part])
    response = _canned_response(
        output=[message_item],
        status="completed",
    )
    provider, _client = _make_provider(response)

    request = CompletionRequest(
        model="gpt-5",
        max_tokens=64,
        messages=[Message(role="user", content=[TextBlock(text="hi")])],
    )
    result = _complete(provider, request)

    assert result.stop_reason == "end_turn"


@pytest.mark.parametrize(
    ("status", "incomplete_details", "has_function_call", "expected"),
    [
        ("completed", None, True, "tool_use"),
        (
            "incomplete",
            SimpleNamespace(reason="max_output_tokens"),
            False,
            "max_tokens",
        ),
        ("completed", None, False, "end_turn"),
    ],
)
def test_stop_reason_priority_table(
    status: str,
    incomplete_details: object | None,
    has_function_call: bool,
    expected: str,
) -> None:
    """Priority: function_call > incomplete/max_output_tokens > end_turn."""
    output: list[object] = []
    if has_function_call:
        output.append(
            SimpleNamespace(
                type="function_call",
                call_id="c",
                name="t",
                arguments="{}",
            )
        )
    else:
        output.append(
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="x")],
            )
        )
    response = _canned_response(
        output=output,
        status=status,
        incomplete_details=incomplete_details,
    )
    provider, _client = _make_provider(response)

    request = CompletionRequest(
        model="gpt-5",
        max_tokens=64,
        messages=[Message(role="user", content=[TextBlock(text="hi")])],
    )
    result = _complete(provider, request)

    assert result.stop_reason == expected


# ---------------------------------------------------------------------------
# Reasoning items / ThinkingBlock round-trip
# ---------------------------------------------------------------------------


def test_thinking_block_outgoing_emits_reasoning_input_item() -> None:
    """A ThinkingBlock with signature serializes to a `reasoning` input item
    with id, summary_text, and (when present) encrypted_content preserved.

    Tested on the first call, where the full conversation (including any
    history-replayed reasoning items) is sent as `input`.
    """
    provider, client = _make_provider(_canned_response())

    request = CompletionRequest(
        model="gpt-5",
        max_tokens=64,
        messages=[
            Message(role="user", content=[TextBlock(text="hi")]),
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(
                        thinking="I should think.",
                        signature="rs_abc",
                        encrypted_content="enc_payload",
                    ),
                    TextBlock(text="My answer."),
                ],
            ),
        ],
    )
    _complete(provider, request)

    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hi"}],
        },
        {
            "type": "reasoning",
            "id": "rs_abc",
            "summary": [{"type": "summary_text", "text": "I should think."}],
            "encrypted_content": "enc_payload",
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "My answer."}],
        },
    ]


def test_thinking_block_without_signature_is_dropped() -> None:
    """The Responses API requires `id` on a reasoning input item; an
    unsigned ThinkingBlock has no replayable id and must be dropped."""
    provider, client = _make_provider(_canned_response())

    request = CompletionRequest(
        model="gpt-5",
        max_tokens=64,
        messages=[
            Message(role="user", content=[TextBlock(text="hi")]),
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="orphan, no signature"),
                    TextBlock(text="answer"),
                ],
            ),
        ],
    )
    _complete(provider, request)

    kwargs = client.responses.create.call_args.kwargs
    assert all(item["type"] != "reasoning" for item in kwargs["input"])


def test_thinking_ordering_interleaved_with_function_calls_and_messages() -> None:
    """Emission order: text, reasoning, function_call, function_call_output.

    First-call shape, where the entire history is serialized to `input`.
    """
    provider, client = _make_provider(_canned_response())

    request = CompletionRequest(
        model="gpt-5",
        max_tokens=64,
        messages=[
            Message(role="user", content=[TextBlock(text="run a tool")]),
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="Plan: call bash.", signature="rs_1"),
                    ToolUseBlock(id="call_1", name="bash", input={"cmd": "ls"}),
                ],
            ),
            Message(
                role="user",
                content=[ToolResultBlock(tool_use_id="call_1", content="ok")],
            ),
        ],
    )
    _complete(provider, request)

    kwargs = client.responses.create.call_args.kwargs
    types_in_order = [item["type"] for item in kwargs["input"]]
    assert types_in_order == [
        "message",
        "reasoning",
        "function_call",
        "function_call_output",
    ]


@pytest.mark.parametrize(
    ("budget", "expected_effort"),
    [
        (1, "low"),
        (2047, "low"),
        (2048, "medium"),
        (8191, "medium"),
        (8192, "high"),
        (32_000, "high"),
        (32_768, "xhigh"),
        (50_000, "xhigh"),
    ],
)
def test_budget_buckets_to_effort(budget: int, expected_effort: str) -> None:
    provider, client = _make_provider(_canned_response())

    _complete(provider,
        CompletionRequest(
            model="gpt-5",
            max_tokens=64,
            thinking=True,
            budget=budget,
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        )
    )

    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["reasoning"] == {"effort": expected_effort}


def test_thinking_false_omits_reasoning_kwarg() -> None:
    provider, client = _make_provider(_canned_response())

    _complete(provider,
        CompletionRequest(
            model="gpt-5",
            max_tokens=64,
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        )
    )

    assert "reasoning" not in client.responses.create.call_args.kwargs


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh"])
def test_effort_passes_through(effort: str) -> None:
    provider, client = _make_provider(_canned_response())

    _complete(provider,
        CompletionRequest(
            model="gpt-5",
            max_tokens=64,
            thinking=True,
            effort=effort,  # type: ignore[arg-type]
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        )
    )

    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["reasoning"] == {"effort": effort}


def test_effort_max_clamps_to_xhigh() -> None:
    provider, client = _make_provider(_canned_response())

    _complete(provider,
        CompletionRequest(
            model="gpt-5",
            max_tokens=64,
            thinking=True,
            effort="max",
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        )
    )

    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["reasoning"] == {"effort": "xhigh"}


def test_effort_takes_precedence_over_budget() -> None:
    provider, client = _make_provider(_canned_response())

    _complete(provider,
        CompletionRequest(
            model="gpt-5",
            max_tokens=64,
            thinking=True,
            effort="high",
            budget=2048,
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        )
    )

    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["reasoning"] == {"effort": "high"}


def test_thinking_true_no_effort_no_budget_omits_reasoning() -> None:
    provider, client = _make_provider(_canned_response())

    _complete(provider,
        CompletionRequest(
            model="gpt-5",
            max_tokens=64,
            thinking=True,
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        )
    )

    assert "reasoning" not in client.responses.create.call_args.kwargs


def test_reasoning_response_item_becomes_thinking_block() -> None:
    """A `reasoning` output item maps back to one ThinkingBlock with id as
    signature, summary parts concatenated as `thinking`, and
    `encrypted_content` preserved."""
    reasoning_item = SimpleNamespace(
        type="reasoning",
        id="rs_out_42",
        summary=[
            SimpleNamespace(type="summary_text", text="First, "),
            SimpleNamespace(type="summary_text", text="then second."),
        ],
        encrypted_content="enc_out",
    )
    text_part = SimpleNamespace(type="output_text", text="The answer.")
    message_item = SimpleNamespace(type="message", content=[text_part])
    response = _canned_response(output=[reasoning_item, message_item])
    provider, _client = _make_provider(response)

    result = _complete(provider,
        CompletionRequest(
            model="gpt-5",
            max_tokens=64,
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        )
    )

    assert len(result.content) == 2
    thinking, text = result.content
    assert isinstance(thinking, ThinkingBlock)
    assert thinking.signature == "rs_out_42"
    assert thinking.thinking == "First, then second."
    assert thinking.encrypted_content == "enc_out"
    assert isinstance(text, TextBlock)


def test_reasoning_response_without_encrypted_content() -> None:
    """`encrypted_content` is optional; missing on the wire -> None."""
    reasoning_item = SimpleNamespace(
        type="reasoning",
        id="rs_99",
        summary=[SimpleNamespace(type="summary_text", text="thinking...")],
    )
    response = _canned_response(output=[reasoning_item])
    provider, _client = _make_provider(response)

    result = _complete(provider,
        CompletionRequest(
            model="gpt-5",
            max_tokens=64,
            messages=[Message(role="user", content=[TextBlock(text="hi")])],
        )
    )
    assert len(result.content) == 1
    block = result.content[0]
    assert isinstance(block, ThinkingBlock)
    assert block.encrypted_content is None


def test_reasoning_items_not_replayed_in_chained_delta() -> None:
    """Under `previous_response_id` chaining, the server already has all
    prior reasoning items in its chain, so the next-call delta must NOT
    include them. Verifies the simplification described in the module
    docstring: outgoing deltas only carry the loop's freshly appended
    messages (typically a function_call_output), never replayed reasoning.
    """
    # Turn 1: server emits a reasoning item + a function_call.
    reasoning_item = SimpleNamespace(
        type="reasoning",
        id="rs_chain",
        summary=[SimpleNamespace(type="summary_text", text="reasoning")],
        encrypted_content="enc_blob",
    )
    function_call_item = SimpleNamespace(
        type="function_call",
        call_id="call_1",
        name="bash",
        arguments=json.dumps({"cmd": "ls"}),
    )
    client = MagicMock()
    client.responses.create.return_value = _canned_response(
        response_id="resp_chain_1",
        output=[reasoning_item, function_call_item],
    )
    provider = OpenAIResponsesProvider(async_client=client)

    user_msg = Message(role="user", content=[TextBlock(text="go")])
    first = _complete(provider,
        CompletionRequest(
            model="gpt-5",
            max_tokens=64,
            messages=[user_msg],
        )
    )

    # Sanity: the response decoded a ThinkingBlock so the loop's history
    # stays faithful.
    assert any(isinstance(b, ThinkingBlock) for b in first.content)

    # Turn 2: the loop appends the assistant turn (with the ThinkingBlock
    # carried verbatim) and the user-tool-result, then calls again. The
    # delta must contain ONLY the function_call_output — no reasoning item,
    # no replayed assistant content, no replayed user input.
    client.responses.create.return_value = _canned_response(response_id="resp_chain_2")
    _complete(provider,
        CompletionRequest(
            model="gpt-5",
            max_tokens=64,
            messages=[
                user_msg,
                Message(role="assistant", content=list(first.content)),
                Message(
                    role="user",
                    content=[ToolResultBlock(tool_use_id="call_1", content="ok")],
                ),
            ],
        )
    )

    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["previous_response_id"] == "resp_chain_1"
    assert kwargs["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "ok",
        }
    ]
    # Belt-and-suspenders: no reasoning item smuggled into the delta.
    assert all(item["type"] != "reasoning" for item in kwargs["input"])


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def _text_delta_event(delta: str) -> SimpleNamespace:
    return SimpleNamespace(type="response.output_text.delta", delta=delta)


def _reasoning_summary_delta_event(delta: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="response.reasoning_summary_text.delta",
        delta=delta,
    )


def _function_call_added_event(call_id: str, name: str) -> SimpleNamespace:
    item = SimpleNamespace(type="function_call", call_id=call_id, name=name)
    return SimpleNamespace(type="response.output_item.added", item=item)


def _function_call_args_delta_event(delta: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="response.function_call_arguments.delta",
        delta=delta,
    )


def _completed_event(response: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(type="response.completed", response=response)


def _stream_responses_provider(
    events: list[SimpleNamespace],
) -> tuple[OpenAIResponsesProvider, MagicMock]:
    client = MagicMock()
    client.responses.create.return_value = iter(events)
    return OpenAIResponsesProvider(async_client=client), client


def test_stream_text_only_emits_deltas_then_complete() -> None:
    final_response = _canned_response(
        response_id="resp_stream_1",
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="Hi there.")],
            )
        ],
    )
    events = [
        _text_delta_event("Hi "),
        _text_delta_event("there."),
        _completed_event(final_response),
    ]
    provider, client = _stream_responses_provider(events)

    emitted = list(
        _stream(provider,
            CompletionRequest(
                model="gpt-5",
                max_tokens=64,
                messages=[Message(role="user", content=[TextBlock(text="hi")])],
            )
        )
    )

    text_deltas = [e for e in emitted if isinstance(e, TextDelta)]
    assert [d.text for d in text_deltas] == ["Hi ", "there."]

    complete = next(e for e in emitted if isinstance(e, StreamComplete))
    assert complete.response.stop_reason == "end_turn"
    assert isinstance(complete.response.content[0], TextBlock)
    assert complete.response.content[0].text == "Hi there."

    # `stream=True` was passed.
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["stream"] is True


def test_stream_function_call_emits_start_then_arg_continuations() -> None:
    final_response = _canned_response(
        response_id="resp_stream_2",
        output=[
            SimpleNamespace(
                type="function_call",
                call_id="call_42",
                name="bash",
                arguments=json.dumps({"cmd": "ls"}),
            )
        ],
    )
    events = [
        _function_call_added_event("call_42", "bash"),
        _function_call_args_delta_event('{"cmd"'),
        _function_call_args_delta_event(': "ls"}'),
        _completed_event(final_response),
    ]
    provider, _client = _stream_responses_provider(events)

    emitted = list(
        _stream(provider,
            CompletionRequest(
                model="gpt-5",
                max_tokens=64,
                messages=[Message(role="user", content=[TextBlock(text="run ls")])],
            )
        )
    )

    tool_deltas = [e for e in emitted if isinstance(e, ToolUseDelta)]
    assert len(tool_deltas) == 3
    assert tool_deltas[0].id == "call_42"
    assert tool_deltas[0].name == "bash"
    assert tool_deltas[0].partial_json is None
    for d in tool_deltas[1:]:
        assert d.id == "call_42"
        assert d.name is None
    assert "".join(d.partial_json or "" for d in tool_deltas[1:]) == '{"cmd": "ls"}'

    complete = next(e for e in emitted if isinstance(e, StreamComplete))
    assert complete.response.stop_reason == "tool_use"
    block = complete.response.content[0]
    assert isinstance(block, ToolUseBlock)
    assert block.input == {"cmd": "ls"}


def test_stream_reasoning_summary_delta_emits_thinking_delta() -> None:
    reasoning_item = SimpleNamespace(
        type="reasoning",
        id="rs_1",
        summary=[SimpleNamespace(type="summary_text", text="thinking step")],
    )
    text_item = SimpleNamespace(
        type="message",
        content=[SimpleNamespace(type="output_text", text="answer")],
    )
    final_response = _canned_response(
        response_id="resp_stream_thinking",
        output=[reasoning_item, text_item],
    )
    events = [
        _reasoning_summary_delta_event("thinking "),
        _reasoning_summary_delta_event("step"),
        _text_delta_event("answer"),
        _completed_event(final_response),
    ]
    provider, _client = _stream_responses_provider(events)

    emitted = list(
        _stream(provider,
            CompletionRequest(
                model="gpt-5",
                max_tokens=64,
                thinking=True,
                budget=8192,
                messages=[Message(role="user", content=[TextBlock(text="think")])],
            )
        )
    )

    thinking_deltas = [e for e in emitted if isinstance(e, ThinkingDelta)]
    assert [d.thinking for d in thinking_deltas] == ["thinking ", "step"]

    response = next(e for e in emitted if isinstance(e, StreamComplete)).response
    assert isinstance(response.content[0], ThinkingBlock)


def test_stream_reasoning_text_delta_also_emits_thinking_delta() -> None:
    """The SDK exposes both `reasoning_summary_text.delta` and
    `reasoning_text.delta` (different reasoning surfaces). Both must
    translate to ThinkingDelta so the streaming contract covers either."""
    final_response = _canned_response(
        response_id="resp_stream_rtext",
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="ok")],
            )
        ],
    )
    events = [
        SimpleNamespace(type="response.reasoning_text.delta", delta="raw thought"),
        _text_delta_event("ok"),
        _completed_event(final_response),
    ]
    provider, _client = _stream_responses_provider(events)

    emitted = list(
        _stream(provider,
            CompletionRequest(
                model="gpt-5",
                max_tokens=64,
                messages=[Message(role="user", content=[TextBlock(text="think")])],
            )
        )
    )
    thinking_deltas = [e for e in emitted if isinstance(e, ThinkingDelta)]
    assert [d.thinking for d in thinking_deltas] == ["raw thought"]


def test_stream_stateful_chain_advances_across_two_streams() -> None:
    """Streaming must update `_previous_response_id` and `_seen_messages_count`
    on `response.completed`, mirroring `complete()` exactly. After two
    consecutive streams the second call must carry the first's
    `previous_response_id` and send only the loop-appended delta."""
    client = MagicMock()
    provider = OpenAIResponsesProvider(async_client=client)

    # --- Stream 1 ---------------------------------------------------------
    final_resp_1 = _canned_response(
        response_id="resp_chain_stream_1",
        output=[
            SimpleNamespace(
                type="function_call",
                call_id="call_1",
                name="bash",
                arguments=json.dumps({"cmd": "ls"}),
            )
        ],
    )
    client.responses.create.return_value = iter(
        [
            _function_call_added_event("call_1", "bash"),
            _function_call_args_delta_event(json.dumps({"cmd": "ls"})),
            _completed_event(final_resp_1),
        ]
    )

    user_msg = Message(role="user", content=[TextBlock(text="go")])
    events_1 = list(
        _stream(provider,
            CompletionRequest(
                model="gpt-5",
                max_tokens=64,
                messages=[user_msg],
            )
        )
    )
    complete_1 = next(e for e in events_1 if isinstance(e, StreamComplete))
    assert complete_1.response.stop_reason == "tool_use"

    # State updated on stream completion, identically to complete().
    assert provider._previous_response_id == "resp_chain_stream_1"
    assert provider._seen_messages_count == 1 + 1

    kw1 = client.responses.create.call_args.kwargs
    assert "previous_response_id" not in kw1
    assert kw1["stream"] is True

    # --- Stream 2 ---------------------------------------------------------
    final_resp_2 = _canned_response(
        response_id="resp_chain_stream_2",
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="done")],
            )
        ],
    )
    client.responses.create.return_value = iter(
        [
            _text_delta_event("done"),
            _completed_event(final_resp_2),
        ]
    )

    assistant_msg = Message(
        role="assistant",
        content=[ToolUseBlock(id="call_1", name="bash", input={"cmd": "ls"})],
    )
    user_tool_result = Message(
        role="user",
        content=[ToolResultBlock(tool_use_id="call_1", content="ok")],
    )
    list(
        _stream(provider,
            CompletionRequest(
                model="gpt-5",
                max_tokens=64,
                messages=[user_msg, assistant_msg, user_tool_result],
            )
        )
    )

    kw2 = client.responses.create.call_args.kwargs
    assert kw2["previous_response_id"] == "resp_chain_stream_1"
    # Delta is only the freshly appended tool_result.
    assert kw2["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "ok",
        }
    ]
    # Cursor advanced by len(messages) + 1.
    assert provider._seen_messages_count == 3 + 1
    assert provider._previous_response_id == "resp_chain_stream_2"
