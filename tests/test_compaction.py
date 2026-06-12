from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator
from typing import Any

import anyio

from wattle.compaction import amaybe_compact_messages
from wattle.providers import (
    CompletionRequest,
    CompletionResponse,
    Message,
    Provider,
    StreamComplete,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


class _ScriptedProvider(Provider):
    def __init__(self, responses: list[CompletionResponse]) -> None:
        self.responses: deque[CompletionResponse] = deque(responses)
        self.requests: list[CompletionRequest] = []
        self.reset_count = 0

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        raise NotImplementedError

    async def astream(self, request: CompletionRequest) -> AsyncIterator[Any]:
        self.requests.append(request)
        yield StreamComplete(response=self.responses.popleft())

    async def areset_conversation(self) -> None:
        self.reset_count += 1


def _message(index: int, text: str | None = None) -> Message:
    return Message(
        role="user" if index % 2 else "assistant",
        content=[TextBlock(text=text or f"message {index}")],
    )


async def _call_compact(kwargs: dict[str, object]) -> tuple[list[Message], object]:
    return await amaybe_compact_messages(**kwargs)  # type: ignore[arg-type]


def _compact(**kwargs: object) -> tuple[list[Message], object]:
    return anyio.run(_call_compact, kwargs)


def test_compaction_keeps_summary_and_last_messages() -> None:
    provider = _ScriptedProvider(
        [
            CompletionResponse(
                content=[TextBlock(text="summary of middle")],
                stop_reason="end_turn",
            )
        ]
    )
    messages = [_message(i, "x" * 80) for i in range(25)]

    request_messages, state = _compact(
        provider=provider,
        model="test-model",
        system="system prompt",
        messages=messages,
        tools=[],
        max_tokens=10,
        context_window=100,
        state=None,
    )

    assert state is not None
    assert state.summary == "summary of middle"
    assert state.summarized_until == 24
    assert state.first_kept_index == 24
    assert "summary of middle" in cast_text(request_messages[0])
    assert request_messages[1:] == messages[24:]
    assert provider.reset_count == 2
    assert provider.requests[0].messages[0].content[0].text.count("x" * 80) == 24


def test_compaction_waits_until_threshold() -> None:
    provider = _ScriptedProvider([])
    messages = [_message(i, "small") for i in range(25)]

    request_messages, state = _compact(
        provider=provider,
        model="test-model",
        system=None,
        messages=messages,
        tools=[],
        max_tokens=10,
        context_window=10_000,
        state=None,
    )

    assert request_messages == messages
    assert state is None
    assert provider.requests == []
    assert provider.reset_count == 0


def test_large_output_cap_does_not_start_compaction_below_threshold() -> None:
    provider = _ScriptedProvider([])
    messages = [_message(i, "small") for i in range(3)]

    request_messages, state = _compact(
        provider=provider,
        model="test-model",
        system=None,
        messages=messages,
        tools=[],
        max_tokens=900,
        context_window=1_000,
        state=None,
    )

    assert request_messages == messages
    assert state is None
    assert provider.requests == []
    assert provider.reset_count == 0


def test_provider_usage_below_threshold_ignores_large_output_cap() -> None:
    provider = _ScriptedProvider([])
    messages = [_message(i, "small") for i in range(3)]

    request_messages, state = _compact(
        provider=provider,
        model="test-model",
        system=None,
        messages=messages,
        tools=[],
        max_tokens=900,
        context_window=1_000,
        state=None,
        provider_context_tokens=799,
    )

    assert request_messages == messages
    assert state is None
    assert provider.requests == []
    assert provider.reset_count == 0


def test_compaction_does_not_mutate_saved_history() -> None:
    provider = _ScriptedProvider(
        [
            CompletionResponse(
                content=[TextBlock(text="summary")],
                stop_reason="end_turn",
            )
        ]
    )
    messages = [_message(i, "y" * 80) for i in range(25)]
    original = list(messages)

    _compact(
        provider=provider,
        model="test-model",
        system=None,
        messages=messages,
        tools=[],
        max_tokens=10,
        context_window=100,
        state=None,
    )

    assert messages == original


def test_compaction_tracks_read_and_modified_files() -> None:
    provider = _ScriptedProvider(
        [
            CompletionResponse(
                content=[TextBlock(text="summary")],
                stop_reason="end_turn",
            )
        ]
    )
    messages = [
        Message(
            role="assistant",
            content=[
                ToolUseBlock(id="read_1", name="read", input={"path": "src/app.py"}),
                ToolUseBlock(id="edit_1", name="edit", input={"path": "src/app.py"}),
                ToolUseBlock(id="write_1", name="write", input={"path": "notes.md"}),
            ],
        ),
        *[_message(i, "q" * 80) for i in range(25)],
    ]

    _request_messages, state = _compact(
        provider=provider,
        model="test-model",
        system=None,
        messages=messages,
        tools=[],
        max_tokens=10,
        context_window=100,
        state=None,
    )

    assert state is not None
    assert state.read_files == ("src/app.py",)
    assert state.modified_files == ("notes.md", "src/app.py")


def test_compaction_updates_previous_summary_with_new_middle_messages() -> None:
    provider = _ScriptedProvider(
        [
            CompletionResponse(
                content=[TextBlock(text="initial summary")],
                stop_reason="end_turn",
            ),
            CompletionResponse(
                content=[TextBlock(text="updated summary")],
                stop_reason="end_turn",
            ),
        ]
    )
    messages = [_message(i, "z" * 80) for i in range(25)]
    _request_messages, state = _compact(
        provider=provider,
        model="test-model",
        system=None,
        messages=messages,
        tools=[],
        max_tokens=10,
        context_window=100,
        state=None,
    )
    assert state is not None

    extended = [*messages, _message(25, "new middle"), _message(26, "new last")]
    request_messages, state = _compact(
        provider=provider,
        model="test-model",
        system=None,
        messages=extended,
        tools=[],
        max_tokens=10,
        context_window=100,
        state=state,
    )

    assert state is not None
    assert state.summary == "updated summary"
    assert state.summarized_until == 25
    assert state.first_kept_index == 25
    assert "updated summary" in cast_text(request_messages[0])
    assert request_messages[1:] == extended[25:]
    assert "initial summary" in cast_text(provider.requests[1].messages[0])
    assert "user:\nnew middle" not in cast_text(provider.requests[1].messages[0])


def test_existing_compaction_reuses_summary_when_projection_is_below_threshold() -> None:
    provider = _ScriptedProvider(
        [
            CompletionResponse(
                content=[TextBlock(text="initial summary")],
                stop_reason="end_turn",
            ),
            CompletionResponse(
                content=[TextBlock(text="unexpected refresh")],
                stop_reason="end_turn",
            ),
        ]
    )
    messages = [_message(i, "x" * 80) for i in range(50)]
    request_messages, state = _compact(
        provider=provider,
        model="test-model",
        system=None,
        messages=messages,
        tools=[],
        max_tokens=10,
        context_window=1_000,
        state=None,
    )
    assert state is not None
    assert "initial summary" in cast_text(request_messages[0])
    initial_requests = list(provider.requests)
    initial_reset_count = provider.reset_count

    extended = [*messages, _message(50, "new small tail"), _message(51, "another small tail")]
    request_messages, next_state = _compact(
        provider=provider,
        model="test-model",
        system=None,
        messages=extended,
        tools=[],
        max_tokens=10,
        context_window=1_000,
        state=state,
    )

    assert next_state == state
    assert provider.requests == initial_requests
    assert provider.reset_count == initial_reset_count
    assert "initial summary" in cast_text(request_messages[0])
    assert request_messages[1:] == extended[state.first_kept_index :]


def test_existing_compaction_refreshes_when_projection_crosses_trigger_ratio() -> None:
    provider = _ScriptedProvider(
        [
            CompletionResponse(
                content=[TextBlock(text="initial summary")],
                stop_reason="end_turn",
            ),
            CompletionResponse(
                content=[TextBlock(text="updated summary")],
                stop_reason="end_turn",
            ),
        ]
    )
    messages = [_message(i, "x" * 80) for i in range(50)]
    _request_messages, state = _compact(
        provider=provider,
        model="test-model",
        system=None,
        messages=messages,
        tools=[],
        max_tokens=10,
        context_window=1_000,
        state=None,
    )
    assert state is not None

    extended = [*messages, *[_message(i, "y" * 80) for i in range(50, 80)]]
    request_messages, next_state = _compact(
        provider=provider,
        model="test-model",
        system=None,
        messages=extended,
        tools=[],
        max_tokens=10,
        context_window=1_000,
        state=state,
    )

    assert next_state is not None
    assert next_state != state
    assert next_state.summary == "updated summary"
    assert next_state.summarized_until > state.summarized_until
    assert "updated summary" in cast_text(request_messages[0])
    assert len(provider.requests) == 2


def test_existing_compaction_refreshes_when_provider_usage_crosses_trigger_ratio() -> None:
    provider = _ScriptedProvider(
        [
            CompletionResponse(
                content=[TextBlock(text="initial summary")],
                stop_reason="end_turn",
            ),
            CompletionResponse(
                content=[TextBlock(text="provider pressure summary")],
                stop_reason="end_turn",
            ),
        ]
    )
    messages = [_message(i, "x" * 80) for i in range(50)]
    _request_messages, state = _compact(
        provider=provider,
        model="test-model",
        system=None,
        messages=messages,
        tools=[],
        max_tokens=10,
        context_window=1_000,
        state=None,
    )
    assert state is not None

    extended = [*messages, _message(50, "a")]
    request_messages, next_state = _compact(
        provider=provider,
        model="test-model",
        system=None,
        messages=extended,
        tools=[],
        max_tokens=10,
        context_window=1_000,
        state=state,
        provider_context_tokens=800,
    )

    assert next_state is not None
    assert next_state != state
    assert next_state.summary == "provider pressure summary"
    assert next_state.summarized_until == len(extended) - 1
    assert "provider pressure summary" in cast_text(request_messages[0])
    assert request_messages[-1] == extended[-1]
    assert len(provider.requests) == 2


def test_provider_usage_can_start_first_compaction() -> None:
    provider = _ScriptedProvider(
        [
            CompletionResponse(
                content=[TextBlock(text="provider pressure summary")],
                stop_reason="end_turn",
            ),
        ]
    )
    messages = [_message(i, "small") for i in range(3)]

    request_messages, state = _compact(
        provider=provider,
        model="test-model",
        system=None,
        messages=messages,
        tools=[],
        max_tokens=10,
        context_window=1_000,
        state=None,
        provider_context_tokens=800,
    )

    assert state is not None
    assert state.summary == "provider pressure summary"
    assert state.summarized_until == len(messages) - 1
    assert "provider pressure summary" in cast_text(request_messages[0])
    assert request_messages[-1] == messages[-1]
    assert len(provider.requests) == 1


def test_provider_pressure_compaction_preserves_live_tool_continuation() -> None:
    provider = _ScriptedProvider(
        [
            CompletionResponse(
                content=[TextBlock(text="initial summary")],
                stop_reason="end_turn",
            ),
            CompletionResponse(
                content=[TextBlock(text="provider pressure summary")],
                stop_reason="end_turn",
            ),
        ]
    )
    messages = [_message(i, "x" * 80) for i in range(50)]
    _request_messages, state = _compact(
        provider=provider,
        model="test-model",
        system=None,
        messages=messages,
        tools=[],
        max_tokens=10,
        context_window=1_000,
        state=None,
    )
    assert state is not None
    live_tool_use = Message(
        role="assistant",
        content=[ToolUseBlock(id="tool_1", name="read", input={"path": "README.md"})],
    )
    live_tool_result = Message(
        role="user",
        content=[ToolResultBlock(tool_use_id="tool_1", content="tool output")],
    )

    request_messages, next_state = _compact(
        provider=provider,
        model="test-model",
        system=None,
        messages=[*messages, live_tool_use, live_tool_result],
        tools=[],
        max_tokens=10,
        context_window=1_000,
        state=state,
        provider_context_tokens=800,
    )

    assert next_state is not None
    assert next_state != state
    assert request_messages[-2:] == [live_tool_use, live_tool_result]
    assert next_state.summarized_until == len(messages)
    assert "provider pressure summary" in cast_text(request_messages[0])


def test_forced_existing_compaction_refreshes_even_when_projection_is_below_threshold() -> None:
    provider = _ScriptedProvider(
        [
            CompletionResponse(
                content=[TextBlock(text="initial summary")],
                stop_reason="end_turn",
            ),
            CompletionResponse(
                content=[TextBlock(text="forced summary")],
                stop_reason="end_turn",
            ),
        ]
    )
    messages = [_message(i, "x" * 80) for i in range(50)]
    _request_messages, state = _compact(
        provider=provider,
        model="test-model",
        system=None,
        messages=messages,
        tools=[],
        max_tokens=10,
        context_window=1_000,
        state=None,
    )
    assert state is not None

    extended = [*messages, _message(50, "new small tail")]
    request_messages, next_state = _compact(
        provider=provider,
        model="test-model",
        system=None,
        messages=extended,
        tools=[],
        max_tokens=10,
        context_window=1_000,
        state=state,
        force=True,
    )

    assert next_state is not None
    assert next_state != state
    assert next_state.summary == "forced summary"
    assert "forced summary" in cast_text(request_messages[0])
    assert len(provider.requests) == 2


def cast_text(message: Message) -> str:
    block = message.content[0]
    assert isinstance(block, TextBlock)
    return block.text
