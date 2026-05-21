from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from typing import Any

from wattle.compaction import maybe_compact_messages
from wattle.providers import (
    CompletionRequest,
    CompletionResponse,
    Message,
    Provider,
    StreamComplete,
    TextBlock,
    ToolUseBlock,
)


class _ScriptedProvider(Provider):
    def __init__(self, responses: list[CompletionResponse]) -> None:
        self.responses: deque[CompletionResponse] = deque(responses)
        self.requests: list[CompletionRequest] = []
        self.reset_count = 0

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        raise NotImplementedError

    def stream(self, request: CompletionRequest) -> Iterator[Any]:
        self.requests.append(request)
        yield StreamComplete(response=self.responses.popleft())

    def reset_conversation(self) -> None:
        self.reset_count += 1


def _message(index: int, text: str | None = None) -> Message:
    return Message(
        role="user" if index % 2 else "assistant",
        content=[TextBlock(text=text or f"message {index}")],
    )


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

    request_messages, state = maybe_compact_messages(
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

    request_messages, state = maybe_compact_messages(
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

    maybe_compact_messages(
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

    _request_messages, state = maybe_compact_messages(
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
    _request_messages, state = maybe_compact_messages(
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
    request_messages, state = maybe_compact_messages(
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


def cast_text(message: Message) -> str:
    block = message.content[0]
    assert isinstance(block, TextBlock)
    return block.text
