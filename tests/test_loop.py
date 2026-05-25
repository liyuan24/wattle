"""Smoke test: agent loop drives a real tool through a scripted provider."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator

from wattle import request_preparation
from wattle.loop import run, run_streaming
from wattle.permissions import PermissionGate, PermissionMode
from wattle.providers import (
    CompletionRequest,
    CompletionResponse,
    IncompleteStreamError,
    Message,
    Provider,
    StreamComplete,
    StreamEvent,
    StubProvider,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseDelta,
)
from wattle.runtime import WattleRuntime
from wattle.tools import BashTool


def test_loop_runs_tool_and_terminates() -> None:
    bash = BashTool()
    tools = {bash.name: bash}

    scripted = [
        # Turn 1: assistant asks to run `echo hello`.
        CompletionResponse(
            content=[
                TextBlock(text="Let me check."),
                ToolUseBlock(
                    id="call_1",
                    name="bash",
                    input={"command": "echo hello"},
                ),
            ],
            stop_reason="tool_use",
            usage={"input_tokens": 10, "output_tokens": 5, "cached_tokens": 4},
        ),
        # Turn 2: assistant sees the tool output and stops.
        CompletionResponse(
            content=[TextBlock(text="The command printed 'hello'.")],
            stop_reason="end_turn",
            usage={"input_tokens": 20, "output_tokens": 8},
        ),
    ]
    provider = StubProvider(scripted)

    messages_out: list[Message] = []
    result = run(
        provider=provider,
        tools_by_name=tools,
        system="You are a helpful assistant.",
        user_input="Run echo hello and tell me what it prints.",
        model="stub-model",
        max_tokens=1024,
        messages_out=messages_out,
    )

    # Loop made exactly two provider calls.
    assert len(provider.requests) == 2

    # First request: just the user input, system propagated, tool spec sent.
    first = provider.requests[0]
    assert first.system == "You are a helpful assistant."
    assert first.model == "stub-model"
    assert first.max_tokens == 1024
    assert len(first.messages) == 1
    assert first.messages[0].role == "user"
    assert isinstance(first.messages[0].content[0], TextBlock)
    assert any(spec["name"] == "bash" for spec in first.tools)

    # Second request: original user, assistant tool_use, user tool_result.
    second = provider.requests[1]
    assert [m.role for m in second.messages] == ["user", "assistant", "user"]
    assert second.messages[1].input_tokens == 10
    assert second.messages[1].output_tokens == 5
    assert second.messages[1].cached_tokens == 4
    tool_result_msg = second.messages[2]
    assert len(tool_result_msg.content) == 1
    block = tool_result_msg.content[0]
    assert isinstance(block, ToolResultBlock)
    assert block.tool_use_id == "call_1"
    assert block.is_error is False
    assert "hello" in block.content

    # Final response is the second scripted one.
    assert result.stop_reason == "end_turn"
    assert isinstance(result.content[0], TextBlock)
    assert "hello" in result.content[0].text
    assert [message.role for message in messages_out] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert messages_out[1].input_tokens == 10
    assert messages_out[1].output_tokens == 5
    assert messages_out[1].cached_tokens == 4
    assert messages_out[3].input_tokens == 20
    assert messages_out[3].output_tokens == 8


def test_loop_retries_context_length_error_with_compacted_history() -> None:
    class HugeTool:
        name = "huge"
        description = "returns a large payload"
        input_schema = {"type": "object", "properties": {}}

        def spec(self) -> dict:
            return {
                "name": self.name,
                "description": self.description,
                "input_schema": self.input_schema,
            }

        def run(self, **_: object) -> str:
            return "x" * 30_000

    class ContextRetryProvider(Provider):
        def __init__(self) -> None:
            self.complete_requests: list[CompletionRequest] = []
            self.summary_requests: list[CompletionRequest] = []

        def complete(self, request: CompletionRequest) -> CompletionResponse:
            self.complete_requests.append(request)
            if len(self.complete_requests) == 1:
                return CompletionResponse(
                    content=[ToolUseBlock(id="call_1", name="huge", input={})],
                    stop_reason="tool_use",
                )
            if len(self.complete_requests) == 2:
                raise RuntimeError("context_length_exceeded")
            return CompletionResponse(
                content=[TextBlock(text="recovered")],
                stop_reason="end_turn",
            )

        def stream(self, request: CompletionRequest) -> Iterator[StreamEvent]:
            self.summary_requests.append(request)
            yield StreamComplete(
                response=CompletionResponse(
                    content=[TextBlock(text="summary of huge tool result")],
                    stop_reason="end_turn",
                )
            )

    provider = ContextRetryProvider()

    result = run(
        provider=provider,
        tools_by_name={"huge": HugeTool()},  # type: ignore[dict-item]
        system=None,
        user_input="run huge",
        model="stub-model",
    )

    assert result.content == [TextBlock(text="recovered")]
    assert len(provider.complete_requests) == 3
    raw_retry = provider.complete_requests[1]
    compacted_retry = provider.complete_requests[2]
    assert "x" * 100 in cast_request_text(raw_retry)
    compacted_text = cast_request_text(compacted_retry)
    assert "summary of huge tool result" in compacted_text
    assert "x" * 100 not in compacted_text
    assert len(provider.summary_requests) == 1


def test_loop_reports_tool_errors_as_blocks() -> None:
    """Tool exceptions become ToolResultBlock(is_error=True), not raises."""

    class BoomTool:
        name = "boom"
        description = "always raises"
        input_schema = {"type": "object", "properties": {}}

        def spec(self) -> dict:
            return {
                "name": self.name,
                "description": self.description,
                "input_schema": self.input_schema,
            }

        def run(self, **_: object) -> str:
            raise ValueError("kaboom")

    scripted = [
        CompletionResponse(
            content=[ToolUseBlock(id="x1", name="boom", input={})],
            stop_reason="tool_use",
        ),
        CompletionResponse(
            content=[TextBlock(text="ok, noted")],
            stop_reason="end_turn",
        ),
    ]
    provider = StubProvider(scripted)
    boom = BoomTool()

    result = run(
        provider=provider,
        tools_by_name={boom.name: boom},  # type: ignore[dict-item]
        system=None,
        user_input="break it",
        model="stub-model",
    )

    second = provider.requests[1]
    err_block = second.messages[2].content[0]
    assert isinstance(err_block, ToolResultBlock)
    assert err_block.is_error is True
    assert "kaboom" in err_block.content
    assert result.stop_reason == "end_turn"


def test_loop_adds_monitor_events_to_tool_followup_message() -> None:
    runtime = WattleRuntime()

    class EmitEventTool:
        name = "emit_event"
        description = "emit a monitor event"
        input_schema = {"type": "object", "properties": {}}

        def __init__(self) -> None:
            self.runtime = runtime

        def spec(self) -> dict:
            return {
                "name": self.name,
                "description": self.description,
                "input_schema": self.input_schema,
            }

        def run(self, **_: object) -> str:
            runtime.events.publish(
                {
                    "event_type": "pattern_match",
                    "severity": "warning",
                    "monitor_id": "monitor-1",
                    "task_id": "shell-1",
                    "status": "running",
                    "summary": "matched ERROR",
                    "tail": "ERROR here",
                }
            )
            return "event emitted"

    tool = EmitEventTool()
    scripted = [
        CompletionResponse(
            content=[ToolUseBlock(id="emit-1", name=tool.name, input={})],
            stop_reason="tool_use",
        ),
        CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn"),
    ]
    provider = StubProvider(scripted)

    run(
        provider=provider,
        tools_by_name={tool.name: tool},  # type: ignore[dict-item]
        system=None,
        user_input="emit",
        model="stub-model",
    )

    followup = provider.requests[1].messages[2]
    assert isinstance(followup.content[0], ToolResultBlock)
    assert isinstance(followup.content[1], TextBlock)
    assert followup.content[1].text == "Monitor event: matched ERROR"


def test_loop_unknown_tool_is_an_error_block() -> None:
    scripted = [
        CompletionResponse(
            content=[ToolUseBlock(id="y1", name="ghost", input={})],
            stop_reason="tool_use",
        ),
        CompletionResponse(
            content=[TextBlock(text="done")],
            stop_reason="end_turn",
        ),
    ]
    provider = StubProvider(scripted)

    result = run(
        provider=provider,
        tools_by_name={},
        system=None,
        user_input="call a missing tool",
        model="stub-model",
    )

    err_block = provider.requests[1].messages[2].content[0]
    assert isinstance(err_block, ToolResultBlock)
    assert err_block.is_error is True
    assert "ghost" in err_block.content
    assert result.stop_reason == "end_turn"


def test_loop_read_only_blocks_bash_without_changing_system_prompt() -> None:
    bash = BashTool()
    provider = StubProvider(
        [
            CompletionResponse(
                content=[ToolUseBlock(id="call_1", name="bash", input={"command": "echo hi"})],
                stop_reason="tool_use",
            ),
            CompletionResponse(content=[TextBlock(text="blocked")], stop_reason="end_turn"),
        ]
    )

    run(
        provider=provider,
        tools_by_name={bash.name: bash},
        system="stable system",
        user_input="run echo hi",
        model="stub-model",
        permission_gate=PermissionGate(PermissionMode.READ_ONLY),
    )

    assert provider.requests[0].system == "stable system"
    assert provider.requests[1].system == "stable system"
    err_block = provider.requests[1].messages[2].content[0]
    assert isinstance(err_block, ToolResultBlock)
    assert err_block.is_error is True
    assert "read-only mode" in err_block.content


def test_loop_preserves_thinking_blocks_in_history() -> None:
    """The loop is content-block-agnostic: a ThinkingBlock emitted by the
    provider in turn 1 must be passed back verbatim in turn 2's request and
    show up in the final response."""
    bash = BashTool()
    tools = {bash.name: bash}

    scripted = [
        CompletionResponse(
            content=[
                ThinkingBlock(
                    thinking="I need to call bash.",
                    signature="sig_loop_1",
                ),
                ToolUseBlock(
                    id="call_1",
                    name="bash",
                    input={"command": "echo hi"},
                ),
            ],
            stop_reason="tool_use",
        ),
        CompletionResponse(
            content=[
                ThinkingBlock(
                    thinking="The output looks fine.",
                    signature="sig_loop_2",
                ),
                TextBlock(text="all done"),
            ],
            stop_reason="end_turn",
        ),
    ]
    provider = StubProvider(scripted)

    result = run(
        provider=provider,
        tools_by_name=tools,
        system=None,
        user_input="run echo hi",
        model="stub-model",
    )

    # Second request must contain the assistant message exactly as emitted,
    # ThinkingBlock first, signature intact, ToolUseBlock second.
    second = provider.requests[1]
    assistant_msg = second.messages[1]
    assert assistant_msg.role == "assistant"
    assert len(assistant_msg.content) == 2
    thinking = assistant_msg.content[0]
    assert isinstance(thinking, ThinkingBlock)
    assert thinking.thinking == "I need to call bash."
    assert thinking.signature == "sig_loop_1"
    assert isinstance(assistant_msg.content[1], ToolUseBlock)

    # Final response also carries its ThinkingBlock untouched.
    assert isinstance(result.content[0], ThinkingBlock)
    assert result.content[0].signature == "sig_loop_2"


# ---------------------------------------------------------------------------
# Streaming loop
# ---------------------------------------------------------------------------


class StubStreamProvider(Provider):
    """Replays a scripted list of `(events, terminal_response)` per turn.

    Each entry is a list of `StreamEvent`s; the loop's `stream()` yields
    them in order and ends with the explicit terminal `StreamComplete` that
    carries the assembled `CompletionResponse`. Tests inject the terminal
    `StreamComplete` themselves so they control both the per-turn deltas
    and the resulting response.
    """

    def __init__(self, scripted: Iterable[list[StreamEvent]]) -> None:
        self._scripted: deque[list[StreamEvent]] = deque(scripted)
        self.requests: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest) -> CompletionResponse:  # pragma: no cover
        raise NotImplementedError("StubStreamProvider only supports stream().")

    def stream(self, request: CompletionRequest) -> Iterator[StreamEvent]:
        self.requests.append(request)
        if not self._scripted:
            raise RuntimeError(
                "StubStreamProvider exhausted: loop made more calls than the test scripted."
            )
        events = self._scripted.popleft()
        yield from events


def test_run_streaming_drives_tools_and_forwards_events() -> None:
    bash = BashTool()

    turn1_response = CompletionResponse(
        content=[
            TextBlock(text="Let me check."),
            ToolUseBlock(
                id="call_1",
                name="bash",
                input={"command": "echo hello"},
            ),
        ],
        stop_reason="tool_use",
    )
    turn2_response = CompletionResponse(
        content=[TextBlock(text="The command printed 'hello'.")],
        stop_reason="end_turn",
    )

    scripted: list[list[StreamEvent]] = [
        [
            TextDelta(text="Let me "),
            TextDelta(text="check."),
            ToolUseDelta(id="call_1", name="bash", partial_json=None),
            ToolUseDelta(id="call_1", name=None, partial_json='{"command":'),
            ToolUseDelta(id="call_1", name=None, partial_json=' "echo hello"}'),
            StreamComplete(response=turn1_response),
        ],
        [
            TextDelta(text="The command "),
            TextDelta(text="printed 'hello'."),
            StreamComplete(response=turn2_response),
        ],
    ]
    provider = StubStreamProvider(scripted)

    captured: list[StreamEvent] = []
    result = run_streaming(
        provider=provider,
        tools_by_name={bash.name: bash},
        system="You are a helpful assistant.",
        user_input="Run echo hello and tell me what it prints.",
        model="stub-model",
        max_tokens=1024,
        on_event=captured.append,
    )

    # Two streamed turns — same as the non-streaming smoke test.
    assert len(provider.requests) == 2

    # Every scripted event reached the on_event callback in order.
    assert captured == [event for turn in scripted for event in turn]

    # The terminal response is the second turn's StreamComplete payload.
    assert result is turn2_response

    # Loop drove the tool: second request carries the user-tool-result.
    second = provider.requests[1]
    assert [m.role for m in second.messages] == ["user", "assistant", "user"]
    tool_result = second.messages[2].content[0]
    assert isinstance(tool_result, ToolResultBlock)
    assert tool_result.tool_use_id == "call_1"
    assert tool_result.is_error is False
    assert "hello" in tool_result.content


def test_run_streaming_retries_incomplete_stream_without_saving_partial(
    monkeypatch,
) -> None:
    class IncompleteThenSuccessProvider(Provider):
        def __init__(self) -> None:
            self.requests: list[CompletionRequest] = []

        def complete(self, request: CompletionRequest) -> CompletionResponse:  # pragma: no cover
            raise NotImplementedError

        def stream(self, request: CompletionRequest) -> Iterator[StreamEvent]:
            self.requests.append(request)
            if len(self.requests) == 1:
                yield TextDelta(text="partial")
                raise IncompleteStreamError("stream closed before response.completed")
            response = CompletionResponse(
                content=[TextBlock(text="final")],
                stop_reason="end_turn",
            )
            yield TextDelta(text="final")
            yield StreamComplete(response=response)

    monkeypatch.setattr(request_preparation, "_stream_retry_delay", lambda _attempt: 0.0)
    provider = IncompleteThenSuccessProvider()
    captured: list[StreamEvent] = []

    result = run_streaming(
        provider=provider,
        tools_by_name={},
        system=None,
        user_input="hello",
        model="stub-model",
        on_event=captured.append,
    )

    assert result.content == [TextBlock(text="final")]
    assert len(provider.requests) == 2
    assert provider.requests[0].messages == provider.requests[1].messages
    assert [event.text for event in captured if isinstance(event, TextDelta)] == [
        "partial",
        "final",
    ]


def cast_request_text(request: CompletionRequest) -> str:
    parts: list[str] = []
    for message in request.messages:
        for block in message.content:
            if isinstance(block, TextBlock):
                parts.append(block.text)
            elif isinstance(block, ToolResultBlock):
                parts.append(block.content)
            elif isinstance(block, ToolUseBlock):
                parts.append(block.name)
    return "\n".join(parts)
