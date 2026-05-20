from __future__ import annotations

from wattle.providers.base import (
    CompletionResponse,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from wattle.turns import append_turn_step, build_turn_step


def test_no_tool_queued_user_text_follows_assistant_response() -> None:
    response = CompletionResponse(
        content=[TextBlock(text="first done")],
        stop_reason="end_turn",
    )

    step = build_turn_step(
        response,
        pending_user_blocks=[TextBlock(text="queued")],
    )

    assert step.assistant == Message(
        role="assistant",
        content=[TextBlock(text="first done")],
    )
    assert step.followup_user == Message(
        role="user",
        content=[TextBlock(text="queued")],
    )
    assert step.continue_running is True

    messages = [Message(role="user", content=[TextBlock(text="first")])]
    append_turn_step(messages, step)

    assert [message.role for message in messages] == ["user", "assistant", "user"]
    assert messages[2].content == [TextBlock(text="queued")]


def test_tool_use_queued_user_text_is_folded_with_tool_results() -> None:
    response = CompletionResponse(
        content=[
            ToolUseBlock(
                id="call_1",
                name="bash",
                input={"command": "printf tool-output"},
            )
        ],
        stop_reason="tool_use",
    )
    result = ToolResultBlock(tool_use_id="call_1", content="tool-output")

    step = build_turn_step(
        response,
        tool_results=[result],
        pending_user_blocks=[TextBlock(text="queued")],
    )

    assert step.assistant == Message(
        role="assistant",
        content=[
            ToolUseBlock(
                id="call_1",
                name="bash",
                input={"command": "printf tool-output"},
            )
        ],
    )
    assert step.followup_user == Message(
        role="user",
        content=[result, TextBlock(text="queued")],
    )
    assert step.continue_running is True

    messages = [Message(role="user", content=[TextBlock(text="first")])]
    append_turn_step(messages, step)

    assert [message.role for message in messages] == ["user", "assistant", "user"]
    assert messages[2].content == [result, TextBlock(text="queued")]
