from __future__ import annotations

from wattle.hooks import FINAL_AUDIT_REMINDER, FinalAuditTurnStopHook, TurnStopContext
from wattle.providers import Message, TextBlock, ToolResultBlock


def test_final_audit_hook_runs_once_after_tool_result() -> None:
    hook = FinalAuditTurnStopHook()
    messages = (
        Message(role="user", content=[TextBlock(text="run it")]),
        Message(role="assistant", content=[TextBlock(text="done")]),
        Message(
            role="user",
            content=[
                ToolResultBlock(
                    tool_use_id="call_1",
                    content="observed output",
                    is_error=False,
                )
            ],
        ),
        Message(role="assistant", content=[TextBlock(text="complete")]),
    )

    continuation = hook.on_turn_stop(
        TurnStopContext(messages=messages, last_response=None, has_pending_user_input=False)
    )

    assert continuation is not None
    assert continuation.reason == "final_audit"
    assert continuation.content == (TextBlock(text=FINAL_AUDIT_REMINDER),)
    assert "derived file, command, or artifact" in FINAL_AUDIT_REMINDER
    assert "requested final observable state" in FINAL_AUDIT_REMINDER

    with_reminder = (
        *messages,
        Message(role="user", content=[TextBlock(text=FINAL_AUDIT_REMINDER)]),
        Message(role="assistant", content=[TextBlock(text="still complete")]),
    )
    assert hook.on_turn_stop(
        TurnStopContext(
            messages=with_reminder,
            last_response=None,
            has_pending_user_input=False,
        )
    ) is None


def test_final_audit_hook_skips_without_tool_result_or_with_pending_user_input() -> None:
    hook = FinalAuditTurnStopHook()
    messages = (
        Message(role="user", content=[TextBlock(text="hello")]),
        Message(role="assistant", content=[TextBlock(text="hi")]),
    )

    assert hook.on_turn_stop(
        TurnStopContext(messages=messages, last_response=None, has_pending_user_input=False)
    ) is None

    tool_messages = (
        *messages,
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="call_1", content="ok")],
        ),
    )
    assert hook.on_turn_stop(
        TurnStopContext(
            messages=tool_messages,
            last_response=None,
            has_pending_user_input=True,
        )
    ) is None


def test_final_audit_hook_skips_after_active_task_guidance() -> None:
    hook = FinalAuditTurnStopHook()
    messages = (
        Message(role="user", content=[TextBlock(text="start")]),
        Message(role="assistant", content=[TextBlock(text="running tool")]),
        Message(
            role="user",
            content=[
                ToolResultBlock(tool_use_id="call_1", content="ok"),
                TextBlock(
                    text=(
                        "The user sent the following while you were working. "
                        "Treat it as additional guidance for the active task."
                    )
                ),
            ],
        ),
        Message(role="assistant", content=[TextBlock(text="continued active task")]),
    )

    assert hook.on_turn_stop(
        TurnStopContext(messages=messages, last_response=None, has_pending_user_input=False)
    ) is None


def test_final_audit_hook_does_not_repeat_after_audit_tool_result() -> None:
    hook = FinalAuditTurnStopHook()
    messages = (
        Message(role="user", content=[TextBlock(text="start")]),
        Message(role="assistant", content=[TextBlock(text="running tool")]),
        Message(role="user", content=[ToolResultBlock(tool_use_id="call_1", content="ok")]),
        Message(role="assistant", content=[TextBlock(text="almost done")]),
        Message(role="user", content=[TextBlock(text=FINAL_AUDIT_REMINDER)]),
        Message(role="assistant", content=[TextBlock(text="checking")]),
        Message(role="user", content=[ToolResultBlock(tool_use_id="call_2", content="ok")]),
        Message(role="assistant", content=[TextBlock(text="done")]),
    )

    assert hook.on_turn_stop(
        TurnStopContext(messages=messages, last_response=None, has_pending_user_input=False)
    ) is None
