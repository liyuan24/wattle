"""Tests for conversation-history helper semantics."""

from __future__ import annotations

from wattle.message_history import (
    active_task_guidance_text_blocks,
    interrupted_user_text_blocks,
    monitor_event_text_blocks,
    queued_user_text_blocks,
)
from wattle.providers import TextBlock


def test_single_queued_user_text_is_unchanged() -> None:
    assert queued_user_text_blocks(["hello"]) == [TextBlock(text="hello")]


def test_multiple_queued_user_texts_are_labeled() -> None:
    assert queued_user_text_blocks(["first", "second"]) == [
        TextBlock(text="[queued user message 1 of 2]\nfirst"),
        TextBlock(text="[queued user message 2 of 2]\nsecond"),
    ]


def test_active_task_guidance_wraps_single_mid_tool_input() -> None:
    blocks = active_task_guidance_text_blocks(["hello"])

    assert len(blocks) == 1
    assert "additional guidance for the active task" in blocks[0].text
    assert "Continue the active task" in blocks[0].text
    assert "[guidance message 1 of 1]\nhello" in blocks[0].text


def test_interrupted_user_texts_are_labeled_with_interrupt_context() -> None:
    assert interrupted_user_text_blocks(["old"], ["new"]) == [
        TextBlock(
            text=(
                "The previous assistant response was interrupted before completion. "
                "Answer each active user message below now, in order.\n\n"
                "[active user message 1 of 2; interrupted before completion]\nold\n\n"
                "[active user message 2 of 2; new message after interruption]\nnew"
            )
        ),
    ]


def test_monitor_events_render_as_queued_user_text_blocks() -> None:
    blocks = monitor_event_text_blocks(
        [
            {
                "event_type": "pattern_match",
                "severity": "warning",
                "monitor_id": "monitor-1",
                "task_id": "shell-1",
                "status": "running",
                "summary": "matched ERROR",
                "tail": "ERROR here",
            }
        ]
    )

    assert len(blocks) == 1
    assert blocks[0].text == "Monitor event: matched ERROR"


def test_monitor_event_without_summary_uses_compact_fallback() -> None:
    blocks = monitor_event_text_blocks(
        [
            {
                "event_type": "terminal_summary",
                "status": "completed",
                "tail": "verbose output omitted",
            }
        ]
    )

    assert blocks == [TextBlock(text="Monitor event: terminal_summary (completed)")]
