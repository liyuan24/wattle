"""Helpers for preserving conversation-history semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from wattle.providers.base import TextBlock


def queued_user_text_blocks(texts: Sequence[str]) -> list[TextBlock]:
    """Convert queued user inputs into history-safe text blocks.

    A single queued input is still the user's original text. When multiple
    inputs are batched into one Wattle user message, each block gets a small
    model-visible label so providers that preserve content-block structure and
    providers that must flatten text both retain the message boundaries.
    """
    if len(texts) <= 1:
        return [TextBlock(text=text) for text in texts]

    total = len(texts)
    return [
        TextBlock(text=f"[queued user message {index} of {total}]\n{text}")
        for index, text in enumerate(texts, start=1)
    ]


def interrupted_user_text_blocks(
    interrupted: Sequence[str],
    interrupting: Sequence[str],
) -> list[TextBlock]:
    """Convert an interrupted turn plus new input into one explicit user turn.

    The retry is intentionally a single model-facing text block. Consecutive
    user blocks can be interpreted as ordinary chat history where the last
    message wins; this wording makes the interrupted request active again.
    """
    total = len(interrupted) + len(interrupting)
    if total == 0:
        return []

    sections = [
        (
            "The previous assistant response was interrupted before completion. "
            "Answer each active user message below now, in order."
        )
    ]
    index = 1
    for text in interrupted:
        sections.append(
            f"[active user message {index} of {total}; "
            f"interrupted before completion]\n{text}"
        )
        index += 1
    for text in interrupting:
        sections.append(
            f"[active user message {index} of {total}; "
            f"new message after interruption]\n{text}"
        )
        index += 1
    return [TextBlock(text="\n\n".join(sections))]


def monitor_event_text(event: Mapping[str, object]) -> str:
    """Render one monitor event as a compact synthetic user message."""

    summary = str(event.get("summary") or "").strip()
    if summary:
        return f"Monitor event: {summary}"

    event_type = str(event.get("event_type") or "update").strip()
    status = str(event.get("status") or "").strip()
    if status:
        return f"Monitor event: {event_type} ({status})"
    return f"Monitor event: {event_type}"


def monitor_event_texts(events: Sequence[Mapping[str, object]]) -> list[str]:
    """Render monitor events into texts suitable for queued user blocks."""

    return [monitor_event_text(event) for event in events]


def monitor_event_text_blocks(events: Sequence[Mapping[str, object]]) -> list[TextBlock]:
    """Render monitor events using the same batching semantics as queued input."""

    return queued_user_text_blocks(monitor_event_texts(events))
