"""Lightweight lifecycle hook contracts for Wattle turns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from wattle.providers import CompletionResponse, ContentBlock, Message


@dataclass(frozen=True, slots=True)
class TurnStopContext:
    """State visible to hooks after one complete Wattle turn stops."""

    messages: tuple[Message, ...]
    last_response: CompletionResponse | None
    has_pending_user_input: bool = False


@dataclass(frozen=True, slots=True)
class HookContinuation:
    """A hook-requested follow-up user message to send immediately."""

    content: tuple[ContentBlock, ...]
    reason: str


class TurnStopHook(Protocol):
    """Hook invoked after a full assistant turn has stopped."""

    name: str

    def on_turn_stop(self, context: TurnStopContext) -> HookContinuation | None:
        """Return a continuation to start another turn, or ``None`` to stay idle."""
        ...
