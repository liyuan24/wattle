"""Provider-agnostic conversation turn assembly.

This module only deals with Wattle's normalized provider types. Tool dispatch,
stream rendering, and provider state stay with their callers; the shared bit is
how one completed assistant response is folded back into message history.
"""

from __future__ import annotations

from collections.abc import Iterable, MutableSequence
from dataclasses import dataclass

from .providers.base import (
    CompletionResponse,
    ContentBlock,
    Message,
)


@dataclass(frozen=True, slots=True)
class TurnStep:
    """Messages to append after one provider response.

    `assistant` is always appended. `followup_user` is appended when the caller
    has tool results, queued user text, or both. `continue_running` tells an
    agent loop whether that follow-up user message should be sent to the model
    immediately.
    """

    assistant: Message
    followup_user: Message | None = None
    continue_running: bool = False


def build_turn_step(
    response: CompletionResponse,
    *,
    tool_results: Iterable[ContentBlock] = (),
    pending_user_blocks: Iterable[ContentBlock] = (),
) -> TurnStep:
    """Build the history update for a completed assistant response.

    The response is always represented as an assistant message. When the model
    stopped for tool use, tool results and queued user text are folded into one
    follow-up user message. When the model ended without tool use, queued user
    text becomes the next user message on its own.
    """

    assistant = Message(
        role="assistant",
        content=list(response.content),
        input_tokens=response.usage.get("input_tokens", 0),
        output_tokens=response.usage.get("output_tokens", 0),
        cached_tokens=_cached_tokens_from_usage(response.usage),
    )
    pending = list(pending_user_blocks)

    content = (
        [*tool_results, *pending]
        if response.stop_reason == "tool_use"
        else pending
    )

    followup = Message(role="user", content=content) if content else None
    return TurnStep(
        assistant=assistant,
        followup_user=followup,
        continue_running=followup is not None,
    )


def _cached_tokens_from_usage(usage: dict[str, int]) -> int:
    for key in ("cached_tokens", "cache_read_input_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return 0


def append_turn_step(messages: MutableSequence[Message], step: TurnStep) -> None:
    """Append a `TurnStep` to an existing message history in order."""

    messages.append(step.assistant)
    if step.followup_user is not None:
        messages.append(step.followup_user)
