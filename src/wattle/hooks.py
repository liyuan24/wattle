"""Lightweight lifecycle hook contracts for Wattle turns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from wattle.providers import (
    CompletionResponse,
    ContentBlock,
    Message,
    TextBlock,
    ToolResultBlock,
)

FINAL_AUDIT_REMINDER = (
    "[system reminder]\n"
    "Before choosing the next action, verify that any derived file, command, "
    "or artifact still matches the user's required interface and preserves "
    "the meaning of the observed inputs.\n\n"
    "Before your next action or final answer, check whether the user's request "
    "is actually complete. If a lightweight verification is useful, run it. "
    "If not, be clear about what was and was not verified.\n\n"
    "Before finalizing, inspect the final changed or new files. Before removing "
    "anything, distinguish validation-only artifacts from files, services, "
    "processes, data, or external effects that are part of the requested final "
    "observable state. Remove artifacts "
    "created only for validation or temporary work unless the user explicitly "
    "asked for them. Deliver only files required by the task, then re-check the "
    "final state."
)
ACTIVE_TASK_GUIDANCE_MARKER = "additional guidance for the active task"


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


class FinalAuditTurnStopHook(TurnStopHook):
    """Run one final audit after a tool-using turn stops."""

    name = "final_audit"

    def on_turn_stop(self, context: TurnStopContext) -> HookContinuation | None:
        if context.has_pending_user_input:
            return None
        if not _has_tool_result_since_last_final_audit(context.messages):
            return None
        return HookContinuation(
            content=(TextBlock(text=FINAL_AUDIT_REMINDER),),
            reason="final_audit",
        )


def default_turn_stop_hooks() -> tuple[TurnStopHook, ...]:
    """Return hooks that run for normal Wattle turns."""

    return (FinalAuditTurnStopHook(),)


def _has_tool_result_since_last_final_audit(messages: tuple[Message, ...]) -> bool:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        has_final_audit = any(
            isinstance(block, TextBlock) and block.text == FINAL_AUDIT_REMINDER
            for block in message.content
        )
        if has_final_audit:
            return False
        has_active_task_guidance = any(
            isinstance(block, TextBlock)
            and ACTIVE_TASK_GUIDANCE_MARKER in block.text
            for block in message.content
        )
        if has_active_task_guidance:
            return False
        if any(isinstance(block, ToolResultBlock) for block in message.content):
            return not _tool_result_belongs_to_final_audit(messages, index)
    return False


def _tool_result_belongs_to_final_audit(
    messages: tuple[Message, ...],
    tool_result_index: int,
) -> bool:
    for message in range(tool_result_index - 1, -1, -1):
        prior = messages[message]
        if prior.role != "user":
            continue
        return any(
            isinstance(block, TextBlock) and block.text == FINAL_AUDIT_REMINDER
            for block in prior.content
        )
    return False
