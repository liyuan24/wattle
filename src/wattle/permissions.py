"""Runtime tool permission gates.

Permission modes are intentionally enforced after the provider returns tool
calls. They do not modify the system prompt or any other model-visible request
prefix.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from wattle.providers import ToolUseBlock


class PermissionMode(StrEnum):
    YOLO = "yolo"
    READ_ONLY = "read_only"
    ASK = "ask_for_permission"


class PermissionAnswer(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    ALLOW_ALL = "allow_all"


@dataclass(frozen=True, slots=True)
class PermissionResult:
    allowed: bool
    denial: str | None = None


PermissionPrompt = Callable[[ToolUseBlock], PermissionAnswer]

READ_ONLY_TOOLS = frozenset({"read"})


class PermissionGate:
    """Stateful runtime gate for tool execution."""

    def __init__(
        self,
        mode: PermissionMode = PermissionMode.YOLO,
        prompt: PermissionPrompt | None = None,
    ) -> None:
        self.mode = mode
        self.prompt = prompt
        self._allow_all = False

    def check(self, block: ToolUseBlock) -> PermissionResult:
        if self.mode == PermissionMode.YOLO or self._allow_all:
            return PermissionResult(allowed=True)

        if self.mode == PermissionMode.READ_ONLY:
            if block.name in READ_ONLY_TOOLS:
                return PermissionResult(allowed=True)
            return PermissionResult(
                allowed=False,
                denial=f"Tool blocked by read-only mode: {block.name} is not allowed.",
            )

        if self.prompt is None:
            return PermissionResult(
                allowed=False,
                denial="Permission denied: no permission prompt is available.",
            )

        answer = self.prompt(block)
        if answer == PermissionAnswer.ALLOW_ALL:
            self._allow_all = True
            return PermissionResult(allowed=True)
        if answer == PermissionAnswer.ALLOW:
            return PermissionResult(allowed=True)
        return PermissionResult(allowed=False, denial="Permission denied by user.")


def parse_permission_mode(value: str) -> PermissionMode:
    try:
        return PermissionMode(value)
    except ValueError as exc:
        raise ValueError(f"unknown permission mode: {value!r}") from exc


def tool_permission_summary(block: ToolUseBlock) -> str:
    if block.name == "bash":
        return f"bash: {block.input.get('command', '<missing command>')}"
    if block.name in {"read", "write", "edit"}:
        return f"{block.name}: {block.input.get('path', '<missing path>')}"
    return f"{block.name}: {block.input}"
