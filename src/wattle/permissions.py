"""Runtime tool permission gates.

Permission modes are intentionally enforced after the provider returns tool
calls. They do not modify the system prompt or any other model-visible request
prefix.
"""

from __future__ import annotations

import shlex
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

READ_ONLY_TOOLS = frozenset({"read", "view_image", "update_plan"})
_READ_ONLY_SHELL_COMMANDS = frozenset({"pwd", "ls", "rg"})
_READ_ONLY_GIT_SUBCOMMANDS = frozenset(
    {
        "branch",
        "diff",
        "log",
        "rev-parse",
        "show",
        "status",
    }
)
_SHELL_CONTROL_CHARS = frozenset(";&|<>()`$\n\r")


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
            if is_read_only_tool_call(block):
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
        command = block.input.get("command", "<missing command>")
        details = []
        if block.input.get("background"):
            details.append("background")
        if block.input.get("tty"):
            details.append("tty")
        if "timeout" in block.input:
            details.append(f"timeout={block.input['timeout']}")
        suffix = f" ({', '.join(details)})" if details else ""
        return f"bash: {command}{suffix}"
    if block.name == "edit":
        path = block.input.get("path", "<missing path>")
        edits = block.input.get("edits")
        if isinstance(edits, list):
            return f"edit: {path} ({len(edits)} edit{'s' if len(edits) != 1 else ''})"
        return f"edit: {path}"
    if block.name == "write":
        path = block.input.get("path", "<missing path>")
        content = block.input.get("content", "")
        if isinstance(content, str):
            lines = content.count("\n") + (1 if content else 0)
            return f"write: {path} ({lines} line{'s' if lines != 1 else ''})"
        return f"write: {path}"
    if block.name in {"read", "view_image"}:
        return f"{block.name}: {block.input.get('path', '<missing path>')}"
    if block.name == "monitor":
        command = block.input.get("command", "<missing command>")
        description = block.input.get("description")
        if isinstance(description, str) and description:
            return f"monitor: {description} ({command})"
        return f"monitor: {command}"
    return f"{block.name}: {block.input}"


def is_read_only_tool_call(block: ToolUseBlock) -> bool:
    if block.name in READ_ONLY_TOOLS:
        return True
    if block.name not in {"bash", "monitor"}:
        return False
    command = block.input.get("command")
    return isinstance(command, str) and is_safe_read_only_shell_command(command)


def is_safe_read_only_shell_command(command: str) -> bool:
    stripped = command.strip()
    if not stripped:
        return False
    if any(char in stripped for char in _SHELL_CONTROL_CHARS):
        return False
    try:
        parts = shlex.split(stripped)
    except ValueError:
        return False
    if not parts:
        return False
    executable = parts[0]
    if executable in _READ_ONLY_SHELL_COMMANDS:
        return True
    if executable == "git" and len(parts) >= 2:
        return parts[1] in _READ_ONLY_GIT_SUBCOMMANDS
    return False
