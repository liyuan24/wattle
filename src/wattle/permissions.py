"""Runtime tool permission helpers.

Wattle now supports only yolo mode: tool calls are allowed without a runtime
permission prompt or read-only gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from wattle.providers import ToolUseBlock


class PermissionMode(StrEnum):
    YOLO = "yolo"


@dataclass(frozen=True, slots=True)
class PermissionResult:
    allowed: bool
    denial: str | None = None


class PermissionGate:
    """Runtime gate kept for API compatibility; yolo always allows tools."""

    def __init__(self, mode: PermissionMode = PermissionMode.YOLO) -> None:
        self.mode = PermissionMode.YOLO

    def check(self, block: ToolUseBlock) -> PermissionResult:
        return PermissionResult(allowed=True)


def parse_permission_mode(value: str) -> PermissionMode:
    if value == PermissionMode.YOLO.value:
        return PermissionMode.YOLO
    raise ValueError(f"unknown permission mode: {value!r}")
