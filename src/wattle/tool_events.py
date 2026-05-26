from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class ToolRunEvent:
    """Runtime-only tool progress event for terminal UI rendering."""

    tool_use_id: str
    tool_name: str
    kind: Literal["started", "output", "completed"]
    text: str = ""
    stream: Literal["stdout", "stderr", "combined"] = "combined"
