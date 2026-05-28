from __future__ import annotations

import pytest

from wattle.permissions import PermissionGate, PermissionMode, parse_permission_mode
from wattle.providers import ToolUseBlock


def _tool(name: str, **input_args: object) -> ToolUseBlock:
    return ToolUseBlock(id="call_1", name=name, input=input_args)


def test_yolo_permission_allows_all_tool_calls() -> None:
    gate = PermissionGate(PermissionMode.YOLO)

    assert gate.check(_tool("read", path="README.md")).allowed
    assert gate.check(_tool("write", path="README.md", content="x")).allowed
    assert gate.check(_tool("bash", command="rm -rf build")).allowed


def test_parse_permission_mode_only_accepts_yolo() -> None:
    assert parse_permission_mode("yolo") == PermissionMode.YOLO
    with pytest.raises(ValueError):
        parse_permission_mode("read_only")
    with pytest.raises(ValueError):
        parse_permission_mode("ask_for_permission")
