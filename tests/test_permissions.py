from __future__ import annotations

from wattle.permissions import (
    PermissionGate,
    PermissionMode,
    is_safe_read_only_shell_command,
    tool_permission_summary,
)
from wattle.providers import ToolUseBlock


def _tool(name: str, **input_args: object) -> ToolUseBlock:
    return ToolUseBlock(id="call_1", name=name, input=input_args)


def test_read_only_permission_allows_read_image_and_safe_shell() -> None:
    gate = PermissionGate(PermissionMode.READ_ONLY)

    assert gate.check(_tool("read", path="README.md")).allowed
    assert gate.check(_tool("view_image", path="shot.png")).allowed
    assert gate.check(_tool("bash", command="pwd")).allowed
    assert gate.check(_tool("bash", command="git status --short")).allowed


def test_read_only_permission_blocks_writes_and_unsafe_shell() -> None:
    gate = PermissionGate(PermissionMode.READ_ONLY)

    assert not gate.check(_tool("write", path="README.md", content="x")).allowed
    assert not gate.check(_tool("bash", command="git status; rm -rf build")).allowed
    assert not gate.check(_tool("bash", command="python script.py")).allowed


def test_safe_read_only_shell_command_is_conservative() -> None:
    assert is_safe_read_only_shell_command("rg TODO src")
    assert is_safe_read_only_shell_command("git show HEAD")
    assert not is_safe_read_only_shell_command("sed -i s/a/b/g file")
    assert not is_safe_read_only_shell_command("ls && rm file")


def test_tool_permission_summary_previews_common_write_tools() -> None:
    assert tool_permission_summary(_tool("write", path="a.txt", content="one\ntwo")) == (
        "write: a.txt (2 lines)"
    )
    assert tool_permission_summary(
        _tool("edit", path="a.txt", edits=[{"old": "a", "new": "b"}])
    ) == "edit: a.txt (1 edit)"
