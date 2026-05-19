from __future__ import annotations

from pathlib import Path

import pytest

from willow.loop import dispatch_tool_blocks
from willow.providers import ImageBlock, ToolResultBlock, ToolUseBlock
from willow.runtime import WillowRuntime
from willow.tools import TOOLS_BY_NAME, build_tools
from willow.tools.bash import BashTool
from willow.tools.edit import EditTool
from willow.tools.image import ViewImageTool
from willow.tools.monitor import MonitorTool
from willow.tools.read import ReadTool
from willow.tools.subagent import (
    CloseAgentTool,
    SendInputTool,
    SpawnAgentTool,
    WaitAgentTool,
)
from willow.tools.write import WriteTool


def test_edit_tool_is_registered() -> None:
    assert isinstance(TOOLS_BY_NAME["edit"], EditTool)


def test_edit_tool_schema_avoids_top_level_composition_keywords() -> None:
    schema = EditTool.spec()["input_schema"]

    assert schema["type"] == "object"
    assert "anyOf" not in schema
    assert "oneOf" not in schema
    assert "allOf" not in schema


def test_grep_tool_is_not_registered() -> None:
    assert "grep" not in TOOLS_BY_NAME


def test_build_tools_shares_runtime_between_bash_and_monitor(tmp_path) -> None:
    runtime = WillowRuntime(root=tmp_path)
    tools = build_tools(runtime)

    bash = tools["bash"]
    monitor = tools["monitor"]
    spawn_agent = tools["spawn_agent"]
    wait_agent = tools["wait_agent"]

    assert isinstance(bash, BashTool)
    assert isinstance(monitor, MonitorTool)
    assert isinstance(spawn_agent, SpawnAgentTool)
    assert isinstance(tools["view_image"], ViewImageTool)
    assert isinstance(tools["send_input"], SendInputTool)
    assert isinstance(wait_agent, WaitAgentTool)
    assert isinstance(tools["close_agent"], CloseAgentTool)
    assert bash.runtime is runtime
    assert monitor.runtime is runtime
    assert spawn_agent.runtime is runtime
    assert wait_agent.runtime is runtime


def test_default_tools_share_runtime_between_bash_and_monitor() -> None:
    bash = TOOLS_BY_NAME["bash"]
    monitor = TOOLS_BY_NAME["monitor"]
    spawn_agent = TOOLS_BY_NAME["spawn_agent"]

    assert isinstance(bash, BashTool)
    assert isinstance(monitor, MonitorTool)
    assert isinstance(TOOLS_BY_NAME["view_image"], ViewImageTool)
    assert bash.runtime is monitor.runtime
    assert spawn_agent.runtime is bash.runtime


def test_write_tool_returns_unified_diff_for_new_file(tmp_path) -> None:
    path = tmp_path / "funny.txt"

    output = WriteTool().run(str(path), "alpha\nbeta\n")

    assert path.read_text() == "alpha\nbeta\n"
    assert "Wrote 11 bytes" in output
    assert "+++ " in output
    assert "+alpha" in output
    assert "+beta" in output


def test_edit_tool_replaces_text_and_returns_diff(tmp_path) -> None:
    path = tmp_path / "story.txt"
    path.write_text("hello old world\n")

    output = EditTool().run(str(path), "old", "new")

    assert path.read_text() == "hello new world\n"
    assert "Edited" in output
    assert "-hello old world" in output
    assert "+hello new world" in output


def test_edit_tool_applies_multiple_replacements_in_one_call(tmp_path) -> None:
    path = tmp_path / "story.txt"
    path.write_text("alpha\nbeta\ngamma\n")

    output = EditTool().run(
        str(path),
        edits=[
            {"old_text": "alpha", "new_text": "one"},
            {"old_text": "gamma", "new_text": "three"},
        ],
    )

    assert path.read_text() == "one\nbeta\nthree\n"
    assert "2 replacements across 2 edits" in output
    assert "-alpha" in output
    assert "+one" in output
    assert "-gamma" in output
    assert "+three" in output


def test_edit_tool_tolerates_whitespace_only_line_drift(tmp_path) -> None:
    path = tmp_path / "story.txt"
    path.write_text("start\n\nend\n")

    output = EditTool().run(str(path), "start\n    \nend\n", "done\n")

    assert path.read_text() == "done\n"
    assert "Edited" in output


def test_edit_tool_batch_failure_does_not_write_partial_changes(tmp_path) -> None:
    path = tmp_path / "story.txt"
    path.write_text("alpha\nbeta\n")

    with pytest.raises(ValueError, match=r"edit 2"):
        EditTool().run(
            str(path),
            edits=[
                {"old_text": "alpha", "new_text": "one"},
                {"old_text": "missing", "new_text": "two"},
            ],
        )

    assert path.read_text() == "alpha\nbeta\n"


def test_read_tool_large_output_is_externalized(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "large.txt"
    path.write_text("\n".join(f"line {i}" for i in range(5000)))

    output = ReadTool().run(str(path))
    fields = dict(line.split(": ", 1) for line in output.splitlines() if ": " in line)
    full_output_path = Path(fields["full_output_path"])
    full_output = full_output_path.read_text()

    assert output.startswith("[output truncated:")
    assert str(full_output_path).startswith(str(tmp_path / ".willow" / "artifacts"))
    assert "     1\tline 0" in full_output
    assert "  5000\tline 4999" in full_output
    assert len(output) < len(full_output)


def test_view_image_tool_returns_latest_debug_image(tmp_path: Path) -> None:
    debug_dir = tmp_path / "debug_images"
    debug_dir.mkdir()
    old = debug_dir / "old.png"
    latest = debug_dir / "latest.png"
    old.write_bytes(b"old")
    latest.write_bytes(b"new")

    tool = ViewImageTool(cwd=tmp_path)
    block = tool.run(latest_debug=True)

    assert block.type == "image"
    assert block.path == str(latest.resolve())
    assert block.media_type == "image/png"
    assert block.filename == "latest.png"


def test_view_image_dispatch_attaches_image_block(tmp_path: Path) -> None:
    image = tmp_path / "shot.png"
    image.write_bytes(b"png")
    block = ToolUseBlock(
        id="call_1",
        name="view_image",
        input={"path": str(image)},
    )

    content = dispatch_tool_blocks(
        block,
        {"view_image": ViewImageTool(cwd=tmp_path)},
    )

    assert isinstance(content[0], ToolResultBlock)
    assert "Attached image: shot.png" in content[0].content
    assert isinstance(content[1], ImageBlock)
    assert content[1].path == str(image.resolve())
