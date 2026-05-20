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
    assert "replace_all" not in schema["properties"]
    assert "replace_all" not in schema["properties"]["edits"]["items"]["properties"]


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


def test_edit_tool_applies_multiple_replacements_against_original_file(tmp_path) -> None:
    path = tmp_path / "story.txt"
    path.write_text("foo\nbar\nbaz\n")

    EditTool().run(
        str(path),
        edits=[
            {"old_text": "foo\n", "new_text": "foo bar\n"},
            {"old_text": "bar\n", "new_text": "BAR\n"},
        ],
    )

    assert path.read_text() == "foo bar\nBAR\nbaz\n"


def test_edit_tool_rejects_duplicate_old_text(tmp_path) -> None:
    path = tmp_path / "story.txt"
    path.write_text("foo foo foo\n")

    with pytest.raises(ValueError, match="include more context"):
        EditTool().run(str(path), "foo", "bar")

    assert path.read_text() == "foo foo foo\n"


def test_edit_tool_rejects_overlapping_replacements(tmp_path) -> None:
    path = tmp_path / "story.txt"
    path.write_text("one\ntwo\nthree\n")

    with pytest.raises(ValueError, match="overlap"):
        EditTool().run(
            str(path),
            edits=[
                {"old_text": "one\ntwo\n", "new_text": "ONE\nTWO\n"},
                {"old_text": "two\nthree\n", "new_text": "TWO\nTHREE\n"},
            ],
        )

    assert path.read_text() == "one\ntwo\nthree\n"


def test_edit_tool_rejects_replace_all_in_edits(tmp_path) -> None:
    path = tmp_path / "story.txt"
    path.write_text("foo\n")

    with pytest.raises(ValueError, match="replace_all is no longer supported"):
        EditTool().run(
            str(path),
            edits=[{"old_text": "foo", "new_text": "bar", "replace_all": True}],
        )

    assert path.read_text() == "foo\n"


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


def test_edit_tool_preserves_crlf_line_endings(tmp_path) -> None:
    path = tmp_path / "story.txt"
    path.write_bytes(b"alpha\r\nbeta\r\n")

    EditTool().run(str(path), "alpha\n", "ALPHA\n")

    assert path.read_bytes() == b"ALPHA\r\nbeta\r\n"


def test_edit_tool_preserves_utf8_bom(tmp_path) -> None:
    path = tmp_path / "story.txt"
    path.write_text("\ufeffhello\n")

    EditTool().run(str(path), "hello", "hi")

    assert path.read_text() == "\ufeffhi\n"


def test_read_tool_truncates_large_file_with_continuation_hint(tmp_path: Path) -> None:
    path = tmp_path / "large.txt"
    path.write_text("\n".join(f"line {i}" for i in range(2500)))

    output = ReadTool(cwd=tmp_path).run("large.txt")

    assert output.startswith(f"path: {path.resolve()}\nlines: 1-2000 of 2500")
    assert "     1\tline 0" in output
    assert "  2000\tline 1999" in output
    assert "  2001\tline 2000" not in output
    assert "[Read incomplete: showing lines 1-2000 of 2500. Use offset=2001 to continue.]" in output


def test_read_tool_limit_adds_remaining_lines_hint(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("\n".join(f"line {i}" for i in range(10)))

    output = ReadTool(cwd=tmp_path).run("notes.txt", limit=3)

    assert "lines: 1-3 of 10" in output
    assert "     3\tline 2" in output
    assert "     4\tline 3" not in output
    assert "[Read incomplete: 7 more lines in file. Use offset=4 to continue.]" in output


def test_read_tool_offset_beyond_end_is_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "short.txt"
    path.write_text("one\ntwo\n")

    with pytest.raises(ValueError, match=r"Offset 10 is beyond end of file \(2 lines total\)"):
        ReadTool(cwd=tmp_path).run("short.txt", offset=10)


def test_read_tool_rejects_invalid_offset_and_limit(tmp_path: Path) -> None:
    path = tmp_path / "short.txt"
    path.write_text("one\n")
    tool = ReadTool(cwd=tmp_path)

    with pytest.raises(ValueError, match="offset must be 1 or greater"):
        tool.run("short.txt", offset=0)
    with pytest.raises(ValueError, match="limit must be 1 or greater"):
        tool.run("short.txt", limit=0)


def test_read_tool_accepts_at_prefixed_paths(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello\n")

    output = ReadTool(cwd=tmp_path).run("@notes.txt")

    assert f"path: {path.resolve()}" in output
    assert "     1\thello" in output


def test_read_tool_expands_home_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    path = home / "notes.txt"
    path.write_text("hello\n")
    monkeypatch.setenv("HOME", str(home))

    output = ReadTool(cwd=tmp_path).run("~/notes.txt")

    assert f"path: {path.resolve()}" in output
    assert "     1\thello" in output


def test_read_tool_normalizes_unicode_spaces_in_paths(tmp_path: Path) -> None:
    path = tmp_path / "notes draft.txt"
    path.write_text("hello\n")

    output = ReadTool(cwd=tmp_path).run("notes\u00a0draft.txt")

    assert f"path: {path.resolve()}" in output
    assert "     1\thello" in output


def test_read_tool_byte_cap_preserves_complete_lines_with_continuation(tmp_path: Path) -> None:
    path = tmp_path / "wide.txt"
    path.write_text("\n".join(f"{index} " + ("x" * 1000) for index in range(100)))

    output = ReadTool(cwd=tmp_path).run("wide.txt")

    assert "lines: 1-51 of 100" in output
    assert "    51\t50 " in output
    assert "    52\t51 " not in output
    assert "[Read incomplete: showing lines 1-51 of 100. Use offset=52 to continue.]" in output


def test_read_tool_huge_single_line_is_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "minified.js"
    path.write_text("x" * (60 * 1024))

    with pytest.raises(ValueError, match=r"Line 1 is 60\.0 KB, exceeding the 50\.0 KB read limit"):
        ReadTool(cwd=tmp_path).run("minified.js")


def test_read_tool_directory_error_is_clear(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError, match="Path is a directory, not a file"):
        ReadTool(cwd=tmp_path).run(".")


def test_read_tool_rejects_binary_files(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"abc\x00def")

    with pytest.raises(ValueError, match="File appears to be binary"):
        ReadTool(cwd=tmp_path).run("data.bin")


def test_read_tool_utf8_decode_error_is_clear(tmp_path: Path) -> None:
    path = tmp_path / "latin1.txt"
    path.write_bytes(b"\xff")

    with pytest.raises(UnicodeDecodeError, match="could not decode"):
        ReadTool(cwd=tmp_path).run("latin1.txt")


def test_view_image_tool_requires_explicit_path(tmp_path: Path) -> None:
    image = tmp_path / "shot.png"
    image.write_bytes(b"png")

    tool = ViewImageTool(cwd=tmp_path)
    block = tool.run(path="shot.png")

    assert block.type == "image"
    assert block.path == str(image.resolve())
    assert block.media_type == "image/png"
    assert block.filename == "shot.png"


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
