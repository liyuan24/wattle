"""End-to-end PTY tests for Wattle's live TUI."""

from __future__ import annotations

import textwrap
from pathlib import Path

from pty_harness import PtySession


def _slow_wattle_child_code(
    *,
    first_delay: float = 1.0,
    later_delay: float = 0.2,
    prompt: str | None = "start",
) -> str:
    return textwrap.dedent(
        f"""
        import argparse
        import time

        from wattle.permissions import PermissionMode
        from wattle.providers import (
            CompletionResponse,
            Provider,
            StreamComplete,
            TextBlock,
            TextDelta,
        )
        from wattle.tui import WattleApp


        class SlowProvider(Provider):
            def __init__(self):
                self.calls = 0

            def complete(self, request):
                self.calls += 1
                return CompletionResponse(
                    content=[TextBlock(text=f"done {{self.calls}}")],
                    stop_reason="end_turn",
                    usage={{}},
                )

            def stream(self, request):
                self.calls += 1
                time.sleep({first_delay!r} if self.calls == 1 else {later_delay!r})
                yield TextDelta(text=f"done {{self.calls}}")
                yield StreamComplete(
                    CompletionResponse(
                        content=[TextBlock(text=f"done {{self.calls}}")],
                        stop_reason="end_turn",
                        usage={{}},
                    )
                )


        args = argparse.Namespace(
            provider="openai_responses",
            model="gpt-5.5",
            max_tokens=4096,
            thinking=False,
            effort=None,
            prompt={prompt!r},
            persist_session=False,
            permission_mode=PermissionMode.YOLO,
        )
        raise SystemExit(WattleApp(args, SlowProvider()).run())
        """
    )


def _clear_wattle_child_code() -> str:
    return textwrap.dedent(
        """
        import argparse

        from wattle.permissions import PermissionMode
        from wattle.providers import (
            CompletionResponse,
            Provider,
            StreamComplete,
            TextBlock,
            TextDelta,
        )
        from wattle.tui import WattleApp


        class ClearProvider(Provider):
            def __init__(self):
                self.calls = 0

            def complete(self, request):
                self.calls += 1
                return CompletionResponse(
                    content=[TextBlock(text=f"done {self.calls}")],
                    stop_reason="end_turn",
                    usage={"input_tokens": 10 * self.calls, "output_tokens": self.calls},
                )

            def stream(self, request):
                self.calls += 1
                yield TextDelta(text=f"done {self.calls}")
                yield StreamComplete(
                    CompletionResponse(
                        content=[TextBlock(text=f"done {self.calls}")],
                        stop_reason="end_turn",
                        usage={
                            "input_tokens": 10 * self.calls,
                            "output_tokens": self.calls,
                        },
                    )
                )


        args = argparse.Namespace(
            provider="openai_responses",
            model="gpt-5.5",
            max_tokens=4096,
            thinking=False,
            effort=None,
            prompt=None,
            persist_session=False,
            permission_mode=PermissionMode.YOLO,
        )
        raise SystemExit(WattleApp(args, ClearProvider()).run())
        """
    )


def _subagent_wait_child_code() -> str:
    return textwrap.dedent(
        """
        import argparse
        import time

        from wattle.permissions import PermissionMode
        from wattle.providers import (
            CompletionResponse,
            Provider,
            StreamComplete,
            TextBlock,
            TextDelta,
            ToolUseBlock,
            ToolUseDelta,
        )
        from wattle.tui import WattleApp


        def find_subagent_id(messages):
            for message in messages:
                for block in message.content:
                    content = getattr(block, "content", "")
                    if not isinstance(content, str):
                        continue
                    for line in content.splitlines():
                        if line.startswith("subagent_id: "):
                            return line.removeprefix("subagent_id: ")
            return "missing-subagent"


        class ChildProvider(Provider):
            def complete(self, request):
                time.sleep(0.8)
                return CompletionResponse(
                    content=[TextBlock(text="child result")],
                    stop_reason="end_turn",
                    usage={},
                )

            def stream(self, request):
                response = self.complete(request)
                yield TextDelta(text="child result")
                yield StreamComplete(response)


        class ParentProvider(Provider):
            def __init__(self):
                self.calls = 0

            def fork(self):
                return ChildProvider()

            def complete(self, request):
                return CompletionResponse(
                    content=[TextBlock(text="done")],
                    stop_reason="end_turn",
                    usage={},
                )

            def stream(self, request):
                self.calls += 1
                if self.calls == 1:
                    yield ToolUseDelta(id="spawn_1", name="spawn_agent", partial_json=None)
                    yield StreamComplete(
                        CompletionResponse(
                            content=[
                                ToolUseBlock(
                                    id="spawn_1",
                                    name="spawn_agent",
                                    input={
                                        "task": "inspect prompt waiting state",
                                        "agent_type": "explorer",
                                    },
                                )
                            ],
                            stop_reason="tool_use",
                            usage={},
                        )
                    )
                    return
                if self.calls == 2:
                    subagent_id = find_subagent_id(request.messages)
                    yield ToolUseDelta(id="wait_1", name="wait_agent", partial_json=None)
                    yield StreamComplete(
                        CompletionResponse(
                            content=[
                                ToolUseBlock(
                                    id="wait_1",
                                    name="wait_agent",
                                    input={
                                        "subagent_id": subagent_id,
                                        "timeout_seconds": 3,
                                    },
                                )
                            ],
                            stop_reason="tool_use",
                            usage={},
                        )
                    )
                    return
                yield TextDelta(text="done")
                yield StreamComplete(
                    CompletionResponse(
                        content=[TextBlock(text="done")],
                        stop_reason="end_turn",
                        usage={},
                    )
                )


        args = argparse.Namespace(
            provider="openai_responses",
            model="gpt-5.5",
            max_tokens=4096,
            thinking=False,
            effort="xhigh",
            prompt="delegate",
            persist_session=False,
            permission_mode=PermissionMode.YOLO,
        )
        raise SystemExit(WattleApp(args, ParentProvider()).run())
        """
    )


def _research_child_code() -> str:
    return textwrap.dedent(
        """
        import argparse

        from wattle.permissions import PermissionMode
        from wattle.providers import (
            CompletionResponse,
            Provider,
            StreamComplete,
            TextBlock,
            TextDelta,
            ToolUseBlock,
            ToolUseDelta,
        )
        from wattle.tui import WattleApp


        class ResearchProvider(Provider):
            def __init__(self):
                self.calls = 0

            def complete(self, request):
                return CompletionResponse(
                    content=[TextBlock(text="done")],
                    stop_reason="end_turn",
                    usage={},
                )

            def stream(self, request):
                self.calls += 1
                if self.calls == 1:
                    yield ToolUseDelta(id="read_1", name="read", partial_json=None)
                    yield ToolUseDelta(id="read_2", name="read", partial_json=None)
                    yield ToolUseDelta(id="read_3", name="read", partial_json=None)
                    yield StreamComplete(
                        CompletionResponse(
                            content=[
                                ToolUseBlock(
                                    id="read_1",
                                    name="read",
                                    input={"path": "notes.txt"},
                                ),
                                ToolUseBlock(
                                    id="read_2",
                                    name="read",
                                    input={"path": "other.txt"},
                                ),
                                ToolUseBlock(
                                    id="read_3",
                                    name="read",
                                    input={"path": "third.txt"},
                                ),
                            ],
                            stop_reason="tool_use",
                            usage={},
                        )
                    )
                    return
                yield TextDelta(text="done")
                yield StreamComplete(
                    CompletionResponse(
                        content=[TextBlock(text="done")],
                        stop_reason="end_turn",
                        usage={},
                    )
                )


        args = argparse.Namespace(
            provider="openai_responses",
            model="gpt-5.5",
            max_tokens=4096,
            thinking=False,
            effort=None,
            prompt="research",
            persist_session=False,
            permission_mode=PermissionMode.YOLO,
        )
        raise SystemExit(WattleApp(args, ResearchProvider()).run())
        """
    )


def test_pty_research_tool_calls_render_aggregate(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "other.txt").write_text("more\n", encoding="utf-8")
    (tmp_path / "third.txt").write_text("final\n", encoding="utf-8")

    with PtySession.spawn_python(
        _research_child_code(),
        cwd=tmp_path,
        cols=100,
        rows=30,
    ) as session:
        session.read_until("Researched", timeout=4)
        session.read_until("done", timeout=4)

        screen_text = session.screen.text()
        assert "Read notes.txt" in screen_text
        assert "Read other.txt" in screen_text
        assert "Read third.txt" in screen_text
        assert "read ok - notes.txt" not in screen_text


def test_pty_subagent_waiting_and_completion_notifications(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _subagent_wait_child_code(),
        cwd=tmp_path,
        cols=120,
        rows=36,
    ) as session:
        session.read_until("Spawned Hopper [explorer] (gpt-5.5 xhigh)", timeout=4)
        session.read_until("Waiting for 1 subagent", timeout=4)
        session.read_until("Hopper [explorer] completed", timeout=6)

        screen_text = session.screen.text()
        assert "Waiting for subagent(s)" not in screen_text
        assert "Workspace:" in screen_text
        assert "inspect prompt waiting state" in screen_text
        assert "subagent_id:" not in screen_text


def test_pty_dragged_image_uses_anchor_while_queued_and_after_finish(tmp_path: Path) -> None:
    image = tmp_path / "dragged image.png"
    image.write_bytes(b"fake-png")
    escaped_path = str(image).replace(" ", "\\ ")

    with PtySession.spawn_python(
        _slow_wattle_child_code(first_delay=2.0, later_delay=0.1),
        cwd=tmp_path,
        cols=100,
        rows=30,
    ) as session:
        session.read_until("press esc to interrupt", timeout=3)
        session.write(f"check {escaped_path}\n")
        session.read_until("Messages to be submitted after next tool call", timeout=3)
        session.read_until("check [image#1]", timeout=3)

        screen_text = session.screen.text()
        assert "check [image#1]" in screen_text
        assert "dragged\\ image.png" not in screen_text
        assert str(image) not in screen_text

        session.read_until("Worked for", timeout=5)
        screen_text = session.screen.text()
        assert "check [image#1]" in screen_text
        assert str(image) not in screen_text


def test_pty_queue_command_renders_end_turn_followup_panel(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _slow_wattle_child_code(first_delay=1.5, later_delay=0.1),
        cwd=tmp_path,
        cols=100,
        rows=30,
    ) as session:
        session.read_until("press esc to interrupt", timeout=3)
        session.write("/queue after the full turn\n")
        session.read_until(
            "Messages to be submitted after assistant turn completes",
            timeout=3,
        )
        session.read_until("after the full turn", timeout=3)

        screen_text = session.screen.text()
        assert "Messages to be submitted after next tool call" not in screen_text
        assert "after the full turn" in screen_text


def test_pty_dragged_image_uses_anchor_in_active_input(tmp_path: Path) -> None:
    image = tmp_path / "dragged image.png"
    image.write_bytes(b"fake-png")
    escaped_path = str(image).replace(" ", "\\ ")

    with PtySession.spawn_python(
        _slow_wattle_child_code(first_delay=2.0, later_delay=0.1),
        cwd=tmp_path,
        cols=60,
        rows=30,
    ) as session:
        session.read_until("press esc to interrupt", timeout=3)
        session.write(f"check {escaped_path}")
        session.read_until("check [image#1]", timeout=3)

        screen_text = session.screen.text()
        assert "check [image#1]" in screen_text
        assert "dragged\\ image.png" not in screen_text
        assert str(image) not in screen_text


def _input_box_rows(screen: object) -> tuple[int, int, int]:
    row = screen.find_row_containing(" > ")
    return row - 1, row, row + 1


def _assert_single_three_row_input_box(screen: object) -> None:
    rows = _input_box_rows(screen)
    for row_index in rows:
        backgrounds = screen.row_backgrounds(row_index)
        assert "black" not in backgrounds
        assert all(background == "ansi-235" for background in backgrounds)
    prompt_backgrounds = screen.row_backgrounds(rows[1])
    assert any(background == "ansi-235" for background in prompt_backgrounds)
    before_row = rows[0] - 1
    after_row = rows[2] + 1
    if before_row >= 0:
        assert any(background != "ansi-235" for background in screen.row_backgrounds(before_row))
    if after_row < screen.rows:
        assert any(background != "ansi-235" for background in screen.row_backgrounds(after_row))


def _assert_no_right_side_prompt_or_status_stripes(screen: object) -> None:
    for row_index in range(screen.rows):
        backgrounds = screen.row_backgrounds(row_index)
        for background in ("ansi-235", "ansi-236"):
            if background not in backgrounds:
                continue
            first_index = backgrounds.index(background)
            assert first_index == 0, (
                f"row {row_index} has orphan {background} stripe starting at "
                f"column {first_index}: {screen.row_text(row_index)!r}"
            )


def test_pty_repeated_resize_keeps_black_out_of_input_box(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _slow_wattle_child_code(first_delay=1.4, later_delay=0.1),
        cwd=tmp_path,
        cols=90,
        rows=28,
    ) as session:
        session.read_until("press esc to interrupt", timeout=3)
        session.write("queued input that stays visible during resize")
        session.read_until("queued input", timeout=3)

        for cols in (38, 120, 24, 96):
            session.resize(cols=cols, rows=28)
            session.read_for(0.18)

        row = session.screen.find_row_containing(" > ")
        text = session.screen.row_text(row)
        assert "queued input" in text

        _assert_single_three_row_input_box(session.screen)
        _assert_no_right_side_prompt_or_status_stripes(session.screen)


def test_pty_resize_does_not_leave_reflowed_prompt_box_rows(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _slow_wattle_child_code(first_delay=2.0, later_delay=0.1),
        cwd=tmp_path,
        cols=120,
        rows=30,
    ) as session:
        session.read_until("press esc to interrupt", timeout=3)
        session.read_until(" > ", timeout=3)
        session.resize(cols=40, rows=30)
        session.read_for(0.35)

        _assert_single_three_row_input_box(session.screen)
        _assert_no_right_side_prompt_or_status_stripes(session.screen)


def test_pty_exit_command_works_while_assistant_is_working(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _slow_wattle_child_code(first_delay=3.0, later_delay=0.1),
        cwd=tmp_path,
        cols=100,
        rows=30,
    ) as session:
        session.read_until("press esc to interrupt", timeout=3)
        session.write("/exit\n")
        session.read_until("Goodbye.", timeout=3)

        assert "Messages to be submitted after next tool call" not in session.plain_output
        assert "done 1" not in session.plain_output
        assert session.process.wait(timeout=3) == 0


def test_pty_idle_resize_keeps_statusline_on_terminal_background(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _slow_wattle_child_code(first_delay=0.1, later_delay=0.1),
        cwd=tmp_path,
        cols=44,
        rows=24,
    ) as session:
        session.read_until("Worked for", timeout=3)
        session.resize(cols=96, rows=24)
        session.read_for(0.25)

        row = session.screen.find_row_containing("Context")
        text = session.screen.row_text(row)
        assert text.startswith(" gpt-5.5")
        backgrounds = session.screen.row_backgrounds(row)
        assert all(background is None for background in backgrounds)


def test_pty_idle_screen_preserves_core_visual_contract(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _slow_wattle_child_code(first_delay=0.1, later_delay=0.1, prompt=None),
        cwd=tmp_path,
        cols=96,
        rows=36,
    ) as session:
        session.read_until(">", timeout=3)
        session.write("hello\n")
        session.read_until("done 1", timeout=3)
        session.read_until("Worked for", timeout=3)

        screen_text = session.screen.text()
        assert "Wattle Agent" in screen_text
        assert "model:     gpt-5.5" in screen_text

        user_row = session.screen.find_row_containing(" hello")
        assistant_row = session.screen.find_row_containing(" done 1")
        worked_row = session.screen.find_row_containing("Worked for")
        prompt_rows = _input_box_rows(session.screen)
        status_row = session.screen.find_row_containing("Context")

        assert assistant_row - user_row == 3
        assert worked_row - assistant_row == 2
        assert prompt_rows[0] - worked_row >= 1
        assert status_row == prompt_rows[2] + 1

        for row in (user_row - 1, user_row, user_row + 1):
            assert all(
                background == "ansi-235"
                for background in session.screen.row_backgrounds(row)
            )
        for row in (assistant_row - 1, assistant_row, assistant_row + 1):
            assert all(
                background is None
                for background in session.screen.row_backgrounds(row)
            )
        for row in prompt_rows:
            assert all(
                background == "ansi-235"
                for background in session.screen.row_backgrounds(row)
            )
        assert all(background is None for background in session.screen.row_backgrounds(status_row))


def test_pty_clear_redraws_clean_session_screen(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _clear_wattle_child_code(),
        cwd=tmp_path,
        cols=96,
        rows=36,
    ) as session:
        session.read_until(">", timeout=3)
        session.write("first\n")
        session.read_until("done 1", timeout=3)
        session.read_until("Worked for", timeout=3)

        session.write("/clear\n")
        session.read_until("Last session usage", timeout=3)

        screen_text = session.screen.text()
        assert "Wattle Agent" in screen_text
        assert "Last session usage: last context: 10 tok" in screen_text
        assert "first" not in screen_text
        assert "done 1" not in screen_text
        assert "Conversation cleared." not in screen_text

        session.write("draft")
        session.read_until("draft", timeout=3)
        session.resize(cols=120, rows=42)
        session.read_for(0.35)

        resized_screen_text = session.screen.text()
        assert "Wattle Agent" in resized_screen_text
        assert "Last session usage: last context: 10 tok" in resized_screen_text
        assert "draft" in resized_screen_text
        assert "first" not in resized_screen_text
        assert "done 1" not in resized_screen_text
        assert session.raw_output.count("\x1b[H\x1b[2J\x1b[H") >= 2

        session.write("\x15")
        session.write("second\n")
        session.read_until("done 2", timeout=3)
        assert "done 2" in session.screen.text()


def test_pty_height_shrink_keeps_prompt_near_transcript(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _slow_wattle_child_code(first_delay=0.1, later_delay=0.1, prompt=None),
        cwd=tmp_path,
        cols=120,
        rows=50,
    ) as session:
        session.read_until("Context", timeout=3)

        session.resize(cols=120, rows=24)
        session.read_for(0.35)

        welcome_bottom = session.screen.find_row_containing("└")
        prompt_row = session.screen.find_row_containing(" > ")
        status_row = session.screen.find_row_containing("Context")
        assert prompt_row - welcome_bottom <= 3
        assert status_row - prompt_row <= 2
        assert session.screen.row_text(status_row).startswith(" gpt-5.5")


def test_pty_resize_does_not_leave_large_gap_between_user_and_assistant(
    tmp_path: Path,
) -> None:
    with PtySession.spawn_python(
        _slow_wattle_child_code(first_delay=1.0, later_delay=0.1, prompt=None),
        cwd=tmp_path,
        cols=48,
        rows=60,
    ) as session:
        session.read_until(">", timeout=3)
        session.write("hello\n")
        session.read_until("press esc to interrupt", timeout=3)
        for cols in (28, 96, 36):
            session.resize(cols=cols, rows=60)
            session.read_for(0.18)
        session.read_until("done 1", timeout=4)

        user_row = session.screen.find_row_containing(" hello")
        assistant_row = session.screen.find_row_containing(" done 1")
        assert assistant_row - user_row == 3


def test_pty_transcript_message_reflows_when_terminal_width_grows(
    tmp_path: Path,
) -> None:
    message = "this submitted message should reflow after the terminal gets wider"
    with PtySession.spawn_python(
        _slow_wattle_child_code(first_delay=0.1, later_delay=0.1, prompt=None),
        cwd=tmp_path,
        cols=24,
        rows=40,
    ) as session:
        session.read_until(">", timeout=3)
        session.write(f"{message}\n")
        session.read_until("done 1", timeout=3)

        session.resize(cols=96, rows=40)
        session.read_for(0.25)

        user_row = session.screen.find_row_containing(f" {message}")
        assert session.screen.row_text(user_row).rstrip() == f" {message}"


def test_pty_transcript_rows_do_not_store_zoom_reflow_fill_spaces(
    tmp_path: Path,
) -> None:
    with PtySession.spawn_python(
        _slow_wattle_child_code(first_delay=0.1, later_delay=0.1, prompt=None),
        cwd=tmp_path,
        cols=120,
        rows=32,
    ) as session:
        session.read_until(">", timeout=3)
        session.write("zoom regression\n")
        session.read_until("done 1", timeout=3)
        session.read_until("Worked for", timeout=3)

        user_row = session.screen.find_row_containing(" zoom regression")
        assistant_row = session.screen.find_row_containing(" done 1")
        assert session.screen.row_text(user_row).rstrip() == " zoom regression"
        assert session.screen.row_text(assistant_row).rstrip() == " done 1"

        for row in (user_row, assistant_row):
            backgrounds = session.screen.row_backgrounds(row)
            text = session.screen.row_text(row)
            visible_columns = [index for index, char in enumerate(text) if char != " "]
            assert visible_columns
            expected_background = "ansi-235" if row == user_row else None
            assert all(backgrounds[index] == expected_background for index in visible_columns)
            assert all(background == expected_background for background in backgrounds)
