"""End-to-end PTY tests for Willow's live TUI."""

from __future__ import annotations

import textwrap
from pathlib import Path

from pty_harness import PtySession


def _slow_willow_child_code(
    *,
    first_delay: float = 1.0,
    later_delay: float = 0.2,
    prompt: str | None = "start",
) -> str:
    return textwrap.dedent(
        f"""
        import argparse
        import time

        from willow.permissions import PermissionMode
        from willow.providers import (
            CompletionResponse,
            Provider,
            StreamComplete,
            TextBlock,
            TextDelta,
        )
        from willow.tui import WillowApp


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
        raise SystemExit(WillowApp(args, SlowProvider()).run())
        """
    )


def test_pty_dragged_image_uses_anchor_while_queued_and_after_finish(tmp_path: Path) -> None:
    image = tmp_path / "dragged image.png"
    image.write_bytes(b"fake-png")
    escaped_path = str(image).replace(" ", "\\ ")

    with PtySession.spawn_python(
        _slow_willow_child_code(first_delay=2.0, later_delay=0.1),
        cwd=tmp_path,
        cols=100,
        rows=30,
    ) as session:
        session.read_until("working...", timeout=3)
        session.write(f"check {escaped_path}\n")
        session.read_until("Messages to be submitted after next tool call", timeout=3)

        screen_text = session.screen.text()
        assert "check [Image #1]" in screen_text
        assert "dragged\\ image.png" not in screen_text
        assert str(image) not in screen_text

        session.read_until("Worked for", timeout=5)
        screen_text = session.screen.text()
        assert "check [Image #1]" in screen_text
        assert str(image) not in screen_text


def test_pty_repeated_resize_keeps_black_out_of_input_box(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _slow_willow_child_code(first_delay=1.4, later_delay=0.1),
        cwd=tmp_path,
        cols=90,
        rows=28,
    ) as session:
        session.read_until("working...", timeout=3)
        session.write("queued input that stays visible during resize")
        session.read_until("queued input", timeout=3)

        for cols in (38, 120, 24, 96):
            session.resize(cols=cols, rows=28)
            session.read_for(0.18)

        row = session.screen.find_row_containing(" > ")
        text = session.screen.row_text(row)
        assert "queued input" in text

        input_box_rows = [row - 1, row, row + 1]
        for row_index in input_box_rows:
            backgrounds = session.screen.row_backgrounds(row_index)
            assert "black" not in backgrounds


def test_pty_idle_resize_refills_statusline_to_new_width(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _slow_willow_child_code(first_delay=0.1, later_delay=0.1),
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
        assert all(background == "ansi-236" for background in backgrounds)


def test_pty_resize_does_not_leave_large_gap_between_user_and_assistant(
    tmp_path: Path,
) -> None:
    with PtySession.spawn_python(
        _slow_willow_child_code(first_delay=1.0, later_delay=0.1, prompt=None),
        cwd=tmp_path,
        cols=48,
        rows=60,
    ) as session:
        session.read_until(">", timeout=3)
        session.write("hello\n")
        session.read_until("working...", timeout=3)
        for cols in (28, 96, 36):
            session.resize(cols=cols, rows=60)
            session.read_for(0.18)
        session.read_until("done 1", timeout=4)

        user_row = session.screen.find_row_containing(" hello")
        assistant_row = session.screen.find_row_containing(" done 1")
        assert assistant_row - user_row == 3
