"""PTY-backed helpers for exercising Willow's live terminal UI.

These helpers intentionally sit below the TUI abstraction. They run Willow in a
real pseudo-terminal, feed keyboard input through the master fd, resize the
terminal with TIOCSWINSZ, and keep a small ANSI screen model for assertions
against what a user would see after cursor movement and clears are applied.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import os
import re
import select
import signal
import struct
import subprocess
import sys
import termios
import time
from dataclasses import dataclass
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b[78]")


@dataclass
class Cell:
    char: str = " "
    bg: str | None = None


class TerminalScreen:
    """A deliberately small terminal emulator for Willow's ANSI output."""

    def __init__(self, *, cols: int, rows: int) -> None:
        self.cols = cols
        self.rows = rows
        self.cursor_row = 0
        self.cursor_col = 0
        self.saved_cursor = (0, 0)
        self.bg: str | None = None
        self.pending_escape = ""
        self.cells: list[list[Cell]] = [
            [Cell() for _ in range(cols)] for _ in range(rows)
        ]

    def resize(self, *, cols: int, rows: int) -> None:
        old_cells = self.cells
        self.cols = cols
        self.rows = rows
        self.cells = [[Cell() for _ in range(cols)] for _ in range(rows)]
        for row_index, row in enumerate(old_cells[:rows]):
            for col_index, cell in enumerate(row[:cols]):
                self.cells[row_index][col_index] = Cell(cell.char, cell.bg)
        self.cursor_row = min(self.cursor_row, rows - 1)
        self.cursor_col = min(self.cursor_col, cols - 1)

    def feed(self, text: str) -> None:
        if self.pending_escape:
            text = self.pending_escape + text
            self.pending_escape = ""
        index = 0
        while index < len(text):
            char = text[index]
            if char == "\x1b":
                next_index = self._handle_escape(text, index)
                if next_index == len(text) + 1:
                    self.pending_escape = text[index:]
                    return
                index = next_index
                continue
            if char == "\r":
                self.cursor_col = 0
                index += 1
                continue
            if char == "\n":
                self._newline()
                index += 1
                continue
            if char.isprintable():
                self._put_char(char)
            index += 1

    def text(self) -> str:
        return "\n".join("".join(cell.char for cell in row).rstrip() for row in self.cells)

    def row_text(self, row: int) -> str:
        return "".join(cell.char for cell in self.cells[row])

    def row_backgrounds(self, row: int) -> list[str | None]:
        return [cell.bg for cell in self.cells[row]]

    def find_row_containing(self, needle: str) -> int:
        for index in range(self.rows):
            if needle in self.row_text(index):
                return index
        raise AssertionError(f"screen does not contain {needle!r}\n{self.text()}")

    def _handle_escape(self, text: str, index: int) -> int:
        if text.startswith("\x1b7", index):
            self.saved_cursor = (self.cursor_row, self.cursor_col)
            return index + 2
        if text.startswith("\x1b8", index):
            self.cursor_row, self.cursor_col = self.saved_cursor
            return index + 2
        if not text.startswith("\x1b[", index):
            if index + 1 >= len(text):
                return len(text) + 1
            return index + 1
        match = re.match(r"\x1b\[([0-?]*)([ -/]*)([@-~])", text[index:])
        if match is None:
            if text[index:].startswith("\x1b["):
                return len(text) + 1
            return index + 1
        params, _intermediate, final = match.groups()
        self._handle_csi(params, final)
        return index + len(match.group(0))

    def _handle_csi(self, params: str, final: str) -> None:
        if final == "m":
            self._set_sgr(params)
            return
        if params.startswith(("?", ">")):
            return
        values = [int(value) if value else 0 for value in params.split(";") if value != "?"]
        count = values[0] if values else 1
        if count == 0:
            count = 1
        if final == "A":
            self.cursor_row = max(0, self.cursor_row - count)
        elif final == "B":
            self.cursor_row = min(self.rows - 1, self.cursor_row + count)
        elif final == "C":
            self.cursor_col = min(self.cols - 1, self.cursor_col + count)
        elif final == "D":
            self.cursor_col = max(0, self.cursor_col - count)
        elif final == "K":
            mode = values[0] if values else 0
            if mode == 2:
                self._clear_line()
        elif final == "J":
            mode = values[0] if values else 0
            if mode in {0, 2}:
                self._clear_to_end()
        elif final in {"H", "f"}:
            row = (values[0] - 1) if len(values) >= 1 and values[0] else 0
            col = (values[1] - 1) if len(values) >= 2 and values[1] else 0
            self.cursor_row = max(0, min(self.rows - 1, row))
            self.cursor_col = max(0, min(self.cols - 1, col))

    def _set_sgr(self, params: str) -> None:
        if params.startswith(("?", ">")):
            return
        values = [int(value) if value else 0 for value in params.split(";")]
        if not values:
            values = [0]
        index = 0
        while index < len(values):
            value = values[index]
            if value == 0:
                self.bg = None
            elif value == 40:
                self.bg = "black"
            elif value == 48 and index + 2 < len(values) and values[index + 1] == 5:
                self.bg = f"ansi-{values[index + 2]}"
                index += 2
            index += 1

    def _put_char(self, char: str) -> None:
        self.cells[self.cursor_row][self.cursor_col] = Cell(char, self.bg)
        if self.cursor_col < self.cols - 1:
            self.cursor_col += 1

    def _newline(self) -> None:
        if self.cursor_row == self.rows - 1:
            self.cells.pop(0)
            self.cells.append([Cell() for _ in range(self.cols)])
        else:
            self.cursor_row += 1

    def _clear_line(self) -> None:
        self.cells[self.cursor_row] = [Cell(bg=self.bg) for _ in range(self.cols)]

    def _clear_to_end(self) -> None:
        for row in range(self.cursor_row, self.rows):
            start = self.cursor_col if row == self.cursor_row else 0
            for col in range(start, self.cols):
                self.cells[row][col] = Cell(bg=self.bg)


class PtySession:
    def __init__(
        self,
        process: subprocess.Popen[bytes],
        master_fd: int,
        *,
        cols: int,
        rows: int,
    ) -> None:
        self.process = process
        self.master_fd = master_fd
        self.output = bytearray()
        self.screen = TerminalScreen(cols=cols, rows=rows)

    @classmethod
    def spawn(
        cls,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
        cols: int = 100,
        rows: int = 30,
    ) -> PtySession:
        master_fd, slave_fd = os.openpty()
        session = cls.__new__(cls)
        try:
            _set_pty_size(slave_fd, cols=cols, rows=rows)
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env={**os.environ, **(env or {})},
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                close_fds=True,
                start_new_session=True,
            )
        finally:
            os.close(slave_fd)
        fcntl.fcntl(master_fd, fcntl.F_SETFL, os.O_NONBLOCK)
        session.__init__(process, master_fd, cols=cols, rows=rows)
        return session

    @classmethod
    def spawn_python(
        cls,
        code: str,
        *,
        cwd: Path,
        cols: int = 100,
        rows: int = 30,
        env: dict[str, str] | None = None,
    ) -> PtySession:
        pythonpath = str(Path(__file__).resolve().parents[1] / "src")
        merged_env = {"PYTHONPATH": pythonpath, **(env or {})}
        return cls.spawn(
            [sys.executable, "-c", code],
            cwd=cwd,
            env=merged_env,
            cols=cols,
            rows=rows,
        )

    def __enter__(self) -> PtySession:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.close()

    def write(self, text: str) -> None:
        os.write(self.master_fd, text.encode())

    def resize(self, *, cols: int, rows: int | None = None) -> None:
        rows = rows or self.screen.rows
        _set_pty_size(self.master_fd, cols=cols, rows=rows)
        self.screen.resize(cols=cols, rows=rows)
        try:
            os.killpg(self.process.pid, signal.SIGWINCH)
        except ProcessLookupError:
            return

    def read_available(self) -> str:
        chunks: list[bytes] = []
        while True:
            readable, _, _ = select.select([self.master_fd], [], [], 0)
            if not readable:
                break
            try:
                chunk = os.read(self.master_fd, 65536)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    break
                raise
            if not chunk:
                break
            chunks.append(chunk)
        if not chunks:
            return ""
        data = b"".join(chunks)
        self.output.extend(data)
        text = data.decode(errors="ignore")
        self.screen.feed(text)
        return text

    def read_for(self, seconds: float) -> str:
        deadline = time.monotonic() + seconds
        collected: list[str] = []
        while time.monotonic() < deadline:
            collected.append(self.read_available())
            time.sleep(0.01)
        collected.append(self.read_available())
        return "".join(collected)

    def read_until(self, needle: str, *, timeout: float = 3.0) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.read_available()
            if (
                needle in self.raw_output
                or needle in self.plain_output
                or needle in self.screen.text()
            ):
                return self.raw_output
            if self.process.poll() is not None:
                self.read_available()
                break
            time.sleep(0.01)
        raise AssertionError(f"timed out waiting for {needle!r}\n{self.plain_output}")

    @property
    def raw_output(self) -> str:
        return self.output.decode(errors="ignore")

    @property
    def plain_output(self) -> str:
        return ANSI_RE.sub("", self.raw_output).replace("\r", "")

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1)
        with contextlib.suppress(OSError):
            os.close(self.master_fd)


def _set_pty_size(fd: int, *, cols: int, rows: int) -> None:
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
