"""Low-level ANSI row rendering helpers for Wattle's TUI."""

from __future__ import annotations

import textwrap

from wattle.tui_flowers import Flower, gradient_style

RESET = "\x1b[0m"


def wrap_terminal_line(line: str, width: int) -> list[str]:
    return textwrap.wrap(
        line,
        width=max(1, width),
        break_long_words=True,
        break_on_hyphens=False,
        drop_whitespace=False,
        replace_whitespace=False,
    ) or [""]


def terminal_line_width(width: int) -> int:
    return max(1, width)


def styled_terminal_line(text: str, style: str, width: int) -> str:
    visible_width = terminal_line_width(width)
    line = text[:visible_width]
    return f"\r\x1b[?7l\x1b[0m\x1b[2K{style}{line}\x1b[K{RESET}\x1b[?7h"


def styled_transcript_line(
    text: str,
    style: str,
    *,
    fill_remainder: bool = False,
) -> str:
    fill = "\x1b[K" if fill_remainder else ""
    return f"\r\x1b[?7l\x1b[0m\x1b[2K\x1b[?7h{style}{text}{fill}{RESET}"


def filled_terminal_line(rendered: str, fill_style: str, _width: int) -> str:
    return f"\r\x1b[?7l\x1b[0m\x1b[2K{rendered}{RESET}\x1b[?7h"


def running_terminal_line(
    text: str,
    width: int,
    *,
    frame: int,
    flower: Flower | None = None,
) -> str:
    visible_width = terminal_line_width(width)
    line = text[:visible_width].ljust(visible_width)
    highlight = frame % (visible_width + 8) - 4
    flower_index = line.find(flower.shape) if flower is not None else -1
    parts = ["\r\x1b[?7l\x1b[40;38;5;255m\x1b[2K"]
    for index, char in enumerate(line):
        if index == flower_index:
            style = gradient_style(flower, frame=frame)
        else:
            distance = abs(index - highlight)
            if distance == 0:
                style = "\x1b[40;38;5;51;1m"
            elif distance == 1:
                style = "\x1b[40;38;5;87;1m"
            elif distance <= 3:
                style = "\x1b[40;38;5;159m"
            elif distance <= 5:
                style = "\x1b[40;38;5;251m"
            else:
                style = "\x1b[40;38;5;255m"
        parts.append(f"{style}{char}")
    parts.append(f"{RESET}\x1b[?7h")
    return "".join(parts)


def black_terminal_line(text: str, width: int) -> str:
    visible_width = terminal_line_width(width)
    line = text[:visible_width].ljust(visible_width)
    return f"\r\x1b[?7l\x1b[40;38;5;255m\x1b[2K{line}{RESET}\x1b[?7h"
