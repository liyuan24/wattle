"""Willow's native terminal chat UI.

The main transcript is append-only stdout. Willow never enters the alternate
screen, never owns terminal scrollback, and never repaints old conversation
history. That is the key difference from the previous app-owned transcript:
native terminal scrollback remains the source of truth.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import mimetypes
import os
import queue
import re
import select
import shlex
import shutil
import signal
import sys
import termios
import textwrap
import threading
import time as _time
import tty
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TextIO, cast
from urllib.parse import unquote, urlparse

from willow.auth import login_openai_codex
from willow.compaction import RuntimeCompaction
from willow.loop import dispatch_tool_blocks_async
from willow.message_history import (
    interrupted_user_text_blocks,
    monitor_event_text_blocks,
    monitor_event_texts,
    queued_user_text_blocks,
)
from willow.models import (
    ModelChoice,
    available_model_choices,
    find_model_choice,
    render_model_choices,
)
from willow.permissions import (
    PermissionAnswer,
    PermissionGate,
    PermissionMode,
    tool_permission_summary,
)
from willow.providers import (
    CompletionResponse,
    ContentBlock,
    ImageBlock,
    Message,
    Provider,
    StreamComplete,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseDelta,
)
from willow.request_preparation import (
    RequestPreparer,
    astream_with_recovery,
    context_window_for_model,
)
from willow.session import (
    SessionCompaction,
    SessionEntry,
    SessionRecord,
    SessionSettings,
    default_session_dir,
    default_session_path,
    list_sessions,
    load_session,
    new_session,
    resolve_session_path,
    save_session,
)
from willow.skills import (
    expand_skill_invocation,
    load_available_skills,
    render_skill_suggestions,
)
from willow.system_prompt import build_system_prompt
from willow.tools import DEFAULT_RUNTIME, TOOLS_BY_NAME
from willow.tools.base import Tool
from willow.tui_flowers import Flower, flower_for_elapsed, gradient_style
from willow.turns import append_turn_step, build_turn_step

time = SimpleNamespace(
    monotonic=_time.monotonic,
    strftime=_time.strftime,
    gmtime=_time.gmtime,
)

with contextlib.suppress(ImportError):
    import readline  # noqa: F401

SLASH_COMMAND_HINTS: tuple[tuple[str, str], ...] = (
    ("/branch", "copy this conversation into a new session branch"),
    ("/clear", "reset conversation history"),
    ("/exit", "exit the TUI"),
    ("/help", "show commands and settings"),
    ("/login", "authenticate OpenAI Codex with OAuth"),
    ("/model", "choose what model to use"),
    ("/quit", "exit the TUI"),
    ("/resume", "switch to a saved session"),
    ("/session", "show persistence and saved session path"),
    ("/status", "show session and status details"),
    ("/statusline", "toggle status snapshots in the prompt"),
)
BUILTIN_SLASH_COMMANDS = frozenset(command for command, _description in SLASH_COMMAND_HINTS)
LOGIN_PROVIDER_CHOICES: tuple[tuple[str, str], ...] = (
    ("openai-codex", "ChatGPT Plus/Pro Codex OAuth"),
)
PASTE_PLACEHOLDER_MIN_CHARS = 200
MAX_IMAGE_ATTACHMENT_BYTES = 20 * 1024 * 1024
SUPPORTED_IMAGE_MEDIA_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/webp", "image/gif"}
)
ESCAPE_SEQUENCE_TIMEOUT_SECONDS = 0.05
KEYBOARD_ENHANCEMENT_ENABLE = "\x1b[>1u\x1b[>4;2m"
KEYBOARD_ENHANCEMENT_DISABLE = "\x1b[<u\x1b[>4m"
SHIFT_ENTER_SEQUENCES = (
    "\x1b[13;2u",
    "\x1b[13;2~",
    "\x1b[27;2;13~",
)
KNOWN_ESCAPE_SEQUENCES = (
    "\x1b[200~",
    "\x1b[201~",
    *SHIFT_ENTER_SEQUENCES,
    "\x1bb",
    "\x1bB",
    "\x1bf",
    "\x1bF",
    "\x1b[1;3D",
    "\x1b[1;5D",
    "\x1b[1;3C",
    "\x1b[1;5C",
    "\x1b[A",
    "\x1bOA",
    "\x1b[B",
    "\x1bOB",
    "\x1b[D",
    "\x1b[C",
    "\x1b[H",
    "\x1b[1~",
    "\x1b[F",
    "\x1b[4~",
)

RESET = "\x1b[0m"
DIM = "\x1b[2m"
BOLD = "\x1b[1m"
USER_STYLE = "\x1b[48;5;235;38;5;231m"
MESSAGE_BLOCK_VERTICAL_PADDING = 1
ASSISTANT_STYLE = "\x1b[40;38;5;255m"
THINKING_STYLE = "\x1b[38;5;245;3m"
WORKED_DURATION_STYLE = "\x1b[38;5;240m"
TOOL_STYLE = "\x1b[48;5;58;38;5;230m"
TOOL_MARKER_STYLE = "\x1b[38;5;82m"
TOOL_MARKER = "|"
TOOL_TITLE_STYLE = "\x1b[38;5;255;1m"
TOOL_PREVIEW_STYLE = "\x1b[38;5;245m"
QUIET_SUCCESS_TOOL_NAMES = frozenset({"wait_agent", "send_input", "close_agent"})
DIFF_ADD_STYLE = "\x1b[48;5;22;38;5;231m"
DIFF_DELETE_STYLE = "\x1b[48;5;52;38;5;231m"
DIFF_META_STYLE = "\x1b[38;5;245m"
SEPARATOR_STYLE = "\x1b[38;5;240m"
ERROR_TEXT_STYLE = "\x1b[38;5;203;1m"
WELCOME_BORDER_STYLE = "\x1b[38;5;244m"
WELCOME_LOGO_STYLE = "\x1b[38;5;107;1m"
WELCOME_TITLE_STYLE = "\x1b[38;5;255;1m"
WELCOME_LABEL_STYLE = "\x1b[38;5;245m"
WELCOME_VALUE_STYLE = "\x1b[38;5;255;1m"
STATUS_STYLE = "\x1b[48;5;236;38;5;248m"
SUBAGENT_WAIT_TITLE_STYLE = "\x1b[48;5;236;38;5;255m"
STATUS_MODEL_STYLE = "\x1b[48;5;236;38;5;82m"
STATUS_TOKEN_STYLE = "\x1b[48;5;236;38;5;203m"
COMPACTION_STYLE = "\x1b[48;5;54;38;5;231;1m"
PROMPT_STYLE = "\x1b[48;5;235;38;5;231m"
ERROR_STYLE = "\x1b[48;5;52;38;5;231m"
PROMPT_MARKER_STYLE = "\x1b[48;5;235;38;5;51;1m"
SELECTED_ROW_STYLE = "\x1b[48;5;240;38;5;255;1m"
COMPACTION_FRAMES = ("◐", "◓", "◑", "◒")
WAIT_AGENT_RUNNING_TITLE = "Waiting for subagent"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b[78]")

WILLOW_LOGO_LINES: tuple[str, ...] = (
    "   \\ | /   ",
    " ~~ \\|/ ~~ ",
    "     Y     ",
)
WELCOME_TITLE = "Willow Agent"


def _abbreviate(content: str, limit: int = 200) -> str:
    flat = content.replace("\n", " ").strip()
    return flat if len(flat) <= limit else flat[:limit] + "..."


def _one_line(content: object, limit: int = 160) -> str:
    flat = " ".join(str(content).split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def _compact_lines(content: str, *, max_lines: int = 4, max_width: int = 110) -> list[str]:
    lines = [line.rstrip() for line in content.splitlines()]
    if not lines:
        return []
    selected = lines[:max_lines]
    rendered = [
        line if len(line) <= max_width else line[: max_width - 3] + "..."
        for line in selected
    ]
    omitted = len(lines) - len(selected)
    if omitted > 0:
        rendered.append(f"... +{omitted} lines")
    return rendered


def _key_value_lines(content: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in content.splitlines():
        key, separator, value = line.partition(": ")
        if separator:
            fields[key] = value
    return fields


def _spawn_agent_title(block: ToolUseBlock, result: ToolResultBlock) -> str:
    fields = _key_value_lines(result.content)
    name = fields.get("name") or fields.get("subagent_id") or "subagent"
    role = fields.get("role") or "subagent"
    model = fields.get("model") or _one_line(block.input.get("model", "default"))
    effort = fields.get("effort") or ""
    model_detail = model if effort in {"", "default", "None"} else f"{model} {effort}"
    return f"Spawned {name} [{role}] ({model_detail})"


def _spawn_agent_detail(block: ToolUseBlock, result: ToolResultBlock) -> str:
    fields = _key_value_lines(result.content)
    workspace = fields.get("workspace") or str(Path.cwd())
    task = fields.get("task") or _one_line(block.input.get("task", ""))
    if task:
        return _one_line(f"└ Workspace: {workspace}. {task}", limit=180)
    return _one_line(f"└ Workspace: {workspace}", limit=180)


def _subagent_event_title(event: Mapping[str, object]) -> str:
    name = str(event.get("name") or event.get("subagent_id") or "subagent")
    role = str(event.get("role") or "subagent")
    status = str(event.get("status") or "updated")
    return f"{name} [{role}] {status}"


def _subagent_event_detail(event: Mapping[str, object]) -> str:
    task = str(event.get("task") or event.get("summary") or "").strip()
    return _one_line(f"└ {task}", limit=180) if task else ""


def _active_subagent_snapshots(snapshots: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        snapshot
        for snapshot in snapshots
        if snapshot.get("status") in {"pending", "running", "closing"}
    ]


def _tool_action_title(block: ToolUseBlock, *, is_error: bool = False) -> str:
    if is_error:
        if block.name in {"read", "write", "edit"}:
            return f"{block.name} error - {_tool_arg(block, 'path', '<missing path>')}"
        return f"{block.name} error"
    if block.name == "bash":
        return f"bash ok - ran {_tool_arg(block, 'command', '<missing command>')}"
    if block.name == "read":
        path = _tool_arg(block, "path", "<missing path>")
        offset = block.input.get("offset")
        limit = block.input.get("limit")
        if offset is not None and limit is not None:
            return f"read ok - {path} lines {offset}-{int(offset) + int(limit) - 1}"
        if offset is not None:
            return f"read ok - {path} from line {offset}"
        return f"read ok - {path}"
    if block.name == "write":
        return f"write ok - {_tool_arg(block, 'path', '<missing path>')}"
    if block.name == "edit":
        return f"edit ok - {_tool_arg(block, 'path', '<missing path>')}"
    return f"{block.name} ok"


def _tool_running_title(block: ToolUseBlock) -> str:
    if block.name == "bash":
        return f"running bash - {_tool_arg(block, 'command', '<missing command>')}"
    if block.name in {"read", "write", "edit"}:
        return f"running {block.name} - {_tool_arg(block, 'path', '<missing path>')}"
    if block.name == "wait_agent":
        return WAIT_AGENT_RUNNING_TITLE
    return f"running {block.name}"


def _tool_edit_title(block: ToolUseBlock, *, path: str, added: int, deleted: int) -> str:
    if block.name == "edit":
        verb = "edited"
    elif deleted == 0:
        verb = "added"
    else:
        verb = "wrote"
    return f"{block.name} ok - {verb} {path} (+{added} -{deleted})"


def _tool_arg(block: ToolUseBlock, name: str, default: str) -> str:
    value = block.input.get(name, default)
    return _one_line(value)


def _suppress_successful_tool_result(
    block: ToolUseBlock,
    result: ToolResultBlock,
) -> bool:
    return block.name in QUIET_SUCCESS_TOOL_NAMES and not result.is_error


def _first_tool_result(
    block: ToolUseBlock,
    blocks: list[ContentBlock],
) -> ToolResultBlock:
    for result in blocks:
        if isinstance(result, ToolResultBlock):
            return result
    return ToolResultBlock(
        tool_use_id=block.id,
        content=f"Tool returned no textual result: {block.name!r}",
        is_error=True,
    )


def _diff_counts(diff_lines: list[str]) -> tuple[int, int]:
    added = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    deleted = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
    return added, deleted


def _diff_starts_from_empty_old_file(diff_lines: list[str]) -> bool:
    return any(line.startswith("@@ -0,0 ") for line in diff_lines)


def _diff_preview_lines(
    diff_lines: list[str], *, max_changes: int | None = 12
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    old_line = 0
    new_line = 0
    shown_changes = 0
    omitted_changes = 0
    for line in diff_lines:
        if line.startswith("@@"):
            parts = line.split()
            if len(parts) >= 3:
                old_line = _parse_hunk_start(parts[1])
                new_line = _parse_hunk_start(parts[2])
            continue
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("+"):
            shown_changes, omitted_changes = _append_diff_row(
                rows,
                shown_changes=shown_changes,
                omitted_changes=omitted_changes,
                max_changes=max_changes,
                kind="add",
                line=f"{new_line:>5} +{line[1:]}",
            )
            new_line += 1
        elif line.startswith("-"):
            shown_changes, omitted_changes = _append_diff_row(
                rows,
                shown_changes=shown_changes,
                omitted_changes=omitted_changes,
                max_changes=max_changes,
                kind="delete",
                line=f"{old_line:>5} -{line[1:]}",
            )
            old_line += 1
        elif line.startswith(" "):
            old_line += 1
            new_line += 1
    if omitted_changes:
        rows.append(("meta", f"... +{omitted_changes} changed lines"))
    return rows


def _append_diff_row(
    rows: list[tuple[str, str]],
    *,
    shown_changes: int,
    omitted_changes: int,
    max_changes: int | None,
    kind: str,
    line: str,
) -> tuple[int, int]:
    if max_changes is None or shown_changes < max_changes:
        rows.append((kind, line))
        shown_changes += 1
    else:
        omitted_changes += 1
    return shown_changes, omitted_changes


def _parse_hunk_start(token: str) -> int:
    value = token[1:].split(",", 1)[0]
    try:
        return int(value)
    except ValueError:
        return 0


def _format_tokens(tokens: int) -> str:
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M tok"
    if tokens >= 1_000:
        return f"{tokens / 1_000:.1f}k tok"
    return f"{tokens} tok"


def _render_statusline(
    *,
    model: str,
    context_tokens: int | None,
    context_window: int | None,
    input_tokens: int,
    cached_tokens: int,
    output_tokens: int,
    cwd: str,
) -> str:
    prefix = f"{model} | Context"
    if context_tokens is None:
        context_tokens = 0
    if context_window is None:
        usage = f"{prefix} unknown ({_format_tokens(context_tokens)}) | window: unknown"
    else:
        percent = (context_tokens / context_window) * 100 if context_window else 0
        usage = (
            f"{prefix} {percent:.1f}% used ({_format_tokens(context_tokens)}) | "
            f"window: {_format_tokens(context_window)}"
        )
    if input_tokens > 0 or output_tokens > 0:
        usage = (
            f"{usage} | input: {_format_tokens(input_tokens)} | "
            f"cached total: {_format_tokens(cached_tokens)} | "
            f"output: {_format_tokens(output_tokens)}"
        )
    return f"{usage} | cwd: {cwd}"


def _cached_tokens_from_usage(usage: dict[str, int]) -> int:
    for key in ("cached_tokens", "cache_read_input_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return 0


def _style_statusline_text(text: str) -> str:
    """Apply statusline segment colors without changing visible text."""
    separator = " | "
    first_sep = text.find(separator)
    if first_sep > 0:
        text = (
            f"{STATUS_MODEL_STYLE}{text[:first_sep]}{STATUS_STYLE}"
            f"{text[first_sep:]}"
        )

    context_start = text.find("Context")
    if context_start == -1:
        return text

    cwd_start = text.find(" | cwd:", context_start)
    if cwd_start == -1:
        cwd_start = len(text.rstrip())

    return (
        f"{text[:context_start]}"
        f"{STATUS_TOKEN_STYLE}{text[context_start:cwd_start]}{STATUS_STYLE}"
        f"{text[cwd_start:]}"
    )


def _context_window_for_model(model: str) -> int | None:
    return context_window_for_model(model)


def _render_command_hints(value: str) -> str:
    stripped = value.strip()
    if not stripped.startswith("/"):
        return ""
    prefix = stripped.split(maxsplit=1)[0]
    matches = [
        f"{command}  {description}"
        for command, description in SLASH_COMMAND_HINTS
        if command.startswith(prefix)
    ]
    return "\n".join(matches)


def _render_input_hints(value: str, cwd: str | Path | None = None) -> str:
    rows: list[str] = []
    file_hints = _render_file_hints(value, cwd or Path.cwd())
    if file_hints:
        rows.extend(file_hints.splitlines())
        return "\n".join(rows)
    command_hints = _render_command_hints(value)
    if command_hints:
        rows.extend(command_hints.splitlines())
    skill_hints = render_skill_suggestions(value, cwd or Path.cwd())
    if skill_hints:
        rows.extend(skill_hints.splitlines())
    return "\n".join(rows)


def _render_file_hints(value: str, cwd: str | Path, *, limit: int = 8) -> str:
    token = _active_at_token(value)
    if token is None:
        return ""
    root = Path(cwd)
    query = token[1:]
    matches: list[tuple[int, str]] = []
    for path in _iter_project_files(root):
        rel = path.relative_to(root).as_posix()
        score = _fuzzy_file_score(query, rel)
        if score is not None:
            matches.append((score, rel))
    matches.sort(key=lambda item: (item[0], len(item[1]), item[1]))
    return "\n".join(_format_at_file_hint(rel) for _score, rel in matches[:limit])


def _active_at_token(value: str) -> str | None:
    stripped = value.rstrip()
    if not stripped:
        return None
    token = stripped.split()[-1]
    return token if token.startswith("@") else None


def _iter_project_files(root: Path) -> Iterator[Path]:
    ignored_dirs = {
        ".cache",
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignored_dirs for part in path.relative_to(root).parts):
            continue
        yield path


def _fuzzy_file_score(query: str, candidate: str) -> int | None:
    if not query:
        return 0
    query_lower = query.lower()
    candidate_lower = candidate.lower()
    if query_lower in candidate_lower:
        return candidate_lower.index(query_lower)
    position = -1
    score = 0
    for char in query_lower:
        next_position = candidate_lower.find(char, position + 1)
        if next_position == -1:
            return None
        score += next_position - position
        position = next_position
    return score + 100


def _format_at_file_hint(relative_path: str) -> str:
    return f"@{shlex.quote(relative_path)}"


def _is_incomplete_escape_sequence(text: str) -> bool:
    return any(
        sequence != text and sequence.startswith(text)
        for sequence in KNOWN_ESCAPE_SEQUENCES
    )


def _hint_command(row: str) -> str:
    if row.startswith("@"):
        return row
    return row.split(maxsplit=1)[0]


def _render_model_picker_rows(
    choices: list[ModelChoice],
    *,
    current_model: str,
    selected_index: int,
    width: int,
    styles_enabled: bool,
) -> list[str]:
    if not choices:
        text = " No models available. Add provider auth to ~/.willow/auth.json."
        return [f"{STATUS_STYLE}{text[:width].ljust(width)}{RESET}" if styles_enabled else text]

    model_width = max(len(choice.model) for choice in choices)
    rows: list[str] = []
    for index, choice in enumerate(choices):
        marker = ">" if index == selected_index else " "
        current = " current" if choice.model == current_model else ""
        line = f" {marker} {choice.model:<{model_width}}{current:<8}  {choice.description}"
        line = line[:width].ljust(width)
        if styles_enabled and index == selected_index:
            rows.append(f"{SELECTED_ROW_STYLE}{line}{RESET}")
        elif styles_enabled:
            rows.append(f"{STATUS_STYLE}{line}{RESET}")
        else:
            rows.append(line)
    return rows


def _render_login_picker_rows(
    *,
    selected_index: int,
    width: int,
    styles_enabled: bool,
) -> list[str]:
    provider_width = max(len(provider) for provider, _description in LOGIN_PROVIDER_CHOICES)
    rows: list[str] = []
    for index, (provider, description) in enumerate(LOGIN_PROVIDER_CHOICES):
        marker = ">" if index == selected_index else " "
        line = f" {marker} {provider:<{provider_width}}  {description}"
        line = line[:width].ljust(width)
        if styles_enabled and index == selected_index:
            rows.append(f"{SELECTED_ROW_STYLE}{line}{RESET}")
        elif styles_enabled:
            rows.append(f"{STATUS_STYLE}{line}{RESET}")
        else:
            rows.append(line)
    return rows


def _render_input_hint_rows(
    rows: list[str],
    *,
    selected_index: int,
    width: int,
    styles_enabled: bool,
) -> list[str]:
    rendered: list[str] = []
    for index, row in enumerate(rows):
        line = f" {row}"[:width].ljust(width)
        if styles_enabled and index == selected_index:
            rendered.append(f"{SELECTED_ROW_STYLE}{line}{RESET}")
        elif styles_enabled:
            rendered.append(f"{STATUS_STYLE}{line}{RESET}")
        else:
            rendered.append(line)
    return rendered


def _apply_hint_to_input(
    buffer: str,
    hint_row: str,
    *,
    append_space_when_empty: bool = False,
) -> str:
    selected_command = _hint_command(hint_row)
    if selected_command.startswith("@"):
        replacement = selected_command[1:]
        return _replace_active_at_token(
            buffer,
            replacement,
            append_space=append_space_when_empty,
        )
    stripped = buffer.strip()
    if not stripped:
        return f"{selected_command} " if append_space_when_empty else selected_command
    parts = stripped.split(maxsplit=1)
    rest = f" {parts[1]}" if len(parts) == 2 else ""
    if not rest and append_space_when_empty:
        rest = " "
    return f"{selected_command}{rest}"


def _replace_active_at_token(
    buffer: str,
    replacement: str,
    *,
    append_space: bool = False,
) -> str:
    trailing = len(buffer) - len(buffer.rstrip())
    searchable = buffer.rstrip()
    token_start = searchable.rfind("@")
    if token_start == -1:
        suffix = "" if trailing else (" " if append_space else "")
        return f"{buffer}{replacement}{suffix}"
    suffix = " " * trailing
    if append_space and not suffix:
        suffix = " "
    return f"{searchable[:token_start]}{replacement}{suffix}"


def _strip_control(text: str) -> str:
    return text.replace("\r", "\\r").replace("\n", "\\n")


def _strip_prompt_control(text: str) -> str:
    return text.replace("\r", "\\r")


@dataclass(frozen=True)
class _PromptInputRender:
    text: str
    cursor: int


@dataclass(frozen=True)
class _PromptInputLines:
    lines: list[str]
    cursor_line: int
    cursor_column: int


@dataclass(frozen=True)
class _PromptFrame:
    rows: list[str]
    width: int
    cursor_line_index: int
    cursor_column: int


@dataclass(frozen=True)
class _RenderedTextLine:
    text: str
    style: str


@dataclass(frozen=True)
class _PathReference:
    path: Path
    start: int
    end: int


def _merge_pasted_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted((start, end) for start, end in ranges if end > start):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _shift_pasted_ranges(
    ranges: list[tuple[int, int]],
    *,
    start: int,
    inserted_length: int,
) -> list[tuple[int, int]]:
    if inserted_length <= 0:
        return ranges
    shifted: list[tuple[int, int]] = []
    for range_start, range_end in ranges:
        if range_start >= start:
            shifted.append((range_start + inserted_length, range_end + inserted_length))
        elif range_start < start < range_end:
            shifted.append((range_start, range_end + inserted_length))
        else:
            shifted.append((range_start, range_end))
    return shifted


def _render_prompt_input(
    buffer: str,
    pasted_ranges: list[tuple[int, int]],
    cursor: int,
) -> _PromptInputRender:
    display_parts: list[str] = []
    display_cursor: int | None = None
    position = 0
    buffer_length = len(buffer)

    def append_segment(segment: str) -> None:
        display_parts.append(_strip_prompt_control(segment))

    for start, end in _merge_pasted_ranges(pasted_ranges):
        start = max(0, min(start, buffer_length))
        end = max(start, min(end, buffer_length))
        if end <= position:
            continue
        if cursor <= start and display_cursor is None:
            display_cursor = len("".join(display_parts)) + len(
                _strip_prompt_control(buffer[position:cursor])
            )
        append_segment(buffer[position:start])
        placeholder = f"[Pasted Content {end - start} chars]"
        if start < cursor <= end and display_cursor is None:
            display_cursor = len("".join(display_parts)) + len(placeholder)
        display_parts.append(placeholder)
        position = end

    if display_cursor is None:
        display_cursor = len("".join(display_parts)) + len(
            _strip_prompt_control(buffer[position:cursor])
        )
    append_segment(buffer[position:])
    return _PromptInputRender(text="".join(display_parts), cursor=display_cursor)


def _wrap_prompt_input(
    preview: str,
    cursor: int,
    *,
    first_width: int,
    continuation_width: int,
) -> _PromptInputLines:
    lines = [""]
    positions: list[tuple[int, int] | None] = [None] * (len(preview) + 1)

    def current_width() -> int:
        return first_width if len(lines) == 1 else continuation_width

    for index, ch in enumerate(preview):
        if ch != "\n" and len(lines[-1]) >= current_width():
            lines.append("")
        positions[index] = (len(lines) - 1, len(lines[-1]))
        if ch == "\n":
            lines.append("")
        else:
            lines[-1] += ch
        positions[index + 1] = (len(lines) - 1, len(lines[-1]))

    cursor = max(0, min(cursor, len(preview)))
    cursor_position = positions[cursor]
    if cursor_position is None:
        cursor_position = (0, 0)
    return _PromptInputLines(
        lines=lines,
        cursor_line=cursor_position[0],
        cursor_column=cursor_position[1],
    )


def _display_cwd(path: Path | None = None) -> str:
    cwd = path or Path.cwd()
    home = Path.home()
    try:
        return f"~/{cwd.relative_to(home)}"
    except ValueError:
        return str(cwd)


def _short_session_id(entry: SessionEntry) -> str:
    return entry.record.metadata.id or entry.path.stem


def _session_label(entry: SessionEntry) -> str:
    metadata = entry.record.metadata
    title = metadata.title or _first_user_text(entry.record) or "(untitled)"
    return (
        f"{_short_session_id(entry):<12} "
        f"{entry.record.settings.model:<18} "
        f"{len(entry.record.messages):>3} msgs  "
        f"{metadata.updated_at}  "
        f"{title}"
    )


def _first_user_text(record: SessionRecord, *, limit: int = 48) -> str | None:
    for message in record.messages:
        if message.role != "user":
            continue
        for block in message.content:
            if isinstance(block, TextBlock) and block.text.strip():
                text = " ".join(block.text.split())
                return text if len(text) <= limit else text[: limit - 3] + "..."
    return None


def _history_ends_with_tool_results(messages: list[Message]) -> bool:
    if not messages:
        return False
    last = messages[-1]
    return last.role == "user" and any(
        isinstance(block, ToolResultBlock) for block in last.content
    )


def _assistant_message(response: CompletionResponse) -> Message:
    return Message(
        role="assistant",
        content=list(response.content),
        input_tokens=response.usage.get("input_tokens", 0),
        output_tokens=response.usage.get("output_tokens", 0),
        cached_tokens=_cached_tokens_from_usage(response.usage),
    )


def _cached_tokens_from_usage(usage: dict[str, int]) -> int:
    for key in ("cached_tokens", "cache_read_input_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return 0


def _turn_error_text(error: BaseException) -> str:
    text = str(error).strip()
    if text:
        return text
    return type(error).__name__


def _input_history_from_messages(messages: list[Message]) -> list[str]:
    history: list[str] = []
    for message in messages:
        if message.role != "user":
            continue
        text_blocks = [
            block.text.strip()
            for block in message.content
            if isinstance(block, TextBlock) and block.text.strip()
        ]
        if not text_blocks:
            continue
        text = "\n\n".join(text_blocks)
        if not history or history[-1] != text:
            history.append(text)
    return history


def _wrap_terminal_line(line: str, width: int) -> list[str]:
    return textwrap.wrap(
        line,
        width=max(1, width),
        break_long_words=True,
        break_on_hyphens=False,
        drop_whitespace=False,
        replace_whitespace=False,
    ) or [""]


def _terminal_line_width(width: int) -> int:
    return max(1, width)


def _styled_terminal_line(text: str, style: str, width: int) -> str:
    visible_width = _terminal_line_width(width)
    line = text[:visible_width]
    return f"\r\x1b[?7l{style}\x1b[2K{line}{RESET}\x1b[?7h"


def _filled_terminal_line(rendered: str, fill_style: str, _width: int) -> str:
    return f"\r\x1b[?7l{fill_style}\x1b[2K{rendered}{RESET}\x1b[?7h"


def _running_terminal_line(
    text: str,
    width: int,
    *,
    frame: int,
    flower: Flower | None = None,
) -> str:
    visible_width = _terminal_line_width(width)
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


def _flower_working_status(elapsed_seconds: int) -> tuple[Flower, str]:
    flower = flower_for_elapsed(elapsed_seconds)
    return flower, f"{flower.shape} {flower.verb}..."


def _black_terminal_line(text: str, width: int) -> str:
    visible_width = _terminal_line_width(width)
    line = text[:visible_width].ljust(visible_width)
    return f"\r\x1b[?7l\x1b[40;38;5;255m\x1b[2K{line}{RESET}\x1b[?7h"


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+(.+)$")
_NUMBERED_RE = re.compile(r"^(\s*)(\d+[.)])\s+(.+)$")
_LINK_RE = re.compile(r"!?\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_STRONG_RE = re.compile(r"(\*\*|__)(.+?)\1")
_EMPHASIS_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)|(?<!_)_([^_\n]+)_(?!_)")
_STRIKE_RE = re.compile(r"~~(.+?)~~")


def _strip_inline_markdown(text: str) -> str:
    def link_replacement(match: re.Match[str]) -> str:
        label = match.group(1).strip()
        target = match.group(2).strip()
        if not label:
            return target
        if match.group(0).startswith("!"):
            return label
        return f"{label} ({target})"

    text = _LINK_RE.sub(link_replacement, text)
    text = _INLINE_CODE_RE.sub(lambda match: match.group(1), text)
    text = _STRONG_RE.sub(lambda match: match.group(2), text)
    text = _EMPHASIS_RE.sub(lambda match: match.group(1) or match.group(2), text)
    text = _STRIKE_RE.sub(lambda match: match.group(1), text)
    return re.sub(r"\\([\\`*_{}\[\]()#+\-.!|>])", r"\1", text)


def _render_markdown_text(text: str, *, width: int) -> list[_RenderedTextLine]:
    rows: list[_RenderedTextLine] = []
    in_code_fence = False

    for raw_line in text.splitlines() or [""]:
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith(("```", "~~~")):
            in_code_fence = not in_code_fence
            continue

        if in_code_fence:
            rows.append(_RenderedTextLine(f"    {line}", TOOL_PREVIEW_STYLE))
            continue

        if not stripped:
            rows.append(_RenderedTextLine("", ASSISTANT_STYLE))
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            rows.append(
                _RenderedTextLine(
                    _strip_inline_markdown(heading.group(2).strip()),
                    f"{ASSISTANT_STYLE}{BOLD}",
                )
            )
            continue

        if re.fullmatch(r"\s{0,3}([-*_])(?:\s*\1){2,}\s*", line):
            rows.append(_RenderedTextLine("─" * max(3, width - 1), SEPARATOR_STYLE))
            continue

        quote_depth = 0
        quote_text = line.lstrip()
        while quote_text.startswith(">"):
            quote_depth += 1
            quote_text = quote_text[1:].lstrip()
        if quote_depth:
            rows.append(
                _RenderedTextLine(
                    f"{'│ ' * quote_depth}{_strip_inline_markdown(quote_text)}",
                    THINKING_STYLE,
                )
            )
            continue

        bullet = _BULLET_RE.match(line)
        if bullet:
            indent = " " * len(bullet.group(1).expandtabs(2))
            rows.append(
                _RenderedTextLine(
                    f"{indent}• {_strip_inline_markdown(bullet.group(2).strip())}",
                    ASSISTANT_STYLE,
                )
            )
            continue

        numbered = _NUMBERED_RE.match(line)
        if numbered:
            indent = " " * len(numbered.group(1).expandtabs(2))
            rows.append(
                _RenderedTextLine(
                    f"{indent}{numbered.group(2)} "
                    f"{_strip_inline_markdown(numbered.group(3).strip())}",
                    ASSISTANT_STYLE,
                )
            )
            continue

        rows.append(_RenderedTextLine(_strip_inline_markdown(line), ASSISTANT_STYLE))

    return rows


def _format_elapsed_compact(seconds: int) -> str:
    seconds = max(0, seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {remaining_seconds}s"
    hours, remaining_minutes = divmod(minutes, 60)
    if remaining_minutes == 0:
        return f"{hours}h"
    return f"{hours}h {remaining_minutes}m"


def _worked_duration_text(started_at: float, *, ended_at: float | None = None) -> str:
    ended = time.monotonic() if ended_at is None else ended_at
    elapsed_seconds = max(0, int(ended - started_at))
    return f"Worked for {_format_elapsed_compact(elapsed_seconds)}"


def _image_summary(block: ImageBlock) -> str:
    size = _format_bytes(block.size_bytes)
    return f"[image] {block.filename} ({block.media_type}, {size})"


def _file_context_text(path: Path, *, cwd: Path | None = None) -> str:
    return _relative_file_path(path, cwd=cwd)


def _relative_file_path(path: Path, *, cwd: Path | None = None) -> str:
    root = (cwd or Path.cwd()).resolve()
    return Path(os.path.relpath(path.resolve(), root)).as_posix()


def _format_bytes(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"


def _candidate_file_paths(text: str, *, cwd: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()
    for reference in _path_references_from_text(text, cwd=cwd):
        if reference.path in seen:
            continue
        candidates.append(reference.path)
        seen.add(reference.path)
    return candidates


def _path_references_from_text(
    text: str,
    *,
    cwd: Path | None = None,
) -> list[_PathReference]:
    cwd = cwd or Path.cwd()
    references: list[_PathReference] = []
    for token, raw_indexes in _shell_like_tokens_with_indexes(text):
        path, start, end = _token_to_path_span(token, raw_indexes, cwd=cwd)
        if path is not None and path.is_file():
            if (
                start > 0
                and end < len(text)
                and text[start - 1] == text[end]
                and text[start - 1] in {"'", '"'}
            ):
                start -= 1
                end += 1
            references.append(_PathReference(path=path, start=start, end=end))
    return references


def _replace_text_ranges(
    text: str,
    replacements: list[tuple[int, int, str]],
) -> str:
    if not replacements:
        return text
    chunks: list[str] = []
    cursor = 0
    for start, end, replacement in sorted(replacements):
        if start < cursor:
            continue
        chunks.append(text[cursor:start])
        chunks.append(replacement)
        cursor = end
    chunks.append(text[cursor:])
    return "".join(chunks)


def _image_placeholder_text(
    text: str,
    *,
    image_index_start: int = 1,
) -> tuple[str, int]:
    replacements: list[tuple[int, int, str]] = []
    image_index = image_index_start
    for reference in _path_references_from_text(text):
        media_type, _encoding = mimetypes.guess_type(reference.path.name)
        if media_type not in SUPPORTED_IMAGE_MEDIA_TYPES:
            continue
        replacements.append((reference.start, reference.end, f"[image#{image_index}]"))
        image_index += 1
    return _replace_text_ranges(text, replacements), image_index


def _image_placeholder_prompt_render(
    render: _PromptInputRender,
    *,
    image_index_start: int = 1,
) -> _PromptInputRender:
    text = render.text
    cursor = max(0, min(render.cursor, len(text)))
    replacements: list[tuple[int, int, str]] = []
    image_index = image_index_start
    for reference in _path_references_from_text(text):
        media_type, _encoding = mimetypes.guess_type(reference.path.name)
        if media_type not in SUPPORTED_IMAGE_MEDIA_TYPES:
            continue
        replacements.append((reference.start, reference.end, f"[image#{image_index}]"))
        image_index += 1
    if not replacements:
        return render

    chunks: list[str] = []
    display_cursor: int | None = None
    source_cursor = 0
    for start, end, replacement in sorted(replacements):
        if start < source_cursor:
            continue
        if display_cursor is None and cursor <= start:
            display_cursor = len("".join(chunks)) + (cursor - source_cursor)
        chunks.append(text[source_cursor:start])
        if display_cursor is None and start < cursor <= end:
            display_cursor = len("".join(chunks)) + len(replacement)
        chunks.append(replacement)
        source_cursor = end
    if display_cursor is None:
        display_cursor = len("".join(chunks)) + (cursor - source_cursor)
    chunks.append(text[source_cursor:])
    return _PromptInputRender(text="".join(chunks), cursor=display_cursor)


def _first_text_block_text(
    content: list[ContentBlock],
    *,
    fallback: str,
) -> str:
    if content and isinstance(content[0], TextBlock):
        return content[0].text
    return fallback


def _shell_like_tokens_with_indexes(text: str) -> list[tuple[str, list[int]]]:
    tokens: list[tuple[str, list[int]]] = []
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break

        value: list[str] = []
        raw_indexes: list[int] = []
        quote: str | None = None
        while index < length:
            char = text[index]
            if quote is None and char.isspace():
                break
            if quote is not None and char == quote:
                quote = None
                index += 1
                continue
            if quote is None and char in {"'", '"'}:
                quote = char
                index += 1
                continue
            if char == "\\" and quote != "'" and index + 1 < length:
                index += 1
                value.append(text[index])
                raw_indexes.append(index)
                index += 1
                continue
            value.append(char)
            raw_indexes.append(index)
            index += 1
        if value:
            tokens.append(("".join(value), raw_indexes))
    return tokens


def _token_to_path_span(
    token: str,
    raw_indexes: list[int],
    *,
    cwd: Path,
) -> tuple[Path | None, int, int]:
    start_index = 0
    if token.startswith("@"):
        start_index = 1
    end_index = len(token)
    while end_index > start_index and token[end_index - 1] in ".,;:)":
        end_index -= 1
    if start_index >= end_index:
        return None, 0, 0

    path = _token_to_path(token[start_index:end_index], cwd=cwd)
    if path is None:
        return None, 0, 0
    start = raw_indexes[0] if start_index == 1 else raw_indexes[start_index]
    end = raw_indexes[end_index - 1] + 1
    return path, start, end


def _should_route_slash_command(text: str, *, cwd: Path | None = None) -> bool:
    stripped = text.strip()
    if not stripped.startswith("/"):
        return False
    command = stripped.split(maxsplit=1)[0]
    if command in BUILTIN_SLASH_COMMANDS:
        return True
    return not _candidate_file_paths(stripped, cwd=cwd)


def _token_to_path(token: str, *, cwd: Path) -> Path | None:
    stripped = token.strip().strip(".,;:)")
    if stripped.startswith("@"):
        stripped = stripped[1:]
    if not stripped:
        return None
    parsed = urlparse(stripped)
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path)).expanduser()
    elif parsed.scheme:
        return None
    else:
        path = Path(stripped).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()


class WillowApp:
    """Append-only native terminal session for Willow."""

    def __init__(
        self,
        args: argparse.Namespace,
        provider: Provider,
        *,
        inline_mode: bool = True,
        state: dict[str, object] | None = None,
        input_func: Callable[[str], str] | None = None,
        out: TextIO | None = None,
    ) -> None:
        self.args = args
        self.provider = provider
        self.inline_mode = inline_mode
        self.input_func = input_func or input
        self.out = out or sys.stdout

        self.current_provider_name: str = args.provider
        self.current_model: str = args.model
        self.max_tokens: int = args.max_tokens
        self.thinking: bool = bool(getattr(args, "thinking", False))
        self.effort: str | None = cast(str | None, getattr(args, "effort", None))
        self.messages: list[Message] = []
        self.runtime = DEFAULT_RUNTIME
        self.tool_specs = [tool.spec() for tool in TOOLS_BY_NAME.values()]
        self.permission_mode: PermissionMode = getattr(
            args,
            "permission_mode",
            PermissionMode.YOLO,
        )
        self.permission_gate = PermissionGate(
            self.permission_mode,
            prompt=self._ask_tool_permission,
        )
        self.system: str | None = build_system_prompt(
            tools_by_name=TOOLS_BY_NAME,
            skills=load_available_skills(Path.cwd()),
            permission_mode=self.permission_mode,
        )

        self._statusline_enabled = True
        self._last_context_tokens: int | None = None
        self._total_input_tokens = 0
        self._total_cached_tokens = 0
        self._total_output_tokens = 0
        self._session_record: SessionRecord | None = None
        self._session_path: Path | None = None
        self._force_plain = input_func is not None or out is not None
        self._last_transcript_was_separator = False
        self._resume_history_pending = False
        self._compaction_state: RuntimeCompaction | None = None

        if state is not None:
            self._restore_state(state)

        if bool(getattr(args, "persist_session", False)):
            resume_record = cast(
                SessionRecord | None,
                getattr(args, "_resume_session_record", None),
            )
            resume_path = cast(Path | None, getattr(args, "_resume_session_path", None))
            if resume_record is not None:
                self._session_record = resume_record
                self._session_path = resume_path or default_session_path(resume_record.metadata.id)
                self._resume_history_pending = bool(resume_record.messages)
                self._compaction_state = _runtime_compaction_from_session(resume_record)
            else:
                self._session_record = new_session(
                    provider=self.current_provider_name,
                    model=self.current_model,
                    system=self.system,
                    max_tokens=self.max_tokens,
                    thinking=self.thinking,
                    effort=cast(Any, self.effort),
                )
                self._session_path = default_session_path(self._session_record.metadata.id)
            self._persist_session()

    def run(self, **_ignored_app_kwargs: Any) -> int:
        """Run until EOF, KeyboardInterrupt, /exit, or /quit."""
        if self._can_run_live():
            return self._run_live()
        return self._run_basic()

    def _run_basic(self) -> int:
        """Portable append-only loop used for tests and non-TTY stdio."""
        self._write_welcome_card()
        self._write_resume_history_if_pending()
        self._continue_resumed_turn_if_needed()
        positional_prompt = self._positional_prompt_text()
        if positional_prompt and self._submit_user_text(positional_prompt, render=True):
            self._run_turn_recovering()
        while True:
            try:
                user_input = self.input_func(self._prompt())
            except EOFError:
                self._write_line("")
                break
            except KeyboardInterrupt:
                self._write_line("")
                break

            text = user_input.strip()
            if not text:
                continue
            if self._expand_skill_text(text) is None and _should_route_slash_command(text):
                if self._handle_slash(text):
                    break
                continue

            if self._submit_user_text(text, render=True):
                self._run_turn_recovering()

        self._write_line("Goodbye.")
        return 0

    def _positional_prompt_text(self) -> str:
        prompt = getattr(self.args, "prompt", None)
        return prompt.strip() if isinstance(prompt, str) else ""

    def _submit_user_text(self, text: str, *, render: bool) -> bool:
        expanded_text = self._expand_skill_text(text)
        try:
            content = self._user_content_blocks(expanded_text or text)
        except ValueError as exc:
            self._write_panel("error", str(exc), ERROR_STYLE)
            return False
        if render:
            self._write_block(
                self._user_display_text(
                    text,
                    content,
                    prefer_content_text=expanded_text is None,
                ),
                USER_STYLE,
            )
        self.messages.append(Message(role="user", content=content))
        self._persist_session()
        return True

    def _user_content_blocks(
        self,
        text: str,
        *,
        image_index_start: int = 1,
    ) -> list[ContentBlock]:
        content: list[ContentBlock] = []
        display_text, images = self._image_placeholders_from_text(
            text,
            image_index_start=image_index_start,
        )
        if display_text.strip():
            content.append(TextBlock(text=display_text))
        content.extend(images)
        for path in self._file_context_paths_from_text(text):
            content.append(TextBlock(text=_file_context_text(path)))
        return content

    def _user_display_text(
        self,
        text: str,
        content: list[ContentBlock],
        *,
        prefer_content_text: bool = True,
    ) -> str:
        display_text = text
        if prefer_content_text and content and isinstance(content[0], TextBlock):
            display_text = content[0].text
        return display_text

    def _image_placeholders_from_text(
        self,
        text: str,
        *,
        image_index_start: int,
    ) -> tuple[str, list[ImageBlock]]:
        blocks: list[ImageBlock] = []
        display_text, _next_image_index = _image_placeholder_text(
            text,
            image_index_start=image_index_start,
        )
        for reference in _path_references_from_text(text):
            size = reference.path.stat().st_size
            media_type, _encoding = mimetypes.guess_type(reference.path.name)
            if media_type not in SUPPORTED_IMAGE_MEDIA_TYPES:
                continue
            if size > MAX_IMAGE_ATTACHMENT_BYTES:
                raise ValueError(
                    f"Image is too large: {reference.path} ({_format_bytes(size)}; "
                    f"max {_format_bytes(MAX_IMAGE_ATTACHMENT_BYTES)})"
                )
            stored_path = self._copy_image_asset(reference.path)
            blocks.append(
                ImageBlock(
                    path=str(stored_path),
                    media_type=media_type,
                    filename=reference.path.name,
                    size_bytes=size,
                )
            )
        return display_text, blocks

    def _file_context_paths_from_text(self, text: str) -> list[Path]:
        paths: list[Path] = []
        for path in _candidate_file_paths(text):
            media_type, _encoding = mimetypes.guess_type(path.name)
            if media_type in SUPPORTED_IMAGE_MEDIA_TYPES:
                continue
            paths.append(path)
        return paths

    def _copy_image_asset(self, path: Path) -> Path:
        if self._session_record is None:
            return path
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()[:16]
        asset_dir = default_session_dir() / "assets" / self._session_record.metadata.id
        asset_dir.mkdir(parents=True, exist_ok=True)
        target = asset_dir / f"{digest}{path.suffix.lower()}"
        if not target.exists():
            target.write_bytes(data)
        return target

    def _can_run_live(self) -> bool:
        return (
            not self._force_plain
            and hasattr(sys.stdin, "isatty")
            and sys.stdin.isatty()
            and hasattr(self.out, "isatty")
            and self.out.isatty()
        )

    def _run_live(self) -> int:
        terminal = _LiveTerminal(self)
        return terminal.run()

    def snapshot_state(self) -> dict[str, object]:
        return {
            "current_provider_name": self.current_provider_name,
            "current_model": self.current_model,
            "system": self.system,
            "max_tokens": self.max_tokens,
            "thinking": self.thinking,
            "effort": self.effort,
            "permission_mode": self.permission_mode,
            "messages": list(self.messages),
            "compaction_state": self._compaction_state,
            "statusline_enabled": self._statusline_enabled,
            "last_context_tokens": self._last_context_tokens,
            "total_input_tokens": self._total_input_tokens,
            "total_cached_tokens": self._total_cached_tokens,
            "total_output_tokens": self._total_output_tokens,
        }

    def _restore_state(self, state: dict[str, object]) -> None:
        self.current_provider_name = str(
            state.get("current_provider_name", self.current_provider_name)
        )
        self.current_model = str(state.get("current_model", self.current_model))
        self.system = cast(str | None, state.get("system", self.system))
        self.max_tokens = int(cast(Any, state.get("max_tokens", self.max_tokens)))
        self.thinking = bool(state.get("thinking", self.thinking))
        self.effort = cast(str | None, state.get("effort", self.effort))
        self.permission_mode = cast(
            PermissionMode,
            state.get("permission_mode", self.permission_mode),
        )
        self.permission_gate = PermissionGate(
            self.permission_mode,
            prompt=self._ask_tool_permission,
        )
        self.messages = list(cast(list[Message], state.get("messages", self.messages)))
        self._compaction_state = cast(
            RuntimeCompaction | None,
            state.get("compaction_state"),
        )
        self._statusline_enabled = bool(
            state.get("statusline_enabled", self._statusline_enabled)
        )
        raw_context_tokens = state.get("last_context_tokens", self._last_context_tokens)
        self._last_context_tokens = (
            int(cast(Any, raw_context_tokens)) if raw_context_tokens is not None else None
        )
        self._total_input_tokens = int(
            cast(Any, state.get("total_input_tokens", self._total_input_tokens))
        )
        self._total_cached_tokens = int(
            cast(Any, state.get("total_cached_tokens", self._total_cached_tokens))
        )
        self._total_output_tokens = int(
            cast(Any, state.get("total_output_tokens", self._total_output_tokens))
        )

    def _run_turn(self, *, started_at: float | None = None) -> None:
        if started_at is None:
            started_at = time.monotonic()
        self._configure_subagents()
        preparer = self._request_preparer()
        while True:
            response = self._drive_stream_with_recovery(preparer)
            self._record_usage(response)

            has_tool_uses = any(isinstance(block, ToolUseBlock) for block in response.content)
            if has_tool_uses:
                self._write_separator()
                self._append_assistant_response(response)
                tool_results = (
                    self._dispatch_tools(response)
                    if response.stop_reason == "tool_use"
                    else []
                )
                monitor_events = self.runtime.events.drain()
                pending_monitor = monitor_event_text_blocks(monitor_events)
                if self._append_followup_user([*tool_results, *pending_monitor]):
                    if tool_results:
                        self._write_separator()
                    continue
                self._write_status_snapshot()
                self._write_worked_duration(started_at)
                return

            monitor_events = self.runtime.events.drain()
            pending_monitor = monitor_event_text_blocks(monitor_events)
            step = build_turn_step(
                response,
                pending_user_blocks=pending_monitor,
            )
            append_turn_step(self.messages, step)
            self._persist_session()
            if not step.continue_running:
                self._write_status_snapshot()
                self._write_worked_duration(started_at)
                return

    def _run_turn_recovering(self) -> None:
        started_at = time.monotonic()
        try:
            self._run_turn(started_at=started_at)
        except Exception as exc:  # noqa: BLE001
            self._write_turn_error(exc)
            self._write_worked_duration(started_at)

    def _write_turn_error(self, error: BaseException) -> None:
        self._write_panel("error", _turn_error_text(error), ERROR_STYLE)
        self._write_status_snapshot()

    def _append_assistant_response(self, response: CompletionResponse) -> None:
        self.messages.append(_assistant_message(response))
        self._persist_session()

    def _append_followup_user(self, content: list[ContentBlock]) -> bool:
        if not content:
            return False
        self.messages.append(Message(role="user", content=content))
        self._persist_session()
        return True

    def _request_preparer(
        self,
        *,
        provider: Provider | None = None,
        on_compaction_start: Callable[[], None] | None = None,
        on_compaction_end: Callable[[], None] | None = None,
    ) -> RequestPreparer:
        return RequestPreparer(
            provider=provider or self.provider,
            model=self.current_model,
            system=self.system,
            tools=self.tool_specs,
            max_tokens=self.max_tokens,
            thinking=self.thinking,
            effort=cast(Any, self.effort),
            context_window=_context_window_for_model(self.current_model),
            state=self._compaction_state,
            on_compaction_start=(
                on_compaction_start or self._write_auto_compacting_status
            ),
            on_compaction_end=on_compaction_end,
            on_compaction_record=self._record_compaction_checkpoint,
        )

    def _configure_subagents(self, *, provider: Provider | None = None) -> None:
        self.runtime.subagents.configure(
            provider=provider or self.provider,
            tools_by_name=TOOLS_BY_NAME,
            system=self.system,
            model=self.current_model,
            max_tokens=self.max_tokens,
            permission_gate=self.permission_gate,
            context_window=_context_window_for_model(self.current_model),
            thinking=self.thinking,
            effort=cast(Any, self.effort),
        )

    def _messages_for_request(
        self,
        *,
        on_compaction_start: Callable[[], None] | None = None,
        on_compaction_end: Callable[[], None] | None = None,
    ) -> list[Message]:
        preparer = self._request_preparer(
            on_compaction_start=on_compaction_start,
            on_compaction_end=on_compaction_end,
        )
        prepared = preparer.prepare(self.messages)
        self._compaction_state = preparer.state
        return prepared.request.messages

    def _write_auto_compacting_status(self) -> None:
        self._write_panel("status", "Auto-compacting...", STATUS_STYLE)

    def _drive_stream_with_recovery(self, preparer: RequestPreparer) -> CompletionResponse:
        return asyncio.run(self._adrive_stream_with_recovery(preparer))

    async def _adrive_stream_with_recovery(
        self,
        preparer: RequestPreparer,
    ) -> CompletionResponse:
        final_response: CompletionResponse | None = None
        seen_tools: set[str] = set()
        in_thinking = False
        in_text = False
        wrote_content = False
        text_buffer: list[str] = []

        def flush_text_buffer() -> None:
            nonlocal in_text
            if not text_buffer:
                return
            self._write_block("".join(text_buffer), ASSISTANT_STYLE)
            text_buffer.clear()
            in_text = False

        try:
            async for event in astream_with_recovery(preparer, self.messages):
                if isinstance(event, TextDelta):
                    if in_thinking:
                        self._write(f"{RESET}\n" if self._styles_enabled() else "\n")
                        in_thinking = False
                    text_buffer.append(event.text)
                    in_text = True
                    wrote_content = True
                elif isinstance(event, ThinkingDelta):
                    if not in_thinking:
                        if in_text:
                            flush_text_buffer()
                        if self._styles_enabled():
                            self._write(f"{THINKING_STYLE} thinking {RESET}\n")
                        else:
                            self._write("thinking\n")
                        in_thinking = True
                        wrote_content = True
                    self._write_styled(event.thinking, THINKING_STYLE)
                elif isinstance(event, ToolUseDelta):
                    if event.id not in seen_tools and event.name is not None:
                        seen_tools.add(event.id)
                        if in_thinking or in_text:
                            flush_text_buffer()
                            self._write(f"{RESET}\n" if self._styles_enabled() else "\n")
                            in_thinking = False
                elif isinstance(event, StreamComplete):
                    final_response = event.response
        finally:
            self._compaction_state = preparer.state

        flush_text_buffer()
        if wrote_content:
            self._write(f"{RESET}\n" if self._styles_enabled() else "\n")
        if final_response is None:
            raise RuntimeError("provider.stream() ended without StreamComplete")
        return final_response

    def _dispatch_tools(self, response: CompletionResponse) -> list[ContentBlock]:
        return asyncio.run(self._adispatch_tools(response))

    async def _adispatch_tools(self, response: CompletionResponse) -> list[ContentBlock]:
        results: list[ContentBlock] = []
        for block in response.content:
            if not isinstance(block, ToolUseBlock):
                continue
            blocks = await dispatch_tool_blocks_async(
                block,
                TOOLS_BY_NAME,
                self.permission_gate,
            )
            self._write_tool_result(block, _first_tool_result(block, blocks))
            results.extend(blocks)
        return results

    def _ask_tool_permission(self, block: ToolUseBlock) -> PermissionAnswer:
        summary = tool_permission_summary(block)
        prompt = f"Allow {summary}? y allow, n deny, a allow all"
        self._write_panel("permission", prompt, STATUS_STYLE)
        while True:
            answer = self._read_permission_answer()
            if answer in {"y", "Y"}:
                return PermissionAnswer.ALLOW
            if answer in {"a", "A"}:
                return PermissionAnswer.ALLOW_ALL
            if answer in {"n", "N", "\x1b", ""}:
                return PermissionAnswer.DENY
            self._write_panel("permission", "Press y, n, or a.", STATUS_STYLE)

    def _read_permission_answer(self) -> str:
        if self._force_plain:
            return self.input_func("").strip()[:1]
        try:
            return os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
        except (AttributeError, OSError):
            return ""

    def _write_tool_result(
        self,
        block: ToolUseBlock,
        result: ToolResultBlock,
    ) -> None:
        self._last_transcript_was_separator = False
        if _suppress_successful_tool_result(block, result):
            return
        if block.name in {"write", "edit"} and not result.is_error:
            self._write_edit_result(block, result)
            return
        if block.name == "spawn_agent" and not result.is_error:
            self._write_spawn_agent_result(block, result)
            return
        title = _tool_action_title(block, is_error=result.is_error)
        preview_content = result.content
        if block.name == "bash":
            preview_content = "\n".join(
                line
                for line in result.content.splitlines()
                if not line.startswith("[elapsed ")
            )
        preview = _compact_lines(preview_content, max_lines=4, max_width=110)
        if not preview:
            preview = ["[no output]"]
        if not self._styles_enabled():
            self._write_line(f"{TOOL_MARKER} {title}")
            for line in preview:
                self._write_line(f"  {line}")
            return

        marker_style = ERROR_TEXT_STYLE if result.is_error else TOOL_MARKER_STYLE
        title_style = ERROR_TEXT_STYLE if result.is_error else TOOL_TITLE_STYLE
        self._write(f"{marker_style}{TOOL_MARKER}{RESET} {title_style}{title}{RESET}\n")
        for line in preview:
            self._write(f"{TOOL_PREVIEW_STYLE}  {line}{RESET}\n")

    def _write_spawn_agent_result(
        self,
        block: ToolUseBlock,
        result: ToolResultBlock,
    ) -> None:
        title = _spawn_agent_title(block, result)
        detail = _spawn_agent_detail(block, result)
        if not self._styles_enabled():
            self._write_line(f"{TOOL_MARKER} {title}")
            self._write_line(f"  {detail}")
            return

        self._write(f"{TOOL_MARKER_STYLE}{TOOL_MARKER}{RESET} {TOOL_TITLE_STYLE}{title}{RESET}\n")
        self._write(f"{TOOL_PREVIEW_STYLE}  {detail}{RESET}\n")

    def _write_edit_result(
        self,
        block: ToolUseBlock,
        result: ToolResultBlock,
    ) -> None:
        path = _tool_arg(block, "path", "<missing path>")
        lines = result.content.splitlines()
        diff_lines = lines[1:] if lines else []
        added, deleted = _diff_counts(diff_lines)
        title = _tool_edit_title(block, path=path, added=added, deleted=deleted)
        max_changes = (
            None
            if (
                block.name == "write"
                and added > 0
                and deleted == 0
                and _diff_starts_from_empty_old_file(diff_lines)
            )
            else 12
        )
        rows = _diff_preview_lines(diff_lines, max_changes=max_changes)
        if not rows:
            rows = [("meta", lines[0] if lines else "[no changes]")]

        if not self._styles_enabled():
            self._write_line(f"{TOOL_MARKER} {title}")
            for _, line in rows:
                self._write_line(line)
            return

        self._write(f"{TOOL_MARKER_STYLE}{TOOL_MARKER}{RESET} {TOOL_TITLE_STYLE}{title}{RESET}\n")
        width = self._terminal_width()
        for kind, line in rows:
            if kind == "add":
                style = DIFF_ADD_STYLE
            elif kind == "delete":
                style = DIFF_DELETE_STYLE
            else:
                style = DIFF_META_STYLE
            clipped = line if len(line) <= width else line[: width - 3] + "..."
            if kind in {"add", "delete"}:
                self._write(f"{style}{clipped.ljust(width)}{RESET}\n")
            else:
                self._write(f"{style}{clipped}{RESET}\n")

    def _write_separator(self) -> None:
        if self._last_transcript_was_separator:
            return
        if not self._styles_enabled():
            self._write_line("---")
            self._last_transcript_was_separator = True
            return
        self._write(f"\r{SEPARATOR_STYLE}{'─' * self._terminal_width()}{RESET}\n")
        self._last_transcript_was_separator = True

    def _handle_slash(self, text: str) -> bool:
        cmd, _, rest = text.partition(" ")
        rest = rest.strip()
        if cmd in ("/exit", "/quit"):
            return True
        if cmd == "/clear":
            self.messages = []
            self._compaction_state = None
            self._last_context_tokens = None
            self._total_input_tokens = 0
            self._total_cached_tokens = 0
            self._total_output_tokens = 0
            self._clear_compaction_records()
            self._write_panel("system", "Conversation cleared.", STATUS_STYLE)
            self._persist_session()
            return False
        if cmd == "/help":
            self._print_help()
            return False
        if cmd == "/login":
            self._handle_login(rest)
            return False
        if cmd == "/branch":
            self._handle_branch()
            return False
        if cmd == "/resume":
            self._handle_resume(rest)
            return False
        if cmd in ("/session", "/status"):
            self._write_session_status()
            return False
        if cmd == "/statusline":
            if rest == "off":
                self._statusline_enabled = False
                self._write_panel("system", "Status snapshots disabled.", STATUS_STYLE)
            else:
                self._statusline_enabled = True
                self._write_status_snapshot(force=True)
            return False
        if cmd == "/model":
            self._handle_model(rest)
            return False
        self._write_panel("error", f"Unknown command: {cmd}", ERROR_STYLE)
        return False

    def _handle_branch(self) -> None:
        if self._session_record is None:
            self._session_record = new_session(
                provider=self.current_provider_name,
                model=self.current_model,
                system=self.system,
                max_tokens=self.max_tokens,
                thinking=self.thinking,
                effort=cast(Any, self.effort),
            )
            self._session_path = default_session_path(self._session_record.metadata.id)
        self._persist_session()

        assert self._session_record is not None
        parent_record = self._session_record
        parent_session_id = parent_record.metadata.id
        branch_record = new_session(
            provider=self.current_provider_name,
            model=self.current_model,
            system=self.system,
            max_tokens=self.max_tokens,
            thinking=self.thinking,
            effort=cast(Any, self.effort),
            title=parent_record.metadata.title,
            cwd=parent_record.metadata.cwd,
            parent_session_id=parent_session_id,
        )
        branch_record = replace(
            branch_record,
            messages=list(self.messages),
            compactions=list(parent_record.compactions),
        )
        self._session_record = branch_record
        self._session_path = default_session_path(branch_record.metadata.id)
        self._compaction_state = _runtime_compaction_from_session(branch_record)
        self._resume_history_pending = False
        self._persist_session()

        branch_session_id = branch_record.metadata.id
        self._write_panel(
            "branch",
            (
                "Branched conversation. You are now in the new branch "
                f"(session {branch_session_id}).\n\n"
                f"Use /resume {parent_session_id} to return to the original, or run:\n"
                f"  willow -r {parent_session_id}\n"
                "in a new terminal."
            ),
            STATUS_STYLE,
        )

    def _handle_resume(self, rest: str) -> None:
        selector = rest.strip()
        if not selector:
            self._write_panel("error", "Usage: /resume SESSION", ERROR_STYLE)
            return
        try:
            record, path = _load_resume_arg(selector)
            self._switch_to_session(record, path)
        except Exception as exc:  # noqa: BLE001
            self._write_panel("error", f"Could not resume session: {exc}", ERROR_STYLE)
            return
        self._write_panel(
            "resumed",
            f"Resumed session {record.metadata.id}.",
            STATUS_STYLE,
        )

    def _switch_to_session(self, record: SessionRecord, path: Path) -> None:
        if record.settings.provider != self.current_provider_name:
            from willow.cli import _build_provider

            self.provider = _build_provider(record.settings.provider)
        self.current_provider_name = record.settings.provider
        self.current_model = record.settings.model
        self.system = record.settings.system
        self.max_tokens = record.settings.max_tokens
        self.thinking = record.settings.thinking
        self.effort = cast(str | None, record.settings.effort)
        self.messages = list(record.messages)
        self._session_record = record
        self._session_path = path
        self._compaction_state = _runtime_compaction_from_session(record)
        self._resume_history_pending = False
        self._last_context_tokens = next(
            (
                message.input_tokens
                for message in reversed(record.messages)
                if message.role == "assistant" and message.input_tokens > 0
            ),
            None,
        )
        self._total_input_tokens = sum(message.input_tokens for message in record.messages)
        self._total_cached_tokens = sum(message.cached_tokens for message in record.messages)
        self._total_output_tokens = sum(message.output_tokens for message in record.messages)

    def _handle_login(self, rest: str) -> None:
        provider = rest.strip() or "openai-codex"
        if provider != "openai-codex":
            self._write_panel(
                "error",
                "Only /login openai-codex is supported.",
                ERROR_STYLE,
            )
            return

        try:
            credential = login_openai_codex(
                on_auth=lambda url: self._write_panel(
                    "login",
                    f"Open this URL to authenticate OpenAI Codex:\n{url}",
                    STATUS_STYLE,
                ),
                prompt=self.input_func,
                originator="willow",
            )
        except Exception as exc:  # noqa: BLE001
            self._write_panel("error", f"OpenAI Codex login failed: {exc}", ERROR_STYLE)
            return

        expires = (
            time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(credential.expires_at))
            if credential.expires_at is not None
            else "unknown"
        )
        self._write_panel(
            "login",
            f"OpenAI Codex OAuth saved to {credential.source}.\nExpires: {expires}",
            STATUS_STYLE,
        )

    def _expand_skill_text(self, text: str) -> str | None:
        command = text.strip().split(maxsplit=1)[0]
        if command in BUILTIN_SLASH_COMMANDS:
            return None
        return expand_skill_invocation(text, Path.cwd())

    def _handle_model(self, rest: str) -> None:
        choices = available_model_choices()
        if not rest:
            self._write(render_model_choices(choices, current_model=self.current_model))
            self._write_line("")
            return

        choice = find_model_choice(rest, choices)
        if choice is None:
            self.current_model = rest
            self._write_panel("model", f"Model set to {self.current_model!r}.", STATUS_STYLE)
            self._persist_session()
            return
        self._apply_model_choice(choice)

    def _apply_model_choice(self, choice: ModelChoice) -> None:
        self.current_model = choice.model
        if choice.provider != self.current_provider_name:
            from willow.cli import _build_provider

            self.provider = _build_provider(choice.provider)
            self.current_provider_name = choice.provider
        self._write_panel(
            "model",
            f"Model set to {self.current_model!r} "
            f"using provider {self.current_provider_name!r}.",
            STATUS_STYLE,
        )
        self._persist_session()

    def _print_help(self) -> None:
        self._write_line("Commands:")
        self._write_line("  /branch           Copy conversation into a new session branch.")
        self._write_line("  /exit, /quit       Exit Willow.")
        self._write_line("  /clear             Reset conversation history.")
        self._write_line("  /help              Show this message.")
        self._write_line("  /login [openai-codex] Authenticate OpenAI Codex.")
        self._write_line("  /model [name|#]    List or switch models.")
        self._write_line("  /resume SESSION    Switch to a saved session.")
        self._write_line("  /session, /status  Show persistence and session status.")
        self._write_line("  /statusline [on|off] Toggle status snapshots.")
        self._write_line("")
        self._write_line("Settings:")
        self._write_line(f"  provider:      {self.current_provider_name}")
        self._write_line(f"  model:         {self.current_model}")
        self._write_line(f"  max_tokens:    {self.max_tokens}")
        self._write_line(f"  thinking:      {'on' if self.thinking else 'off'}")
        if self.effort is not None:
            self._write_line(f"  effort:        {self.effort}")
        self._write_line(f"  mode:          {self.permission_mode.value}")
        self._write_line(f"  message count: {len(self.messages)}")
        self._write_line(f"  persistence:   {self._session_persistence_text()}")
        self._write_line(f"  session:       {self._session_path_text()}")

    def _write_session_status(self) -> None:
        lines = [
            f"persistence: {self._session_persistence_text()}",
            f"session: {self._session_path_text()}",
            f"mode: {self.permission_mode.value}",
            f"thinking: {'on' if self.thinking else 'off'}",
            f"effort: {self.effort or '(provider default)'}",
            f"messages: {len(self.messages)}",
            f"statusline: {'on' if self._statusline_enabled else 'off'}",
            self._status_text(),
        ]
        self._write_panel("status", "\n".join(lines), STATUS_STYLE)

    def _session_persistence_text(self) -> str:
        return "enabled" if self._session_record is not None else "disabled"

    def _session_path_text(self) -> str:
        return str(self._session_path) if self._session_path is not None else "(not saved)"

    def _record_usage(self, response: CompletionResponse) -> None:
        input_tokens = response.usage.get("input_tokens", 0)
        output_tokens = response.usage.get("output_tokens", 0)
        cached_tokens = _cached_tokens_from_usage(response.usage)
        self._total_input_tokens += input_tokens
        self._total_cached_tokens += cached_tokens
        self._total_output_tokens += output_tokens
        self._last_context_tokens = input_tokens if input_tokens > 0 else None

    def _status_text(self) -> str:
        return _render_statusline(
            model=self.current_model,
            context_tokens=self._last_context_tokens,
            context_window=_context_window_for_model(self.current_model),
            input_tokens=self._total_input_tokens,
            cached_tokens=self._total_cached_tokens,
            output_tokens=self._total_output_tokens,
            cwd=_display_cwd(),
        )

    def _write_status_snapshot(self, *, force: bool = False) -> None:
        if self._can_run_live():
            return
        if self._statusline_enabled or force:
            self._write_panel("status", self._status_text(), STATUS_STYLE)

    def _write_worked_duration(self, started_at: float) -> None:
        self._last_transcript_was_separator = False
        text = _worked_duration_text(started_at)
        if self._styles_enabled():
            self._write(f"{WORKED_DURATION_STYLE}{text}{RESET}\n")
        else:
            self._write_line(text)

    def _write_resume_history_if_pending(self) -> None:
        if not self._resume_history_pending:
            return
        self._resume_history_pending = False
        self._write_panel(
            "resumed",
            f"Loaded {len(self.messages)} saved message(s) from {self._session_path_text()}.",
            STATUS_STYLE,
        )
        self._write_history_transcript()

    def _continue_resumed_turn_if_needed(self) -> None:
        if _history_ends_with_tool_results(self.messages):
            self._write_panel(
                "resumed",
                "Continuing from saved tool result.",
                STATUS_STYLE,
            )
            self._run_turn()

    def _write_history_transcript(self) -> None:
        for index, message in enumerate(self.messages):
            if index:
                self._write_separator()
            self._write_history_message(message)
        if self.messages:
            self._write_separator()

    def _write_history_message(self, message: Message) -> None:
        text_parts: list[str] = []
        tool_results: list[ToolResultBlock] = []
        tool_uses: list[ToolUseBlock] = []
        images: list[ImageBlock] = []
        thinking_parts: list[str] = []
        redacted_thinking = 0
        for block in message.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
            elif isinstance(block, ImageBlock):
                images.append(block)
            elif isinstance(block, ThinkingBlock):
                thinking_parts.append(block.thinking)
            elif isinstance(block, ToolUseBlock):
                tool_uses.append(block)
            elif isinstance(block, ToolResultBlock):
                tool_results.append(block)
            else:
                redacted_thinking += 1

        if thinking_parts:
            self._write_panel("thinking", "\n".join(thinking_parts), THINKING_STYLE)
        elif redacted_thinking:
            self._write_panel(
                "thinking",
                f"[{redacted_thinking} redacted thinking block(s)]",
                THINKING_STYLE,
            )

        text = "\n".join(part for part in text_parts if part)
        if images:
            image_text = "\n".join(_image_summary(block) for block in images)
            text = f"{text}\n{image_text}" if text else image_text
        if text:
            self._write_block(text, USER_STYLE if message.role == "user" else ASSISTANT_STYLE)
        elif not tool_uses and not tool_results and not thinking_parts and not redacted_thinking:
            self._write_panel(message.role, "[empty message]", STATUS_STYLE)

        for block in tool_uses:
            self._write_panel("tool use", tool_permission_summary(block), TOOL_TITLE_STYLE)
        for block in tool_results:
            label = "tool error" if block.is_error else "tool result"
            preview = "\n".join(_compact_lines(block.content, max_lines=6, max_width=110))
            style = ERROR_STYLE if block.is_error else TOOL_PREVIEW_STYLE
            self._write_panel(label, preview or "[no output]", style)

    def _write_welcome_card(self) -> None:
        rows = [
            ("model:", self.current_model),
            ("directory:", _display_cwd()),
        ]
        content_width = max(
            len(WELCOME_TITLE),
            *(len(line) for line in WILLOW_LOGO_LINES),
            *(len(label) + 2 + len(value) for label, value in rows),
        )
        width = min(self._terminal_width() - 2, max(36, content_width + 4))
        inner_width = width - 2

        if not self._styles_enabled():
            self._write_line(f"┌{'─' * inner_width}┐")
            for line in WILLOW_LOGO_LINES:
                self._write_line(f"│ {line.center(inner_width - 1)}│")
            self._write_line(f"│ {WELCOME_TITLE.ljust(inner_width - 1)}│")
            self._write_line(f"│ {' '.ljust(inner_width - 1)}│")
            for label, value in rows:
                self._write_line(f"│ {label:<10} {value.ljust(inner_width - 12)}│")
            self._write_line(f"└{'─' * inner_width}┘")
            return

        self._write(f"{WELCOME_BORDER_STYLE}┌{'─' * inner_width}┐{RESET}\n")
        for line in WILLOW_LOGO_LINES:
            self._write(
                f"{WELCOME_BORDER_STYLE}│{RESET} "
                f"{WELCOME_LOGO_STYLE}{line.center(inner_width - 1)}{RESET}"
                f"{WELCOME_BORDER_STYLE}│{RESET}\n"
            )
        title = WELCOME_TITLE
        title_padding = inner_width - 1 - len(title)
        self._write(
            f"{WELCOME_BORDER_STYLE}│{RESET} "
            f"{WELCOME_TITLE_STYLE}{title}{RESET}"
            f"{' ' * max(0, title_padding)}"
            f"{WELCOME_BORDER_STYLE}│{RESET}\n"
        )
        self._write(
            f"{WELCOME_BORDER_STYLE}│{RESET}{' ' * inner_width}"
            f"{WELCOME_BORDER_STYLE}│{RESET}\n"
        )
        for label, value in rows:
            body_width = inner_width - 1
            label_text = f"{label:<10}"
            value_width = max(0, body_width - len(label_text) - 1)
            if len(value) <= value_width:
                visible_value = value
            elif value_width <= 3:
                visible_value = value[-value_width:] if value_width else ""
            else:
                visible_value = "..." + value[-(value_width - 3) :]
            self._write(
                f"{WELCOME_BORDER_STYLE}│{RESET} "
                f"{WELCOME_LABEL_STYLE}{label_text}{RESET} "
                f"{WELCOME_VALUE_STYLE}{visible_value.ljust(value_width)}{RESET}"
                f"{WELCOME_BORDER_STYLE}│{RESET}\n"
            )
        self._write(f"{WELCOME_BORDER_STYLE}└{'─' * inner_width}┘{RESET}\n")

    def _prompt(self) -> str:
        if not self._statusline_enabled:
            return "> "
        return f"willow [{self._status_text()}] > "

    def _persist_session(self) -> None:
        if self._session_record is None:
            return
        self._session_record = replace(
            self._session_record,
            settings=SessionSettings(
                provider=self.current_provider_name,
                model=self.current_model,
                system=self.system,
                max_tokens=self.max_tokens,
                thinking=self.thinking,
                effort=cast(Any, self.effort),
            ),
            messages=list(self.messages),
            compactions=list(self._session_record.compactions),
        )
        self._session_path = save_session(self._session_record, self._session_path)

    def _record_compaction_checkpoint(
        self,
        state: RuntimeCompaction,
        reason: str,
        tokens_before: int,
        tokens_after: int,
    ) -> None:
        if self._session_record is None:
            return
        compaction = SessionCompaction(
            summary=state.summary,
            first_kept_message_index=state.first_kept_index,
            summarized_until_message_index=state.summarized_until,
            created_after_message_index=len(self.messages),
            reason=reason,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )
        self._session_record = replace(
            self._session_record,
            compactions=[*self._session_record.compactions, compaction],
        )
        self._persist_session()

    def _clear_compaction_records(self) -> None:
        if self._session_record is None:
            return
        self._session_record = replace(self._session_record, compactions=[])

    def _write(self, text: str) -> None:
        self.out.write(text)
        self.out.flush()

    def _write_line(self, text: str) -> None:
        self._write(f"{text}\n")

    def _styles_enabled(self) -> bool:
        return (
            not self._force_plain
            and hasattr(self.out, "isatty")
            and self.out.isatty()
        )

    def _terminal_width(self) -> int:
        return max(1, shutil.get_terminal_size((88, 24)).columns)

    def _write_styled(self, text: str, style: str) -> None:
        self._last_transcript_was_separator = False
        if self._styles_enabled():
            self._write(f"{style}{text}{RESET}")
        else:
            self._write(text)

    def _write_panel(self, label: str, text: str, style: str) -> None:
        self._last_transcript_was_separator = False
        if not self._styles_enabled():
            self._write_line(f"[{label}] {text}")
            return

        width = self._terminal_width()
        header = f" {label} "
        self._write(f"{_styled_terminal_line(header, style, width)}\n")
        for line in (text.splitlines() or [""]):
            body = f" {line}"
            for wrapped in _wrap_terminal_line(body, width):
                content = wrapped
                if label == "status":
                    content = _style_statusline_text(content)
                    self._write(f"{_filled_terminal_line(content, style, width)}\n")
                else:
                    self._write(f"{_styled_terminal_line(content, style, width)}\n")

    def _write_block(self, text: str, style: str) -> None:
        self._last_transcript_was_separator = False
        if not self._styles_enabled():
            if style == ASSISTANT_STYLE:
                for row in _render_markdown_text(text, width=self._terminal_width()):
                    self._write_line(row.text)
            else:
                self._write_line(text)
            return

        width = self._terminal_width()
        should_pad = style in {USER_STYLE, ASSISTANT_STYLE}
        self._write("\r")
        if should_pad:
            blank = f"{_styled_terminal_line('', style, width)}\n"
            self._write(blank * MESSAGE_BLOCK_VERTICAL_PADDING)
        rows = (
            _render_markdown_text(text, width=width)
            if style == ASSISTANT_STYLE
            else [_RenderedTextLine(line, style) for line in (text.splitlines() or [""])]
        )
        for row in rows:
            body = f" {row.text}"
            for wrapped in _wrap_terminal_line(body, width):
                self._write(f"{_styled_terminal_line(wrapped, row.style, width)}\n")
        if should_pad:
            blank = f"{_styled_terminal_line('', style, width)}\n"
            self._write(blank * MESSAGE_BLOCK_VERTICAL_PADDING)


class _LiveTerminal:
    """Raw-mode prompt that stays active while provider streaming runs."""

    def __init__(self, app: WillowApp) -> None:
        self.app = app
        try:
            self.fd = sys.stdin.fileno()
        except (AttributeError, OSError):
            self.fd = -1
        self.events: queue.Queue[tuple[int, str, Any]] = queue.Queue()
        self._unsubscribe_monitor_events = self.app.runtime.events.subscribe(
            lambda event: self.events.put((-1, "monitor_event", event))
        )
        self.buffer = ""
        self.cursor = 0
        self.running = True
        self.streaming = False
        self.worker: threading.Thread | None = None
        self.pending_user_inputs: list[str] = []
        self.pending_monitor_inputs: list[str] = []
        self.interrupted_user_inputs: list[str] = []
        self.compacting = False
        self._last_compaction_frame_at = 0.0
        self._last_running_frame_at = 0.0
        self.working_started_at: float | None = None
        self.prompt_lines = 0
        self.prompt_width = 0
        self.prompt_cursor_line_index = 0
        self.prompt_cursor_column = 0
        self.prompt_cursor_offset_from_bottom = 0
        self.resize_pending = False
        self.turn_started_at: float | None = None
        self.pending_paste_chunks: list[str] | None = None
        self.pending_escape_sequence: str | None = None
        self.pending_escape_started_at = 0.0
        self.pasted_ranges: list[tuple[int, int]] = []
        self.in_text = False
        self.in_thinking = False
        self.seen_tools: set[str] = set()
        self.stream_text: list[str] = []
        self.stream_thinking: list[str] = []
        self.stream_tool_names: list[str] = []
        self.active_tool_status: str | None = None
        self.active_turn_id = 0
        self.model_picker_choices: list[ModelChoice] | None = None
        self.model_picker_selected = 0
        self.login_picker_selected = 0
        self.input_hint_rows: list[str] | None = None
        self.input_hint_source = ""
        self.input_hint_selected = 0
        self.input_history = _input_history_from_messages(self.app.messages)
        self.input_history_index: int | None = None
        self.input_history_draft = ""

    def _reset_prompt_state(self) -> None:
        self.prompt_lines = 0
        self.prompt_width = 0
        self.prompt_cursor_line_index = 0
        self.prompt_cursor_column = 0
        self.prompt_cursor_offset_from_bottom = 0

    def run(self) -> int:
        self.app._write_welcome_card()
        self.app._write_resume_history_if_pending()
        with self._raw_terminal():
            if _history_ends_with_tool_results(self.app.messages):
                self.app._write_panel(
                    "resumed",
                    "Continuing from saved tool result.",
                    STATUS_STYLE,
                )
                self._start_worker()
            else:
                positional_prompt = self.app._positional_prompt_text()
                if positional_prompt and self.app._submit_user_text(
                    positional_prompt,
                    render=True,
                ):
                    self._start_worker()
            self._draw_prompt()
            try:
                while self.running:
                    self._read_available_input()
                    self._drain_events()
                    should_animate = (
                        self.compacting
                        and time.monotonic() - self._last_compaction_frame_at >= 0.12
                    )
                    if should_animate:
                        self._last_compaction_frame_at = time.monotonic()
                        self._draw_prompt()
                    should_animate_running = self._running_status_active() and (
                        time.monotonic() - self._last_running_frame_at >= 0.04
                    )
                    if should_animate_running:
                        self._last_running_frame_at = time.monotonic()
                        self._redraw_running_status_line()
                    if self._prompt_needs_resize_repaint():
                        self._draw_prompt(force_reflow_clear=self.resize_pending)
                        self.resize_pending = False
                if (
                    not self.streaming
                    and self.worker is None
                    and (
                        self.pending_user_inputs
                        or self.pending_monitor_inputs
                        or self.interrupted_user_inputs
                    )
                ):
                    self._start_queued_turn()
            except KeyboardInterrupt:
                self.running = False
        self._clear_prompt()
        self._unsubscribe_monitor_events()
        self.app._write_line("Goodbye.")
        return 0

    @contextmanager
    def _raw_terminal(self):
        old = termios.tcgetattr(self.fd)
        old_winch_handler = signal.getsignal(signal.SIGWINCH)

        def mark_resize_pending(_signum: int, _frame: object) -> None:
            self.resize_pending = True

        try:
            signal.signal(signal.SIGWINCH, mark_resize_pending)
            tty.setcbreak(self.fd)
            self.app._write(f"\x1b[?2004h{KEYBOARD_ENHANCEMENT_ENABLE}\x1b[6 q")
            yield
        finally:
            signal.signal(signal.SIGWINCH, old_winch_handler)
            self.app._write(
                f"\x1b[?2004l{KEYBOARD_ENHANCEMENT_DISABLE}\x1b[0 q\x1b[0m\x1b[?25h"
            )
            termios.tcsetattr(self.fd, termios.TCSADRAIN, old)

    def _prompt_needs_resize_repaint(self) -> bool:
        if self.prompt_lines == 0:
            return False
        return self.resize_pending or _terminal_line_width(
            self.app._terminal_width()
        ) != self.prompt_width

    def _read_available_input(self) -> None:
        readable, _, _ = select.select([self.fd], [], [], 0.03)
        if not readable:
            self._flush_pending_escape_if_expired()
            return
        data = os.read(self.fd, 4096).decode(errors="ignore")
        if self.pending_escape_sequence is not None:
            data = self.pending_escape_sequence + data
            self.pending_escape_sequence = None
        index = 0
        while index < len(data):
            if self.pending_paste_chunks is not None:
                end = data.find("\x1b[201~", index)
                if end == -1:
                    self.pending_paste_chunks.append(data[index:])
                    return
                self.pending_paste_chunks.append(data[index:end])
                self._insert_pasted_text("".join(self.pending_paste_chunks))
                self.pending_paste_chunks = None
                index = end + 6
                self._draw_prompt()
                continue
            if data.startswith("\x1b[200~", index):
                end = data.find("\x1b[201~", index + 6)
                if end == -1:
                    self.pending_paste_chunks = [data[index + 6 :]]
                    index = len(data)
                else:
                    pasted = data[index + 6 : end]
                    index = end + 6
                    self._insert_pasted_text(pasted)
                    self._draw_prompt()
                continue
            ch = data[index]
            index += 1
            if ch in ("\r", "\n"):
                self._submit_buffer()
            elif ch == "\t":
                if not self._complete_selected_hint():
                    self._insert_text(ch)
                self._draw_prompt()
            elif ch == "\x03" or (ch == "\x04" and not self.buffer):
                self.running = False
            elif ch == "\x15":
                self.buffer = ""
                self.cursor = 0
                self.pasted_ranges = []
                self._draw_prompt()
            elif ch in ("\x7f", "\b"):
                if self.cursor > 0:
                    self.buffer = self.buffer[: self.cursor - 1] + self.buffer[self.cursor :]
                    self.pasted_ranges = []
                    self.cursor -= 1
                self._draw_prompt()
            elif ch == "\x1b":
                candidate = data[index - 1 :]
                if _is_incomplete_escape_sequence(candidate):
                    self.pending_escape_sequence = candidate
                    self.pending_escape_started_at = time.monotonic()
                    return
                shift_enter = next(
                    (
                        sequence
                        for sequence in SHIFT_ENTER_SEQUENCES
                        if data.startswith(sequence[1:], index)
                    ),
                    None,
                )
                if shift_enter is not None:
                    self._insert_text("\n")
                    index += len(shift_enter) - 1
                    self._draw_prompt()
                elif data.startswith("b", index) or data.startswith("B", index):
                    self._move_cursor_word_left()
                    index += 1
                    self._draw_prompt()
                elif data.startswith("f", index) or data.startswith("F", index):
                    self._move_cursor_word_right()
                    index += 1
                    self._draw_prompt()
                elif data.startswith("[1;3D", index) or data.startswith("[1;5D", index):
                    self._move_cursor_word_left()
                    index += 5
                    self._draw_prompt()
                elif data.startswith("[1;3C", index) or data.startswith("[1;5C", index):
                    self._move_cursor_word_right()
                    index += 5
                    self._draw_prompt()
                elif data.startswith("[A", index) or data.startswith("OA", index):
                    self._move_picker_or_history(-1)
                    index += 2
                    self._draw_prompt()
                elif data.startswith("[B", index) or data.startswith("OB", index):
                    self._move_picker_or_history(1)
                    index += 2
                    self._draw_prompt()
                elif data.startswith("[D", index):
                    self.cursor = max(0, self.cursor - 1)
                    index += 2
                    self._draw_prompt()
                elif data.startswith("[C", index):
                    self.cursor = min(len(self.buffer), self.cursor + 1)
                    index += 2
                    self._draw_prompt()
                elif data.startswith("[H", index) or data.startswith("[1~", index):
                    self.cursor = 0
                    index += 2 if data.startswith("[H", index) else 3
                    self._draw_prompt()
                elif data.startswith("[F", index) or data.startswith("[4~", index):
                    self.cursor = len(self.buffer)
                    index += 2 if data.startswith("[F", index) else 3
                    self._draw_prompt()
                elif self.streaming and (index >= len(data) or data[index] != "["):
                    self._interrupt_with_buffer_if_possible()
                else:
                    # Ignore unsupported escape sequences; this keeps arrow keys harmless.
                    while index < len(data) and data[index].isalpha() is False:
                        index += 1
                    if index < len(data):
                        index += 1
            elif ch.isprintable():
                end = index
                while end < len(data) and data[end].isprintable():
                    end += 1
                self._insert_text(ch + data[index:end])
                index = end
                self._draw_prompt()

    def _flush_pending_escape_if_expired(self) -> None:
        pending = self.pending_escape_sequence
        if pending is None:
            return
        elapsed = time.monotonic() - self.pending_escape_started_at
        if elapsed < ESCAPE_SEQUENCE_TIMEOUT_SECONDS:
            return
        self.pending_escape_sequence = None
        if pending == "\x1b" and self.streaming:
            self._interrupt_with_buffer_if_possible()

    def _insert_text(self, text: str) -> None:
        self.buffer = self.buffer[: self.cursor] + text + self.buffer[self.cursor :]
        self.pasted_ranges = _shift_pasted_ranges(
            self.pasted_ranges,
            start=self.cursor,
            inserted_length=len(text),
        )
        self.cursor += len(text)

    def _insert_pasted_text(self, text: str) -> None:
        start = self.cursor
        self._insert_text(text)
        if len(text) >= PASTE_PLACEHOLDER_MIN_CHARS or "\n" in text:
            self.pasted_ranges = _merge_pasted_ranges(
                [*self.pasted_ranges, (start, start + len(text))]
            )

    def _move_cursor_word_left(self) -> None:
        cursor = self.cursor
        while cursor > 0 and self.buffer[cursor - 1].isspace():
            cursor -= 1
        while cursor > 0 and not self.buffer[cursor - 1].isspace():
            cursor -= 1
        self.cursor = cursor

    def _move_cursor_word_right(self) -> None:
        cursor = self.cursor
        length = len(self.buffer)
        while cursor < length and self.buffer[cursor].isspace():
            cursor += 1
        while cursor < length and not self.buffer[cursor].isspace():
            cursor += 1
        self.cursor = cursor

    def _model_picker_active(self) -> bool:
        return not self.streaming and self.buffer.strip() == "/model"

    def _login_picker_active(self) -> bool:
        return not self.streaming and self.buffer.strip() == "/login"

    def _ensure_model_picker(self) -> list[ModelChoice] | None:
        if not self._model_picker_active():
            self.model_picker_choices = None
            self.model_picker_selected = 0
            return None
        if self.model_picker_choices is None:
            choices = available_model_choices()
            self.model_picker_choices = choices
            current_index = next(
                (
                    index
                    for index, choice in enumerate(choices)
                    if choice.model == self.app.current_model
                ),
                0,
            )
            self.model_picker_selected = min(current_index, max(0, len(choices) - 1))
        return self.model_picker_choices

    def _move_model_picker_selection(self, delta: int) -> None:
        choices = self._ensure_model_picker()
        if not choices:
            return
        self.model_picker_selected = max(
            0,
            min(len(choices) - 1, self.model_picker_selected + delta),
        )

    def _move_login_picker_selection(self, delta: int) -> None:
        self.login_picker_selected = max(
            0,
            min(len(LOGIN_PROVIDER_CHOICES) - 1, self.login_picker_selected + delta),
        )

    def _input_hints_active(self) -> bool:
        return (
            not self.streaming
            and not self._model_picker_active()
            and not self._login_picker_active()
        )

    def _ensure_input_hints(self) -> list[str]:
        if not self._input_hints_active():
            self.input_hint_rows = None
            self.input_hint_source = ""
            self.input_hint_selected = 0
            return []
        hints = _render_input_hints(self.buffer, Path.cwd())
        rows = hints.splitlines() if hints else []
        source = "\n".join(rows)
        if source != self.input_hint_source:
            self.input_hint_source = source
            self.input_hint_rows = rows
            self.input_hint_selected = 0
        if rows:
            self.input_hint_selected = min(self.input_hint_selected, len(rows) - 1)
        else:
            self.input_hint_selected = 0
        return rows

    def _move_input_hint_selection(self, delta: int) -> None:
        rows = self._ensure_input_hints()
        if not rows:
            return
        self.input_hint_selected = max(
            0,
            min(len(rows) - 1, self.input_hint_selected + delta),
        )

    def _move_picker_selection(self, delta: int) -> None:
        if self._model_picker_active():
            self._move_model_picker_selection(delta)
        elif self._login_picker_active():
            self._move_login_picker_selection(delta)
        else:
            self._move_input_hint_selection(delta)

    def _move_picker_or_history(self, delta: int) -> None:
        if self._model_picker_active() or self._login_picker_active():
            self._move_picker_selection(delta)
            return

        if self._ensure_input_hints():
            self._move_input_hint_selection(delta)
            return

        self._move_input_history(delta)

    def _move_input_history(self, delta: int) -> None:
        if not self.input_history:
            return
        if self.input_history_index is None:
            if delta >= 0:
                return
            self.input_history_draft = self.buffer
            next_index = len(self.input_history) - 1
        else:
            next_index = self.input_history_index + delta

        if next_index < 0:
            next_index = 0
        if next_index >= len(self.input_history):
            self.input_history_index = None
            self._replace_input_buffer(self.input_history_draft)
            return

        self.input_history_index = next_index
        self._replace_input_buffer(self.input_history[next_index])

    def _replace_input_buffer(self, text: str) -> None:
        self.buffer = text
        self.cursor = len(text)
        self.pasted_ranges = []
        self.input_hint_rows = None
        self.input_hint_source = ""
        self.input_hint_selected = 0

    def _record_input_history(self, text: str) -> None:
        if not self.input_history or self.input_history[-1] != text:
            self.input_history.append(text)
        self.input_history_index = None
        self.input_history_draft = ""

    def _complete_selected_hint(self) -> bool:
        if self._model_picker_active():
            choices = self._ensure_model_picker()
            if not choices:
                return False
            selected = choices[self.model_picker_selected]
            self.buffer = f"/model {selected.model}"
            self.cursor = len(self.buffer)
            self.pasted_ranges = []
            self.model_picker_choices = None
            self.model_picker_selected = 0
            return True

        if self._login_picker_active():
            provider, _description = LOGIN_PROVIDER_CHOICES[self.login_picker_selected]
            self.buffer = f"/login {provider}"
            self.cursor = len(self.buffer)
            self.pasted_ranges = []
            self.login_picker_selected = 0
            return True

        rows = self._ensure_input_hints()
        if not rows:
            return False
        selected = rows[self.input_hint_selected]
        self.buffer = _apply_hint_to_input(
            self.buffer,
            selected,
            append_space_when_empty=True,
        )
        self.cursor = len(self.buffer)
        self.pasted_ranges = []
        self.input_hint_rows = None
        self.input_hint_source = ""
        self.input_hint_selected = 0
        return True

    def _submit_buffer(self, *, use_selected_hint: bool = True) -> None:
        if self._model_picker_active():
            self._submit_model_picker_selection()
            return
        if self._login_picker_active():
            self._submit_login_picker_selection()
            return
        if use_selected_hint and self._submit_input_hint_selection():
            return
        text = self.buffer.strip()
        self.buffer = ""
        self.cursor = 0
        self.pasted_ranges = []
        self._clear_prompt()
        if not text:
            self._draw_prompt()
            return
        self._record_input_history(text)
        if self.streaming:
            self.pending_user_inputs.append(text)
            self._draw_prompt()
            return
        expanded_text = self.app._expand_skill_text(text)
        if (
            expanded_text is None
            and _should_route_slash_command(text)
            and not self.streaming
        ):
            should_exit = self.app._handle_slash(text)
            if should_exit:
                self.running = False
                return
            self._draw_prompt()
            return

        message_text = expanded_text or text
        try:
            message_content = self.app._user_content_blocks(message_text)
        except ValueError as exc:
            self.app._write_panel("error", str(exc), ERROR_STYLE)
            self._draw_prompt()
            return
        self.app._write_block(
            self.app._user_display_text(
                text,
                message_content,
                prefer_content_text=expanded_text is None,
            ),
            USER_STYLE,
        )
        if self.interrupted_user_inputs:
            submitted_text = _first_text_block_text(message_content, fallback=message_text)
            content = interrupted_user_text_blocks(
                self.interrupted_user_inputs,
                [submitted_text],
            )
            content.extend(message_content[1:])
            self.interrupted_user_inputs = []
        else:
            content = message_content
        self.app.messages.append(Message(role="user", content=content))
        self.app._persist_session()
        self._start_worker()
        self._draw_prompt()

    def _submit_input_hint_selection(self) -> bool:
        rows = self._ensure_input_hints()
        if not rows:
            return False
        selected = rows[self.input_hint_selected]
        selected_command = _hint_command(selected)
        if selected_command.startswith("@"):
            self.buffer = _apply_hint_to_input(
                self.buffer,
                selected,
                append_space_when_empty=True,
            )
            self.cursor = len(self.buffer)
            self.pasted_ranges = []
            self.input_hint_rows = None
            self.input_hint_source = ""
            self.input_hint_selected = 0
            self._draw_prompt()
            return True
        text = _apply_hint_to_input(self.buffer, selected)
        if text.strip() in {"/login", "/model"}:
            self.buffer = text.strip()
            self.cursor = len(self.buffer)
            self.pasted_ranges = []
            self.input_hint_rows = None
            self.input_hint_source = ""
            self.input_hint_selected = 0
            self._draw_prompt()
            return True
        self.buffer = text
        self.cursor = len(self.buffer)
        self.pasted_ranges = []
        self.input_hint_rows = None
        self.input_hint_source = ""
        self.input_hint_selected = 0
        self._submit_buffer(use_selected_hint=False)
        return True

    def _submit_model_picker_selection(self) -> None:
        choices = self._ensure_model_picker()
        selected_index = self.model_picker_selected
        self.buffer = ""
        self.cursor = 0
        self.pasted_ranges = []
        self.model_picker_choices = None
        self.model_picker_selected = 0
        self._clear_prompt()
        if not choices:
            self.app._write_panel(
                "model",
                "No models available. Add provider auth to ~/.willow/auth.json.",
                ERROR_STYLE,
            )
            self._draw_prompt()
            return
        selected = choices[selected_index]
        self.app._apply_model_choice(selected)
        self._draw_prompt()

    def _submit_login_picker_selection(self) -> None:
        provider, _description = LOGIN_PROVIDER_CHOICES[self.login_picker_selected]
        self.buffer = ""
        self.cursor = 0
        self.pasted_ranges = []
        self.login_picker_selected = 0
        self._clear_prompt()
        self.app._handle_login(provider)
        self._draw_prompt()

    def _interrupt_with_buffer_if_possible(self) -> None:
        text = self.buffer.strip()
        if text:
            self.buffer = ""
            self.cursor = 0
            self.pasted_ranges = []
            self.pending_user_inputs.append(text)
        if self.pending_user_inputs:
            self._interrupt_and_send_queued()
        else:
            self._interrupt_current_turn()

    def _interrupt_current_turn(self) -> None:
        if not self.streaming:
            self._draw_prompt()
            return
        self.active_turn_id += 1
        self._clear_stream_buffers()
        self._clear_prompt()
        interrupted = self._take_trailing_text_user_message()
        if interrupted:
            self.interrupted_user_inputs.extend(interrupted)
            self.app._persist_session()
        self.streaming = False
        self.compacting = False
        self.active_tool_status = None
        self.working_started_at = None
        self.turn_started_at = None
        self.app._write_panel(
            "status",
            "Interrupted current turn; it will be sent again with your next message.",
            STATUS_STYLE,
        )
        self._reset_provider_for_interrupt()
        self._draw_prompt()

    def _start_queued_turn(self) -> None:
        self._append_pending_user_message(render=True)
        self._start_worker()
        self._draw_prompt()

    def _append_pending_user_message(self, *, render: bool) -> None:
        interrupted = self.interrupted_user_inputs
        pending = self.pending_user_inputs
        monitor_pending = self.pending_monitor_inputs
        self.interrupted_user_inputs = []
        self.pending_user_inputs = []
        self.pending_monitor_inputs = []
        prepared_pending, attachment_blocks = self._prepare_pending_user_inputs(
            pending,
            render=render,
        )
        if interrupted:
            user_blocks = interrupted_user_text_blocks(interrupted, prepared_pending)
        else:
            user_blocks = queued_user_text_blocks(prepared_pending)
        blocks = [
            *user_blocks,
            *attachment_blocks,
            *(TextBlock(text=text) for text in monitor_pending),
        ]
        if not blocks:
            return
        self.app.messages.append(Message(role="user", content=list(blocks)))
        self.app._persist_session()

    def _interrupt_and_send_queued(self) -> None:
        if not self.streaming or not self.pending_user_inputs:
            return
        self.active_turn_id += 1
        self._clear_stream_buffers()
        self._clear_prompt()
        self.app._write_panel(
            "status",
            "Interrupted current turn; sending queued input now.",
            STATUS_STYLE,
        )
        self.turn_started_at = None
        self._append_interrupted_user_message(render=True)
        self._reset_provider_for_interrupt()
        self._start_worker()
        self._draw_prompt()

    def _append_interrupted_user_message(self, *, render: bool) -> None:
        pending = self.pending_user_inputs
        monitor_pending = self.pending_monitor_inputs
        self.pending_user_inputs = []
        self.pending_monitor_inputs = []
        interrupted = [*self.interrupted_user_inputs, *self._take_trailing_text_user_message()]
        self.interrupted_user_inputs = []
        prepared_pending, attachment_blocks = self._prepare_pending_user_inputs(
            pending,
            render=render,
        )
        user_blocks = (
            interrupted_user_text_blocks(interrupted, prepared_pending)
            if interrupted
            else queued_user_text_blocks(prepared_pending)
        )
        blocks = [
            *user_blocks,
            *attachment_blocks,
            *(TextBlock(text=text) for text in monitor_pending),
        ]
        if not blocks:
            return
        self.app.messages.append(Message(role="user", content=list(blocks)))
        self.app._persist_session()

    def _prepare_pending_user_inputs(
        self,
        pending: list[str],
        *,
        render: bool,
    ) -> tuple[list[str], list[ContentBlock]]:
        prepared_pending: list[str] = []
        attachment_blocks: list[ContentBlock] = []
        image_index = 1
        for text in pending:
            try:
                content = self.app._user_content_blocks(
                    text,
                    image_index_start=image_index,
                )
            except ValueError as exc:
                if render:
                    self.app._write_panel("error", str(exc), ERROR_STYLE)
                continue
            image_index += sum(isinstance(block, ImageBlock) for block in content)
            prepared_pending.append(_first_text_block_text(content, fallback=text))
            attachment_blocks.extend(content[1:])
            if render:
                self.app._write_block(self.app._user_display_text(text, content), USER_STYLE)
        return prepared_pending, attachment_blocks

    def _take_trailing_text_user_message(self) -> list[str]:
        if not self.app.messages or self.app.messages[-1].role != "user":
            return []
        message = self.app.messages[-1]
        if not all(isinstance(block, TextBlock) for block in message.content):
            return []
        self.app.messages.pop()
        return [cast(TextBlock, block).text for block in message.content]

    def _reset_provider_for_interrupt(self) -> None:
        from willow.cli import _build_provider

        self.app.provider = _build_provider(self.app.current_provider_name)

    def _start_worker(self) -> None:
        self.streaming = True
        if self.turn_started_at is None:
            self.turn_started_at = time.monotonic()
        self.working_started_at = time.monotonic()
        self.active_turn_id += 1
        self.in_text = False
        self.in_thinking = False
        self.seen_tools = set()
        self._clear_stream_buffers()
        turn_id = self.active_turn_id
        provider = self.app.provider
        self.app._configure_subagents(provider=provider)
        self.worker = threading.Thread(
            target=self._worker_main,
            args=(turn_id, provider),
            daemon=True,
        )
        self.worker.start()

    def _worker_main(self, turn_id: int, provider: Provider) -> None:
        asyncio.run(self._worker_main_async(turn_id, provider))

    async def _worker_main_async(self, turn_id: int, provider: Provider) -> None:
        preparer = self.app._request_preparer(
            provider=provider,
            on_compaction_start=lambda: self.events.put((turn_id, "compact_start", None)),
            on_compaction_end=lambda: self.events.put((turn_id, "compact_end", None)),
        )
        messages = list(self.app.messages)
        try:
            final: CompletionResponse | None = None
            async for event in astream_with_recovery(preparer, messages):
                self.events.put((turn_id, "stream", event))
                if isinstance(event, StreamComplete):
                    final = event.response
            if final is None:
                raise RuntimeError("provider.stream() ended without StreamComplete")
            self.events.put((turn_id, "complete", final))
        except Exception as exc:  # noqa: BLE001
            self.events.put((turn_id, "error", exc))
        finally:
            self.app._compaction_state = preparer.state

    def _drain_events(self) -> None:
        while True:
            try:
                turn_id, kind, payload = self.events.get_nowait()
            except queue.Empty:
                return
            if kind == "monitor_event":
                self._queue_monitor_event(cast(dict[str, object], payload))
                continue
            if turn_id != self.active_turn_id:
                continue
            if kind == "stream":
                self._render_stream_event(payload)
            elif kind == "complete":
                self._clear_prompt()
                self.worker = None
                self._finish_response(cast(CompletionResponse, payload))
            elif kind == "error":
                self._clear_prompt()
                self._flush_stream_buffer()
                self.app._write_turn_error(cast(BaseException, payload))
                if self.turn_started_at is not None:
                    self.app._write_worked_duration(self.turn_started_at)
                self.streaming = False
                self.compacting = False
                self.worker = None
                self.active_tool_status = None
                self.working_started_at = None
                self.turn_started_at = None
            elif kind == "compact_start":
                self.compacting = True
                self._last_compaction_frame_at = 0.0
                self._draw_prompt()
            elif kind == "compact_end":
                self.compacting = False
                self._draw_prompt()
            if self.running and kind != "stream":
                self._draw_prompt()

    def _queue_monitor_event(self, event: dict[str, object]) -> None:
        if event.get("event_type") == "subagent":
            self._write_subagent_event(event)
            if self.running:
                self._draw_prompt()
            return
        texts = monitor_event_texts([event])
        if not texts:
            return
        self.pending_monitor_inputs.extend(texts)
        if self.streaming or self.worker is not None:
            return
        self._start_queued_turn()

    def _write_subagent_event(self, event: Mapping[str, object]) -> None:
        self._clear_prompt()
        title = _subagent_event_title(event)
        detail = _subagent_event_detail(event)
        if not self.app._styles_enabled():
            self.app._write_line(f"{TOOL_MARKER} {title}")
            if detail:
                self.app._write_line(f"  {detail}")
            return
        self.app._write(
            f"{TOOL_MARKER_STYLE}{TOOL_MARKER}{RESET} "
            f"{TOOL_TITLE_STYLE}{title}{RESET}\n"
        )
        if detail:
            self.app._write(f"{TOOL_PREVIEW_STYLE}  {detail}{RESET}\n")

    def _render_stream_event(self, event: Any) -> None:
        if isinstance(event, TextDelta):
            self.stream_text.append(event.text)
        elif isinstance(event, ThinkingDelta):
            self.stream_thinking.append(event.thinking)
        elif (
            isinstance(event, ToolUseDelta)
            and event.id not in self.seen_tools
            and event.name is not None
        ):
            self.seen_tools.add(event.id)
            self.stream_tool_names.append(event.name)

    def _finish_response(self, response: CompletionResponse) -> None:
        self._flush_stream_buffer()
        self.in_text = False
        self.in_thinking = False
        self.app._record_usage(response)
        has_tool_uses = any(isinstance(block, ToolUseBlock) for block in response.content)
        if has_tool_uses:
            self.app._write_separator()
            self.app._append_assistant_response(response)
        tool_results = (
            self._dispatch_tools_with_prompt(response)
            if response.stop_reason == "tool_use"
            else []
        )
        if not self.streaming:
            if has_tool_uses and tool_results:
                self.app._append_followup_user(list(tool_results))
            if self.turn_started_at is not None:
                self.app._write_worked_duration(self.turn_started_at)
                self.turn_started_at = None
            return
        interrupted_texts = self.interrupted_user_inputs
        pending_texts = self.pending_user_inputs
        pending_monitor_texts = self.pending_monitor_inputs
        prepared_pending, attachment_blocks = self._prepare_pending_user_inputs(
            pending_texts,
            render=True,
        )
        if interrupted_texts:
            user_blocks = interrupted_user_text_blocks(interrupted_texts, prepared_pending)
        else:
            user_blocks = queued_user_text_blocks(prepared_pending)
        pending = [
            *user_blocks,
            *attachment_blocks,
            *(TextBlock(text=text) for text in pending_monitor_texts),
        ]
        self.interrupted_user_inputs = []
        self.pending_user_inputs = []
        self.pending_monitor_inputs = []
        if has_tool_uses:
            followup_content = [*tool_results, *pending]
            continue_running = self.app._append_followup_user(followup_content)
        else:
            step = build_turn_step(
                response,
                pending_user_blocks=pending,
            )
            append_turn_step(self.app.messages, step)
            self.app._persist_session()
            continue_running = step.continue_running
        if continue_running:
            if tool_results:
                self.app._write_separator()
            self._start_worker()
        else:
            self.streaming = False
            self.working_started_at = None
            self.app._write_status_snapshot()
            if self.turn_started_at is not None:
                self.app._write_worked_duration(self.turn_started_at)
                self.turn_started_at = None

    def _dispatch_tools_with_prompt(
        self,
        response: CompletionResponse,
    ) -> list[ContentBlock]:
        results: list[ContentBlock] = []
        for block in response.content:
            if not isinstance(block, ToolUseBlock):
                continue
            blocks = self._dispatch_tool_with_animated_prompt(block)
            result = _first_tool_result(block, blocks)
            self.app._write_tool_result(block, result)
            results.extend(blocks)
        return results

    def _dispatch_tool_with_animated_prompt(self, block: ToolUseBlock) -> list[ContentBlock]:
        tool = TOOLS_BY_NAME.get(block.name)
        if tool is None:
            return [
                ToolResultBlock(
                    tool_use_id=block.id,
                    content=f"Unknown tool: {block.name!r}",
                    is_error=True,
                )
            ]
        permission = self.app.permission_gate.check(block)
        if not permission.allowed:
            return [
                ToolResultBlock(
                    tool_use_id=block.id,
                    content=permission.denial or "Tool execution denied.",
                    is_error=True,
                )
            ]

        result_box: list[list[ContentBlock]] = []

        def run_tool() -> None:
            result_box.append(self._run_tool_without_permission(block, tool))

        self.active_tool_status = _tool_running_title(block)
        self._last_running_frame_at = 0.0
        self._draw_prompt()
        worker = threading.Thread(
            target=run_tool,
            name=f"willow-tool-{block.name}",
            daemon=True,
        )
        worker.start()
        try:
            while worker.is_alive():
                if self.fd >= 0:
                    self._read_available_input()
                if not self.streaming:
                    return [
                        ToolResultBlock(
                            tool_use_id=block.id,
                            content="Interrupted by user.",
                            is_error=True,
                        )
                    ]
                self._last_running_frame_at = time.monotonic()
                self._redraw_running_status_line()
                worker.join(timeout=0.04)
        finally:
            if self.streaming:
                worker.join()
            self._clear_prompt()
            self.active_tool_status = None
        if result_box:
            return result_box[0]
        return [
            ToolResultBlock(
                tool_use_id=block.id,
                content=f"Tool ended without a result: {block.name!r}",
                is_error=True,
            )
        ]

    @staticmethod
    def _run_tool_without_permission(block: ToolUseBlock, tool: Tool) -> list[ContentBlock]:
        return asyncio.run(dispatch_tool_blocks_async(block, {tool.name: tool}))

    def _flush_stream_buffer(self) -> None:
        thinking = "".join(self.stream_thinking)
        text = "".join(self.stream_text)
        if thinking:
            self.app._write_block(thinking, THINKING_STYLE)
        if text:
            self.app._write_block(text, ASSISTANT_STYLE)
        self._clear_stream_buffers()

    def _clear_stream_buffers(self) -> None:
        self.stream_thinking = []
        self.stream_text = []
        self.stream_tool_names = []

    def _draw_prompt(self, *, force_reflow_clear: bool = False) -> None:
        frame = self._build_prompt_frame()
        self._write_prompt_frame(frame, force_reflow_clear=force_reflow_clear)

    def _build_prompt_frame(self) -> _PromptFrame:
        rows: list[str] = []
        width = self.app._terminal_width()
        line_width = _terminal_line_width(width)
        status = self.app._status_text()
        rendered_input = _image_placeholder_prompt_render(
            _render_prompt_input(self.buffer, self.pasted_ranges, self.cursor)
        )
        preview = rendered_input.text
        cursor = min(len(preview), rendered_input.cursor)
        active_subagents = self._active_subagent_snapshots()
        if self.compacting:
            frame = COMPACTION_FRAMES[int(time.monotonic() * 8) % len(COMPACTION_FRAMES)]
            line = f" {frame} Auto-compacting..."
            rows.append(_styled_terminal_line(line, COMPACTION_STYLE, width))
        elif self.streaming and not self._suppress_running_status_line(active_subagents):
            rows.append(_black_terminal_line("", width))
            rows.append(self._running_status_line(width))
            rows.append(_black_terminal_line("", width))
        if active_subagents:
            count = len(active_subagents)
            noun = "subagent" if count == 1 else "subagents"
            rows.append(
                _styled_terminal_line(
                    f" Waiting for {count} {noun}",
                    SUBAGENT_WAIT_TITLE_STYLE,
                    width,
                )
            )
            for snapshot in active_subagents[:3]:
                name = str(snapshot.get("display_name") or snapshot.get("subagent_id"))
                role = str(snapshot.get("role") or "subagent")
                task = _one_line(str(snapshot.get("task") or ""), limit=max(20, line_width - 8))
                line = f"  ↳ {name} [{role}] {task}"
                rows.append(_styled_terminal_line(line, STATUS_STYLE, width))
            omitted = count - 3
            if omitted > 0:
                rows.append(_styled_terminal_line(f"  ... +{omitted} more", STATUS_STYLE, width))
        if self.interrupted_user_inputs:
            title = " Interrupted messages to be sent with your next message"
            rows.append(_styled_terminal_line(title, STATUS_STYLE, width))
            for text in self.interrupted_user_inputs:
                line = f"  ↳ {_strip_control(text)}"
                rows.append(_styled_terminal_line(line, STATUS_STYLE, width))
        if self.pending_user_inputs:
            title = " Messages to be submitted after next tool call"
            hint = " (press esc to interrupt and send immediately)"
            rows.append(_styled_terminal_line(title, STATUS_STYLE, width))
            rows.append(_styled_terminal_line(hint, STATUS_STYLE, width))
            image_index = 1
            for text in self.pending_user_inputs:
                pending_preview, image_index = _image_placeholder_text(
                    text,
                    image_index_start=image_index,
                )
                line = f"  ↳ {_strip_control(pending_preview)}"
                rows.append(_styled_terminal_line(line, STATUS_STYLE, width))
        prefix = " > "
        continuation_prefix = " " * len(prefix)
        first_input_width = max(1, line_width - len(prefix))
        continuation_width = max(1, line_width - len(continuation_prefix))
        wrapped_input = _wrap_prompt_input(
            preview,
            cursor,
            first_width=first_input_width,
            continuation_width=continuation_width,
        )
        input_lines = wrapped_input.lines
        cursor_line = wrapped_input.cursor_line
        cursor_column = wrapped_input.cursor_column
        if cursor_line == 0:
            cursor_column += len(prefix)
        else:
            cursor_column += len(continuation_prefix)
        input_box_start = len(rows)
        rows.append(_styled_terminal_line("", PROMPT_STYLE, width))
        prompt_line_index = input_box_start + 1 + cursor_line
        for line_index, line in enumerate(input_lines):
            display = f"{prefix}{line}" if line_index == 0 else f"{continuation_prefix}{line}"
            rows.append(_styled_terminal_line(display, PROMPT_STYLE, width))
        rows.append(_styled_terminal_line("", PROMPT_STYLE, width))
        model_picker_choices = self._ensure_model_picker()
        if model_picker_choices is not None:
            picker_rows = _render_model_picker_rows(
                model_picker_choices,
                current_model=self.app.current_model,
                selected_index=self.model_picker_selected,
                width=line_width,
                styles_enabled=self.app._styles_enabled(),
            )
            rows.extend(_filled_terminal_line(row, STATUS_STYLE, width) for row in picker_rows)
        elif self._login_picker_active():
            login_rows = _render_login_picker_rows(
                selected_index=self.login_picker_selected,
                width=line_width,
                styles_enabled=self.app._styles_enabled(),
            )
            rows.extend(_filled_terminal_line(row, STATUS_STYLE, width) for row in login_rows)
        else:
            input_hints = self._ensure_input_hints()
            if input_hints:
                hint_rows = _render_input_hint_rows(
                    input_hints,
                    selected_index=self.input_hint_selected,
                    width=line_width,
                    styles_enabled=self.app._styles_enabled(),
                )
                rows.extend(_filled_terminal_line(row, STATUS_STYLE, width) for row in hint_rows)
            elif self.app._statusline_enabled:
                line = f" {status}"[:line_width].ljust(line_width)
                statusline = f"{STATUS_STYLE}{_style_statusline_text(line)}{RESET}"
                rows.append(_filled_terminal_line(statusline, STATUS_STYLE, width))
        return _PromptFrame(
            rows=rows,
            width=line_width,
            cursor_line_index=prompt_line_index,
            cursor_column=cursor_column,
        )

    def _active_subagent_snapshots(self) -> list[dict[str, object]]:
        subagents = getattr(self.app.runtime, "_subagents", None)
        if subagents is None:
            return []
        return _active_subagent_snapshots(subagents.snapshots())

    def _write_prompt_frame(
        self,
        frame: _PromptFrame,
        *,
        force_reflow_clear: bool = False,
    ) -> None:
        can_overwrite_in_place = (
            not force_reflow_clear
            and self.prompt_lines > 0
            and self.prompt_width == frame.width
            and len(frame.rows) >= self.prompt_lines
        )
        parts: list[str] = []
        if can_overwrite_in_place:
            parts.append("\x1b[?25l")
            if self.prompt_cursor_line_index:
                parts.append(f"\x1b[{self.prompt_cursor_line_index}A")
            parts.append("\r")
        else:
            parts.append(self._clear_prompt_sequence(force_reflow_clear=force_reflow_clear))
            parts.append("\x1b[?25l")
        parts.append("\n".join(frame.rows))
        self.prompt_lines = len(frame.rows)
        self.prompt_width = frame.width
        self.prompt_cursor_line_index = frame.cursor_line_index
        self.prompt_cursor_column = frame.cursor_column
        self.prompt_cursor_offset_from_bottom = (
            self.prompt_lines - self.prompt_cursor_line_index - 1
        )
        parts.append(self._place_prompt_cursor_sequence(frame.cursor_column))
        self.app._write("".join(parts))

    def _running_status_line(self, width: int) -> str:
        if self.app._styles_enabled():
            frame = int(time.monotonic() * 48)
            if self.active_tool_status is not None:
                return _running_terminal_line(
                    f" {self.active_tool_status}",
                    width,
                    frame=frame,
                )
            running_text, flower = self._working_status_text()
            return _running_terminal_line(
                f" {running_text}",
                width,
                frame=frame,
                flower=flower,
            )
        running_text, _flower = self._working_status_text()
        if self.active_tool_status is not None:
            running_text = self.active_tool_status
        return _black_terminal_line(f" {running_text}", width)

    def _working_status_text(self) -> tuple[str, Flower]:
        started_at = self.working_started_at or time.monotonic()
        elapsed_seconds = max(0, int(time.monotonic() - started_at))
        elapsed = _format_elapsed_compact(elapsed_seconds)
        flower, text = _flower_working_status(elapsed_seconds)
        return f"{text} ({elapsed}, press esc to interrupt)", flower

    def _running_status_active(self) -> bool:
        return (
            self.streaming
            and not self.compacting
            and not self._suppress_running_status_line(self._active_subagent_snapshots())
        )

    def _suppress_running_status_line(
        self,
        active_subagents: list[dict[str, object]],
    ) -> bool:
        return self.active_tool_status == WAIT_AGENT_RUNNING_TITLE and bool(active_subagents)

    def _redraw_running_status_line(self) -> None:
        if not self._running_status_active() or self.prompt_lines == 0:
            return
        width = self.app._terminal_width()
        if _terminal_line_width(width) != self.prompt_width:
            self._draw_prompt()
            return
        parts = ["\x1b7\x1b[?25l"]
        running_status_line_index = 1
        rows_above_cursor = self.prompt_cursor_line_index - running_status_line_index
        if rows_above_cursor > 0:
            parts.append(f"\x1b[{rows_above_cursor}A")
        parts.append("\r")
        parts.append(self._running_status_line(width))
        parts.append("\x1b8\x1b[?25h")
        self.app._write("".join(parts))

    def _place_prompt_cursor(self, column: int) -> None:
        self.app._write(self._place_prompt_cursor_sequence(column))

    def _place_prompt_cursor_sequence(self, column: int) -> str:
        parts: list[str] = []
        if self.prompt_cursor_offset_from_bottom:
            parts.append(f"\x1b[{self.prompt_cursor_offset_from_bottom}A")
        parts.append("\r")
        if column:
            parts.append(f"\x1b[{column}C")
        parts.append("\x1b[?25h")
        return "".join(parts)

    def _clear_prompt(self) -> None:
        sequence = self._clear_prompt_sequence()
        if sequence:
            self.app._write(sequence)

    def _clear_prompt_sequence(self, *, force_reflow_clear: bool = False) -> str:
        if self.prompt_lines == 0:
            return ""
        rows_to_clear = self.prompt_lines
        rows_down_to_prompt_bottom = self.prompt_cursor_offset_from_bottom
        parts: list[str] = []
        if rows_down_to_prompt_bottom:
            parts.append(f"\x1b[{rows_down_to_prompt_bottom}B")
            self.prompt_cursor_offset_from_bottom = 0
        parts.append("\x1b[?25l\r\x1b[J")
        for _ in range(rows_to_clear - 1):
            parts.append("\x1b[1A\r\x1b[2K")
        parts.append("\x1b[?25h")
        self._reset_prompt_state()
        return "".join(parts)


def _state_from_session(record: SessionRecord) -> dict[str, object]:
    last_context_tokens = next(
        (
            message.input_tokens
            for message in reversed(record.messages)
            if message.role == "assistant" and message.input_tokens > 0
        ),
        None,
    )
    return {
        "current_provider_name": record.settings.provider,
        "current_model": record.settings.model,
        "system": record.settings.system,
        "max_tokens": record.settings.max_tokens,
        "thinking": record.settings.thinking,
        "effort": record.settings.effort,
        "messages": list(record.messages),
        "compaction_state": _runtime_compaction_from_session(record),
        "statusline_enabled": True,
        "last_context_tokens": last_context_tokens,
        "total_input_tokens": sum(message.input_tokens for message in record.messages),
        "total_cached_tokens": sum(message.cached_tokens for message in record.messages),
        "total_output_tokens": sum(message.output_tokens for message in record.messages),
    }


def _runtime_compaction_from_session(record: SessionRecord) -> RuntimeCompaction | None:
    if not record.compactions:
        return None
    compaction = record.compactions[-1]
    return RuntimeCompaction(
        summary=compaction.summary,
        summarized_until=compaction.summarized_until_message_index,
        first_kept_index=compaction.first_kept_message_index,
    )


def _apply_resumed_session(
    args: argparse.Namespace,
    record: SessionRecord,
    path: Path,
) -> dict[str, object]:
    args.provider = record.settings.provider
    args.model = record.settings.model
    args.max_tokens = record.settings.max_tokens
    args.thinking = record.settings.thinking
    args.effort = record.settings.effort
    args._resume_session_record = record
    args._resume_session_path = path
    return _state_from_session(record)


def _load_resume_arg(selector: str) -> tuple[SessionRecord, Path]:
    path = resolve_session_path(selector)
    return load_session(path), path


def _select_resume_session(entries: list[SessionEntry]) -> SessionEntry | None:
    if not entries:
        return None
    if not (
        hasattr(sys.stdin, "isatty")
        and sys.stdin.isatty()
        and hasattr(sys.stdout, "isatty")
        and sys.stdout.isatty()
    ):
        return entries[0]
    return _run_resume_picker(entries)


def _run_resume_picker(entries: list[SessionEntry]) -> SessionEntry | None:
    width = max(72, shutil.get_terminal_size((88, 24)).columns)
    selected = 0
    rows = min(len(entries), max(1, shutil.get_terminal_size((88, 24)).lines - 6))

    def draw() -> None:
        offset = min(max(0, selected - rows + 1), max(0, len(entries) - rows))
        sys.stdout.write("\x1b[?25l\x1b[2J\x1b[H")
        sys.stdout.write("Resume Willow Session\n")
        sys.stdout.write("Enter resumes the highlighted session. Esc starts a new session.\n\n")
        for visible_idx, entry in enumerate(entries[offset : offset + rows]):
            idx = offset + visible_idx
            marker = ">" if idx == selected else " "
            text = _session_label(entry)
            if len(text) > width - 4:
                text = text[: width - 7] + "..."
            if idx == selected:
                sys.stdout.write(
                    f"{USER_STYLE} {marker} {text.ljust(width - 4)} {RESET}\n"
                )
            else:
                sys.stdout.write(f" {marker} {text}\n")
        sys.stdout.flush()

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        draw()
        while True:
            data = os.read(fd, 8).decode(errors="ignore")
            if data in ("\r", "\n"):
                return entries[selected]
            if data == "\x1b":
                return None
            if data in ("\x1b[A", "\x1bOA"):
                selected = max(0, selected - 1)
                draw()
            elif data in ("\x1b[B", "\x1bOB"):
                selected = min(len(entries) - 1, selected + 1)
                draw()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write("\x1b[0m\x1b[?25h\x1b[2J\x1b[H")
        sys.stdout.flush()


def _resolve_resume(args: argparse.Namespace) -> dict[str, object] | None:
    resume = getattr(args, "resume", None)
    if resume is None:
        return None
    if resume:
        record, path = _load_resume_arg(str(resume))
        return _apply_resumed_session(args, record, path)

    selected = _select_resume_session(list_sessions())
    if selected is None:
        return None
    return _apply_resumed_session(args, selected.record, selected.path)


def run_tui(args: argparse.Namespace) -> int:
    """Entry point wired in from `willow.cli`."""
    from willow.cli import _build_provider

    try:
        state = _resolve_resume(args)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"Could not resume session: {exc}\n")
        sys.stderr.flush()
        return 1

    provider = _build_provider(args.provider)
    args.persist_session = True
    app = WillowApp(args, provider, inline_mode=True, state=state)
    return app.run() or 0
