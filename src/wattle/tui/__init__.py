"""Wattle's native terminal chat UI.

The main transcript is append-only stdout. Wattle never enters the alternate
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
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, TextIO, cast
from urllib.parse import unquote, urlparse

from pygments import lex as pygments_lex
from pygments.lexers import get_lexer_by_name, get_lexer_for_filename
from pygments.token import Token
from pygments.util import ClassNotFound

from wattle.auth import login_openai_codex, save_api_key_credential
from wattle.clipboard import ClipboardImage, read_clipboard_image
from wattle.command_summary import (
    CommandSummary,
    is_research_summary,
    render_summary_action,
    summarize_tool_call,
)
from wattle.compaction import RuntimeCompaction
from wattle.loop import dispatch_tool_blocks_async
from wattle.message_history import (
    active_task_guidance_text_blocks,
    interrupted_user_text_blocks,
    monitor_event_text_blocks,
    monitor_event_texts,
    queued_user_text_blocks,
)
from wattle.models import (
    XIAOMI_DEFAULT_MODEL,
    XIAOMI_TOKEN_PLAN_SGP_PROVIDER,
    ModelChoice,
    available_model_choices,
    effort_levels_for_model,
    find_model_choice,
    model_supports_modality,
    render_model_choices,
)
from wattle.permissions import PermissionGate, PermissionMode
from wattle.providers import (
    CompletionRequest,
    CompletionResponse,
    ContentBlock,
    ImageBlock,
    IncompleteStreamError,
    Message,
    OpenAICodexResponsesProvider,
    Provider,
    StreamComplete,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseDelta,
    TransientProviderError,
)
from wattle.request_preparation import (
    RequestPreparer,
    astream_with_recovery,
    context_window_for_model,
    project_messages_for_model_modalities,
)
from wattle.session import (
    SessionCompaction,
    SessionEntry,
    SessionRecord,
    SessionSettings,
    default_session_dir,
    default_session_path,
    filter_session_entries,
    list_session_entries,
    load_session,
    new_session,
    resolve_session_path,
    save_session,
)
from wattle.settings import (
    DEFAULT_TUI_STATUSLINE_FIELDS,
    TuiSettings,
    load_settings,
    update_settings,
)
from wattle.skills import (
    expand_skill_invocation,
    load_available_skills,
    render_skill_suggestions,
)
from wattle.system_prompt import build_system_prompt
from wattle.tool_events import ToolRunEvent
from wattle.tools import DEFAULT_RUNTIME, TOOLS_BY_NAME
from wattle.tools.base import Tool
from wattle.tools.plan import PlanUpdate, parse_plan_update_input
from wattle.tui import terminal as terminal_rendering
from wattle.tui_flowers import Flower, flower_for_elapsed
from wattle.turns import append_turn_step, build_turn_step
from wattle.version import get_wattle_version

_default_terminal_line = terminal_rendering.default_terminal_line
_filled_terminal_line = terminal_rendering.filled_terminal_line
_running_terminal_line = terminal_rendering.running_terminal_line
_styled_terminal_line = terminal_rendering.styled_terminal_line
_styled_transcript_line = terminal_rendering.styled_transcript_line
_terminal_line_width = terminal_rendering.terminal_line_width
_wrap_terminal_line = terminal_rendering.wrap_terminal_line

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
    ("/compact", "compact conversation history now"),
    ("/effort", "choose reasoning effort"),
    ("/exit", "exit the TUI"),
    ("/help", "show commands and settings"),
    ("/login", "authenticate OpenAI Codex with OAuth"),
    ("/model", "choose what model to use"),
    ("/queue", "while streaming, send after the assistant turn completes"),
    ("/quit", "exit the TUI"),
    ("/resume", "switch to a saved session"),
    ("/session", "show persistence and saved session path"),
    ("/status", "show session and status details"),
    ("/statusline", "configure the bottom statusline"),
)

DEFAULT_LOGIN_CALLBACK_TIMEOUT_SECONDS = 300.0
SSH_LOGIN_CALLBACK_TIMEOUT_SECONDS = 2.0
SSH_LOGIN_CALLBACK_HINT = (
    "SSH detected. If your browser shows localhost refused to connect, copy the "
    "full callback URL from the browser address bar and paste it into Wattle "
    "when prompted."
)


def _running_over_ssh(env: Mapping[str, str] = os.environ) -> bool:
    return any(env.get(name) for name in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"))


def _login_callback_timeout_seconds() -> float:
    if _running_over_ssh():
        return SSH_LOGIN_CALLBACK_TIMEOUT_SECONDS
    return DEFAULT_LOGIN_CALLBACK_TIMEOUT_SECONDS


BUILTIN_SLASH_COMMANDS = frozenset(command for command, _description in SLASH_COMMAND_HINTS)
LOGIN_PROVIDER_CHOICES: tuple[tuple[str, str], ...] = (
    ("openai-codex", "ChatGPT Plus/Pro Codex OAuth"),
    (XIAOMI_TOKEN_PLAN_SGP_PROVIDER, "Xiaomi Token Plan SGP API key"),
)
API_KEY_LOGIN_PROVIDERS: Mapping[str, tuple[str, str, str]] = {
    XIAOMI_TOKEN_PLAN_SGP_PROVIDER: (
        XIAOMI_TOKEN_PLAN_SGP_PROVIDER,
        "Xiaomi Token Plan SGP",
        XIAOMI_DEFAULT_MODEL,
    ),
}
PASTE_PLACEHOLDER_MIN_CHARS = 500
MAX_IMAGE_ATTACHMENT_BYTES = 20 * 1024 * 1024
SUPPORTED_IMAGE_MEDIA_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/webp", "image/gif"}
)
ESCAPE_SEQUENCE_TIMEOUT_SECONDS = 0.05
KEYBOARD_ENHANCEMENT_ENABLE = "\x1b[>1u\x1b[>4;2m"
KEYBOARD_ENHANCEMENT_DISABLE = "\x1b[<u\x1b[>4m"
CTRL_V_KEY_SEQUENCES = (
    "\x16",
    "\x1b[118;5u",
    "\x1b[86;5u",
    "\x1b[27;5;118~",
    "\x1b[27;5;86~",
)
VISIBLE_SCREEN_CLEAR = "\x1b[H\x1b[2J\x1b[H"
TERMINAL_HISTORY_CLEAR = "\x1b[3J"
SHIFT_ENTER_SEQUENCES = (
    "\x1b[13;2u",
    "\x1b[13;2~",
    "\x1b[27;2;13~",
)
SHIFT_TAB_SEQUENCES = (
    "\x1b[Z",
    "\x1b[9;2u",
    "\x1b[27;2;9~",
)
THINKING_LEVELS = ("low", "medium", "high", "xhigh", "max")
KNOWN_ESCAPE_SEQUENCES = (
    "\x1b[200~",
    "\x1b[201~",
    *CTRL_V_KEY_SEQUENCES[1:],
    *SHIFT_ENTER_SEQUENCES,
    *SHIFT_TAB_SEQUENCES,
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


class _UnavailableProvider(Provider):
    def __init__(self, message: str) -> None:
        self.message = message

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        raise RuntimeError(self.message)

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamComplete]:
        raise RuntimeError(self.message)
        yield  # pragma: no cover

RESET = "\x1b[0m"
DIM = "\x1b[2m"
BOLD = "\x1b[1m"
USER_STYLE = "\x1b[48;5;235;38;5;231m"
MESSAGE_BLOCK_VERTICAL_PADDING = 1
ASSISTANT_STYLE = "\x1b[38;5;255m"
THINKING_STYLE = "\x1b[38;5;245m"
WORKED_DURATION_STYLE = "\x1b[38;5;240m"
TOOL_STYLE = "\x1b[48;5;58;38;5;230m"
TOOL_MARKER_STYLE = "\x1b[38;5;82m"
TOOL_MARKER = "|"
TOOL_TITLE_STYLE = "\x1b[38;5;255;1m"
TOOL_PREVIEW_STYLE = "\x1b[38;5;245m"
PLAN_COMPLETED_STYLE = "\x1b[38;5;240m"
PLAN_IN_PROGRESS_STYLE = "\x1b[38;5;80;1m"
PLAN_PENDING_STYLE = TOOL_PREVIEW_STYLE
DIFF_ADD_STYLE = "\x1b[38;5;231m"
DIFF_DELETE_STYLE = "\x1b[38;5;231m"
DIFF_ADD_COUNT_STYLE = "\x1b[38;5;82;1m"
DIFF_DELETE_COUNT_STYLE = "\x1b[38;5;203;1m"
DIFF_META_STYLE = "\x1b[38;5;245m"
DIFF_ADD_LINE_NUMBER_STYLE = "\x1b[48;5;22;38;5;72m"
DIFF_DELETE_LINE_NUMBER_STYLE = "\x1b[48;5;52;38;5;95m"
DIFF_ADD_MARKER_STYLE = "\x1b[48;5;22;38;5;40m"
DIFF_DELETE_MARKER_STYLE = "\x1b[48;5;52;38;5;160m"
DIFF_ADD_CODE_STYLE = "\x1b[48;5;22;38;5;252m"
DIFF_DELETE_CODE_STYLE = "\x1b[48;5;52;38;5;248m"
DIFF_ADD_SYNTAX_COMMENT_STYLE = "\x1b[48;5;22;38;5;108m"
DIFF_DELETE_SYNTAX_COMMENT_STYLE = "\x1b[48;5;52;38;5;138m"
DIFF_ADD_SYNTAX_HEADING_STYLE = "\x1b[48;5;22;38;5;80m"
DIFF_DELETE_SYNTAX_HEADING_STYLE = "\x1b[48;5;52;38;5;174m"
DIFF_ADD_SYNTAX_KEYWORD_STYLE = "\x1b[48;5;22;38;5;177m"
DIFF_DELETE_SYNTAX_KEYWORD_STYLE = "\x1b[48;5;52;38;5;174m"
DIFF_ADD_SYNTAX_NAME_STYLE = "\x1b[48;5;22;38;5;116m"
DIFF_DELETE_SYNTAX_NAME_STYLE = "\x1b[48;5;52;38;5;181m"
DIFF_ADD_SYNTAX_STRING_STYLE = "\x1b[48;5;22;38;5;222m"
DIFF_DELETE_SYNTAX_STRING_STYLE = "\x1b[48;5;52;38;5;180m"
DIFF_ADD_SYNTAX_NUMBER_STYLE = "\x1b[48;5;22;38;5;229m"
DIFF_DELETE_SYNTAX_NUMBER_STYLE = "\x1b[48;5;52;38;5;222m"
DIFF_ADD_SYNTAX_OPERATOR_STYLE = "\x1b[48;5;22;38;5;159m"
DIFF_DELETE_SYNTAX_OPERATOR_STYLE = "\x1b[48;5;52;38;5;181m"
SYNTAX_KEYWORD_STYLE = "\x1b[38;5;177;1m"
SYNTAX_STRING_STYLE = "\x1b[38;5;216m"
SYNTAX_NAME_STYLE = "\x1b[38;5;75;1m"
SYNTAX_CONSTANT_STYLE = "\x1b[38;5;222m"
COMMAND_EXEC_STYLE = "\x1b[38;5;75;1m"
COMMAND_ARG_STYLE = "\x1b[38;5;189;1m"
COMMAND_OPTION_STYLE = "\x1b[38;5;211m"
COMMAND_PATH_STYLE = "\x1b[38;5;159;1m"
COMMAND_OPERATOR_STYLE = "\x1b[38;5;80;1m"
COMMAND_STRING_STYLE = "\x1b[38;5;120m"
SEPARATOR_STYLE = "\x1b[38;5;240m"
ERROR_TEXT_STYLE = "\x1b[38;5;203;1m"
WELCOME_BORDER_STYLE = "\x1b[38;5;244m"
WELCOME_LOGO_STYLE = "\x1b[38;5;107;1m"
WELCOME_TITLE_STYLE = "\x1b[38;5;255;1m"
WELCOME_LABEL_STYLE = "\x1b[38;5;245m"
WELCOME_VALUE_STYLE = "\x1b[38;5;255;1m"
STATUS_STYLE = "\x1b[48;5;236;38;5;248m"
SSH_LOGIN_HINT_STYLE = "\x1b[48;5;236;38;5;82;1m"
SUBAGENT_WAIT_TITLE_STYLE = "\x1b[48;5;236;38;5;255m"
STATUS_MODEL_STYLE = "\x1b[48;5;236;38;5;82m"
STATUS_TOKEN_STYLE = "\x1b[48;5;236;38;5;203m"
STATUSLINE_STYLE = "\x1b[38;5;248m"
STATUSLINE_MODEL_STYLE = "\x1b[38;5;82m"
STATUSLINE_TOKEN_STYLE = "\x1b[38;5;203m"
COMPACTION_STYLE = "\x1b[48;5;54;38;5;231;1m"
PROMPT_STYLE = "\x1b[48;5;235;38;5;231m"
ERROR_STYLE = "\x1b[48;5;52;38;5;231m"
PROMPT_MARKER_STYLE = "\x1b[48;5;235;38;5;51;1m"
SELECTED_ROW_STYLE = "\x1b[48;5;240;38;5;255;1m"
COMPACTION_FRAMES = ("◐", "◓", "◑", "◒")
WAIT_AGENT_RUNNING_TITLE = "Waiting for subagent"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b[78]")
SUBAGENT_VISIBLE_STATUSES = frozenset(
    {"pending", "running", "closing", "completed", "failed"}
)
SUBAGENT_STATUS_LABELS = {
    "pending": "waiting to start",
    "running": "running",
    "completed": "complete",
    "failed": "error",
    "closing": "closing",
    "closed": "closed",
}
SUBAGENT_STATUS_GLYPHS = {
    "pending": "◌",
    "running": "↻",
    "completed": "✓",
    "failed": "!",
    "closing": "×",
    "closed": "×",
}
SUBAGENT_STATUS_ORDER = ("completed", "failed", "running", "pending", "closing")
_EDIT_HUNK_SEPARATOR_ROW = "      ..."
_DIFF_CONTEXT_MARKER_ROW = "      ⋮"


@dataclass(frozen=True)
class _DiffPreviewRow:
    kind: Literal["add", "delete", "context", "meta"]
    old_line: int | None
    new_line: int | None
    text: str


@dataclass
class _EditRenderItem:
    path: str
    tool_name: str
    added: int
    deleted: int
    rows: list[tuple[str, str]]


@dataclass
class _EditRenderGroup:
    path: str
    items: list[_EditRenderItem]

WATTLE_LOGO_LINES: tuple[str, ...] = (
    "   \\ | /   ",
    " ~~ \\|/ ~~ ",
    "     Y     ",
)
WELCOME_TITLE = "Wattle Agent"


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
    rendered = [_truncate_preview_line(line, max_width=max_width) for line in selected]
    omitted = len(lines) - len(selected)
    if omitted > 0:
        rendered.append(f"... +{omitted} lines")
    return rendered


def _compact_head_tail_lines(
    content: str,
    *,
    max_head_lines: int = 2,
    max_tail_lines: int = 2,
    max_width: int = 110,
) -> list[str]:
    lines = [line.rstrip() for line in content.splitlines()]
    if not lines:
        return []
    max_lines = max_head_lines + max_tail_lines
    if len(lines) <= max_lines:
        return [_truncate_preview_line(line, max_width=max_width) for line in lines]

    head = lines[:max_head_lines]
    tail = lines[-max_tail_lines:] if max_tail_lines else []
    omitted = len(lines) - len(head) - len(tail)
    return [
        *[_truncate_preview_line(line, max_width=max_width) for line in head],
        f"... +{omitted} lines",
        *[_truncate_preview_line(line, max_width=max_width) for line in tail],
    ]


def _truncate_preview_line(line: str, *, max_width: int) -> str:
    return line if len(line) <= max_width else line[: max_width - 3] + "..."


def _truncate_cell_text(text: str, max_width: int) -> str:
    if max_width <= 0:
        return ""
    if len(text) <= max_width:
        return text
    if max_width <= 3:
        return text[:max_width]
    return text[: max_width - 3] + "..."


def _bash_preview_content(content: str) -> str:
    lines = [line for line in content.splitlines() if not line.startswith("[elapsed ")]
    try:
        excerpt_start = lines.index("[excerpt]") + 1
        excerpt_end = lines.index("[/excerpt]", excerpt_start)
    except ValueError:
        return "\n".join(lines)
    return "\n".join(lines[excerpt_start:excerpt_end])


def _tail_chars(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[-limit:]


def _plain_terminal_rows(text: str, *, width: int) -> list[str]:
    rows: list[str] = []
    for line in text.splitlines() or [""]:
        rows.extend(_wrap_terminal_line(_strip_control(ANSI_ESCAPE_RE.sub("", line)), width))
    return rows


def _bash_exec_output_rows(
    output: str,
    *,
    width: int,
    running: bool,
) -> list[str]:
    content = _bash_preview_content(output).rstrip("\n")
    if not content:
        return [] if running else ["[no output]"]
    rows = _plain_terminal_rows(content, width=width)
    if running:
        return rows[-5:]
    if len(rows) <= 4:
        return rows
    omitted = len(rows) - 4
    return [*rows[:2], f"... +{omitted} lines", *rows[-2:]]


def _bash_exec_cell_plain_rows(cell: _BashExecCell, *, width: int) -> list[str]:
    status = "Running" if cell.running else "Ran"
    title_prefix = f"{TOOL_MARKER} {status} "
    continuation_prefix = "  │ "
    line_width = _terminal_line_width(width)
    command = _strip_control(cell.command)
    first_width = max(1, line_width - len(title_prefix))
    continuation_width = max(1, line_width - len(continuation_prefix))
    command_rows = _wrap_prompt_input(
        command,
        len(command),
        first_width=first_width,
        continuation_width=continuation_width,
    ).lines
    rows = [f"{title_prefix}{command_rows[0]}"]
    rows.extend(f"{continuation_prefix}{line}" for line in command_rows[1:])
    output_width = max(1, line_width - 4)
    for index, line in enumerate(
        _bash_exec_output_rows(cell.output, width=output_width, running=cell.running)
    ):
        prefix = "  └ " if index == 0 else "    "
        rows.append(f"{prefix}{line}")
    return rows


def _bash_exec_cell_prompt_rows(
    cell: _BashExecCell,
    *,
    width: int,
    styles_enabled: bool,
) -> list[str]:
    rows = _bash_exec_cell_plain_rows(cell, width=width)
    if not styles_enabled:
        return [_default_terminal_line(row, width) for row in rows]
    rendered: list[str] = []
    for index, row in enumerate(rows):
        if index == 0:
            status = "Running" if cell.running else "Ran"
            prefix = f"{TOOL_MARKER} {status} "
            command = row[len(prefix) :]
            rendered_command = _render_shell_command(command) or command
            line = (
                f"{TOOL_MARKER_STYLE}{TOOL_MARKER}{RESET} "
                f"{TOOL_TITLE_STYLE}{status}{RESET} "
                f"{rendered_command}"
            )
            rendered.append(_filled_terminal_line(line, TOOL_PREVIEW_STYLE, width))
        else:
            rendered.append(_styled_terminal_line(row, TOOL_PREVIEW_STYLE, width))
    return rendered


def _key_value_lines(content: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in content.splitlines():
        key, separator, value = line.partition(": ")
        if separator:
            fields[key] = value
    return fields


def _subagent_summary_fields(content: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    lines = content.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        key, separator, value = line.partition(": ")
        if separator:
            fields[key] = value
            index += 1
            continue
        if line in {"result:", "error:"}:
            key = line[:-1]
            index += 1
            body: list[str] = []
            while index < len(lines) and not re.match(r"^[a-z_]+: ", lines[index]):
                body.append(lines[index])
                index += 1
            fields[key] = "\n".join(body).strip()
            continue
        index += 1
    return fields


def _short_subagent_id(subagent_id: object) -> str:
    text = str(subagent_id or "subagent")
    if text.startswith("subagent-") and len(text) > 18:
        return f"subagent-{text[-6:]}"
    return text


def _subagent_label(fields: Mapping[str, object]) -> str:
    name = str(
        fields.get("name")
        or fields.get("display_name")
        or _short_subagent_id(fields.get("subagent_id"))
    )
    role = str(fields.get("role") or "subagent")
    return f"{name} [{role}]"


def _subagent_status_label(status: object) -> str:
    return SUBAGENT_STATUS_LABELS.get(str(status or ""), str(status or "updated"))


def _subagent_status_glyph(status: object) -> str:
    return SUBAGENT_STATUS_GLYPHS.get(str(status or ""), "•")


def _subagent_summary_text(fields: Mapping[str, object]) -> str:
    for key in ("result", "error", "summary", "task"):
        value = fields.get(key)
        if isinstance(value, str) and value.strip():
            return _one_line(value, limit=120)
    return ""


def _subagent_count_summary(snapshots: list[Mapping[str, object]]) -> str:
    counts: dict[str, int] = {}
    for snapshot in snapshots:
        status = str(snapshot.get("status") or "")
        if status:
            counts[status] = counts.get(status, 0) + 1
    parts: list[str] = []
    for status in SUBAGENT_STATUS_ORDER:
        count = counts.get(status, 0)
        if count:
            parts.append(f"{count} {_subagent_status_label(status)}")
    for status, count in sorted(counts.items()):
        if status not in SUBAGENT_STATUS_ORDER:
            parts.append(f"{count} {_subagent_status_label(status)}")
    return ", ".join(parts) if parts else "none"


def _subagent_lifecycle_lines(snapshots: list[Mapping[str, object]]) -> list[str]:
    lines: list[str] = []
    for snapshot in snapshots:
        status = str(snapshot.get("status") or "")
        line = (
            f"  {_subagent_status_glyph(status)} {_subagent_label(snapshot)} "
            f"{_subagent_status_label(status)}"
        )
        summary = _subagent_summary_text(snapshot)
        if summary:
            line = f"{line} · {summary}"
        lines.append(line)
    return lines


def _subagent_result_line(fields: Mapping[str, object]) -> str:
    status = str(fields.get("status") or "")
    label = _subagent_label(fields)
    status_text = _subagent_status_label(status).capitalize()
    summary = _subagent_summary_text(fields)
    line = f"{label}: {status_text}"
    if summary:
        line = f"{line} - {summary}"
    return line


def _spawn_agent_title(block: ToolUseBlock, result: ToolResultBlock) -> str:
    fields = _subagent_summary_fields(result.content)
    name = fields.get("name") or fields.get("subagent_id") or "subagent"
    role = fields.get("role") or "subagent"
    model = fields.get("model") or _one_line(block.input.get("model", "default"))
    effort = fields.get("effort") or ""
    model_detail = model if effort in {"", "default", "None"} else f"{model} {effort}"
    return f"Spawned {name} [{role}] ({model_detail})"


def _spawn_agent_detail(block: ToolUseBlock, result: ToolResultBlock) -> str:
    fields = _subagent_summary_fields(result.content)
    task = fields.get("task") or _one_line(block.input.get("task", ""))
    if task:
        return _one_line(f"└ {task}", limit=180)
    return "└ [no task]"


def _subagent_event_title(event: Mapping[str, object]) -> str:
    return f"{_subagent_label(event)} {_subagent_status_label(event.get('status'))}"


def _subagent_event_detail(event: Mapping[str, object]) -> str:
    summary = _subagent_summary_text(event)
    return _one_line(f"└ {summary}", limit=180) if summary else ""


def _visible_subagent_snapshots(snapshots: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        snapshot
        for snapshot in snapshots
        if snapshot.get("status") in SUBAGENT_VISIBLE_STATUSES
    ]


def _tool_action_title(block: ToolUseBlock, *, is_error: bool = False) -> str:
    if is_error:
        if block.name in {"read", "write", "edit"}:
            return f"{block.name} error - {_tool_arg(block, 'path', '<missing path>')}"
        return f"{block.name} error"
    summary = summarize_tool_call(block.name, block.input)
    if is_research_summary(summary):
        return render_summary_action(summary)
    if block.name == "bash":
        return f"Ran {_tool_arg(block, 'command', '<missing command>')}"
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
    summary = summarize_tool_call(block.name, block.input)
    if is_research_summary(summary):
        return render_summary_action(summary)
    if block.name == "bash":
        return f"running bash - {_tool_arg(block, 'command', '<missing command>')}"
    if block.name in {"read", "write", "edit"}:
        return f"running {block.name} - {_tool_arg(block, 'path', '<missing path>')}"
    if block.name == "wait_agent":
        return WAIT_AGENT_RUNNING_TITLE
    return f"running {block.name}"


def _tool_edit_title(block: ToolUseBlock, *, path: str, added: int, deleted: int) -> str:
    return _edit_title_for_tool_name(block.name, path=path, added=added, deleted=deleted)


def _edit_title_for_tool_name(
    tool_name: str,
    *,
    path: str,
    added: int,
    deleted: int,
) -> str:
    if tool_name == "edit":
        verb = "Edited"
    elif deleted == 0:
        verb = "Added"
    else:
        verb = "Wrote"
    return f"{verb} {path} (+{added} -{deleted})"


def _edit_group_title(group: _EditRenderGroup) -> str:
    added = sum(item.added for item in group.items)
    deleted = sum(item.deleted for item in group.items)
    tool_names = {item.tool_name for item in group.items}
    if len(group.items) == 1:
        return _edit_title_for_tool_name(
            group.items[0].tool_name,
            path=group.path,
            added=added,
            deleted=deleted,
        )
    if tool_names == {"edit"}:
        verb = "Edited"
    elif tool_names == {"write"}:
        verb = "Wrote"
    else:
        verb = "Updated"
    return f"{verb} {group.path} (+{added} -{deleted})"


def _edit_render_item(
    block: ToolUseBlock,
    result: ToolResultBlock,
) -> _EditRenderItem | None:
    if result.is_error or block.name not in {"write", "edit"}:
        return None
    path = _tool_arg(block, "path", "<missing path>")
    lines = result.content.splitlines()
    diff_lines = lines[1:] if lines else []
    added, deleted = _diff_counts(diff_lines)
    rows = _diff_preview_lines(diff_lines, max_changes=None)
    if not rows:
        rows = [("meta", lines[0] if lines else "[no changes]")]
    return _EditRenderItem(
        path=path,
        tool_name=block.name,
        added=added,
        deleted=deleted,
        rows=rows,
    )


def _tool_arg(block: ToolUseBlock, name: str, default: str) -> str:
    value = block.input.get(name, default)
    return _one_line(value)


_SHELL_TOKEN_RE = re.compile(r"""'[^']*'|"[^"]*"|\|\||&&|[|;&()<>]|[^\s|;&()<>]+""")
_PYTHON_KEYWORDS = frozenset(
    {
        "and",
        "as",
        "assert",
        "break",
        "class",
        "continue",
        "def",
        "elif",
        "else",
        "except",
        "finally",
        "for",
        "from",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "not",
        "or",
        "pass",
        "return",
        "try",
        "while",
        "with",
        "yield",
    }
)
_PYTHON_CONSTANTS = frozenset({"False", "None", "True"})
_PYTHON_TOKEN_RE = re.compile(
    r"""('[^']*'|"[^"]*"|\b[A-Za-z_][A-Za-z0-9_]*\b|\d+(?:\.\d+)?)"""
)


def _shell_tokens(command: str) -> list[str]:
    return _SHELL_TOKEN_RE.findall(command)


def _is_shell_operator(token: str) -> bool:
    return token in {"|", "||", "&&", ";", "&", "(", ")", "<", ">", ">>", "2>", "2>&1"}


def _is_shell_path(token: str) -> bool:
    stripped = token.strip("'\"")
    return (
        "/" in stripped
        or stripped.startswith((".", "~"))
        or bool(re.search(r"\.[A-Za-z0-9]{1,8}$", stripped))
    )


def _render_shell_command(command: str) -> str:
    tokens = _shell_tokens(command)
    if not tokens:
        return ""
    rendered: list[str] = []
    expect_command = True
    for token in tokens:
        if _is_shell_operator(token):
            style = COMMAND_OPERATOR_STYLE
            expect_command = token in {"|", "||", "&&", ";"}
        elif expect_command:
            style = COMMAND_EXEC_STYLE
            expect_command = False
        elif token.startswith("-"):
            style = COMMAND_OPTION_STYLE
        elif token.startswith(("'", '"')):
            style = COMMAND_STRING_STYLE
        elif _is_shell_path(token):
            style = COMMAND_PATH_STYLE
        else:
            style = COMMAND_ARG_STYLE
        rendered.append(f"{style}{token}{RESET}")
    return " ".join(rendered)


def _render_python_syntax(text: str, *, base_style: str) -> str:
    parts: list[str] = []
    position = 0
    for match in _PYTHON_TOKEN_RE.finditer(text):
        if match.start() > position:
            parts.append(f"{base_style}{text[position:match.start()]}")
        token = match.group(0)
        if token.startswith(("'", '"')):
            syntax_style = SYNTAX_STRING_STYLE
        elif token in _PYTHON_KEYWORDS:
            syntax_style = SYNTAX_KEYWORD_STYLE
        elif token in _PYTHON_CONSTANTS or re.fullmatch(r"\d+(?:\.\d+)?", token):
            syntax_style = SYNTAX_CONSTANT_STYLE
        else:
            syntax_style = SYNTAX_NAME_STYLE
        parts.append(f"{base_style}{syntax_style}{token}")
        position = match.end()
    if position < len(text):
        parts.append(f"{base_style}{text[position:]}")
    return "".join(parts)


def _render_diff_row(
    kind: str,
    line: str,
    *,
    width: int,
    path: str | None = None,
) -> str:
    if kind not in {"add", "delete"}:
        clipped = line if len(line) <= width else line[: width - 3] + "..."
        return f"{DIFF_META_STYLE}{clipped}{RESET}"

    if kind == "add":
        line_number_style = DIFF_ADD_LINE_NUMBER_STYLE
        marker_style = DIFF_ADD_MARKER_STYLE
        code_style = DIFF_ADD_CODE_STYLE
    else:
        line_number_style = DIFF_DELETE_LINE_NUMBER_STYLE
        marker_style = DIFF_DELETE_MARKER_STYLE
        code_style = DIFF_DELETE_CODE_STYLE

    match = re.match(r"(\s*\d+\s+)([+-])(.*)", line)
    if match is None:
        return _styled_terminal_line(line, code_style, width)

    line_number, marker, code = match.groups()
    prefix_width = len(line_number) + len(marker)
    code_width = max(1, width - prefix_width)
    chunks = _diff_syntax_chunks(code, kind=kind, path=path, width=code_width)

    rows: list[str] = []
    for index, chunk in enumerate(chunks):
        code_rendered = "".join(f"{style}{text}" for style, text in chunk)
        if index == 0:
            rendered = (
                f"{line_number_style}{line_number}"
                f"{marker_style}{marker}"
                f"{code_rendered}"
            )
        else:
            rendered = (
                f"{line_number_style}{' ' * len(line_number)}"
                f"{marker_style} "
                f"{code_rendered}"
            )
        rows.append(_diff_terminal_row(rendered, code_style))
    return "\n".join(rows)


@lru_cache(maxsize=256)
def _diff_lexer_for_path(path: str | None) -> Any | None:
    if not path:
        return None
    try:
        return get_lexer_for_filename(path)
    except ClassNotFound:
        return None


def _diff_syntax_chunks(
    text: str,
    *,
    kind: str,
    path: str | None,
    width: int,
) -> list[list[tuple[str, str]]]:
    spans = _diff_syntax_spans(text, kind=kind, path=path)
    chunks: list[list[tuple[str, str]]] = [[]]
    remaining = width
    for style, value in spans:
        if remaining == 0:
            chunks.append([])
            remaining = width
        while value:
            take = min(len(value), remaining)
            chunks[-1].append((style, value[:take]))
            value = value[take:]
            remaining -= take
            if remaining == 0 and value:
                chunks.append([])
                remaining = width
    return chunks


def _diff_syntax_spans(
    text: str,
    *,
    kind: str,
    path: str | None,
) -> list[tuple[str, str]]:
    base_style = DIFF_ADD_CODE_STYLE if kind == "add" else DIFF_DELETE_CODE_STYLE
    lexer = _diff_lexer_for_path(path)
    if lexer is None:
        return [(base_style, text)]

    spans: list[tuple[str, str]] = []
    for token_type, value in pygments_lex(text, lexer):
        value = value.replace("\n", "")
        if not value:
            continue
        spans.append((_diff_syntax_style(token_type, kind=kind, base_style=base_style), value))
    return spans or [(base_style, text)]


def _diff_syntax_style(token_type: Any, *, kind: str, base_style: str) -> str:
    if token_type in Token.Comment:
        return (
            DIFF_ADD_SYNTAX_COMMENT_STYLE
            if kind == "add"
            else DIFF_DELETE_SYNTAX_COMMENT_STYLE
        )
    if token_type in Token.Generic.Heading or token_type in Token.Generic.Subheading:
        return (
            DIFF_ADD_SYNTAX_HEADING_STYLE
            if kind == "add"
            else DIFF_DELETE_SYNTAX_HEADING_STYLE
        )
    if token_type in Token.Keyword:
        return (
            DIFF_ADD_SYNTAX_KEYWORD_STYLE
            if kind == "add"
            else DIFF_DELETE_SYNTAX_KEYWORD_STYLE
        )
    if token_type in Token.Literal.String:
        return (
            DIFF_ADD_SYNTAX_STRING_STYLE
            if kind == "add"
            else DIFF_DELETE_SYNTAX_STRING_STYLE
        )
    if token_type in Token.Literal.Number:
        return (
            DIFF_ADD_SYNTAX_NUMBER_STYLE
            if kind == "add"
            else DIFF_DELETE_SYNTAX_NUMBER_STYLE
        )
    if token_type in Token.Name:
        return DIFF_ADD_SYNTAX_NAME_STYLE if kind == "add" else DIFF_DELETE_SYNTAX_NAME_STYLE
    if token_type in Token.Operator or token_type in Token.Punctuation:
        return (
            DIFF_ADD_SYNTAX_OPERATOR_STYLE
            if kind == "add"
            else DIFF_DELETE_SYNTAX_OPERATOR_STYLE
        )
    return base_style


@lru_cache(maxsize=256)
def _lexer_for_language(language: str) -> Any | None:
    try:
        return get_lexer_by_name(language)
    except ClassNotFound:
        return None


def _syntax_style(token_type: Any, *, base_style: str) -> str:
    if token_type in Token.Comment:
        return THINKING_STYLE
    if token_type in Token.Keyword:
        return SYNTAX_KEYWORD_STYLE
    if token_type in Token.Literal.String:
        return SYNTAX_STRING_STYLE
    if token_type in Token.Literal.Number:
        return SYNTAX_CONSTANT_STYLE
    if token_type in Token.Name:
        return SYNTAX_NAME_STYLE
    if token_type in Token.Operator or token_type in Token.Punctuation:
        return SYNTAX_CONSTANT_STYLE
    return base_style


def _syntax_spans_for_language(
    text: str,
    language: str,
    *,
    base_style: str,
) -> list[tuple[str, str]]:
    lexer = _lexer_for_language(language)
    if lexer is None:
        return [(base_style, text)]

    spans: list[tuple[str, str]] = []
    for token_type, value in pygments_lex(text, lexer):
        value = value.replace("\n", "")
        if not value:
            continue
        spans.append((_syntax_style(token_type, base_style=base_style), value))
    return spans or [(base_style, text)]


def _diff_terminal_row(rendered: str, fill_style: str) -> str:
    return f"\r\x1b[?7l\x1b[0m\x1b[2K{rendered}{fill_style}\x1b[K{RESET}\x1b[?7h"


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


def _matching_tool_result(
    message: Message | None,
    tool_use_id: str,
) -> ToolResultBlock | None:
    if message is None:
        return None
    for block in message.content:
        if isinstance(block, ToolResultBlock) and block.tool_use_id == tool_use_id:
            return block
    return None


def _history_message_is_fully_suppressed(
    message: Message,
    suppressed_tool_result_ids: set[str],
) -> bool:
    return bool(message.content) and all(
        isinstance(block, ToolResultBlock)
        and block.tool_use_id in suppressed_tool_result_ids
        for block in message.content
    )


def _tool_call_summary(block: ToolUseBlock) -> str:
    if block.name == "bash":
        command = block.input.get("command", "<missing command>")
        details = []
        if block.input.get("background"):
            details.append("background")
        if block.input.get("tty"):
            details.append("tty")
        if "timeout" in block.input:
            details.append(f"timeout={block.input['timeout']}")
        suffix = f" ({', '.join(details)})" if details else ""
        return f"bash: {command}{suffix}"
    if block.name == "edit":
        path = block.input.get("path", "<missing path>")
        edits = block.input.get("edits")
        if isinstance(edits, list):
            return f"edit: {path} ({len(edits)} edit{'s' if len(edits) != 1 else ''})"
        return f"edit: {path}"
    if block.name == "write":
        path = block.input.get("path", "<missing path>")
        content = block.input.get("content", "")
        if isinstance(content, str):
            lines = content.count("\n") + (1 if content else 0)
            return f"write: {path} ({lines} line{'s' if lines != 1 else ''})"
        return f"write: {path}"
    if block.name in {"read", "view_image"}:
        return f"{block.name}: {block.input.get('path', '<missing path>')}"
    if block.name == "monitor":
        command = block.input.get("command", "<missing command>")
        description = block.input.get("description")
        if isinstance(description, str) and description:
            return f"monitor: {description} ({command})"
        return f"monitor: {command}"
    return f"{block.name}: {block.input}"


def _research_summary_for_success(
    block: ToolUseBlock,
    result: ToolResultBlock,
) -> CommandSummary | None:
    if result.is_error:
        return None
    summary = summarize_tool_call(block.name, block.input)
    if not is_research_summary(summary):
        return None
    return summary


def _research_lines(summaries: list[CommandSummary]) -> list[str]:
    return [render_summary_action(summary) for summary in summaries]


def _plan_update_for_success(
    block: ToolUseBlock,
    result: ToolResultBlock,
) -> PlanUpdate | None:
    if block.name != "update_plan" or result.is_error:
        return None
    try:
        return parse_plan_update_input(block.input)
    except ValueError:
        return None


def _plan_marker(status: str) -> str:
    if status == "completed":
        return "[x]"
    if status == "in_progress":
        return "[>]"
    return "[ ]"


def _plan_style(status: str) -> str:
    if status == "completed":
        return PLAN_COMPLETED_STYLE
    if status == "in_progress":
        return PLAN_IN_PROGRESS_STYLE
    return PLAN_PENDING_STYLE


def _diff_counts(diff_lines: list[str]) -> tuple[int, int]:
    added = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    deleted = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
    return added, deleted


def _diff_starts_from_empty_old_file(diff_lines: list[str]) -> bool:
    return any(line.startswith("@@ -0,0 ") for line in diff_lines)


def _diff_preview_lines(
    diff_lines: list[str], *, max_changes: int | None = 12
) -> list[tuple[str, str]]:
    return [
        (row.kind, line)
        for row, line in _format_diff_preview_rows(
            _diff_preview_rows(diff_lines, max_changes=max_changes)
        )
    ]


def _diff_preview_rows(
    diff_lines: list[str], *, max_changes: int | None = 12
) -> list[_DiffPreviewRow]:
    source_rows: list[_DiffPreviewRow] = []
    old_line = 0
    new_line = 0
    for line in diff_lines:
        if line.startswith("@@"):
            parts = line.split()
            if len(parts) >= 3:
                old_line = _parse_hunk_start(parts[1])
                new_line = _parse_hunk_start(parts[2])
                _append_gap_marker_if_needed(
                    source_rows,
                    _DiffPreviewRow(
                        kind="context",
                        old_line=old_line,
                        new_line=new_line,
                        text="",
                    ),
                )
            continue
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("+"):
            source_rows.append(
                _DiffPreviewRow(
                    kind="add",
                    old_line=None,
                    new_line=new_line,
                    text=line[1:],
                )
            )
            new_line += 1
        elif line.startswith("-"):
            source_rows.append(
                _DiffPreviewRow(
                    kind="delete",
                    old_line=old_line,
                    new_line=None,
                    text=line[1:],
                )
            )
            old_line += 1
        elif line.startswith(" "):
            source_rows.append(
                _DiffPreviewRow(
                    kind="context",
                    old_line=old_line,
                    new_line=new_line,
                    text=line[1:],
                )
            )
            old_line += 1
            new_line += 1

    return _limit_diff_preview_rows(source_rows, max_changes=max_changes)


def _limit_diff_preview_rows(
    rows: Sequence[_DiffPreviewRow], *, max_changes: int | None
) -> list[_DiffPreviewRow]:
    if max_changes is None:
        return list(rows)

    limited: list[_DiffPreviewRow] = []
    shown_changes = 0
    omitted_changes = 0
    for row in rows:
        if row.kind not in {"add", "delete"}:
            if omitted_changes:
                limited.append(_omitted_diff_row(omitted_changes))
                omitted_changes = 0
            limited.append(row)
            continue
        if shown_changes >= max_changes:
            omitted_changes += 1
            continue
        if omitted_changes:
            limited.append(_omitted_diff_row(omitted_changes))
            omitted_changes = 0
        limited.append(row)
        shown_changes += 1
    if omitted_changes:
        limited.append(_omitted_diff_row(omitted_changes))
    return limited


def _omitted_diff_row(count: int) -> _DiffPreviewRow:
    return _DiffPreviewRow(
        kind="meta",
        old_line=None,
        new_line=None,
        text=f"... +{count} changed lines",
    )


def _append_gap_marker_if_needed(
    rows: list[_DiffPreviewRow],
    current: _DiffPreviewRow,
) -> None:
    previous = _last_numbered_diff_row(rows)
    if previous is not None and _needs_diff_context_marker(previous, current):
        rows.append(
            _DiffPreviewRow(
                kind="meta",
                old_line=None,
                new_line=None,
                text=_DIFF_CONTEXT_MARKER_ROW,
            )
        )


def _last_numbered_diff_row(rows: Sequence[_DiffPreviewRow]) -> _DiffPreviewRow | None:
    for row in reversed(rows):
        if row.old_line is not None or row.new_line is not None:
            return row
    return None


def _needs_diff_context_marker(
    previous: _DiffPreviewRow | None,
    current: _DiffPreviewRow,
) -> bool:
    if previous is None:
        return False
    previous_old = previous.old_line
    current_old = current.old_line
    previous_new = previous.new_line
    current_new = current.new_line
    old_gap = (
        previous_old is not None
        and current_old is not None
        and current_old > previous_old + 1
    )
    new_gap = (
        previous_new is not None
        and current_new is not None
        and current_new > previous_new + 1
    )
    return old_gap or new_gap


def _format_diff_preview_rows(
    rows: Sequence[_DiffPreviewRow],
) -> list[tuple[_DiffPreviewRow, str]]:
    formatted: list[tuple[_DiffPreviewRow, str]] = []
    for row in rows:
        if row.kind == "meta":
            formatted.append((row, row.text))
            continue
        line_number = _display_diff_line_number(row)
        if row.kind == "add":
            formatted.append((row, f"{line_number:>5} +{row.text}"))
        elif row.kind == "delete":
            formatted.append((row, f"{line_number:>5} -{row.text}"))
        else:
            formatted.append((row, f"{line_number:>5}  {row.text}"))
    return formatted


def _display_diff_line_number(row: _DiffPreviewRow) -> int:
    if row.kind == "add":
        return row.new_line or 0
    if row.kind == "delete":
        return row.old_line or 0
    return row.new_line or row.old_line or 0


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


DEFAULT_STATUSLINE_FIELDS = DEFAULT_TUI_STATUSLINE_FIELDS

STATUSLINE_FIELD_DESCRIPTIONS = (
    ("model", "Current model"),
    ("thinking", "Thinking level / effort"),
    ("context_used", "Context tokens used"),
    ("context_remaining", "Context tokens remaining"),
    ("context_size", "Total input token limit"),
    ("input_tokens", "Current input tokens"),
    ("output_tokens", "Output tokens"),
    ("cached_tokens", "Cached input tokens"),
    ("cwd", "Current working directory"),
    ("quota_5h", "5 hour quota limit"),
    ("quota_1w", "1 week subscription limit"),
)

STATUSLINE_FIELDS = tuple(field for field, _description in STATUSLINE_FIELD_DESCRIPTIONS)

_STATUSLINE_FIELD_ALIASES = {
    "context": "context_used",
    "context_usage": "context_used",
    "context_remaining": "context_remaining",
    "context_left": "context_remaining",
    "context_size": "context_size",
    "context_window": "context_size",
    "total_input_token_limit": "context_size",
    "current_input_tokens": "input_tokens",
    "input": "input_tokens",
    "cached": "cached_tokens",
    "output": "output_tokens",
    "working_directory": "cwd",
    "current_working_directory": "cwd",
    "5_hour_quota_limit": "quota_5h",
    "5_hour_limit": "quota_5h",
    "five_hour_quota_limit": "quota_5h",
    "1_week_limit": "quota_1w",
    "one_week_limit": "quota_1w",
    "weekly_limit": "quota_1w",
    "thinking_level": "thinking",
}


def _normalize_statusline_fields(fields: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if fields is None:
        return DEFAULT_STATUSLINE_FIELDS
    normalized: list[str] = []
    for field in fields:
        key = field.strip().lower().replace("-", "_").replace(" ", "_")
        key = _STATUSLINE_FIELD_ALIASES.get(key, key)
        if key and key not in normalized:
            normalized.append(key)
    return tuple(normalized)


def _render_statusline(
    *,
    model: str,
    context_tokens: int | None,
    context_window: int | None,
    input_tokens: int,
    cached_tokens: int,
    output_tokens: int,
    cwd: str,
    thinking: bool = False,
    effort: str | None = None,
    quota_5h_remaining_percent: int | None = None,
    quota_1w_remaining_percent: int | None = None,
    fields: tuple[str, ...] | list[str] | None = None,
) -> str:
    context_tokens = context_tokens or 0
    status_fields = _normalize_statusline_fields(fields)
    default_shape = status_fields == DEFAULT_STATUSLINE_FIELDS

    if context_window is None:
        context_segment = "Context unknown"
        context_size_segment = "window: unknown"
        context_remaining_segment = "remaining: unknown"
    else:
        percent = (context_tokens / context_window) * 100 if context_window else 0
        remaining = max(context_window - context_tokens, 0)
        context_segment = f"Context {percent:.1f}% used"
        context_size_segment = f"window: {_format_tokens(context_window)}"
        context_remaining_segment = f"remaining: {_format_tokens(remaining)}"

    values: dict[str, str | None] = {
        "model": model,
        "thinking": f"thinking: {effort}" if thinking and effort else "thinking: off",
        "context_used": context_segment,
        "context_remaining": context_remaining_segment,
        "context_size": context_size_segment,
        "input_tokens": f"input: {_format_tokens(input_tokens)}",
        "cached_tokens": f"cached total: {_format_tokens(cached_tokens)}",
        "output_tokens": f"output: {_format_tokens(output_tokens)}",
        "cwd": cwd,
        "quota_5h": _format_quota_segment(
            "5h",
            quota_5h_remaining_percent,
            unknown="5h quota: unknown",
        ),
        "quota_1w": _format_quota_segment(
            "weekly",
            quota_1w_remaining_percent,
            unknown="1w limit: unknown",
        ),
    }

    parts: list[str] = []
    for field in status_fields:
        if (
            default_shape
            and field in {"input_tokens", "cached_tokens", "output_tokens"}
            and input_tokens <= 0
            and output_tokens <= 0
        ):
            continue
        value = values.get(field)
        if value is not None:
            parts.append(value)
    return " | ".join(parts)


def _format_quota_segment(
    label: str,
    remaining_percent: int | None,
    *,
    unknown: str,
) -> str:
    if remaining_percent is None:
        return unknown
    return f"{label} {max(0, min(100, remaining_percent))}%"


def _cached_tokens_from_usage(usage: dict[str, int]) -> int:
    for key in ("cached_tokens", "cache_read_input_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return 0


def _optional_percent(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return None


def _style_statusline_text(text: str) -> str:
    """Apply statusline segment colors without changing visible text."""
    separator = " | "
    first_sep = text.find(separator)
    if first_sep > 0:
        text = (
            f"{STATUSLINE_MODEL_STYLE}{text[:first_sep]}{STATUSLINE_STYLE}"
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
        f"{STATUSLINE_TOKEN_STYLE}{text[context_start:cwd_start]}{STATUSLINE_STYLE}"
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
        text = " No models available. Add provider auth to ~/.wattle/auth.json."
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


def _render_statusline_selector_rows(
    *,
    selected_fields: set[str],
    selected_index: int,
    width: int,
    styles_enabled: bool,
) -> list[str]:
    inner_width = max(1, width - 2)
    title = " Configure statusline "
    top = f"╭─{title}{'─' * max(0, inner_width - len(title) - 1)}╮"[:width].ljust(width)
    bottom = f"╰{'─' * inner_width}╯"[:width].ljust(width)
    content_rows = [
        "Use ↑/↓ to move, x to select/deselect, Enter to save, Esc to cancel",
        "",
    ]
    field_width = max(len(field) for field in STATUSLINE_FIELDS)
    field_rows: list[str] = []
    for index, (field, description) in enumerate(STATUSLINE_FIELD_DESCRIPTIONS):
        marker = ">" if index == selected_index else " "
        checked = "x" if field in selected_fields else " "
        field_rows.append(f" {marker} [{checked}] {field:<{field_width}}  {description}")

    def boxed(row: str) -> str:
        return f"│ {row[: max(0, inner_width - 2)].ljust(max(0, inner_width - 2))} │"[
            :width
        ].ljust(width)

    rows = [top, *(boxed(row) for row in content_rows), *(boxed(row) for row in field_rows), bottom]
    rendered: list[str] = []
    for index, line in enumerate(rows):
        field_index = index - 3
        is_selected = field_index == selected_index
        if styles_enabled and is_selected:
            rendered.append(f"{SELECTED_ROW_STYLE}{line}{RESET}")
        elif styles_enabled:
            rendered.append(f"{STATUS_STYLE}{line}{RESET}")
        else:
            rendered.append(line)
    return rendered


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


@dataclass(slots=True)
class _BashExecCell:
    tool_use_id: str
    command: str
    output: str = ""
    running: bool = True
    is_error: bool = False


@dataclass(frozen=True)
class _RenderedTextLine:
    text: str
    style: str
    ansi_text: str | None = None


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
    preview = entry.preview or metadata.title or _first_user_text(entry.record) or "(untitled)"
    cwd = _short_session_cwd(metadata.cwd)
    return (
        f"{_short_session_id(entry):<12} "
        f"{entry.record.settings.model:<18} "
        f"{len(entry.record.messages):>3} msgs  "
        f"{metadata.updated_at}  "
        f"{cwd}  "
        f"{preview}"
    )


def _short_session_cwd(cwd: str | None) -> str:
    if not cwd:
        return "-"
    path = Path(cwd).expanduser()
    name = path.name
    return name or str(path)


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
    if isinstance(error, TransientProviderError):
        text = str(error).strip() or "temporary provider error"
        return (
            f"Temporary provider error after retries: {text}\n"
            "Conversation history and completed tool results were kept. "
            "Send another message to retry."
        )
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


def _flower_working_status(elapsed_seconds: int) -> tuple[Flower, str]:
    flower = flower_for_elapsed(elapsed_seconds)
    return flower, f"{flower.shape} {flower.verb}..."


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+(.+)$")
_NUMBERED_RE = re.compile(r"^(\s*)(\d+[.)])\s+(.+)$")
_LINK_RE = re.compile(r"!?\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_STRONG_RE = re.compile(r"(\*\*|__)(.+?)\1")
_EMPHASIS_RE = re.compile(
    r"(?<!\*)\*([^*\n]+)\*(?!\*)|(?<![\w_])_([^_\n]+)_(?![\w_])"
)
_STRIKE_RE = re.compile(r"~~(.+?)~~")
_CODE_FENCE_RE = re.compile(r"^\s*(```+|~~~+)\s*([^`]*)$")


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


def _markdown_content_width(width: int) -> int:
    return max(1, width - 1)


def _wrap_markdown_line(
    text: str,
    *,
    width: int,
    style: str,
    first_prefix: str = "",
    subsequent_prefix: str | None = None,
) -> list[_RenderedTextLine]:
    content_width = _markdown_content_width(width)
    subsequent = first_prefix if subsequent_prefix is None else subsequent_prefix
    if not text:
        return [_RenderedTextLine(first_prefix.rstrip(), style)]
    wrapped = textwrap.wrap(
        text,
        width=content_width,
        initial_indent=first_prefix,
        subsequent_indent=subsequent,
        break_long_words=True,
        break_on_hyphens=False,
        drop_whitespace=True,
    )
    return [_RenderedTextLine(line, style) for line in (wrapped or [first_prefix + text])]


def _code_fence_language(info: str) -> str | None:
    language = re.split(r"[\s,]+", info.strip(), maxsplit=1)[0]
    return language or None


def _render_code_line(line: str, *, language: str | None) -> _RenderedTextLine:
    text = f"    {line}"
    if language is None:
        return _RenderedTextLine(text, TOOL_PREVIEW_STYLE)
    spans = _syntax_spans_for_language(line, language, base_style=TOOL_PREVIEW_STYLE)
    if len(spans) == 1 and spans[0] == (TOOL_PREVIEW_STYLE, line):
        return _RenderedTextLine(text, TOOL_PREVIEW_STYLE)
    ansi_text = f"{TOOL_PREVIEW_STYLE}    " + "".join(
        f"{RESET}{style}{value}" for style, value in spans
    )
    return _RenderedTextLine(text, TOOL_PREVIEW_STYLE, ansi_text=ansi_text)


def _render_markdown_text(text: str, *, width: int) -> list[_RenderedTextLine]:
    rows: list[_RenderedTextLine] = []
    lines = text.splitlines() or [""]
    index = 0
    in_code_fence: tuple[str, int, str | None] | None = None

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()

        fence = _CODE_FENCE_RE.match(line)
        if fence is not None:
            marker, info = fence.groups()
            marker_prefix = marker[0]
            if in_code_fence is None:
                in_code_fence = (marker_prefix, len(marker), _code_fence_language(info))
                index += 1
                continue
            elif (
                marker_prefix == in_code_fence[0]
                and len(marker) >= in_code_fence[1]
                and not info.strip()
            ):
                in_code_fence = None
                index += 1
                continue

        if in_code_fence is not None:
            _marker, _marker_length, language = in_code_fence
            rows.append(_render_code_line(line, language=language))
            index += 1
            continue

        if not stripped:
            rows.append(_RenderedTextLine("", ASSISTANT_STYLE))
            index += 1
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            rows.extend(
                _wrap_markdown_line(
                    _strip_inline_markdown(heading.group(2).strip()),
                    width=width,
                    style=f"{ASSISTANT_STYLE}{BOLD}",
                )
            )
            index += 1
            continue

        if re.fullmatch(r"\s{0,3}([-*_])(?:\s*\1){2,}\s*", line):
            rows.append(
                _RenderedTextLine(
                    "─" * max(3, _markdown_content_width(width)),
                    SEPARATOR_STYLE,
                )
            )
            index += 1
            continue

        quote = re.match(r"^(\s*)((?:>\s*)+)(.*)$", line)
        if quote:
            indent = " " * len(quote.group(1).expandtabs(2))
            depth = quote.group(2).count(">")
            prefix = f"{indent}{'│ ' * depth}"
            rows.extend(
                _wrap_markdown_line(
                    _strip_inline_markdown(quote.group(3).strip()),
                    width=width,
                    style=THINKING_STYLE,
                    first_prefix=prefix,
                    subsequent_prefix=prefix,
                )
            )
            index += 1
            continue

        bullet = _BULLET_RE.match(line)
        if bullet:
            indent = " " * len(bullet.group(1).expandtabs(2))
            rows.extend(
                _wrap_markdown_line(
                    _strip_inline_markdown(bullet.group(2).strip()),
                    width=width,
                    style=ASSISTANT_STYLE,
                    first_prefix=f"{indent}• ",
                    subsequent_prefix=f"{indent}  ",
                )
            )
            index += 1
            continue

        numbered = _NUMBERED_RE.match(line)
        if numbered:
            indent = " " * len(numbered.group(1).expandtabs(2))
            marker = numbered.group(2)
            rows.extend(
                _wrap_markdown_line(
                    _strip_inline_markdown(numbered.group(3).strip()),
                    width=width,
                    style=ASSISTANT_STYLE,
                    first_prefix=f"{indent}{marker} ",
                    subsequent_prefix=f"{indent}{' ' * (len(marker) + 1)}",
                )
            )
            index += 1
            continue

        rows.extend(
            _wrap_markdown_line(
                _strip_inline_markdown(line),
                width=width,
                style=ASSISTANT_STYLE,
            )
        )
        index += 1

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
        if path is not None and _path_is_file(path):
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


def _path_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except (OSError, ValueError):
        return False


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


def _content_has_images(content: list[ContentBlock]) -> bool:
    return any(isinstance(block, ImageBlock) for block in content)


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
            if quote is None and char in {"'", '"'} and (not value or value == ["@"]):
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
    try:
        if parsed.scheme == "file":
            path = Path(unquote(parsed.path)).expanduser()
        elif parsed.scheme:
            return None
        else:
            path = Path(stripped).expanduser()
    except RuntimeError:
        return None
    if not path.is_absolute():
        path = cwd / path
    try:
        return path.resolve()
    except (OSError, RuntimeError, ValueError):
        return None


class WattleApp:
    """Append-only native terminal session for Wattle."""

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
        self.cwd = Path.cwd()
        self._settings = load_settings()

        self.current_provider_name: str = args.provider
        self.current_model: str = args.model
        self.max_tokens: int = args.max_tokens
        self.thinking: bool = bool(getattr(args, "thinking", False))
        self.show_thinking_content: bool = bool(self._settings.tui.show_thinking)
        self.effort: str | None = cast(str | None, getattr(args, "effort", None))
        self.enabled_models: tuple[str, ...] = tuple(
            cast(tuple[str, ...], getattr(args, "enabled_models", ()))
        )
        self.compaction_keep_recent_tokens: int = int(
            getattr(args, "compaction_keep_recent_tokens", 20_000)
        )
        self.messages: list[Message] = []
        self.runtime = DEFAULT_RUNTIME
        self.tool_specs = [tool.spec() for tool in self._available_tools().values()]
        self.permission_mode = PermissionMode.YOLO
        self.permission_gate = PermissionGate()
        self.system: str | None = build_system_prompt(
            tools_by_name=self._available_tools(),
            skills=load_available_skills(Path.cwd()),
            permission_mode=self.permission_mode,
        )

        self._statusline_fields = _normalize_statusline_fields(
            cast(tuple[str, ...] | list[str] | None, getattr(args, "statusline_fields", None))
        )
        self._statusline_enabled = bool(self._statusline_fields)
        self._last_context_tokens: int | None = None
        self._total_input_tokens = 0
        self._total_cached_tokens = 0
        self._total_output_tokens = 0
        self._quota_5h_remaining_percent: int | None = None
        self._quota_1w_remaining_percent: int | None = None
        self._session_record: SessionRecord | None = None
        self._session_path: Path | None = None
        self._force_plain = input_func is not None or out is not None
        self._last_transcript_was_separator = False
        self._resume_history_pending = False
        self._compaction_state: RuntimeCompaction | None = None
        self._cleared_empty_screen_active = False
        self._clear_screen_notice: str | None = None
        self._research_run_seen: set[CommandSummary] = set()

        if state is not None:
            self._restore_state(state)
        self._coerce_effort_for_current_model()
        self._prefetch_startup_quota()

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
        return asyncio.run(self._arun_basic())

    async def _arun_basic(self) -> int:
        """Async implementation of the portable append-only loop."""
        self._write_welcome_card()
        self._write_resume_history_if_pending()
        await self._acontinue_resumed_turn_if_needed()
        positional_prompt = self._positional_prompt_text()
        if positional_prompt and self._submit_user_text(positional_prompt, render=True):
            await self._arun_turn_recovering()
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
                if await self._ahandle_slash(text):
                    break
                continue

            if self._submit_user_text(text, render=True):
                await self._arun_turn_recovering()

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
        omitted_images = _content_has_images(content) and not self._current_model_supports_images()
        if render:
            self._cleared_empty_screen_active = False
            self._write_block(
                self._user_display_text(
                    text,
                    content,
                    prefer_content_text=expanded_text is None,
                ),
                USER_STYLE,
            )
            if omitted_images:
                self._write_unsupported_image_notice()
        if omitted_images:
            content = self._project_content_for_current_model(content)
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

    def _current_model_supports_images(self) -> bool:
        return model_supports_modality(self.current_model, "image")

    def _unsupported_image_notice_text(self) -> str:
        return (
            f"Images were not sent because model {self.current_model} "
            "does not support image inputs."
        )

    def _write_unsupported_image_notice(self) -> None:
        self._write_panel("notice", self._unsupported_image_notice_text(), STATUS_STYLE)

    def _project_content_for_current_model(
        self,
        content: list[ContentBlock],
    ) -> list[ContentBlock]:
        message = Message(role="user", content=content)
        return project_messages_for_model_modalities(
            [message],
            model=self.current_model,
        )[0].content

    def _available_tools(self) -> dict[str, Tool]:
        if self._current_model_supports_images():
            return dict(TOOLS_BY_NAME)
        return {name: tool for name, tool in TOOLS_BY_NAME.items() if name != "view_image"}

    def _refresh_model_dependent_context(self) -> None:
        tools_by_name = self._available_tools()
        self.tool_specs = [tool.spec() for tool in tools_by_name.values()]
        self.system = build_system_prompt(
            tools_by_name=tools_by_name,
            skills=load_available_skills(Path.cwd()),
            permission_mode=self.permission_mode,
        )

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
        return self._write_image_asset(data, path.suffix.lower())

    def _save_clipboard_image_asset(self, image: ClipboardImage) -> Path:
        if len(image.data) > MAX_IMAGE_ATTACHMENT_BYTES:
            raise ValueError(
                f"Clipboard image is too large ({_format_bytes(len(image.data))}; "
                f"max {_format_bytes(MAX_IMAGE_ATTACHMENT_BYTES)})"
            )
        return self._write_image_asset(image.data, image.extension)

    def _write_image_asset(self, data: bytes, suffix: str) -> Path:
        digest = hashlib.sha256(data).hexdigest()[:16]
        safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        if self._session_record is not None:
            asset_dir = default_session_dir() / "assets" / self._session_record.metadata.id
            target = asset_dir / f"{digest}{safe_suffix.lower()}"
        else:
            asset_dir = self.cwd / ".wattle" / "clipboard-images"
            target = asset_dir / f"clipboard-{digest}{safe_suffix.lower()}"
        asset_dir.mkdir(parents=True, exist_ok=True)
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
            "show_thinking_content": self.show_thinking_content,
            "effort": self.effort,
            "permission_mode": self.permission_mode,
            "enabled_models": self.enabled_models,
            "compaction_keep_recent_tokens": self.compaction_keep_recent_tokens,
            "messages": list(self.messages),
            "compaction_state": self._compaction_state,
            "statusline_enabled": self._statusline_enabled,
            "statusline_fields": self._statusline_fields,
            "last_context_tokens": self._last_context_tokens,
            "total_input_tokens": self._total_input_tokens,
            "total_cached_tokens": self._total_cached_tokens,
            "total_output_tokens": self._total_output_tokens,
            "quota_5h_remaining_percent": self._quota_5h_remaining_percent,
            "quota_1w_remaining_percent": self._quota_1w_remaining_percent,
        }

    def _restore_state(self, state: dict[str, object]) -> None:
        self.current_provider_name = str(
            state.get("current_provider_name", self.current_provider_name)
        )
        self.current_model = str(state.get("current_model", self.current_model))
        self.system = cast(str | None, state.get("system", self.system))
        self._refresh_model_dependent_context()
        self.max_tokens = int(cast(Any, state.get("max_tokens", self.max_tokens)))
        self.thinking = bool(state.get("thinking", self.thinking))
        self.show_thinking_content = bool(
            state.get("show_thinking_content", self.show_thinking_content)
        )
        self.effort = cast(str | None, state.get("effort", self.effort))
        self.permission_mode = PermissionMode.YOLO
        self.enabled_models = tuple(
            cast(tuple[str, ...], state.get("enabled_models", self.enabled_models))
        )
        self.compaction_keep_recent_tokens = int(
            cast(
                Any,
                state.get(
                    "compaction_keep_recent_tokens",
                    self.compaction_keep_recent_tokens,
                ),
            )
        )
        self.permission_gate = PermissionGate()
        self.messages = list(cast(list[Message], state.get("messages", self.messages)))
        self._compaction_state = cast(
            RuntimeCompaction | None,
            state.get("compaction_state"),
        )
        self._statusline_enabled = bool(
            state.get("statusline_enabled", self._statusline_enabled)
        )
        self._statusline_fields = _normalize_statusline_fields(
            cast(
                tuple[str, ...] | list[str] | None,
                state.get("statusline_fields", self._statusline_fields),
            )
        )
        self._statusline_enabled = bool(self._statusline_fields) and self._statusline_enabled
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
        self._quota_5h_remaining_percent = _optional_percent(
            state.get("quota_5h_remaining_percent", self._quota_5h_remaining_percent)
        )
        self._quota_1w_remaining_percent = _optional_percent(
            state.get("quota_1w_remaining_percent", self._quota_1w_remaining_percent)
        )

    def _run_turn(self, *, started_at: float | None = None) -> None:
        asyncio.run(self._arun_turn(started_at=started_at))

    async def _arun_turn(self, *, started_at: float | None = None) -> None:
        if started_at is None:
            started_at = time.monotonic()
        self._configure_subagents()
        preparer = self._request_preparer()
        while True:
            response = await self._adrive_stream_with_recovery(preparer)
            self._record_usage(response)

            has_tool_uses = any(isinstance(block, ToolUseBlock) for block in response.content)
            if has_tool_uses:
                self._write_separator()
                self._append_assistant_response(response)
                tool_results = (
                    await self._adispatch_tools(response)
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
        asyncio.run(self._arun_turn_recovering())

    async def _arun_turn_recovering(self) -> None:
        started_at = time.monotonic()
        try:
            await self._arun_turn(started_at=started_at)
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
        on_stream_retry: Callable[[int, int, BaseException, float], None] | None = None,
    ) -> RequestPreparer:
        self._coerce_effort_for_current_model()
        self._refresh_model_dependent_context()
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
            compaction_keep_recent_tokens=self.compaction_keep_recent_tokens,
            provider_context_tokens=lambda: self._last_context_tokens,
            on_stream_retry=on_stream_retry or self._write_stream_retry_status,
        )

    def _configure_subagents(self, *, provider: Provider | None = None) -> None:
        self._coerce_effort_for_current_model()
        tools_by_name = self._available_tools()
        self.runtime.subagents.configure(
            provider=provider or self.provider,
            tools_by_name=tools_by_name,
            system=self.system,
            model=self.current_model,
            full_tools_by_name=TOOLS_BY_NAME,
            max_tokens=self.max_tokens,
            permission_gate=self.permission_gate,
            context_window=_context_window_for_model(self.current_model),
            thinking=self.thinking,
            effort=cast(Any, self.effort),
        )

    def _write_auto_compacting_status(self) -> None:
        self._write_panel("status", "Auto-compacting...", STATUS_STYLE)

    def _write_stream_retry_status(
        self,
        attempt: int,
        max_retries: int,
        _exc: BaseException,
        _delay: float,
    ) -> None:
        self._write_panel(
            "status",
            f"Reconnecting... {attempt}/{max_retries}",
            STATUS_STYLE,
        )

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

        previous_retry_callback = preparer.on_stream_retry

        def on_retry(
            attempt: int,
            max_retries: int,
            exc: BaseException,
            delay: float,
        ) -> None:
            nonlocal in_text, wrote_content
            text_buffer.clear()
            in_text = False
            wrote_content = False
            if previous_retry_callback is not None:
                previous_retry_callback(attempt, max_retries, exc, delay)

        preparer.on_stream_retry = on_retry
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
                    if not self.show_thinking_content:
                        continue
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
            preparer.on_stream_retry = previous_retry_callback
            self._compaction_state = preparer.state

        flush_text_buffer()
        if wrote_content:
            self._write(f"{RESET}\n" if self._styles_enabled() else "\n")
        if final_response is None:
            raise RuntimeError("provider stream ended without StreamComplete")
        return final_response

    async def _adispatch_tools(self, response: CompletionResponse) -> list[ContentBlock]:
        results: list[ContentBlock] = []
        pending_research: list[CommandSummary] = []
        pending_edit_group: _EditRenderGroup | None = None

        def flush_research() -> None:
            if pending_research:
                self._write_research_result(pending_research)
                pending_research.clear()

        def flush_edit_group() -> None:
            nonlocal pending_edit_group
            if pending_edit_group is not None:
                self._write_edit_result_group(pending_edit_group)
                pending_edit_group = None

        for block in response.content:
            if not isinstance(block, ToolUseBlock):
                continue
            blocks = await dispatch_tool_blocks_async(
                block,
                self._available_tools(),
                self.permission_gate,
            )
            result = _first_tool_result(block, blocks)
            edit_item = _edit_render_item(block, result)
            summary = _research_summary_for_success(block, result)
            if edit_item is not None:
                flush_research()
                if pending_edit_group is None:
                    pending_edit_group = _EditRenderGroup(
                        path=edit_item.path,
                        items=[edit_item],
                    )
                elif pending_edit_group.path == edit_item.path:
                    pending_edit_group.items.append(edit_item)
                else:
                    flush_edit_group()
                    pending_edit_group = _EditRenderGroup(
                        path=edit_item.path,
                        items=[edit_item],
                    )
            elif summary is not None:
                flush_edit_group()
                pending_research.append(summary)
            else:
                flush_edit_group()
                flush_research()
                self._write_tool_result(block, result)
            results.extend(blocks)
        flush_edit_group()
        flush_research()
        return results

    def _write_research_result(self, summaries: list[CommandSummary]) -> None:
        unique_summaries: list[CommandSummary] = []
        for summary in summaries:
            if summary in self._research_run_seen:
                continue
            self._research_run_seen.add(summary)
            unique_summaries.append(summary)
        if not unique_summaries:
            return
        self._last_transcript_was_separator = False
        lines = _research_lines(unique_summaries)
        if not self._styles_enabled():
            self._write_line(f"{TOOL_MARKER} Researched")
            self._write_line(f"  └ {lines[0]}")
            for line in lines[1:]:
                self._write_line(f"    {line}")
            return

        self._write(
            f"{TOOL_MARKER_STYLE}{TOOL_MARKER}{RESET} "
            f"{TOOL_TITLE_STYLE}Researched{RESET}\n"
        )
        self._write(f"{TOOL_PREVIEW_STYLE}  └ {lines[0]}{RESET}\n")
        for line in lines[1:]:
            self._write(f"{TOOL_PREVIEW_STYLE}    {line}{RESET}\n")

    def _write_plan_update(self, update: PlanUpdate) -> None:
        self._research_run_seen.clear()
        self._last_transcript_was_separator = False
        width = self._terminal_width()

        rows: list[tuple[str, str]] = []
        if update.explanation:
            rows.extend(
                (line, TOOL_PREVIEW_STYLE)
                for line in _wrap_terminal_line(f"  {update.explanation}", width)
            )
        for step in update.plan:
            prefix = f"  - {_plan_marker(step.status)} "
            wrapped = _wrap_terminal_line(f"{prefix}{step.step}", width)
            rows.extend((line, _plan_style(step.status)) for line in wrapped)

        if not self._styles_enabled():
            self._write_line(f"{TOOL_MARKER} Updated Plan")
            for line, _style in rows:
                self._write_line(line)
            return

        self._write(
            f"{TOOL_MARKER_STYLE}{TOOL_MARKER}{RESET} "
            f"{TOOL_TITLE_STYLE}Updated Plan{RESET}\n"
        )
        for line, style in rows:
            self._write(f"{style}{line}{RESET}\n")

    def _write_tool_result(
        self,
        block: ToolUseBlock,
        result: ToolResultBlock,
    ) -> None:
        self._research_run_seen.clear()
        self._last_transcript_was_separator = False
        if result.is_error:
            if block.name in {"spawn_agent", "wait_agent", "send_input", "close_agent"}:
                self._write_subagent_tool_error(block, result)
                return
            return
        plan_update = _plan_update_for_success(block, result)
        if plan_update is not None:
            self._write_plan_update(plan_update)
            return
        if block.name in {"write", "edit"} and not result.is_error:
            self._write_edit_result(block, result)
            return
        if block.name == "spawn_agent" and not result.is_error:
            self._write_spawn_agent_result(block, result)
            return
        if block.name == "wait_agent" and not result.is_error:
            self._write_wait_agent_result(block, result)
            return
        if block.name == "send_input" and not result.is_error:
            self._write_send_input_result(block, result)
            return
        if block.name == "close_agent" and not result.is_error:
            self._write_close_agent_result(block, result)
            return
        if block.name == "bash":
            self._write_bash_exec_result(block, result)
            return
        title = _tool_action_title(block, is_error=result.is_error)
        preview_content = result.content
        if block.name == "bash":
            preview_content = _bash_preview_content(result.content)
            preview = _compact_head_tail_lines(preview_content, max_width=110)
        else:
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
        if block.name == "bash":
            command = _tool_arg(block, "command", "<missing command>")
            rendered_title = (
                f"{title_style}Ran{RESET} {_render_shell_command(command)}"
            )
        else:
            rendered_title = f"{title_style}{title}{RESET}"
        self._write(f"{marker_style}{TOOL_MARKER}{RESET} {rendered_title}\n")
        for line in preview:
            self._write(f"{TOOL_PREVIEW_STYLE}  {line}{RESET}\n")

    def _write_bash_exec_result(
        self,
        block: ToolUseBlock,
        result: ToolResultBlock,
    ) -> None:
        cell = _BashExecCell(
            tool_use_id=block.id,
            command=_tool_arg(block, "command", "<missing command>"),
            output=result.content,
            running=False,
            is_error=result.is_error,
        )
        width = self._terminal_width()
        rows = _bash_exec_cell_plain_rows(cell, width=width)
        if not self._styles_enabled():
            for row in rows:
                self._write_line(row)
            return
        marker_style = ERROR_TEXT_STYLE if result.is_error else TOOL_MARKER_STYLE
        title_style = ERROR_TEXT_STYLE if result.is_error else TOOL_TITLE_STYLE
        for index, row in enumerate(rows):
            if index == 0:
                prefix = f"{TOOL_MARKER} Ran "
                command = row[len(prefix) :]
                rendered_command = _render_shell_command(command) or command
                self._write(
                    f"{marker_style}{TOOL_MARKER}{RESET} "
                    f"{title_style}Ran{RESET} {rendered_command}\n"
                )
            else:
                self._write(f"{TOOL_PREVIEW_STYLE}{row}{RESET}\n")

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

    def _write_wait_agent_begin(self, block: ToolUseBlock) -> None:
        snapshots = self._subagent_snapshots_for_tool(block)
        if not snapshots:
            subagent_id = block.input.get("subagent_id")
            snapshots = [{"subagent_id": subagent_id, "role": "subagent"}]
        count = len(snapshots)
        noun = "agent" if count == 1 else "agents"
        title = f"Waiting for {count} {noun}"
        lines = [f"└ {_subagent_label(snapshots[0])}"]
        lines.extend(f"  {_subagent_label(snapshot)}" for snapshot in snapshots[1:])
        self._write_subagent_lifecycle_block(title, lines)

    def _write_wait_agent_result(
        self,
        block: ToolUseBlock,
        result: ToolResultBlock,
    ) -> None:
        snapshots = self._subagent_snapshots_for_tool(block)
        if snapshots:
            lines = []
            for index, snapshot in enumerate(snapshots):
                prefix = "└" if index == 0 else " "
                lines.append(f"{prefix} {_subagent_result_line(snapshot)}")
        else:
            fields = _subagent_summary_fields(result.content)
            lines = [f"└ {_subagent_result_line(fields)}"]
        self._write_subagent_lifecycle_block("Finished waiting", lines)

    def _write_send_input_result(
        self,
        _block: ToolUseBlock,
        result: ToolResultBlock,
    ) -> None:
        fields = _subagent_summary_fields(result.content)
        self._write_subagent_lifecycle_block(
            f"Sent input to {_subagent_label(fields)}",
            [],
        )

    def _write_close_agent_result(
        self,
        _block: ToolUseBlock,
        result: ToolResultBlock,
    ) -> None:
        fields = _subagent_summary_fields(result.content)
        self._write_subagent_lifecycle_block(
            f"Closed {_subagent_label(fields)}",
            [],
        )

    def _write_subagent_tool_error(
        self,
        block: ToolUseBlock,
        result: ToolResultBlock,
    ) -> None:
        title = f"{block.name} error"
        preview = _compact_lines(result.content, max_lines=4, max_width=110) or ["[no output]"]
        if not self._styles_enabled():
            self._write_line(f"{TOOL_MARKER} {title}")
            for line in preview:
                self._write_line(f"  {line}")
            return
        self._write(f"{ERROR_TEXT_STYLE}{TOOL_MARKER}{RESET} {ERROR_TEXT_STYLE}{title}{RESET}\n")
        for line in preview:
            self._write(f"{TOOL_PREVIEW_STYLE}  {line}{RESET}\n")

    def _write_subagent_lifecycle_block(self, title: str, lines: list[str]) -> None:
        if not self._styles_enabled():
            self._write_line(f"{TOOL_MARKER} {title}")
            for line in lines:
                self._write_line(f"  {line}")
            return
        self._write(
            f"{TOOL_MARKER_STYLE}{TOOL_MARKER}{RESET} "
            f"{TOOL_TITLE_STYLE}{title}{RESET}\n"
        )
        for line in lines:
            self._write(f"{TOOL_PREVIEW_STYLE}  {line}{RESET}\n")

    def _subagent_snapshots_for_tool(
        self,
        block: ToolUseBlock,
    ) -> list[dict[str, object]]:
        target_id = str(block.input.get("subagent_id") or "")
        visible = self._visible_subagent_snapshots()
        if target_id and any(snapshot.get("subagent_id") == target_id for snapshot in visible):
            if len(visible) > 1:
                return visible
            return [
                snapshot
                for snapshot in visible
                if snapshot.get("subagent_id") == target_id
            ]
        if not target_id and len(visible) > 1:
            return visible
        if target_id:
            for snapshot in self._subagent_snapshots():
                if snapshot.get("subagent_id") == target_id:
                    return [snapshot]
            return []
        return visible

    def _subagent_snapshots(self) -> list[dict[str, object]]:
        subagents = getattr(self.runtime, "_subagents", None)
        if subagents is None:
            return []
        return list(subagents.snapshots())

    def _visible_subagent_snapshots(self) -> list[dict[str, object]]:
        return _visible_subagent_snapshots(self._subagent_snapshots())

    def _write_edit_result(
        self,
        block: ToolUseBlock,
        result: ToolResultBlock,
    ) -> None:
        item = _edit_render_item(block, result)
        if item is None:
            return
        self._write_edit_result_group(_EditRenderGroup(path=item.path, items=[item]))

    def _write_edit_result_group(self, group: _EditRenderGroup) -> None:
        self._research_run_seen.clear()
        self._last_transcript_was_separator = False
        title = _edit_group_title(group)

        if not self._styles_enabled():
            self._write_line(f"{TOOL_MARKER} {title}")
            for index, item in enumerate(group.items):
                if index > 0:
                    self._write_line(_EDIT_HUNK_SEPARATOR_ROW)
                for _, line in item.rows:
                    self._write_line(line)
            return

        verb_path = title.rsplit(" (", 1)[0]
        verb, _, title_path = verb_path.partition(" ")
        added = sum(item.added for item in group.items)
        deleted = sum(item.deleted for item in group.items)
        self._write(
            f"{TOOL_MARKER_STYLE}{TOOL_MARKER}{RESET} "
            f"{TOOL_TITLE_STYLE}{verb} {RESET}"
            f"{COMMAND_PATH_STYLE}{title_path}{RESET}"
            f"{TOOL_TITLE_STYLE} ({RESET}"
            f"{DIFF_ADD_COUNT_STYLE}+{added}{RESET} "
            f"{DIFF_DELETE_COUNT_STYLE}-{deleted}{RESET}"
            f"{TOOL_TITLE_STYLE}){RESET}\n"
        )
        width = self._terminal_width()
        for index, item in enumerate(group.items):
            if index > 0:
                self._write(
                    f"{_render_diff_row('meta', _EDIT_HUNK_SEPARATOR_ROW, width=width)}\n"
                )
            for kind, line in item.rows:
                self._write(f"{_render_diff_row(kind, line, width=width, path=item.path)}\n")

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
            self._handle_clear()
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
                self._set_statusline_fields(())
                self._write_statusline_update()
            elif rest == "on":
                self._set_statusline_fields(DEFAULT_STATUSLINE_FIELDS)
                self._write_statusline_update()
            else:
                self._write_panel(
                    "statusline",
                    "Interactive statusline configuration is available in the live TUI.",
                    STATUS_STYLE,
                )
            return False
        if cmd == "/model":
            self._handle_model(rest)
            return False
        if cmd == "/effort":
            self._handle_effort(rest)
            return False
        if cmd == "/compact":
            self._handle_compact(rest)
            return False
        self._write_panel("error", f"Unknown command: {cmd}", ERROR_STYLE)
        return False

    async def _ahandle_slash(self, text: str) -> bool:
        cmd, _, rest = text.partition(" ")
        rest = rest.strip()
        if cmd in ("/exit", "/quit"):
            return True
        if cmd == "/clear":
            await self._ahandle_clear()
            return False
        if cmd == "/compact":
            await self._ahandle_compact(rest)
            return False
        return self._handle_slash(text)

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
                f"  wattle -r {parent_session_id}\n"
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
            from wattle.cli import _build_provider

            self.provider = _build_provider(record.settings.provider)
        self.current_provider_name = record.settings.provider
        self.current_model = record.settings.model
        self._refresh_model_dependent_context()
        self.max_tokens = record.settings.max_tokens
        self.thinking = record.settings.thinking
        self.effort = cast(str | None, record.settings.effort)
        self._coerce_effort_for_current_model()
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
        if provider in API_KEY_LOGIN_PROVIDERS:
            _vendor, display_name, _default_model = API_KEY_LOGIN_PROVIDERS[provider]
            try:
                api_key = self.input_func(f"Enter API key for {display_name}: ")
            except EOFError:
                self._write_panel("error", f"{display_name} API key login cancelled.", ERROR_STYLE)
                return
            self._save_api_key_login(provider, api_key)
            return

        if provider != "openai-codex":
            self._write_panel(
                "error",
                f"Unsupported login provider: {provider}",
                ERROR_STYLE,
            )
            return

        callback_timeout_seconds = _login_callback_timeout_seconds()
        running_over_ssh = _running_over_ssh()

        def write_auth_url(url: str) -> None:
            self._write_panel(
                "login",
                f"Open this URL to authenticate OpenAI Codex:\n{url}",
                STATUS_STYLE,
            )
            if running_over_ssh:
                self._write_panel("login", SSH_LOGIN_CALLBACK_HINT, SSH_LOGIN_HINT_STYLE)

        try:
            credential = login_openai_codex(
                on_auth=write_auth_url,
                prompt=self.input_func,
                callback_timeout_seconds=callback_timeout_seconds,
                originator="wattle",
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
        from wattle.cli import _build_provider

        try:
            self.current_provider_name = "openai_codex"
            self.provider = _build_provider(self.current_provider_name)
            self._configure_subagents(provider=self.provider)
            self._persist_user_settings(
                provider=self.current_provider_name,
                model=self.current_model,
            )
            self._persist_session()
        except Exception as exc:  # noqa: BLE001
            self._write_panel("error", f"OpenAI Codex provider reload failed: {exc}", ERROR_STYLE)

    def _save_api_key_login(self, provider: str, api_key: str) -> None:
        login_config = API_KEY_LOGIN_PROVIDERS.get(provider)
        if login_config is None:
            self._write_panel(
                "error",
                f"Unsupported API-key login provider: {provider}",
                ERROR_STYLE,
            )
            return
        vendor, display_name, _default_model = login_config
        try:
            credential = save_api_key_credential(vendor, api_key)
        except Exception as exc:  # noqa: BLE001
            self._write_panel("error", f"{display_name} API key login failed: {exc}", ERROR_STYLE)
            return

        self._write_panel(
            "login",
            f"{display_name} API key saved to {credential.source}.",
            STATUS_STYLE,
        )
        from wattle.cli import _build_provider

        try:
            self.current_provider_name = provider
            self.provider = _build_provider(self.current_provider_name)
            self._configure_subagents(provider=self.provider)
            self._persist_user_settings(
                provider=self.current_provider_name,
                model=self.current_model,
            )
            self._persist_session()
        except Exception as exc:  # noqa: BLE001
            self._write_panel(
                "error",
                f"{display_name} provider reload failed: {exc}",
                ERROR_STYLE,
            )

    def _expand_skill_text(self, text: str) -> str | None:
        command = text.strip().split(maxsplit=1)[0]
        if command in BUILTIN_SLASH_COMMANDS:
            return None
        return expand_skill_invocation(text, Path.cwd())

    def _handle_model(self, rest: str) -> None:
        choices = self._selectable_model_choices()
        verb, _, arg = rest.partition(" ")
        if verb == "next":
            self._cycle_model()
            return
        if verb == "enabled":
            text = "\n".join(self.enabled_models) if self.enabled_models else "(all available)"
            self._write_panel("model", f"Enabled models:\n{text}", STATUS_STYLE)
            return
        if verb in {"enable", "disable"}:
            self._handle_enabled_model_change(verb, arg.strip())
            return
        if not rest:
            self._write(render_model_choices(choices, current_model=self.current_model))
            self._write_line("")
            return

        choice = find_model_choice(rest, choices)
        if choice is None:
            self.current_model = rest
            self._coerce_effort_for_current_model()
            self._refresh_model_dependent_context()
            self._write_panel("model", f"Model set to {self.current_model!r}.", STATUS_STYLE)
            self._persist_user_settings(
                provider=self.current_provider_name,
                model=self.current_model,
                thinking=self.thinking,
                effort=self.effort,
            )
            self._persist_session()
            return
        self._apply_model_choice(choice)

    def _apply_model_choice(self, choice: ModelChoice) -> None:
        self.current_model = choice.model
        if choice.provider != self.current_provider_name:
            from wattle.cli import _build_provider

            self.provider = _build_provider(choice.provider)
            self.current_provider_name = choice.provider
        self._coerce_effort_for_current_model()
        self._refresh_model_dependent_context()
        self._persist_user_settings(
            provider=self.current_provider_name,
            model=self.current_model,
            thinking=self.thinking,
            effort=self.effort,
        )
        self._write_panel(
            "model",
            f"Model set to {self.current_model!r} "
            f"using provider {self.current_provider_name!r}.",
            STATUS_STYLE,
        )
        self._persist_session()

    def _selectable_model_choices(self) -> list[ModelChoice]:
        choices = available_model_choices()
        if not self.enabled_models:
            return choices
        enabled = set(self.enabled_models)
        return [choice for choice in choices if choice.model in enabled]

    def _cycle_model(self) -> None:
        choices = self._selectable_model_choices()
        if not choices:
            self._write_panel("error", "No enabled models are available.", ERROR_STYLE)
            return
        models = [choice.model for choice in choices]
        try:
            index = models.index(self.current_model)
        except ValueError:
            index = -1
        self._apply_model_choice(choices[(index + 1) % len(choices)])

    def _handle_enabled_model_change(self, verb: str, selector: str) -> None:
        if not selector:
            self._write_panel("error", f"Usage: /model {verb} MODEL_OR_NUMBER", ERROR_STYLE)
            return
        all_choices = available_model_choices()
        choice = find_model_choice(selector, all_choices)
        model = choice.model if choice is not None else selector
        enabled = list(self.enabled_models)
        if verb == "enable":
            if model not in enabled:
                enabled.append(model)
            message = f"Enabled model {model!r}."
        else:
            enabled = [item for item in enabled if item != model]
            message = f"Disabled model {model!r}."
        self.enabled_models = tuple(enabled)
        self._persist_user_settings(enabled_models=self.enabled_models)
        self._write_panel("model", message, STATUS_STYLE)

    def _handle_effort(self, rest: str) -> None:
        effort = rest.strip()
        effort_levels = self._effort_levels_for_current_model()
        choices_text = ", ".join((*effort_levels, "off"))
        if not effort:
            current = self.effort if self.thinking and self.effort else "off"
            self._write_panel(
                "effort",
                f"Current effort: {current}\nChoices for {self.current_model}: {choices_text}",
                STATUS_STYLE,
            )
            return
        if effort == "off":
            self.thinking = False
            self.effort = None
        elif effort in effort_levels:
            self.thinking = True
            self.effort = effort
        else:
            self._write_panel(
                "error",
                f"Usage for {self.current_model}: /effort {'|'.join((*effort_levels, 'off'))}",
                ERROR_STYLE,
            )
            return
        self._persist_user_settings(thinking=self.thinking, effort=self.effort)
        self._persist_session()
        self._write_panel(
            "effort",
            f"Effort set to {self.effort or '(provider default)'}.",
            STATUS_STYLE,
        )

    def _effort_levels_for_current_model(self) -> tuple[str, ...]:
        return tuple(effort_levels_for_model(self.current_model))

    def _coerce_effort_for_current_model(self) -> None:
        if not self.thinking or self.effort is None:
            return
        effort_levels = self._effort_levels_for_current_model()
        if self.effort in effort_levels:
            return
        self.effort = effort_levels[-1] if effort_levels else None
        self.thinking = self.effort is not None

    def _cycle_thinking_level(self) -> None:
        effort_levels = self._effort_levels_for_current_model()
        if not effort_levels:
            next_effort: str | None = None
        elif not self.thinking or self.effort not in effort_levels:
            next_effort = effort_levels[0]
        else:
            current_index = effort_levels.index(self.effort)
            next_effort = (
                effort_levels[current_index + 1]
                if current_index + 1 < len(effort_levels)
                else None
            )

        self.thinking = next_effort is not None
        self.effort = next_effort
        self._persist_user_settings(thinking=self.thinking, effort=self.effort)
        self._persist_session()

    def _handle_compact(self, rest: str) -> None:
        if not self.messages:
            self._write_panel("compact", "No conversation history to compact.", STATUS_STYLE)
            return
        preparer = self._request_preparer(
            on_compaction_start=lambda: self._write_panel(
                "compact",
                "Compacting...",
                STATUS_STYLE,
            )
        )
        prepared = asyncio.run(
            preparer.aprepare(
                self.messages,
                force_compaction=True,
                reset_provider=False,
                compaction_instructions=rest or None,
            )
        )
        self._compaction_state = preparer.state
        self._write_panel(
            "compact",
            f"Compacted request projection to {prepared.context_tokens} token estimate.",
            STATUS_STYLE,
        )

    async def _ahandle_compact(self, rest: str) -> None:
        if not self.messages:
            self._write_panel("compact", "No conversation history to compact.", STATUS_STYLE)
            return
        preparer = self._request_preparer(
            on_compaction_start=lambda: self._write_panel(
                "compact",
                "Compacting...",
                STATUS_STYLE,
            )
        )
        prepared = await preparer.aprepare(
            self.messages,
            force_compaction=True,
            reset_provider=False,
            compaction_instructions=rest or None,
        )
        self._compaction_state = preparer.state
        self._write_panel(
            "compact",
            f"Compacted request projection to {prepared.context_tokens} token estimate.",
            STATUS_STYLE,
        )

    def _persist_user_settings(self, **changes: Any) -> None:
        if "tui" not in changes:
            changes["tui"] = TuiSettings(
                statusline=self._statusline_fields,
                show_thinking=self.show_thinking_content,
            )
        self._settings = update_settings(**changes)

    def _set_statusline_fields(self, fields: tuple[str, ...]) -> None:
        self._statusline_fields = _normalize_statusline_fields(list(fields))
        self._statusline_enabled = bool(self._statusline_fields)
        self._persist_user_settings(
            tui=TuiSettings(
                statusline=self._statusline_fields,
                show_thinking=self.show_thinking_content,
            )
        )

    def _write_statusline_update(self) -> None:
        if self._statusline_enabled:
            self._write_panel(
                "statusline",
                "Statusline updated: " + " | ".join(self._statusline_fields),
                STATUS_STYLE,
            )
        else:
            self._write_panel("statusline", "Statusline disabled.", STATUS_STYLE)

    def _print_help(self) -> None:
        self._write_line("Commands:")
        self._write_line("  /branch           Copy conversation into a new session branch.")
        self._write_line("  /compact [notes]  Compact conversation history now.")
        self._write_line("  /effort [level]   Set reasoning effort.")
        self._write_line("  /exit, /quit       Exit Wattle.")
        self._write_line("  /clear             Reset conversation history.")
        self._write_line("  /help              Show this message.")
        self._write_line("  /login [openai-codex] Authenticate OpenAI Codex.")
        self._write_line("  /model [name|#|next] List or switch models.")
        self._write_line("  /resume SESSION    Switch to a saved session.")
        self._write_line("  /session, /status  Show persistence and session status.")
        self._write_line("  /statusline          Configure the bottom statusline.")
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
        self._record_quota_usage(response.usage)

    def _record_quota_usage(self, usage: Mapping[str, int]) -> None:
        quota_5h = _optional_percent(usage.get("quota_5h_remaining_percent"))
        quota_1w = _optional_percent(usage.get("quota_1w_remaining_percent"))
        if quota_5h is not None:
            self._quota_5h_remaining_percent = quota_5h
        if quota_1w is not None:
            self._quota_1w_remaining_percent = quota_1w

    def _prefetch_startup_quota(self) -> None:
        if not isinstance(self.provider, OpenAICodexResponsesProvider):
            return
        try:
            usage = self.provider.fetch_quota_usage()
        except Exception:
            return
        self._record_quota_usage(usage)

    def _handle_clear(self) -> None:
        previous_usage = self._previous_session_usage_text()
        self._persist_session()
        self.messages = []
        self._compaction_state = None
        self._resume_history_pending = False
        self._last_transcript_was_separator = False
        self._last_context_tokens = None
        self._total_input_tokens = 0
        self._total_cached_tokens = 0
        self._total_output_tokens = 0
        self._quota_5h_remaining_percent = None
        self._quota_1w_remaining_percent = None
        asyncio.run(self.provider.areset_conversation())

        if self._session_record is not None:
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

        self._cleared_empty_screen_active = True
        self._clear_screen_notice = previous_usage
        self._write_cleared_session_screen()

    async def _ahandle_clear(self) -> None:
        previous_usage = self._previous_session_usage_text()
        self._persist_session()
        self.messages = []
        self._compaction_state = None
        self._resume_history_pending = False
        self._last_transcript_was_separator = False
        self._last_context_tokens = None
        self._total_input_tokens = 0
        self._total_cached_tokens = 0
        self._total_output_tokens = 0
        self._quota_5h_remaining_percent = None
        self._quota_1w_remaining_percent = None
        await self.provider.areset_conversation()

        if self._session_record is not None:
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

        self._cleared_empty_screen_active = True
        self._clear_screen_notice = previous_usage
        self._write_cleared_session_screen()

    def _write_cleared_session_screen(self) -> None:
        self._last_transcript_was_separator = False
        self._write(VISIBLE_SCREEN_CLEAR)
        self._write(TERMINAL_HISTORY_CLEAR)
        self._write_welcome_card()
        if self._clear_screen_notice is not None:
            self._write_panel("previous session", self._clear_screen_notice, STATUS_STYLE)

    def _previous_session_usage_text(self) -> str | None:
        if self._total_input_tokens <= 0 and self._total_output_tokens <= 0:
            return None
        parts = [
            f"input: {_format_tokens(self._total_input_tokens)}",
            f"cached total: {_format_tokens(self._total_cached_tokens)}",
            f"output: {_format_tokens(self._total_output_tokens)}",
        ]
        if self._last_context_tokens is not None:
            parts.insert(0, f"last context: {_format_tokens(self._last_context_tokens)}")
        return "Last session usage: " + " | ".join(parts)

    def _status_text(self) -> str:
        return _render_statusline(
            model=self.current_model,
            context_tokens=self._last_context_tokens,
            context_window=_context_window_for_model(self.current_model),
            input_tokens=self._total_input_tokens,
            cached_tokens=self._total_cached_tokens,
            output_tokens=self._total_output_tokens,
            cwd=_display_cwd(),
            thinking=self.thinking,
            effort=self.effort,
            quota_5h_remaining_percent=self._quota_5h_remaining_percent,
            quota_1w_remaining_percent=self._quota_1w_remaining_percent,
            fields=self._statusline_fields,
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

    async def _acontinue_resumed_turn_if_needed(self) -> None:
        if _history_ends_with_tool_results(self.messages):
            self._write_panel(
                "resumed",
                "Continuing from saved tool result.",
                STATUS_STYLE,
            )
            await self._arun_turn()

    def _write_history_transcript(self) -> None:
        self._write_history_messages(self.messages, with_separators=True)

    def _write_history_messages(
        self,
        messages: list[Message],
        *,
        with_separators: bool,
    ) -> None:
        suppressed_tool_result_ids: set[str] = set()
        wrote_message = False
        for index, message in enumerate(messages):
            if _history_message_is_fully_suppressed(message, suppressed_tool_result_ids):
                continue
            if with_separators and wrote_message:
                self._write_separator()
            next_message = messages[index + 1] if index + 1 < len(messages) else None
            displayed = self._write_history_message(
                message,
                next_message=next_message,
                suppressed_tool_result_ids=suppressed_tool_result_ids,
            )
            wrote_message = wrote_message or displayed
        if with_separators and wrote_message:
            self._write_separator()

    def _write_history_message(
        self,
        message: Message,
        *,
        next_message: Message | None = None,
        suppressed_tool_result_ids: set[str] | None = None,
    ) -> bool:
        suppressed_ids = (
            suppressed_tool_result_ids
            if suppressed_tool_result_ids is not None
            else set()
        )
        text_parts: list[str] = []
        tool_results: list[ToolResultBlock] = []
        tool_uses: list[ToolUseBlock] = []
        images: list[ImageBlock] = []
        thinking_parts: list[str] = []
        redacted_thinking = 0
        displayed = False
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

        if self.show_thinking_content:
            if thinking_parts:
                self._write_panel("thinking", "\n".join(thinking_parts), THINKING_STYLE)
                displayed = True
            elif redacted_thinking:
                self._write_panel(
                    "thinking",
                    f"[{redacted_thinking} redacted thinking block(s)]",
                    THINKING_STYLE,
                )
                displayed = True

        text = "\n".join(part for part in text_parts if part)
        if images and (message.role != "user" or not text):
            image_text = "\n".join(_image_summary(block) for block in images)
            text = f"{text}\n{image_text}" if text else image_text
        if text:
            self._write_block(text, USER_STYLE if message.role == "user" else ASSISTANT_STYLE)
            displayed = True
        elif not tool_uses and not tool_results and not thinking_parts and not redacted_thinking:
            self._write_panel(message.role, "[empty message]", STATUS_STYLE)
            displayed = True

        pending_research: list[CommandSummary] = []
        pending_edit_group: _EditRenderGroup | None = None

        def flush_research() -> None:
            nonlocal displayed
            if pending_research:
                self._write_research_result(pending_research)
                pending_research.clear()
                displayed = True

        def flush_edit_group() -> None:
            nonlocal pending_edit_group, displayed
            if pending_edit_group is not None:
                self._write_edit_result_group(pending_edit_group)
                pending_edit_group = None
                displayed = True

        def flush_semantic_groups() -> None:
            flush_edit_group()
            flush_research()

        for block in tool_uses:
            result = _matching_tool_result(next_message, block.id)
            if result is None:
                flush_semantic_groups()
                self._write_panel("tool use", _tool_call_summary(block), TOOL_TITLE_STYLE)
                displayed = True
                continue
            if result.is_error:
                flush_semantic_groups()
                self._write_tool_result(block, result)
                suppressed_ids.add(block.id)
                displayed = True
                continue
            plan_update = _plan_update_for_success(block, result)
            if plan_update is not None:
                flush_semantic_groups()
                self._write_plan_update(plan_update)
                suppressed_ids.add(block.id)
                displayed = True
                continue
            edit_item = _edit_render_item(block, result)
            if edit_item is not None:
                flush_research()
                if pending_edit_group is None:
                    pending_edit_group = _EditRenderGroup(
                        path=edit_item.path,
                        items=[edit_item],
                    )
                elif pending_edit_group.path == edit_item.path:
                    pending_edit_group.items.append(edit_item)
                else:
                    flush_edit_group()
                    pending_edit_group = _EditRenderGroup(
                        path=edit_item.path,
                        items=[edit_item],
                    )
                suppressed_ids.add(block.id)
                continue
            summary = _research_summary_for_success(block, result)
            if summary is not None:
                flush_edit_group()
                pending_research.append(summary)
                suppressed_ids.add(block.id)
                continue
            flush_semantic_groups()
            self._write_tool_result(block, result)
            suppressed_ids.add(block.id)
            displayed = True
        flush_semantic_groups()
        for block in tool_results:
            if block.tool_use_id in suppressed_ids:
                continue
            label = "tool error" if block.is_error else "tool result"
            preview = "\n".join(_compact_lines(block.content, max_lines=6, max_width=110))
            style = ERROR_STYLE if block.is_error else TOOL_PREVIEW_STYLE
            self._write_panel(label, preview or "[no output]", style)
            displayed = True
        return displayed

    def _write_welcome_card(self) -> None:
        rows = [
            ("model:", self.current_model),
            ("directory:", _display_cwd()),
        ]
        title = f"{WELCOME_TITLE} {get_wattle_version()}"
        max_text_width = max(0, self._terminal_width() - 5)
        visible_title = _truncate_cell_text(title, max_text_width)
        visible_logo_lines = tuple(
            _truncate_cell_text(line, max_text_width) for line in WATTLE_LOGO_LINES
        )
        terminal_width = self._terminal_width()
        if terminal_width <= 5:
            tiny_title = _truncate_cell_text(title, terminal_width)
            if self._styles_enabled():
                self._write(f"{WELCOME_TITLE_STYLE}{tiny_title}{RESET}\n")
            else:
                self._write_line(tiny_title)
            return

        content_width = max(
            len(visible_title),
            *(len(line) for line in visible_logo_lines),
            *(len(label) + 2 + len(value) for label, value in rows),
        )
        width = min(terminal_width - 2, max(36, content_width + 4))
        inner_width = width - 2

        if not self._styles_enabled():
            self._write_line(f"┌{'─' * inner_width}┐")
            for line in visible_logo_lines:
                self._write_line(f"│ {line.center(inner_width - 1)}│")
            self._write_line(f"│ {visible_title.ljust(inner_width - 1)}│")
            self._write_line(f"│ {' '.ljust(inner_width - 1)}│")
            for label, value in rows:
                body_width = inner_width - 1
                label_text = _truncate_cell_text(f"{label:<10}", body_width)
                value_width = max(0, body_width - len(label_text) - 1)
                visible_value = _truncate_cell_text(value, value_width)
                separator = " " if value_width else ""
                self._write_line(
                    f"│ {label_text}{separator}{visible_value.ljust(value_width)}│"
                )
            self._write_line(f"└{'─' * inner_width}┘")
            return

        self._write(f"{WELCOME_BORDER_STYLE}┌{'─' * inner_width}┐{RESET}\n")
        for line in visible_logo_lines:
            self._write(
                f"{WELCOME_BORDER_STYLE}│{RESET} "
                f"{WELCOME_LOGO_STYLE}{line.center(inner_width - 1)}{RESET}"
                f"{WELCOME_BORDER_STYLE}│{RESET}\n"
            )
        title_padding = inner_width - 1 - len(visible_title)
        self._write(
            f"{WELCOME_BORDER_STYLE}│{RESET} "
            f"{WELCOME_TITLE_STYLE}{visible_title}{RESET}"
            f"{' ' * max(0, title_padding)}"
            f"{WELCOME_BORDER_STYLE}│{RESET}\n"
        )
        self._write(
            f"{WELCOME_BORDER_STYLE}│{RESET}{' ' * inner_width}"
            f"{WELCOME_BORDER_STYLE}│{RESET}\n"
        )
        for label, value in rows:
            body_width = inner_width - 1
            label_text = _truncate_cell_text(f"{label:<10}", body_width)
            value_width = max(0, body_width - len(label_text) - 1)
            if len(value) <= value_width:
                visible_value = value
            elif value_width <= 3:
                visible_value = value[-value_width:] if value_width else ""
            else:
                visible_value = "..." + value[-(value_width - 3) :]
            separator = " " if value_width else ""
            self._write(
                f"{WELCOME_BORDER_STYLE}│{RESET} "
                f"{WELCOME_LABEL_STYLE}{label_text}{RESET}{separator}"
                f"{WELCOME_VALUE_STYLE}{visible_value.ljust(value_width)}{RESET}"
                f"{WELCOME_BORDER_STYLE}│{RESET}\n"
            )
        self._write(f"{WELCOME_BORDER_STYLE}└{'─' * inner_width}┘{RESET}\n")

    def _prompt(self) -> str:
        if not self._statusline_enabled:
            return "> "
        return f"wattle [{self._status_text()}] > "

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
            read_files=list(state.read_files),
            modified_files=list(state.modified_files),
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
        self._research_run_seen.clear()
        self._last_transcript_was_separator = False
        if not self._styles_enabled():
            if style == ASSISTANT_STYLE:
                for row in _render_markdown_text(text, width=self._terminal_width()):
                    self._write_line(row.text)
            else:
                self._write_line(text)
            return

        width = self._terminal_width()
        if style == ASSISTANT_STYLE:
            self._write("\r")
            self._write("\n" * MESSAGE_BLOCK_VERTICAL_PADDING)
            for row in _render_markdown_text(text, width=width):
                if row.ansi_text is not None:
                    self._write(f"\r {row.ansi_text}{RESET}\n")
                else:
                    self._write(f"\r{row.style} {row.text}{RESET}\n")
            self._write("\n" * MESSAGE_BLOCK_VERTICAL_PADDING)
            return

        self._write("\r")
        should_pad = style == USER_STYLE
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
            if row.ansi_text is not None:
                self._write(f"\r {row.ansi_text}{RESET}\n")
                continue
            self._write(
                f"{_styled_transcript_line(body, row.style, fill_remainder=should_pad)}\n"
            )
        if should_pad:
            blank = f"{_styled_terminal_line('', style, width)}\n"
            self._write(blank * MESSAGE_BLOCK_VERTICAL_PADDING)


class _LiveTerminal:
    """Raw-mode prompt that stays active while provider streaming runs."""

    def __init__(self, app: WattleApp) -> None:
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
        self.turn_followup_user_inputs: list[str] = []
        self.pending_monitor_inputs: list[str] = []
        self.interrupted_user_inputs: list[str] = []
        self.compacting = False
        self._last_compaction_frame_at = 0.0
        self._last_running_frame_at = 0.0
        self.working_started_at: float | None = None
        self.last_worked_duration_text: str | None = None
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
        self.inflight_tool_results: list[ToolResultBlock] = []
        self.inflight_tools_by_name: dict[str, Tool] = {}
        self.active_tool_status: str | None = None
        self.active_exec_cells: dict[str, _BashExecCell] = {}
        self._tool_event_queue: queue.Queue[ToolRunEvent] = queue.Queue()
        self.active_turn_id = 0
        self.model_picker_choices: list[ModelChoice] | None = None
        self.model_picker_selected = 0
        self.login_picker_selected = 0
        self.statusline_selector_active = False
        self.statusline_selector_selected = 0
        self.statusline_selector_fields: set[str] = set()
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
                        if self.resize_pending and self.app._cleared_empty_screen_active:
                            self._redraw_cleared_session_screen()
                        else:
                            self._draw_prompt(force_reflow_clear=self.resize_pending)
                        self.resize_pending = False
                if (
                    not self.streaming
                    and self.worker is None
                    and (
                        self.pending_user_inputs
                        or self.turn_followup_user_inputs
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
            attrs = termios.tcgetattr(self.fd)
            attrs[3] &= ~termios.IEXTEN
            termios.tcsetattr(self.fd, termios.TCSADRAIN, attrs)
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
            if self.statusline_selector_active:
                index = self._handle_statusline_selector_input(ch, data, index)
                self._draw_prompt()
                continue
            if ch in ("\r", "\n"):
                self._submit_buffer()
            elif ch == "\t":
                if self.streaming:
                    self._queue_buffer_for_end_of_turn()
                elif not self._complete_selected_hint():
                    self._insert_text(ch)
                self._draw_prompt()
            elif ch == "\x03" or (ch == "\x04" and not self.buffer):
                self.running = False
            elif ch == "\x16":
                self._paste_clipboard_image_or_insert_literal(ch)
                self._draw_prompt()
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
                elif (
                    shift_tab := next(
                        (
                            sequence
                            for sequence in SHIFT_TAB_SEQUENCES
                            if data.startswith(sequence[1:], index)
                        ),
                        None,
                    )
                ):
                    self.app._cycle_thinking_level()
                    index += len(shift_tab) - 1
                    self._draw_prompt()
                elif (
                    ctrl_v := next(
                        (
                            sequence
                            for sequence in CTRL_V_KEY_SEQUENCES[1:]
                            if data.startswith(sequence[1:], index)
                        ),
                        None,
                    )
                ):
                    index += len(ctrl_v) - 1
                    self._paste_clipboard_image_or_insert_literal("\x16")
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

    def _open_statusline_selector(self) -> None:
        self.statusline_selector_active = True
        self.statusline_selector_selected = 0
        self.statusline_selector_fields = set(self.app._statusline_fields)
        self.buffer = ""
        self.cursor = 0
        self.pasted_ranges = []

    def _handle_statusline_selector_input(self, ch: str, data: str, index: int) -> int:
        if ch in ("\r", "\n"):
            selected = tuple(
                field for field in STATUSLINE_FIELDS if field in self.statusline_selector_fields
            )
            self.statusline_selector_active = False
            self._clear_prompt()
            self.app._set_statusline_fields(selected)
            self.app._write_statusline_update()
            return index
        if ch in {"x", "X"}:
            field = STATUSLINE_FIELDS[self.statusline_selector_selected]
            if field in self.statusline_selector_fields:
                self.statusline_selector_fields.remove(field)
            else:
                self.statusline_selector_fields.add(field)
            return index
        if ch == "\x1b":
            if data.startswith("[A", index) or data.startswith("OA", index):
                self._move_statusline_selector(-1)
                return index + 2
            if data.startswith("[B", index) or data.startswith("OB", index):
                self._move_statusline_selector(1)
                return index + 2
            self.statusline_selector_active = False
            self._clear_prompt()
            self.app._write_panel("statusline", "Statusline unchanged.", STATUS_STYLE)
            while index < len(data) and not data[index].isalpha():
                index += 1
            return index + 1 if index < len(data) else index
        return index

    def _move_statusline_selector(self, delta: int) -> None:
        self.statusline_selector_selected = max(
            0,
            min(len(STATUSLINE_FIELDS) - 1, self.statusline_selector_selected + delta),
        )

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
        if len(text) >= PASTE_PLACEHOLDER_MIN_CHARS:
            self.pasted_ranges = _merge_pasted_ranges(
                [*self.pasted_ranges, (start, start + len(text))]
            )

    def _paste_clipboard_image_or_insert_literal(self, fallback: str) -> None:
        image = read_clipboard_image()
        if image is None:
            self._insert_text(fallback)
            return
        try:
            path = self.app._save_clipboard_image_asset(image)
        except ValueError as exc:
            self.app._write_panel("error", str(exc), ERROR_STYLE)
            return
        text = shlex.quote(str(path))
        if self.cursor > 0 and not self.buffer[self.cursor - 1].isspace():
            text = f" {text}"
        if self.cursor < len(self.buffer) and not self.buffer[self.cursor].isspace():
            text = f"{text} "
        self._insert_text(text)

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
        expanded_text = self.app._expand_skill_text(text)
        if expanded_text is None and text == "/statusline":
            self._open_statusline_selector()
            self._draw_prompt()
            return
        if expanded_text is None and _should_route_slash_command(text):
            if self._handle_live_queue_command(text):
                return
            should_exit = self.app._handle_slash(text)
            if should_exit:
                self.running = False
                return
            if text.partition(" ")[0] == "/clear":
                self.input_history = _input_history_from_messages(self.app.messages)
                self.input_history_index = None
                self.input_history_draft = ""
            self._draw_prompt()
            return
        if self.streaming:
            self.pending_user_inputs.append(text)
            self._draw_prompt()
            return

        message_text = expanded_text or text
        try:
            message_content = self.app._user_content_blocks(message_text)
        except ValueError as exc:
            self.app._write_panel("error", str(exc), ERROR_STYLE)
            self._draw_prompt()
            return
        self.last_worked_duration_text = None
        self.app._cleared_empty_screen_active = False
        self.app._write_block(
            self.app._user_display_text(
                text,
                message_content,
                prefer_content_text=expanded_text is None,
            ),
            USER_STYLE,
        )
        omitted_images = (
            _content_has_images(message_content)
            and not self.app._current_model_supports_images()
        )
        if omitted_images:
            self.app._write_unsupported_image_notice()
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
        if omitted_images:
            content = self.app._project_content_for_current_model(content)
        self.app.messages.append(Message(role="user", content=content))
        self.app._persist_session()
        self._start_worker()
        self._draw_prompt()

    def _queue_buffer_for_end_of_turn(self) -> None:
        text = self.buffer.strip()
        self.buffer = ""
        self.cursor = 0
        self.pasted_ranges = []
        if not text:
            return
        self._record_input_history(text)
        self.turn_followup_user_inputs.append(text)

    def _redraw_cleared_session_screen(self) -> None:
        self.app._write_cleared_session_screen()
        self._reset_prompt_state()
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
                "No models available. Add provider auth to ~/.wattle/auth.json.",
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
        if self.pending_user_inputs or self.turn_followup_user_inputs:
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
        self.inflight_tool_results = []
        self.inflight_tools_by_name = {}
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
        pending = [*self.pending_user_inputs, *self.turn_followup_user_inputs]
        monitor_pending = self.pending_monitor_inputs
        self.interrupted_user_inputs = []
        self.pending_user_inputs = []
        self.turn_followup_user_inputs = []
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
        if (
            not self.streaming
            or not (self.pending_user_inputs or self.turn_followup_user_inputs)
        ):
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
        self.inflight_tool_results = []
        self._append_interrupted_user_message(render=True)
        self._reset_provider_for_interrupt()
        self._start_worker()
        self._draw_prompt()

    def _append_interrupted_user_message(self, *, render: bool) -> None:
        pending = [*self.pending_user_inputs, *self.turn_followup_user_inputs]
        monitor_pending = self.pending_monitor_inputs
        self.pending_user_inputs = []
        self.turn_followup_user_inputs = []
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
            omitted_images = (
                _content_has_images(content)
                and not self.app._current_model_supports_images()
            )
            prepared_pending.append(_first_text_block_text(content, fallback=text))
            if omitted_images:
                projected = self.app._project_content_for_current_model(content)
                attachment_blocks.extend(
                    block for block in projected[1:] if not isinstance(block, ImageBlock)
                )
            else:
                attachment_blocks.extend(content[1:])
            if render:
                self.app._write_block(self.app._user_display_text(text, content), USER_STYLE)
                if omitted_images:
                    self.app._write_unsupported_image_notice()
        return prepared_pending, attachment_blocks

    def _handle_live_queue_command(self, text: str) -> bool:
        cmd, _, rest = text.partition(" ")
        if cmd != "/queue":
            return False
        queued = rest.strip()
        if not queued:
            self.app._write_panel(
                "error",
                "Usage: /queue <message>",
                ERROR_STYLE,
            )
            self._draw_prompt()
            return True
        if not self.streaming:
            self.app._write_panel(
                "error",
                "/queue is only available while an assistant turn is streaming.",
                ERROR_STYLE,
            )
            self._draw_prompt()
            return True
        self.turn_followup_user_inputs.append(queued)
        self._draw_prompt()
        return True

    def _take_trailing_text_user_message(self) -> list[str]:
        if not self.app.messages or self.app.messages[-1].role != "user":
            return []
        message = self.app.messages[-1]
        if not all(isinstance(block, TextBlock) for block in message.content):
            return []
        self.app.messages.pop()
        return [cast(TextBlock, block).text for block in message.content]

    def _reset_provider_for_interrupt(self) -> None:
        from wattle.cli import _build_provider

        self.app.provider = _build_provider(self.app.current_provider_name)

    def _start_worker(self) -> None:
        self.streaming = True
        self.last_worked_duration_text = None
        if self.turn_started_at is None:
            self.turn_started_at = time.monotonic()
        self.working_started_at = time.monotonic()
        self.active_turn_id += 1
        self.in_text = False
        self.in_thinking = False
        self.seen_tools = set()
        self.inflight_tool_results = []
        self._clear_stream_buffers()
        turn_id = self.active_turn_id
        provider = self.app.provider
        tools_by_name = self.app._available_tools()
        self.inflight_tools_by_name = tools_by_name
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
            on_stream_retry=lambda attempt, max_retries, exc, delay: self.events.put(
                (turn_id, "stream_retry", (attempt, max_retries, exc, delay))
            ),
        )
        messages = list(self.app.messages)
        try:
            final: CompletionResponse | None = None
            async for event in astream_with_recovery(preparer, messages):
                self.events.put((turn_id, "stream", event))
                if isinstance(event, StreamComplete):
                    final = event.response
            if final is None:
                raise RuntimeError("provider stream ended without StreamComplete")
            self.events.put((turn_id, "complete", (final, preparer.state)))
        except Exception as exc:  # noqa: BLE001
            self.events.put((turn_id, "error", (exc, preparer.state)))

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
                response, compaction_state = cast(
                    tuple[CompletionResponse, RuntimeCompaction | None],
                    payload,
                )
                self.app._compaction_state = compaction_state
                self._finish_response(response)
            elif kind == "error":
                self._clear_prompt()
                error, compaction_state = cast(
                    tuple[BaseException, RuntimeCompaction | None],
                    payload,
                )
                self.app._compaction_state = compaction_state
                if isinstance(error, (IncompleteStreamError, TransientProviderError)):
                    self._clear_stream_buffers()
                else:
                    self._flush_stream_buffer()
                self.app._write_turn_error(error)
                self._remember_worked_duration()
                self.streaming = False
                self.compacting = False
                self.inflight_tool_results = []
                self.inflight_tools_by_name = {}
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
            elif kind == "stream_retry":
                self._clear_prompt()
                self._clear_stream_buffers()
                attempt, max_retries, exc, delay = cast(
                    tuple[int, int, BaseException, float],
                    payload,
                )
                self.app._write_stream_retry_status(attempt, max_retries, exc, delay)
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
        self.inflight_tool_results = []
        tool_results = (
            self._dispatch_tools_with_prompt(response)
            if response.stop_reason == "tool_use"
            else []
        )
        if not self.streaming:
            if has_tool_uses and tool_results:
                self.app._append_followup_user(list(tool_results))
            self.inflight_tool_results = []
            self.inflight_tools_by_name = {}
            self._remember_worked_duration()
            return
        pending_monitor_texts = self.pending_monitor_inputs
        if has_tool_uses:
            pending_texts = self.pending_user_inputs
            prepared_pending, attachment_blocks = self._prepare_pending_user_inputs(
                pending_texts,
                render=True,
            )
            pending = [
                *active_task_guidance_text_blocks(prepared_pending),
                *attachment_blocks,
                *(TextBlock(text=text) for text in pending_monitor_texts),
            ]
            self.pending_user_inputs = []
            self.pending_monitor_inputs = []
            followup_content = [*tool_results, *pending]
            continue_running = self.app._append_followup_user(followup_content)
        else:
            interrupted_texts = self.interrupted_user_inputs
            pending_texts = [*self.pending_user_inputs, *self.turn_followup_user_inputs]
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
            self.turn_followup_user_inputs = []
            self.pending_monitor_inputs = []
            step = build_turn_step(
                response,
                pending_user_blocks=pending,
            )
            append_turn_step(self.app.messages, step)
            self.app._persist_session()
            continue_running = step.continue_running
        if continue_running:
            self.inflight_tool_results = []
            if tool_results:
                self.app._write_separator()
            self._start_worker()
        else:
            self.inflight_tool_results = []
            self.inflight_tools_by_name = {}
            self.active_tool_status = None
            self.streaming = False
            self.working_started_at = None
            self.app._write_status_snapshot()
            self._remember_worked_duration()

    def _remember_worked_duration(self) -> None:
        if self.turn_started_at is None:
            return
        self.last_worked_duration_text = _worked_duration_text(self.turn_started_at)
        self.turn_started_at = None

    def _dispatch_tools_with_prompt(
        self,
        response: CompletionResponse,
    ) -> list[ContentBlock]:
        results: list[ContentBlock] = []
        pending_research: list[CommandSummary] = []
        pending_edit_group: _EditRenderGroup | None = None

        def flush_research() -> None:
            if pending_research:
                self.app._write_research_result(pending_research)
                pending_research.clear()

        def flush_edit_group() -> None:
            nonlocal pending_edit_group
            if pending_edit_group is not None:
                self.app._write_edit_result_group(pending_edit_group)
                pending_edit_group = None

        for block in response.content:
            if not isinstance(block, ToolUseBlock):
                continue
            blocks = self._dispatch_tool_with_animated_prompt(block)
            result = _first_tool_result(block, blocks)
            self.inflight_tool_results.append(result)
            edit_item = _edit_render_item(block, result)
            summary = _research_summary_for_success(block, result)
            if edit_item is not None:
                flush_research()
                if pending_edit_group is None:
                    pending_edit_group = _EditRenderGroup(
                        path=edit_item.path,
                        items=[edit_item],
                    )
                elif pending_edit_group.path == edit_item.path:
                    pending_edit_group.items.append(edit_item)
                else:
                    flush_edit_group()
                    pending_edit_group = _EditRenderGroup(
                        path=edit_item.path,
                        items=[edit_item],
                    )
            elif summary is not None:
                flush_edit_group()
                pending_research.append(summary)
            else:
                flush_edit_group()
                flush_research()
                self.app._write_tool_result(block, result)
            results.extend(blocks)
        flush_edit_group()
        flush_research()
        return results

    def _dispatch_tool_with_animated_prompt(self, block: ToolUseBlock) -> list[ContentBlock]:
        tools_by_name = self.inflight_tools_by_name or self.app._available_tools()
        tool = tools_by_name.get(block.name)
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
        if block.name == "wait_agent":
            self.app._write_wait_agent_begin(block)
        self._draw_prompt()
        worker = threading.Thread(
            target=run_tool,
            name=f"wattle-tool-{block.name}",
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
                if self._drain_tool_events():
                    self._draw_prompt()
                else:
                    self._redraw_running_status_line()
                worker.join(timeout=0.04)
        finally:
            if self.streaming:
                worker.join()
            self._drain_tool_events()
            self._clear_prompt()
            self.active_tool_status = None
            self.active_exec_cells.pop(block.id, None)
        if result_box:
            return result_box[0]
        return [
            ToolResultBlock(
                tool_use_id=block.id,
                content=f"Tool ended without a result: {block.name!r}",
                is_error=True,
            )
        ]

    def _run_tool_without_permission(
        self,
        block: ToolUseBlock,
        tool: Tool,
    ) -> list[ContentBlock]:
        callback = self._tool_event_queue.put if block.name == "bash" else None
        return asyncio.run(dispatch_tool_blocks_async(block, {tool.name: tool}, None, callback))

    def _drain_tool_events(self) -> bool:
        changed = False
        while True:
            try:
                event = self._tool_event_queue.get_nowait()
            except queue.Empty:
                return changed
            changed = True
            if event.tool_name != "bash":
                continue
            if event.kind == "started":
                self.active_exec_cells[event.tool_use_id] = _BashExecCell(
                    tool_use_id=event.tool_use_id,
                    command=event.text or "<missing command>",
                )
            elif event.kind == "output":
                cell = self.active_exec_cells.get(event.tool_use_id)
                if cell is not None:
                    cell.output = _tail_chars(cell.output + event.text, 12000)
            elif event.kind == "completed":
                cell = self.active_exec_cells.get(event.tool_use_id)
                if cell is not None:
                    cell.running = False

    def _flush_stream_buffer(self) -> None:
        thinking = "".join(self.stream_thinking)
        text = "".join(self.stream_text)
        if thinking and self.app.show_thinking_content:
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
        visible_subagents = self._visible_subagent_snapshots()
        if self.statusline_selector_active:
            rows.extend(
                _render_statusline_selector_rows(
                    selected_fields=self.statusline_selector_fields,
                    selected_index=self.statusline_selector_selected,
                    width=line_width,
                    styles_enabled=self.app._styles_enabled(),
                )
            )
            return _PromptFrame(
                rows=rows,
                width=line_width,
                cursor_line_index=0,
                cursor_column=0,
            )
        if self.compacting:
            frame = COMPACTION_FRAMES[int(time.monotonic() * 8) % len(COMPACTION_FRAMES)]
            line = f" {frame} Auto-compacting..."
            rows.append(_styled_terminal_line(line, COMPACTION_STYLE, width))
        elif self.streaming and self.active_exec_cells:
            rows.append(_default_terminal_line("", width))
            for cell in self.active_exec_cells.values():
                rows.extend(
                    _bash_exec_cell_prompt_rows(
                        cell,
                        width=width,
                        styles_enabled=self.app._styles_enabled(),
                    )
                )
            rows.append(_default_terminal_line("", width))
        elif self.streaming and not self._suppress_running_status_line(visible_subagents):
            rows.append(_default_terminal_line("", width))
            rows.append(self._running_status_line(width))
            rows.append(_default_terminal_line("", width))
        if visible_subagents:
            summary = _subagent_count_summary(visible_subagents)
            rows.append(
                _styled_terminal_line(
                    f" Subagents · {summary}",
                    SUBAGENT_WAIT_TITLE_STYLE,
                    width,
                )
            )
            for line in _subagent_lifecycle_lines(visible_subagents[:3]):
                line = _one_line(line, limit=max(20, line_width - 1))
                rows.append(_styled_terminal_line(line, STATUS_STYLE, width))
            omitted = len(visible_subagents) - 3
            if omitted > 0:
                rows.append(_styled_terminal_line(f"  ... +{omitted} more", STATUS_STYLE, width))
        if self.interrupted_user_inputs:
            title = " Interrupted messages to be sent with your next message"
            rows.append(_styled_terminal_line(title, STATUS_STYLE, width))
            for text in self.interrupted_user_inputs:
                line = f"  ↳ {_strip_control(text)}"
                rows.append(_styled_terminal_line(line, STATUS_STYLE, width))
        queued_image_index = 1
        if self.pending_user_inputs:
            title = " Messages to be submitted after next tool call"
            hint = " (press esc to interrupt and send immediately)"
            rows.append(_styled_terminal_line(title, STATUS_STYLE, width))
            rows.append(_styled_terminal_line(hint, STATUS_STYLE, width))
            for text in self.pending_user_inputs:
                pending_preview, queued_image_index = _image_placeholder_text(
                    text,
                    image_index_start=queued_image_index,
                )
                line = f"  ↳ {_strip_control(pending_preview)}"
                rows.append(_styled_terminal_line(line, STATUS_STYLE, width))
        if self.turn_followup_user_inputs:
            title = " Messages to be submitted after assistant turn completes"
            hint = " (press esc to interrupt and send immediately)"
            rows.append(_styled_terminal_line(title, STATUS_STYLE, width))
            rows.append(_styled_terminal_line(hint, STATUS_STYLE, width))
            for text in self.turn_followup_user_inputs:
                pending_preview, queued_image_index = _image_placeholder_text(
                    text,
                    image_index_start=queued_image_index,
                )
                line = f"  ↳ {_strip_control(pending_preview)}"
                rows.append(_styled_terminal_line(line, STATUS_STYLE, width))
        if self.last_worked_duration_text is not None and not self.streaming:
            text = self.last_worked_duration_text[:line_width].ljust(line_width)
            if self.app._styles_enabled():
                rows.append(f"{WORKED_DURATION_STYLE}{text}{RESET}")
            else:
                rows.append(text)
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
                status_text = (
                    "press Enter to queue after next tool call; Tab for next turn"
                    if self.streaming and preview.strip()
                    else status
                )
                line = f" {status_text}"[:line_width].ljust(line_width)
                statusline = f"{STATUSLINE_STYLE}{_style_statusline_text(line)}{RESET}"
                rows.append(_filled_terminal_line(statusline, STATUSLINE_STYLE, width))
        return _PromptFrame(
            rows=rows,
            width=line_width,
            cursor_line_index=prompt_line_index,
            cursor_column=cursor_column,
        )

    def _visible_subagent_snapshots(self) -> list[dict[str, object]]:
        subagents = getattr(self.app.runtime, "_subagents", None)
        if subagents is None:
            return []
        return _visible_subagent_snapshots(subagents.snapshots())

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
            if force_reflow_clear:
                self._redraw_visible_screen_after_resize()
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
        return _default_terminal_line(f" {running_text}", width)

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
            and not self._suppress_running_status_line(self._visible_subagent_snapshots())
        )

    def _suppress_running_status_line(
        self,
        visible_subagents: list[dict[str, object]],
    ) -> bool:
        return self.active_tool_status == WAIT_AGENT_RUNNING_TITLE and bool(visible_subagents)

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

    def _redraw_visible_screen_after_resize(self) -> None:
        self._reset_prompt_state()
        self.app._write(VISIBLE_SCREEN_CLEAR)
        self.app._write(TERMINAL_HISTORY_CLEAR)
        self.app._write_welcome_card()
        messages = self.app.messages
        if self.inflight_tool_results:
            messages = [
                *messages,
                Message(role="user", content=list(self.inflight_tool_results)),
            ]
        self.app._write_history_messages(messages, with_separators=False)
        self.app._last_transcript_was_separator = False

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
        parts.append("\x1b[?25l\r\x1b[0m\x1b[J")
        for _ in range(rows_to_clear - 1):
            parts.append("\x1b[1A\r\x1b[0m\x1b[2K")
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
        read_files=tuple(compaction.read_files),
        modified_files=tuple(compaction.modified_files),
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
    try:
        path = resolve_session_path(selector)
    except ValueError:
        path = None
    if path is not None and path.exists():
        return load_session(path), path
    match = _resolve_resume_selector(selector)
    if match is not None:
        return match.record, match.path
    if path is None:
        path = resolve_session_path(selector)
    return load_session(path), path


def _resolve_resume_selector(selector: str) -> SessionEntry | None:
    entries = list_session_entries()
    exact_title = [entry for entry in entries if entry.record.metadata.title == selector]
    if len(exact_title) == 1:
        return exact_title[0]
    matches = filter_session_entries(entries, selector)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        if _resume_picker_tty_available():
            return _run_resume_picker(entries, initial_query=selector)
        raise ValueError(_ambiguous_resume_selector_message(selector, matches))
    return None


def _ambiguous_resume_selector_message(selector: str, matches: list[SessionEntry]) -> str:
    shown = ", ".join(
        f"{_short_session_id(entry)} ({entry.preview or '(untitled)'})"
        for entry in matches[:5]
    )
    suffix = "" if len(matches) <= 5 else f", +{len(matches) - 5} more"
    return f"ambiguous resume selector {selector!r}: {shown}{suffix}"


def _resume_picker_tty_available() -> bool:
    return (
        hasattr(sys.stdin, "isatty")
        and sys.stdin.isatty()
        and hasattr(sys.stdout, "isatty")
        and sys.stdout.isatty()
    )


def _select_resume_session(entries: list[SessionEntry]) -> SessionEntry | None:
    if not entries:
        return None
    if not _resume_picker_tty_available():
        return entries[0]
    return _run_resume_picker(entries)


@dataclass
class _ResumePickerState:
    all_entries: list[SessionEntry]
    filtered_entries: list[SessionEntry]
    selected: int = 0
    scroll_top: int = 0
    query: str = ""
    rows: int = 1
    width: int = 72


def _new_resume_picker_state(
    entries: list[SessionEntry],
    *,
    initial_query: str = "",
) -> _ResumePickerState:
    terminal_size = shutil.get_terminal_size((88, 24))
    state = _ResumePickerState(
        all_entries=entries,
        filtered_entries=[],
        query=initial_query,
        rows=max(1, terminal_size.lines - 7),
        width=max(72, terminal_size.columns),
    )
    _apply_resume_picker_filter(state)
    return state


def _apply_resume_picker_filter(state: _ResumePickerState) -> None:
    previous_entry = (
        state.filtered_entries[state.selected]
        if 0 <= state.selected < len(state.filtered_entries)
        else None
    )
    state.filtered_entries = filter_session_entries(state.all_entries, state.query)
    if not state.filtered_entries:
        state.selected = 0
        state.scroll_top = 0
        return
    if previous_entry in state.filtered_entries:
        state.selected = state.filtered_entries.index(previous_entry)
    else:
        state.selected = min(state.selected, len(state.filtered_entries) - 1)
    _clamp_resume_picker_scroll(state)


def _move_resume_picker_selection(state: _ResumePickerState, delta: int) -> None:
    if not state.filtered_entries:
        return
    state.selected = max(0, min(len(state.filtered_entries) - 1, state.selected + delta))
    _clamp_resume_picker_scroll(state)


def _clamp_resume_picker_scroll(state: _ResumePickerState) -> None:
    if state.selected < state.scroll_top:
        state.scroll_top = state.selected
    elif state.selected >= state.scroll_top + state.rows:
        state.scroll_top = state.selected - state.rows + 1
    max_scroll = max(0, len(state.filtered_entries) - state.rows)
    state.scroll_top = max(0, min(state.scroll_top, max_scroll))


def _run_resume_picker(
    entries: list[SessionEntry],
    *,
    initial_query: str = "",
) -> SessionEntry | None:
    state = _new_resume_picker_state(entries, initial_query=initial_query)

    def draw() -> None:
        sys.stdout.write("\x1b[?25l\x1b[2J\x1b[H")
        sys.stdout.write("Resume Wattle Session\n")
        sys.stdout.write(f"Search: {state.query}\n")
        sys.stdout.write("Enter resume  Esc clear/cancel  ↑↓ move\n\n")
        if not state.filtered_entries:
            sys.stdout.write("  No sessions match your search.\n")
            sys.stdout.flush()
            return
        visible = state.filtered_entries[state.scroll_top : state.scroll_top + state.rows]
        for visible_idx, entry in enumerate(visible):
            idx = state.scroll_top + visible_idx
            marker = ">" if idx == state.selected else " "
            text = _session_label(entry)
            if len(text) > state.width - 4:
                text = text[: state.width - 7] + "..."
            if idx == state.selected:
                sys.stdout.write(
                    f"{SELECTED_ROW_STYLE} {marker} {text.ljust(state.width - 4)} {RESET}\n"
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
            data = os.read(fd, 32).decode(errors="ignore")
            should_draw = False
            while data:
                sequence = None
                for candidate in (
                    "\x1b[5~",
                    "\x1b[6~",
                    "\x1b[A",
                    "\x1bOA",
                    "\x1b[B",
                    "\x1bOB",
                    "\x1b[H",
                    "\x1bOH",
                    "\x1b[F",
                    "\x1bOF",
                ):
                    if data.startswith(candidate):
                        sequence = candidate
                        break
                if sequence is not None:
                    data = data[len(sequence) :]
                    if sequence in ("\x1b[A", "\x1bOA"):
                        _move_resume_picker_selection(state, -1)
                    elif sequence in ("\x1b[B", "\x1bOB"):
                        _move_resume_picker_selection(state, 1)
                    elif sequence == "\x1b[5~":
                        _move_resume_picker_selection(state, -state.rows)
                    elif sequence == "\x1b[6~":
                        _move_resume_picker_selection(state, state.rows)
                    elif sequence in ("\x1b[H", "\x1bOH"):
                        state.selected = 0
                        _clamp_resume_picker_scroll(state)
                    elif sequence in ("\x1b[F", "\x1bOF"):
                        state.selected = max(0, len(state.filtered_entries) - 1)
                        _clamp_resume_picker_scroll(state)
                    should_draw = True
                    continue

                char, data = data[0], data[1:]
                if char in ("\r", "\n"):
                    if state.filtered_entries:
                        return state.filtered_entries[state.selected]
                elif char == "\x03":
                    return None
                elif char == "\x1b":
                    if state.query:
                        state.query = ""
                        _apply_resume_picker_filter(state)
                        should_draw = True
                    else:
                        return None
                elif char in ("\x7f", "\b"):
                    if state.query:
                        state.query = state.query[:-1]
                        _apply_resume_picker_filter(state)
                        should_draw = True
                elif char.isprintable():
                    state.query += char
                    _apply_resume_picker_filter(state)
                    should_draw = True
            if should_draw:
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

    selected = _select_resume_session(list_session_entries())
    if selected is None:
        return None
    return _apply_resumed_session(args, selected.record, selected.path)


def run_tui(args: argparse.Namespace) -> int:
    """Entry point wired in from `wattle.cli`."""
    from wattle.cli import _build_provider

    try:
        state = _resolve_resume(args)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"Could not resume session: {exc}\n")
        sys.stderr.flush()
        return 1

    try:
        provider = _build_provider(args.provider)
    except Exception as exc:  # noqa: BLE001
        provider = _UnavailableProvider(str(exc))
    args.persist_session = True
    app = WattleApp(args, provider, inline_mode=True, state=state)
    return app.run() or 0
