"""Conservative semantic summaries for passive inspection tool calls."""

from __future__ import annotations

import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePath
from typing import Any


class CommandSummaryKind(Enum):
    READ_FILE = "read_file"
    LIST_FILES = "list_files"
    SEARCH_TEXT = "search_text"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CommandSummary:
    kind: CommandSummaryKind
    subject: str


RESEARCH_KINDS = frozenset(
    {
        CommandSummaryKind.READ_FILE,
        CommandSummaryKind.LIST_FILES,
        CommandSummaryKind.SEARCH_TEXT,
    }
)

_READ_COMMANDS = {"cat", "bat", "batcat", "less", "more", "head", "tail", "nl"}
_SEARCH_COMMANDS = {"rg", "grep", "egrep", "fgrep", "ag", "ack", "pt"}
_UNSUPPORTED_SHELL_TOKENS = {"|", ">", ">>", "<", "<<", ";", "&", "&&", "||"}


def summarize_tool_call(name: str, input: Mapping[str, Any]) -> CommandSummary:
    """Return a display summary for a tool call, or UNKNOWN if unclear."""
    if name == "read":
        return CommandSummary(
            CommandSummaryKind.READ_FILE,
            _string_arg(input, "path", "<missing path>"),
        )
    if name == "bash":
        return summarize_bash_command(_string_arg(input, "command", ""))
    return CommandSummary(CommandSummaryKind.UNKNOWN, "")


def summarize_bash_command(command: str) -> CommandSummary:
    command = command.strip()
    if not command or _has_unsupported_shell_syntax(command):
        return CommandSummary(CommandSummaryKind.UNKNOWN, command)
    try:
        words = _split_shell_words(command)
    except ValueError:
        return CommandSummary(CommandSummaryKind.UNKNOWN, command)
    if not words or any(word in _UNSUPPORTED_SHELL_TOKENS for word in words):
        return CommandSummary(CommandSummaryKind.UNKNOWN, command)

    executable = _basename(words[0])
    if executable in _READ_COMMANDS:
        subject = _last_non_option_operand(words[1:])
        return _summary_or_unknown(CommandSummaryKind.READ_FILE, subject, command)
    if executable == "sed":
        subject = _sed_read_subject(words[1:])
        return _summary_or_unknown(CommandSummaryKind.READ_FILE, subject, command)
    if executable == "ls" or executable == "tree":
        subject = _last_non_option_operand(words[1:]) or "."
        return CommandSummary(CommandSummaryKind.LIST_FILES, subject)
    if executable == "rg" and "--files" in words[1:]:
        subject = _last_non_option_operand([word for word in words[1:] if word != "--files"]) or "."
        return CommandSummary(CommandSummaryKind.LIST_FILES, subject)
    if executable == "git" and len(words) >= 2:
        if words[1:3] == ["ls-files"] or words[1] == "ls-files":
            subject = _last_non_option_operand(words[2:]) or "."
            return CommandSummary(CommandSummaryKind.LIST_FILES, subject)
        if words[1] == "grep":
            subject = _search_subject(words[2:])
            return _summary_or_unknown(CommandSummaryKind.SEARCH_TEXT, subject, command)
    if executable == "find" and "-type" in words and "f" in words:
        subject = _find_subject(words[1:]) or "."
        return CommandSummary(CommandSummaryKind.LIST_FILES, subject)
    if executable in _SEARCH_COMMANDS:
        subject = _search_subject(words[1:])
        return _summary_or_unknown(CommandSummaryKind.SEARCH_TEXT, subject, command)
    return CommandSummary(CommandSummaryKind.UNKNOWN, command)


def is_research_summary(summary: CommandSummary) -> bool:
    return summary.kind in RESEARCH_KINDS


def render_summary_action(summary: CommandSummary) -> str:
    if summary.kind == CommandSummaryKind.READ_FILE:
        return f"Read {summary.subject}"
    if summary.kind == CommandSummaryKind.LIST_FILES:
        return f"List {summary.subject}"
    if summary.kind == CommandSummaryKind.SEARCH_TEXT:
        return f"Search {summary.subject}"
    return summary.subject


def _string_arg(input: Mapping[str, Any], name: str, default: str) -> str:
    value = input.get(name, default)
    return str(value)


def _basename(executable: str) -> str:
    return PurePath(executable).name


def _split_shell_words(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    return list(lexer)


def _has_unsupported_shell_syntax(command: str) -> bool:
    return "\n" in command


def _last_non_option_operand(args: list[str]) -> str | None:
    operands = [arg for arg in args if arg and not arg.startswith("-")]
    return operands[-1] if operands else None


def _sed_read_subject(args: list[str]) -> str | None:
    if "-n" not in args:
        return None
    return _last_non_option_operand(args)


def _find_subject(args: list[str]) -> str | None:
    for arg in args:
        if arg.startswith("-"):
            break
        return arg
    return None


def _search_subject(args: list[str]) -> str | None:
    skip_next_for = {"-e", "-f", "--regexp", "--file", "-C", "-A", "-B", "-m", "--max-count"}
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in skip_next_for:
            skip_next = True
            continue
        if arg.startswith("--"):
            continue
        if arg.startswith("-"):
            continue
        return arg
    return None


def _summary_or_unknown(
    kind: CommandSummaryKind,
    subject: str | None,
    command: str,
) -> CommandSummary:
    if not subject:
        return CommandSummary(CommandSummaryKind.UNKNOWN, command)
    return CommandSummary(kind, subject)
