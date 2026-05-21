from __future__ import annotations

import pytest

from wattle.command_summary import (
    CommandSummary,
    CommandSummaryKind,
    render_summary_action,
    summarize_bash_command,
    summarize_tool_call,
)


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (
            "cat src/wattle/system_prompt.py",
            CommandSummary(CommandSummaryKind.READ_FILE, "src/wattle/system_prompt.py"),
        ),
        ("head -20 README.md", CommandSummary(CommandSummaryKind.READ_FILE, "README.md")),
        (
            "sed -n '1,20p' pyproject.toml",
            CommandSummary(CommandSummaryKind.READ_FILE, "pyproject.toml"),
        ),
        ("nl README.md", CommandSummary(CommandSummaryKind.READ_FILE, "README.md")),
        ("ls src/wattle", CommandSummary(CommandSummaryKind.LIST_FILES, "src/wattle")),
        ("tree tests", CommandSummary(CommandSummaryKind.LIST_FILES, "tests")),
        ("rg --files src/wattle", CommandSummary(CommandSummaryKind.LIST_FILES, "src/wattle")),
        ("git ls-files", CommandSummary(CommandSummaryKind.LIST_FILES, ".")),
        ("find src -type f", CommandSummary(CommandSummaryKind.LIST_FILES, "src")),
        ("rg 'background|bg' src", CommandSummary(CommandSummaryKind.SEARCH_TEXT, "background|bg")),
        ("grep -R needle tests", CommandSummary(CommandSummaryKind.SEARCH_TEXT, "needle")),
        ("git grep TODO", CommandSummary(CommandSummaryKind.SEARCH_TEXT, "TODO")),
    ],
)
def test_summarize_bash_command_classifies_passive_inspection(
    command: str,
    expected: CommandSummary,
) -> None:
    assert summarize_bash_command(command) == expected


@pytest.mark.parametrize(
    "command",
    [
        "rg TODO | head",
        "cat README.md > copy.txt",
        "ls src && pytest",
        "uv run pytest",
        "python -m compileall src",
        "sed 's/a/b/' file.txt",
    ],
)
def test_summarize_bash_command_keeps_unclear_or_non_inspection_unknown(command: str) -> None:
    summary = summarize_bash_command(command)

    assert summary.kind == CommandSummaryKind.UNKNOWN
    assert summary.subject == command


def test_summarize_read_tool_maps_to_read_file() -> None:
    assert summarize_tool_call("read", {"path": "src/wattle/loop.py"}) == CommandSummary(
        CommandSummaryKind.READ_FILE,
        "src/wattle/loop.py",
    )


def test_render_summary_action_uses_required_labels() -> None:
    assert (
        render_summary_action(CommandSummary(CommandSummaryKind.READ_FILE, "a.py"))
        == "Read a.py"
    )
    assert render_summary_action(CommandSummary(CommandSummaryKind.LIST_FILES, "src")) == "List src"
    assert (
        render_summary_action(CommandSummary(CommandSummaryKind.SEARCH_TEXT, "TODO"))
        == "Search TODO"
    )
