from __future__ import annotations

import pytest

from wattle.command_summary import (
    CommandSummary,
    CommandSummaryKind,
    analyze_shell_chain,
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
        (
            "pwd && git status --short --branch --untracked-files=all && git remote -v",
            CommandSummary(CommandSummaryKind.SHELL_CHAIN, "3 shell commands"),
        ),
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
        "cat README.md > copy.txt",
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


def test_summarize_bash_tool_includes_workdir_for_simple_chain() -> None:
    assert summarize_tool_call(
        "bash",
        {
            "command": "pwd && git status --short --branch --untracked-files=all && git remote -v",
            "workdir": "/repo",
        },
    ) == CommandSummary(CommandSummaryKind.SHELL_CHAIN, "3 shell commands in /repo")


def test_summarize_bash_command_accepts_simple_pipe_chain() -> None:
    assert summarize_bash_command("rg TODO | head") == CommandSummary(
        CommandSummaryKind.SHELL_CHAIN,
        "2 shell commands",
    )


def test_analyze_shell_chain_returns_segments_and_operators() -> None:
    chain = analyze_shell_chain(
        "pwd && git status --short --branch --untracked-files=all && git remote -v"
    )

    assert chain is not None
    assert chain.operators == ("&&", "&&")
    assert chain.segments == (
        ("pwd",),
        ("git", "status", "--short", "--branch", "--untracked-files=all"),
        ("git", "remote", "-v"),
    )


@pytest.mark.parametrize(
    "command",
    [
        "cat README.md > copy.txt",
        "echo $(pwd)",
        "if true; then pwd; fi",
        "(pwd && ls)",
        "python - <<'PY'\nprint('x')\nPY",
    ],
)
def test_analyze_shell_chain_rejects_complex_shell(command: str) -> None:
    assert analyze_shell_chain(command) is None


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
    assert (
        render_summary_action(CommandSummary(CommandSummaryKind.SHELL_CHAIN, "3 shell commands"))
        == "Run 3 shell commands"
    )
