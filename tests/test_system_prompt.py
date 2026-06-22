from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from wattle.permissions import PermissionMode
from wattle.system_prompt import (
    ContextFile,
    build_system_prompt,
    load_context_files,
)
from wattle.tools import TOOLS_BY_NAME


def test_build_system_prompt_uses_wattle_default_without_pi_docs(tmp_path: Path) -> None:
    prompt = build_system_prompt(
        tools_by_name=TOOLS_BY_NAME,
        cwd=tmp_path,
        context_files=[],
        current_date=date(2026, 5, 10),
    )

    assert "I'm Wattle, an AI coding assistant." in prompt
    assert "operating inside pi" not in prompt
    assert "Pi documentation" not in prompt
    assert "- read:" in prompt
    assert "- bash:" in prompt
    assert "Validation discipline:" in prompt
    assert "consumer inputs as part of the final interface" in prompt
    assert "preserves the same payload contents" in prompt
    assert "input contract the delivered result will consume" in prompt
    assert "independently derived oracle" in prompt
    assert "Checking only that a file exists" in prompt
    assert "closest faithful check from the task contract" in prompt
    assert "verifier-minded contradiction pass" in prompt
    assert "plausible alternate interpretations" in prompt
    assert "indexing and ranking conventions" in prompt
    assert "validate artifact-first" in prompt
    assert "script that merely runs" in prompt
    assert "validate the representation contract as a downstream consumer" in prompt
    assert "exact scalar types, container types" in prompt
    assert "visual similarity or a permissive parse is not enough" in prompt
    assert "Derived or modified validation is proxy evidence only" not in prompt
    assert "include deliberately tiny-capacity" not in prompt
    assert "do not treat a barely-over-threshold result as robust" not in prompt
    assert "vary capacity separately from feature or interface choices" not in prompt
    assert "first extract an internal checklist of explicit requirements" in prompt
    assert "Before finalizing, compare the result against that checklist" in prompt
    assert "When instructions specify exact identifiers or literals" in prompt
    assert "Do not infer, rename, abbreviate, expand, or harmonize identifiers" in prompt
    assert "Treat explicit task constraints as higher priority than convenient shortcuts" in prompt
    assert "derive the task contract" in prompt
    assert "assumptions that must be checked" in prompt
    assert "validate heuristic constants or stopping conditions with repeatable evidence" in prompt
    assert "Work until the user's task is actually handled" in prompt
    assert "Do not stop at analysis, a partial implementation" in prompt
    assert "Treat fixable follow-up work discovered during validation" in prompt
    assert "rerun focused validation" in prompt
    assert "runtime permission restriction" in prompt
    assert "When investigating a failure, prioritize reproducing and explaining" in prompt
    assert "keep multiple hypotheses alive until one explains the symptom end to end" in prompt
    assert "For investigation/debugging requests, do not modify existing project files" in prompt
    assert "temporary validation scripts, logs, or repro artifacts" in prompt
    assert "ask for approval before changing existing files" in prompt
    assert "Before making factual claims about observable state" in prompt
    assert "verify with available tools when practical" in prompt
    assert "Do not rely only on memory of prior actions" in prompt
    assert "what you personally did and what the current environment shows" in prompt
    assert "prefer using `rg` or `rg --files`" in prompt
    assert "why or how current-project behavior works, differs, or regressed" in prompt
    assert "inspect relevant repository files before giving a concrete answer" in prompt
    assert "Use tty=true only for commands that require an interactive terminal" in prompt
    assert "try `uv run pytest` before using `python -m compileall`" in prompt
    assert "prefer conventional entrypoint names and process shapes" in prompt
    assert "Before starting a long-running, expensive, or quiet operation" in prompt
    assert "Wattle does not automatically start monitors after background commands" in prompt
    assert "Do not spawn a subagent as the first action" in prompt
    assert "code review, or repository inspection" in prompt
    assert "Before spawning subagents, quickly decide the work split" in prompt
    assert "set agent_type explicitly when the role matters" in prompt
    assert "do not use them to hand off the user's main request" in prompt
    assert "do not duplicate a subagent's assigned work while it is still running" in prompt
    assert "do not spawn replacement subagents for the same task" in prompt
    assert "write the task so the ownership is clear in prose" in prompt
    assert "first find the image path using available file-search tools" in prompt
    assert "Use update_plan sparingly for complex, ambiguous, or long-running work" in prompt
    assert "Do not use it for short routine tasks" in prompt
    assert "brief status sentence" in prompt
    assert "first identify and review the referenced content" in prompt
    assert "whether it is in a file or earlier conversation" in prompt
    assert "do not use update_plan as a substitute for reading or understanding the plan" in prompt
    assert "Each old_text must match exactly and uniquely" in prompt
    assert "send them together in one edit call with the edits array" in prompt
    assert "Merge nearby or overlapping changes into one replacement" in prompt
    assert "Current date: 2026-05-10" in prompt
    assert f"Current working directory: {tmp_path.resolve()}" in prompt


def test_system_prompt_preserves_context_and_runtime_metadata(tmp_path: Path) -> None:
    prompt = build_system_prompt(
        tools_by_name={},
        context_files=[ContextFile(path=tmp_path / "AGENTS.md", content="Project rules.")],
        cwd=tmp_path,
        current_date=date(2026, 5, 10),
    )

    assert prompt.startswith("I'm Wattle, an AI coding assistant.")
    assert "# Project Context" in prompt
    assert f"# AGENTS.md instructions for {tmp_path}" in prompt
    assert "<INSTRUCTIONS>\nProject rules.\n</INSTRUCTIONS>" in prompt
    assert "Use update_plan sparingly for complex, ambiguous, or long-running work" not in prompt
    assert "Current date: 2026-05-10" in prompt


def test_persistence_guideline_is_stable_across_permission_modes(tmp_path: Path) -> None:
    prompts = [
        build_system_prompt(
            tools_by_name=TOOLS_BY_NAME,
            cwd=tmp_path,
            context_files=[],
            permission_mode=mode,
            current_date=date(2026, 5, 10),
        )
        for mode in PermissionMode
    ]

    expected = (
        "Work until the user's task is actually handled. Do not stop at analysis, "
        "a partial implementation, or the first validation failure when the next "
        "step is clear."
    )
    assert all(expected in prompt for prompt in prompts)
    assert len({prompt.split("When investigating a failure", 1)[0] for prompt in prompts}) == 1


def _touch_git(path: Path) -> None:
    (path / ".git").mkdir()


def test_agents_md_in_project_root_is_loaded(tmp_path: Path) -> None:
    user_dir = tmp_path / "home" / ".wattle"
    repo = tmp_path / "repo"
    app = repo / "pkg" / "app"
    app.mkdir(parents=True)
    _touch_git(repo)
    (repo / "AGENTS.md").write_text("Root rules.")

    files = load_context_files(cwd=app, user_dir=user_dir)

    assert [(file.path, file.content) for file in files] == [(repo / "AGENTS.md", "Root rules.")]


def test_agents_override_wins_over_same_directory_agents_md(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _touch_git(repo)
    (repo / "AGENTS.md").write_text("Default rules.")
    (repo / "AGENTS.override.md").write_text("Override rules.")

    files = load_context_files(cwd=repo, user_dir=tmp_path / "missing")

    assert [(file.path.name, file.content) for file in files] == [
        ("AGENTS.override.md", "Override rules.")
    ]


def test_invalid_override_falls_back_to_agents_md(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _touch_git(repo)
    (repo / "AGENTS.override.md").mkdir()
    (repo / "AGENTS.md").write_text("Default rules.")

    files = load_context_files(cwd=repo, user_dir=tmp_path / "missing")

    assert [(file.path.name, file.content) for file in files] == [("AGENTS.md", "Default rules.")]


def test_nested_docs_concatenate_root_first_child_second(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    child = repo / "lib"
    child.mkdir(parents=True)
    _touch_git(repo)
    (repo / "AGENTS.md").write_text("Root rules.")
    (child / "AGENTS.md").write_text("Child rules.")

    files = load_context_files(cwd=child, user_dir=tmp_path / "missing")

    assert [file.content for file in files] == ["Root rules.", "Child rules."]


def test_child_override_does_not_suppress_parent_agents_md(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    child = repo / "lib"
    child.mkdir(parents=True)
    _touch_git(repo)
    (repo / "AGENTS.md").write_text("Root rules.")
    (child / "AGENTS.md").write_text("Child default rules.")
    (child / "AGENTS.override.md").write_text("Child override rules.")

    files = load_context_files(cwd=child, user_dir=tmp_path / "missing")

    assert [(file.path.name, file.content) for file in files] == [
        ("AGENTS.md", "Root rules."),
        ("AGENTS.override.md", "Child override rules."),
    ]


def test_no_git_marker_only_checks_cwd(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    (parent / "AGENTS.md").write_text("Parent rules.")
    (child / "AGENTS.md").write_text("Child rules.")

    files = load_context_files(cwd=child, user_dir=tmp_path / "missing")

    assert [(file.path, file.content) for file in files] == [(child / "AGENTS.md", "Child rules.")]


def test_directories_special_files_and_empty_files_are_ignored(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    child = repo / "child"
    child.mkdir(parents=True)
    _touch_git(repo)
    (repo / "AGENTS.md").mkdir()
    (child / "AGENTS.md").write_text("   \n\t")

    fifo_path = child / "AGENTS.override.md"
    if hasattr(os, "mkfifo"):
        os.mkfifo(fifo_path)

    files = load_context_files(cwd=child, user_dir=tmp_path / "missing")

    assert files == []


def test_byte_cap_truncates_safely(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _touch_git(repo)
    (repo / "AGENTS.md").write_text("abcdef")

    files = load_context_files(cwd=repo, user_dir=tmp_path / "missing", byte_budget=3)

    assert len(files) == 1
    assert files[0].content.startswith("abc")
    assert "truncated due to byte budget" in files[0].content


def test_loaded_context_includes_source_path(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _touch_git(repo)
    (repo / "AGENTS.md").write_text("Project rules.")

    prompt = build_system_prompt(
        tools_by_name={},
        cwd=repo,
        context_files=load_context_files(cwd=repo, user_dir=tmp_path / "missing"),
        current_date=date(2026, 5, 10),
    )

    assert f"# AGENTS.md instructions for {repo}" in prompt
    assert "<INSTRUCTIONS>" in prompt


def test_project_codex_agents_md_is_not_loaded_by_default(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    codex = repo / ".codex"
    codex.mkdir(parents=True)
    _touch_git(repo)
    (codex / "AGENTS.md").write_text("Codex subdir rules.")

    files = load_context_files(cwd=repo, user_dir=tmp_path / "missing")

    assert files == []


def test_wattle_md_is_ignored_even_when_present(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _touch_git(repo)
    (repo / "WATTLE.md").write_text("Wattle rules.")

    files = load_context_files(cwd=repo, user_dir=tmp_path / "missing")

    assert files == []


def test_when_wattle_md_and_agents_md_exist_only_agents_md_appears(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _touch_git(repo)
    (repo / "WATTLE.md").write_text("Wattle rules.")
    (repo / "AGENTS.md").write_text("Agents rules.")

    prompt = build_system_prompt(
        tools_by_name={},
        cwd=repo,
        context_files=load_context_files(cwd=repo, user_dir=tmp_path / "missing"),
        current_date=date(2026, 5, 10),
    )

    assert "Agents rules." in prompt
    assert "Wattle rules." not in prompt
    assert "WATTLE.md" not in prompt


def test_global_agents_load_before_project_agents(tmp_path: Path) -> None:
    user_dir = tmp_path / "home" / ".wattle"
    repo = tmp_path / "repo"
    user_dir.mkdir(parents=True)
    repo.mkdir()
    _touch_git(repo)
    (user_dir / "AGENTS.override.md").write_text("Global override rules.")
    (user_dir / "AGENTS.md").write_text("Global default rules.")
    (repo / "AGENTS.md").write_text("Project rules.")

    files = load_context_files(cwd=repo, user_dir=user_dir)

    assert [(file.path.name, file.content) for file in files] == [
        ("AGENTS.override.md", "Global override rules."),
        ("AGENTS.md", "Project rules."),
    ]
