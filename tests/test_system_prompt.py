from __future__ import annotations

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
    assert "first extract an internal checklist of explicit requirements" in prompt
    assert "Before finalizing, compare the result against that checklist" in prompt
    assert "When instructions specify exact identifiers or literals" in prompt
    assert "Do not infer, rename, abbreviate, expand, or harmonize identifiers" in prompt
    assert "Treat explicit task constraints as higher priority than convenient shortcuts" in prompt
    assert "derive the task contract" in prompt
    assert "assumptions that must be checked" in prompt
    assert "validate heuristic constants or stopping conditions with repeatable evidence" in prompt
    assert "When investigating a failure, prioritize reproducing and explaining" in prompt
    assert "keep multiple hypotheses alive until one explains the symptom end to end" in prompt
    assert "prefer using `rg` or `rg --files`" in prompt
    assert "why or how current-project behavior works, differs, or regressed" in prompt
    assert "inspect relevant repository files before giving a concrete answer" in prompt
    assert "Use tty=true only for commands that require an interactive terminal" in prompt
    assert "try `uv run pytest` before using `python -m compileall`" in prompt
    assert "prefer conventional entrypoint names and process shapes" in prompt
    assert "Before starting a long-running, expensive, or quiet operation" in prompt
    assert "Wattle does not automatically start monitors after background commands" in prompt
    assert "Use spawn_agent for independent, bounded side tasks" in prompt
    assert "set agent_type explicitly when the role matters" in prompt
    assert "do not duplicate their assigned work locally while they are still running" in prompt
    assert "first find the image path using available file-search tools" in prompt
    assert "Each old_text must match exactly and uniquely" in prompt
    assert "send them together in one edit call with the edits array" in prompt
    assert "Merge nearby or overlapping changes into one replacement" in prompt
    assert "Current date: 2026-05-10" in prompt
    assert f"Current working directory: {tmp_path.resolve()}" in prompt


def test_system_prompt_preserves_context_and_runtime_metadata(tmp_path: Path) -> None:
    prompt = build_system_prompt(
        tools_by_name={},
        context_files=[ContextFile(path=tmp_path / "WATTLE.md", content="Project rules.")],
        cwd=tmp_path,
        current_date=date(2026, 5, 10),
    )

    assert prompt.startswith("I'm Wattle, an AI coding assistant.")
    assert "# Project Context" in prompt
    assert "Project rules." in prompt
    assert "Current date: 2026-05-10" in prompt


def test_read_only_mode_adds_read_only_guideline(tmp_path: Path) -> None:
    prompt = build_system_prompt(
        tools_by_name=TOOLS_BY_NAME,
        cwd=tmp_path,
        context_files=[],
        permission_mode=PermissionMode.READ_ONLY,
        current_date=date(2026, 5, 10),
    )

    assert "Read-only mode is active" in prompt


def test_load_context_files_reads_user_and_project_locations(
    tmp_path: Path,
) -> None:
    user_dir = tmp_path / "home" / ".wattle"
    project = tmp_path / "repo" / "pkg"
    user_dir.mkdir(parents=True)
    (user_dir / "WATTLE.md").write_text("User rules.")
    (tmp_path / "repo" / ".wattle").mkdir(parents=True)
    (tmp_path / "repo" / ".wattle" / "WATTLE.md").write_text("Project rules.")
    project.mkdir(parents=True)

    files = load_context_files(cwd=project, user_dir=user_dir)

    assert [file.content for file in files] == ["User rules.", "Project rules."]
