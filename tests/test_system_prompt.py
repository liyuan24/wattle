from __future__ import annotations

from datetime import date
from pathlib import Path

from willow.permissions import PermissionMode
from willow.system_prompt import (
    ContextFile,
    build_system_prompt,
    load_context_files,
)
from willow.tools import TOOLS_BY_NAME


def test_build_system_prompt_uses_willow_default_without_pi_docs(tmp_path: Path) -> None:
    prompt = build_system_prompt(
        tools_by_name=TOOLS_BY_NAME,
        cwd=tmp_path,
        context_files=[],
        current_date=date(2026, 5, 10),
    )

    assert "I'm Willow, an AI coding assistant." in prompt
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
    assert "prefer using `rg` or `rg --files`" in prompt
    assert "Use tty=true only for commands that require an interactive terminal" in prompt
    assert "prefer conventional entrypoint names and process shapes" in prompt
    assert "Before starting a long-running, expensive, or quiet operation" in prompt
    assert "Willow does not automatically start monitors after background commands" in prompt
    assert "Current date: 2026-05-10" in prompt
    assert f"Current working directory: {tmp_path.resolve()}" in prompt


def test_system_prompt_preserves_context_and_runtime_metadata(tmp_path: Path) -> None:
    prompt = build_system_prompt(
        tools_by_name={},
        context_files=[ContextFile(path=tmp_path / "WILLOW.md", content="Project rules.")],
        cwd=tmp_path,
        current_date=date(2026, 5, 10),
    )

    assert prompt.startswith("I'm Willow, an AI coding assistant.")
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
    user_dir = tmp_path / "home" / ".willow"
    project = tmp_path / "repo" / "pkg"
    user_dir.mkdir(parents=True)
    (user_dir / "WILLOW.md").write_text("User rules.")
    (tmp_path / "repo" / ".willow").mkdir(parents=True)
    (tmp_path / "repo" / ".willow" / "WILLOW.md").write_text("Project rules.")
    project.mkdir(parents=True)

    files = load_context_files(cwd=project, user_dir=user_dir)

    assert [file.content for file in files] == ["User rules.", "Project rules."]
