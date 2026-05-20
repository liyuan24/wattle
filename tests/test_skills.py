"""Tests for Wattle skill discovery and explicit invocation."""

from __future__ import annotations

from pathlib import Path

import pytest

from wattle.skills import (
    SkillNotFoundError,
    expand_skill_invocation,
    format_skills_for_system_prompt,
    load_available_skills,
    resolve_skill,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_load_available_skills_discovers_user_and_project_layouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    project = home / "repo" / "project"
    monkeypatch.setattr(Path, "home", lambda: home)
    _write(
        home / ".wattle" / "skills" / "reviewer" / "SKILL.md",
        "---\nname: reviewer\ndescription: Review code carefully.\n---\nbody",
    )
    _write(
        home / "repo" / ".wattle" / "skills" / "planner" / "SKILL.md",
        "# Planner\nBreak work into steps.",
    )
    _write(
        project / ".wattle" / "skills" / "writer" / "SKILL.md",
        "Write tersely.",
    )

    skills = load_available_skills(project)

    assert [(skill.name, skill.description, skill.location) for skill in skills] == [
        ("planner", "Planner", "project"),
        ("reviewer", "Review code carefully.", "user"),
        ("writer", "Write tersely.", "project"),
    ]


def test_direct_markdown_skill_files_are_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    project = home / "project"
    monkeypatch.setattr(Path, "home", lambda: home)
    _write(project / ".wattle" / "skills" / "planner.md", "# Planner")

    assert load_available_skills(project) == []


def test_nearest_project_skill_overrides_parent_and_user_skill_with_same_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    project = home / "repo" / "project"
    monkeypatch.setattr(Path, "home", lambda: home)
    _write(home / ".wattle" / "skills" / "test" / "SKILL.md", "user body")
    _write(home / "repo" / ".wattle" / "skills" / "test" / "SKILL.md", "parent body")
    _write(project / ".wattle" / "skills" / "test" / "SKILL.md", "project body")

    skill = resolve_skill("test", project)

    assert skill.location == "project"
    assert skill.load_content() == "project body"


def test_format_skills_for_system_prompt_lists_metadata_not_bodies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    monkeypatch.setattr(Path, "home", lambda: home)
    _write(
        project / ".wattle" / "skills" / "secret" / "SKILL.md",
        "---\ndescription: Use secret workflow.\n---\nFULL SECRET BODY",
    )

    rendered = format_skills_for_system_prompt(load_available_skills(project))

    assert "secret" in rendered
    assert "Use secret workflow." in rendered
    assert "Invoke a skill directly with /<skill_name>" in rendered
    assert "FULL SECRET BODY" not in rendered


def test_expand_skill_invocation_loads_skill_body_and_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    monkeypatch.setattr(Path, "home", lambda: home)
    _write(project / ".wattle" / "skills" / "writer" / "SKILL.md", "Write tersely.")

    expanded = expand_skill_invocation("/writer draft release notes", project)

    assert expanded is not None
    assert "Use the Wattle skill 'writer'" in expanded
    assert "  <name>writer</name>" in expanded
    skill_path = project / ".wattle" / "skills" / "writer" / "SKILL.md"
    assert f"  <path>{skill_path}</path>" in expanded
    assert "Write tersely." in expanded
    assert "User task:\ndraft release notes" in expanded


def test_resolve_skill_raises_for_unknown_skill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")

    with pytest.raises(SkillNotFoundError):
        resolve_skill("missing", tmp_path)


def test_expand_skill_invocation_returns_none_for_unknown_slash_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")

    assert expand_skill_invocation("/missing do work", tmp_path) is None
