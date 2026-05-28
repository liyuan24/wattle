"""Tests for the release-version bump helper."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bump_release_version.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bump_release_version", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_version_bump_types() -> None:
    script = _load_script()
    version = script.Version.parse("1.2.3")

    assert str(version.bump("patch")) == "1.2.4"
    assert str(version.bump("minor")) == "1.3.0"
    assert str(version.bump("major")) == "2.0.0"


def test_replace_rejects_duplicate_versions() -> None:
    script = _load_script()
    original = (
        '[project]\nname = "wattle"\nversion = "0.1.0"\n\n'
        '[tool.example]\nversion = "9.9.9"\n'
    )

    with pytest.raises(SystemExit, match="exactly one"):
        script._replace_one(script.VERSION_RE, original, "0.2.0", path=script.PYPROJECT)


def test_replace_pyproject_version_once() -> None:
    script = _load_script()
    original = '[project]\nname = "wattle"\nversion = "0.1.0"\n'

    updated = script._replace_one(
        script.VERSION_RE,
        original,
        "0.2.0",
        path=script.PYPROJECT,
    )

    assert updated == '[project]\nname = "wattle"\nversion = "0.2.0"\n'


def test_replace_lockfile_wattle_version_only() -> None:
    script = _load_script()
    original = (
        'name = "other"\nversion = "9.9.9"\n\n'
        'name = "wattle"\nversion = "0.1.0"\n\n'
        'name = "wattle-extra"\nversion = "1.0.0"\n'
    )

    updated = script._replace_one(
        script.LOCK_WATTLE_RE,
        original,
        "0.2.0",
        path=script.UV_LOCK,
    )

    assert 'name = "other"\nversion = "9.9.9"' in updated
    assert 'name = "wattle"\nversion = "0.2.0"' in updated
    assert 'name = "wattle-extra"\nversion = "1.0.0"' in updated


def test_write_version_rolls_back_if_second_file_write_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = _load_script()
    pyproject = tmp_path / "pyproject.toml"
    lockfile = tmp_path / "uv.lock"
    original_pyproject = '[project]\nname = "wattle"\nversion = "0.1.0"\n'
    original_lockfile = 'name = "wattle"\nversion = "0.1.0"\n'
    pyproject.write_text(original_pyproject, encoding="utf-8")
    lockfile.write_text(original_lockfile, encoding="utf-8")
    monkeypatch.setattr(script, "PYPROJECT", pyproject)
    monkeypatch.setattr(script, "UV_LOCK", lockfile)

    real_atomic_write = script._atomic_write

    def fail_lock_write(path: Path, content: str) -> None:
        if path == lockfile and 'version = "0.2.0"' in content:
            raise OSError("simulated lockfile failure")
        real_atomic_write(path, content)

    monkeypatch.setattr(script, "_atomic_write", fail_lock_write)

    with pytest.raises(OSError, match="simulated lockfile failure"):
        script._write_version(script.Version.parse("0.2.0"), dry_run=False)

    assert pyproject.read_text(encoding="utf-8") == original_pyproject
    assert lockfile.read_text(encoding="utf-8") == original_lockfile


def test_latest_version_api_payload() -> None:
    script = _load_script()

    payload = script._latest_version_payload(script.Version.parse("0.2.0"))

    assert payload == {
        "ok": True,
        "version": "0.2.0",
        "tag": "v0.2.0",
        "repo": "https://github.com/liyuan24/wattle",
        "releaseUrl": "https://github.com/liyuan24/wattle/releases/tag/v0.2.0",
        "installUrl": "https://wattleagent.com/install.sh",
    }


def test_write_latest_version_api_writes_extensionless_and_json_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = _load_script()
    extensionless = tmp_path / "docs" / "api" / "latest-version"
    json_file = tmp_path / "docs" / "api" / "latest-version.json"
    monkeypatch.setattr(script, "LATEST_VERSION_API", extensionless)
    monkeypatch.setattr(script, "LATEST_VERSION_JSON", json_file)

    script._write_latest_version_api(script.Version.parse("0.2.0"), dry_run=False)

    assert json.loads(extensionless.read_text(encoding="utf-8")) == (
        script._latest_version_payload(script.Version.parse("0.2.0"))
    )
    assert json_file.read_text(encoding="utf-8") == extensionless.read_text(encoding="utf-8")


def test_release_runs_full_publish_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _load_script()
    commands: list[list[str]] = []
    monkeypatch.setattr(script, "_read_current_version", lambda: script.Version.parse("0.1.0"))
    monkeypatch.setattr(script, "_tag_exists", lambda _tag: False)
    monkeypatch.setattr(script, "_write_version", lambda _version, *, dry_run: None)
    monkeypatch.setattr(
        script,
        "_write_latest_version_api",
        lambda _version, *, dry_run: None,
    )
    monkeypatch.setattr(script, "_require_clean_worktree", lambda: None)
    monkeypatch.setattr(script, "_git_output", lambda _command: "master")
    monkeypatch.setattr(sys, "argv", ["bump_release_version.py", "minor"])

    def record_run(command: list[str], *, dry_run: bool) -> None:
        assert dry_run is False
        commands.append(command)

    monkeypatch.setattr(script, "_run", record_run)

    assert script.main() == 0
    assert commands == [
        ["uv", "run", "pytest"],
        [
            "git",
            "add",
            "pyproject.toml",
            "uv.lock",
            "docs/api/latest-version",
            "docs/api/latest-version.json",
        ],
        ["git", "commit", "-m", "Release v0.2.0"],
        ["git", "tag", "v0.2.0"],
        ["git", "push", "--atomic", "origin", "master", "v0.2.0"],
        [
            "gh",
            "release",
            "create",
            "v0.2.0",
            "--title",
            "Wattle v0.2.0",
            "--generate-notes",
        ],
    ]
    assert ["git", "push", "--atomic", "origin", "master", "v0.2.0"] in commands


def test_release_writes_latest_version_api(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _load_script()
    latest_version_api_calls: list[tuple[object, bool]] = []
    monkeypatch.setattr(script, "_read_current_version", lambda: script.Version.parse("0.1.0"))
    monkeypatch.setattr(script, "_tag_exists", lambda _tag: False)
    monkeypatch.setattr(script, "_write_version", lambda _version, *, dry_run: None)
    monkeypatch.setattr(
        script,
        "_write_latest_version_api",
        lambda version, *, dry_run: latest_version_api_calls.append((version, dry_run)),
    )
    monkeypatch.setattr(script, "_require_clean_worktree", lambda: None)
    monkeypatch.setattr(script, "_git_output", lambda _command: "master")
    monkeypatch.setattr(script, "_run", lambda _command, *, dry_run: None)
    monkeypatch.setattr(sys, "argv", ["bump_release_version.py", "minor"])

    assert script.main() == 0
    assert [(str(version), dry_run) for version, dry_run in latest_version_api_calls] == [
        ("0.2.0", False)
    ]


def test_dry_run_skips_clean_worktree_requirement(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _load_script()
    commands: list[list[str]] = []
    monkeypatch.setattr(script, "_read_current_version", lambda: script.Version.parse("0.1.0"))
    monkeypatch.setattr(script, "_tag_exists", lambda _tag: False)
    monkeypatch.setattr(script, "_write_version", lambda _version, *, dry_run: None)
    monkeypatch.setattr(
        script,
        "_write_latest_version_api",
        lambda _version, *, dry_run: None,
    )
    monkeypatch.setattr(script, "_git_output", lambda _command: "master")
    monkeypatch.setattr(sys, "argv", ["bump_release_version.py", "minor", "--dry-run"])

    def fail_if_clean_worktree_checked() -> None:
        raise AssertionError("dry-run should not require a clean worktree")

    def record_run(command: list[str], *, dry_run: bool) -> None:
        assert dry_run is True
        commands.append(command)

    monkeypatch.setattr(script, "_require_clean_worktree", fail_if_clean_worktree_checked)
    monkeypatch.setattr(script, "_run", record_run)

    assert script.main() == 0
    assert [
        "git",
        "add",
        "pyproject.toml",
        "uv.lock",
        "docs/api/latest-version",
        "docs/api/latest-version.json",
    ] in commands
    assert ["git", "push", "--atomic", "origin", "master", "v0.2.0"] in commands
    assert [
        "gh",
        "release",
        "create",
        "v0.2.0",
        "--title",
        "Wattle v0.2.0",
        "--generate-notes",
    ] in commands
