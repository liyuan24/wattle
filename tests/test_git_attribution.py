from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

from wattle.git_attribution import WATTLE_COAUTHOR_TRAILER, apply_git_attribution_env
from wattle.runtime import WattleRuntime
from wattle.tools.bash import BashTool


def _git(
    repo: Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _init_repo(repo: Path) -> None:
    repo.mkdir()
    assert _git(repo, "init").returncode == 0
    assert _git(repo, "config", "user.name", "Tester").returncode == 0
    assert _git(repo, "config", "user.email", "tester@example.com").returncode == 0


def _make_commit(
    repo: Path, message: str, *, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    tracked = repo / "file.txt"
    tracked.write_text(message, encoding="utf-8")
    assert _git(repo, "add", "file.txt", env=env).returncode == 0
    return _git(repo, "commit", "-m", message, env=env)


def _commit_message(repo: Path) -> str:
    result = _git(repo, "log", "-1", "--pretty=%B")
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_hook_appends_wattle_trailer(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)

    attribution = apply_git_attribution_env(os.environ, cwd=repo, state_root=tmp_path)
    result = _make_commit(repo, "test", env=attribution.env)

    assert result.returncode == 0, result.stderr
    assert WATTLE_COAUTHOR_TRAILER in _commit_message(repo)
    assert not (repo / ".git" / "hooks" / "commit-msg").exists()


def test_hook_is_idempotent_when_trailer_already_exists(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    attribution = apply_git_attribution_env(os.environ, cwd=repo, state_root=tmp_path)

    result = _make_commit(
        repo,
        f"test\n\n{WATTLE_COAUTHOR_TRAILER}",
        env=attribution.env,
    )

    assert result.returncode == 0, result.stderr
    assert _commit_message(repo).count(WATTLE_COAUTHOR_TRAILER) == 1


def test_existing_executable_commit_msg_hook_is_chained(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    hook = repo / ".git" / "hooks" / "commit-msg"
    hook.write_text(
        "#!/bin/sh\nprintf '\\nExisting-Trailer: yes\\n' >> \"$1\"\n",
        encoding="utf-8",
    )
    hook.chmod(hook.stat().st_mode | stat.S_IXUSR)
    attribution = apply_git_attribution_env(os.environ, cwd=repo, state_root=tmp_path)

    result = _make_commit(repo, "test", env=attribution.env)

    assert result.returncode == 0, result.stderr
    message = _commit_message(repo)
    assert "Existing-Trailer: yes" in message
    assert WATTLE_COAUTHOR_TRAILER in message


def test_existing_hook_failure_blocks_commit_and_wattle_trailer(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    hook = repo / ".git" / "hooks" / "commit-msg"
    hook.write_text("#!/bin/sh\nexit 42\n", encoding="utf-8")
    hook.chmod(hook.stat().st_mode | stat.S_IXUSR)
    attribution = apply_git_attribution_env(os.environ, cwd=repo, state_root=tmp_path)

    result = _make_commit(repo, "test", env=attribution.env)

    assert result.returncode == 1
    assert WATTLE_COAUTHOR_TRAILER not in result.stdout
    assert _git(repo, "log", "--oneline").stdout == ""


def test_configured_hooks_path_is_chained(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    _init_repo(repo)
    assert _git(repo, "config", "core.hooksPath", str(hooks)).returncode == 0
    hook = hooks / "commit-msg"
    hook.write_text(
        "#!/bin/sh\nprintf '\\nConfigured-Hook: yes\\n' >> \"$1\"\n",
        encoding="utf-8",
    )
    hook.chmod(hook.stat().st_mode | stat.S_IXUSR)
    attribution = apply_git_attribution_env(os.environ, cwd=repo, state_root=tmp_path)

    result = _make_commit(repo, "test", env=attribution.env)

    assert result.returncode == 0, result.stderr
    message = _commit_message(repo)
    assert "Configured-Hook: yes" in message
    assert WATTLE_COAUTHOR_TRAILER in message


def test_env_injection_preserves_existing_git_config_count(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    env = {
        **os.environ,
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "safe.directory",
        "GIT_CONFIG_VALUE_0": str(repo),
    }

    attribution = apply_git_attribution_env(env, cwd=repo, state_root=tmp_path)

    assert attribution.env["GIT_CONFIG_COUNT"] == "2"
    assert attribution.env["GIT_CONFIG_KEY_0"] == "safe.directory"
    assert attribution.env["GIT_CONFIG_VALUE_0"] == str(repo)
    assert attribution.env["GIT_CONFIG_KEY_1"] == "core.hooksPath"
    assert Path(attribution.env["GIT_CONFIG_VALUE_1"]).name == "v1"
    assert "core.hooksPath=" in attribution.env["GIT_CONFIG_PARAMETERS"]


def test_invalid_existing_git_config_count_skips_with_warning(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    env = {**os.environ, "GIT_CONFIG_COUNT": "invalid"}

    attribution = apply_git_attribution_env(
        env,
        cwd=repo,
        state_root=tmp_path,
        command="git commit -m test",
    )

    assert attribution.env["GIT_CONFIG_COUNT"] == "invalid"
    assert attribution.warnings == (
        "[wattle] Git commit attribution skipped: existing GIT_CONFIG_COUNT is invalid.",
    )


def test_bash_tool_git_commit_adds_wattle_trailer(tmp_path: Path, monkeypatch) -> None:
    settings_path = tmp_path / "settings.json"
    monkeypatch.setenv("WATTLE_SETTINGS_PATH", str(settings_path))
    repo = tmp_path / "repo"
    _init_repo(repo)
    tool = BashTool(runtime=WattleRuntime(root=tmp_path), cwd=repo)

    output = tool.run("echo hello > file.txt && git add file.txt && git commit -m test")

    assert "[exit" not in output
    assert WATTLE_COAUTHOR_TRAILER in _commit_message(repo)


def test_bash_tool_respects_git_commit_attribution_setting(tmp_path: Path, monkeypatch) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text('{"git_commit_attribution": false}\n', encoding="utf-8")
    monkeypatch.setenv("WATTLE_SETTINGS_PATH", str(settings_path))
    repo = tmp_path / "repo"
    _init_repo(repo)
    tool = BashTool(runtime=WattleRuntime(root=tmp_path), cwd=repo)

    output = tool.run("echo hello > file.txt && git add file.txt && git commit -m test")

    assert "[exit" not in output
    assert WATTLE_COAUTHOR_TRAILER not in _commit_message(repo)
