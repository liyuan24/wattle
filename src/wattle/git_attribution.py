from __future__ import annotations

import shlex
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

WATTLE_COAUTHOR_TRAILER = (
    "Co-authored-by: Wattle <287834001+wattle-coding@users.noreply.github.com>"
)
HOOKS_VERSION = "v1"


@dataclass(frozen=True, slots=True)
class GitAttributionEnv:
    env: dict[str, str]
    warnings: tuple[str, ...] = ()


def apply_git_attribution_env(
    env: Mapping[str, str],
    *,
    cwd: Path,
    state_root: Path,
    command: str = "",
    enabled: bool = True,
) -> GitAttributionEnv:
    """Return an environment with Wattle's commit-msg hook injected when safe.

    The hook lives in Wattle state and is routed through Git's environment-based
    config override so the user's repository hooks are not modified. If there is
    an existing effective commit-msg hook, the generated hook chains it before
    appending Wattle's co-author trailer.
    """

    next_env = dict(env)
    if not enabled:
        return GitAttributionEnv(next_env)

    repo = _discover_repo(cwd)
    if repo is None:
        return GitAttributionEnv(next_env)

    if not _git_interpret_trailers_available(cwd):
        return GitAttributionEnv(
            next_env,
            _warning_if_commit(command, "git interpret-trailers is unavailable"),
        )

    existing_hooks_path = _effective_hooks_path(cwd, repo)
    if existing_hooks_path is None:
        return GitAttributionEnv(
            next_env,
            _warning_if_commit(command, "existing Git hooks could not be resolved"),
        )

    hooks_dir = state_root / ".wattle" / "runtime" / "git-hooks" / HOOKS_VERSION
    try:
        hooks_dir.mkdir(parents=True, exist_ok=True)
        hook_path = hooks_dir / "commit-msg"
        hook_path.write_text(
            _hook_script(existing_hooks_path),
            encoding="utf-8",
        )
        hook_path.chmod(0o755)
    except OSError as exc:
        return GitAttributionEnv(
            next_env,
            _warning_if_commit(command, f"managed Git hook could not be written: {exc}"),
        )

    try:
        return GitAttributionEnv(_inject_git_config(next_env, "core.hooksPath", str(hooks_dir)))
    except ValueError:
        return GitAttributionEnv(
            next_env,
            _warning_if_commit(command, "existing GIT_CONFIG_COUNT is invalid"),
        )


@dataclass(frozen=True, slots=True)
class _RepoInfo:
    root: Path
    git_dir: Path


def _discover_repo(cwd: Path) -> _RepoInfo | None:
    root_text = _git_output(cwd, "rev-parse", "--show-toplevel")
    git_dir_text = _git_output(cwd, "rev-parse", "--absolute-git-dir")
    if root_text is None or git_dir_text is None:
        return None
    return _RepoInfo(root=Path(root_text), git_dir=Path(git_dir_text))


def _effective_hooks_path(cwd: Path, repo: _RepoInfo) -> str | None:
    configured = _git_output(cwd, "config", "--path", "--get", "core.hooksPath")
    if configured is None:
        return str(repo.git_dir / "hooks")
    if not configured:
        return None
    return configured


def _git_interpret_trailers_available(cwd: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "interpret-trailers", "--parse"],
            cwd=cwd,
            input="",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


def _git_output(cwd: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _hook_script(existing_hooks_path: str) -> str:
    return "\n".join(
        [
            "#!/bin/sh",
            "set -eu",
            f"existing_hooks_path={shlex.quote(existing_hooks_path)}",
            'existing_hook="$existing_hooks_path/commit-msg"',
            'if [ -x "$existing_hook" ]; then',
            '  "$existing_hook" "$1"',
            "fi",
            "git interpret-trailers --in-place --if-exists doNothing \\",
            f"  --trailer {shlex.quote(WATTLE_COAUTHOR_TRAILER)} \\",
            '  "$1"',
            "",
        ]
    )


def _inject_git_config(env: dict[str, str], key: str, value: str) -> dict[str, str]:
    raw_count = env.get("GIT_CONFIG_COUNT")
    if raw_count is None:
        count = 0
    else:
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise ValueError("invalid GIT_CONFIG_COUNT") from exc
        if count < 0:
            raise ValueError("invalid GIT_CONFIG_COUNT")

    # Git 2.31+ reads the documented GIT_CONFIG_KEY_N/GIT_CONFIG_VALUE_N
    # spelling. Apple's system Git 2.24 reads GIT_CONFIG_PARAMETERS instead,
    # so set both forms while preserving any existing count entries.
    env[f"GIT_CONFIG_KEY_{count}"] = key
    env[f"GIT_CONFIG_VALUE_{count}"] = value
    env["GIT_CONFIG_COUNT"] = str(count + 1)
    env["GIT_CONFIG_PARAMETERS"] = _append_git_config_parameters(
        env.get("GIT_CONFIG_PARAMETERS", ""),
        key,
        value,
    )
    return env


def _append_git_config_parameters(existing: str, key: str, value: str) -> str:
    parameter = f"'{_quote_git_config_parameter(key)}={_quote_git_config_parameter(value)}'"
    return f"{existing} {parameter}".strip() if existing else parameter


def _quote_git_config_parameter(value: str) -> str:
    return value.replace("'", "'\\''")


def _warning_if_commit(command: str, reason: str) -> tuple[str, ...]:
    if _looks_like_git_commit(command):
        return (f"[wattle] Git commit attribution skipped: {reason}.",)
    return ()


def _looks_like_git_commit(command: str) -> bool:
    return "git commit" in command or "git\tcommit" in command
