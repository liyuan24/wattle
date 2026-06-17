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

    next_env = _strip_wattle_git_hook_config(env)
    if not enabled:
        return GitAttributionEnv(next_env)
    if _git_config_count_is_invalid(next_env):
        return GitAttributionEnv(
            next_env,
            _warning_if_commit(command, "existing GIT_CONFIG_COUNT is invalid"),
        )

    repo = _discover_repo(cwd, next_env)
    if repo is None:
        return GitAttributionEnv(next_env)

    if not _git_interpret_trailers_available(cwd, next_env):
        return GitAttributionEnv(
            next_env,
            _warning_if_commit(command, "git interpret-trailers is unavailable"),
        )

    existing_hooks_path = _effective_hooks_path(cwd, repo, next_env)
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


def _git_config_count_is_invalid(env: Mapping[str, str]) -> bool:
    raw_count = env.get("GIT_CONFIG_COUNT")
    if raw_count is None:
        return False
    try:
        return int(raw_count) < 0
    except ValueError:
        return True


def _strip_wattle_git_hook_config(env: Mapping[str, str]) -> dict[str, str]:
    next_env = dict(env)
    raw_count = next_env.get("GIT_CONFIG_COUNT")
    if raw_count is not None:
        try:
            count = int(raw_count)
        except ValueError:
            count = -1
        if count >= 0:
            entries: list[tuple[str, str]] = []
            removed_count_entry = False
            for index in range(count):
                key_name = f"GIT_CONFIG_KEY_{index}"
                value_name = f"GIT_CONFIG_VALUE_{index}"
                key = next_env.get(key_name)
                value = next_env.get(value_name)
                if key == "core.hooksPath" and _is_wattle_hooks_path(value):
                    removed_count_entry = True
                    continue
                if key is not None and value is not None:
                    entries.append((key, value))
            if removed_count_entry:
                for index in range(count):
                    next_env.pop(f"GIT_CONFIG_KEY_{index}", None)
                    next_env.pop(f"GIT_CONFIG_VALUE_{index}", None)
                if entries:
                    next_env["GIT_CONFIG_COUNT"] = str(len(entries))
                    for index, (key, value) in enumerate(entries):
                        next_env[f"GIT_CONFIG_KEY_{index}"] = key
                        next_env[f"GIT_CONFIG_VALUE_{index}"] = value
                else:
                    next_env.pop("GIT_CONFIG_COUNT", None)

    parameters = next_env.get("GIT_CONFIG_PARAMETERS")
    if parameters:
        stripped = _strip_wattle_git_config_parameters(parameters)
        if stripped:
            next_env["GIT_CONFIG_PARAMETERS"] = stripped
        else:
            next_env.pop("GIT_CONFIG_PARAMETERS", None)
    return next_env


def _strip_wattle_git_config_parameters(parameters: str) -> str:
    try:
        tokens = shlex.split(parameters)
    except ValueError:
        return parameters

    kept: list[str] = []
    removed = False
    for token in tokens:
        key, separator, value = token.partition("=")
        if separator and key == "core.hooksPath" and _is_wattle_hooks_path(value):
            removed = True
            continue
        kept.append(token)
    if not removed:
        return parameters

    rebuilt = ""
    for token in kept:
        key, separator, value = token.partition("=")
        if separator:
            rebuilt = _append_git_config_parameters(rebuilt, key, value)
        else:
            quoted = f"'{_quote_git_config_parameter(token)}'"
            rebuilt = f"{rebuilt} {quoted}".strip() if rebuilt else quoted
    return rebuilt


def _is_wattle_hooks_path(value: str | None) -> bool:
    if not value:
        return False
    parts = Path(value).parts
    return any(
        parts[index : index + 4] == (".wattle", "runtime", "git-hooks", HOOKS_VERSION)
        for index in range(len(parts) - 3)
    )


@dataclass(frozen=True, slots=True)
class _RepoInfo:
    root: Path
    git_dir: Path


def _discover_repo(cwd: Path, env: Mapping[str, str]) -> _RepoInfo | None:
    root_text = _git_output(cwd, env, "rev-parse", "--show-toplevel")
    git_dir_text = _git_output(cwd, env, "rev-parse", "--absolute-git-dir")
    if root_text is None or git_dir_text is None:
        return None
    return _RepoInfo(root=Path(root_text), git_dir=Path(git_dir_text))


def _effective_hooks_path(cwd: Path, repo: _RepoInfo, env: Mapping[str, str]) -> str | None:
    configured = _git_output(cwd, env, "config", "--path", "--get", "core.hooksPath")
    if configured is None:
        return str(repo.git_dir / "hooks")
    if not configured:
        return None
    return configured


def _git_interpret_trailers_available(cwd: Path, env: Mapping[str, str]) -> bool:
    try:
        result = subprocess.run(
            ["git", "interpret-trailers", "--parse"],
            cwd=cwd,
            input="",
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


def _git_output(cwd: Path, env: Mapping[str, str], *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env,
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
