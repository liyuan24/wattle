#!/usr/bin/env python3
"""Bump Wattle's release version and publish a GitHub release."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
UV_LOCK = REPO_ROOT / "uv.lock"
LATEST_VERSION_API = REPO_ROOT / "docs" / "api" / "latest-version"
LATEST_VERSION_JSON = REPO_ROOT / "docs" / "api" / "latest-version.json"
REPO_URL = "https://github.com/liyuan24/wattle"
INSTALL_URL = "https://wattleagent.com/install.sh"
VERSION_RE = re.compile(r'(?m)^(version = ")(?P<version>\d+\.\d+\.\d+)(")$')
LOCK_WATTLE_RE = re.compile(
    r'(?m)^(name = "wattle"\nversion = ")(?P<version>\d+\.\d+\.\d+)(")$'
)


@dataclass(frozen=True)
class Version:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, raw: str) -> Version:
        parts = raw.split(".")
        if len(parts) != 3:
            raise ValueError(f"expected MAJOR.MINOR.PATCH version, got {raw!r}")
        return cls(*(int(part) for part in parts))

    def bump(self, bump_type: str) -> Version:
        if bump_type == "patch":
            return Version(self.major, self.minor, self.patch + 1)
        if bump_type == "minor":
            return Version(self.major, self.minor + 1, 0)
        if bump_type == "major":
            return Version(self.major + 1, 0, 0)
        raise ValueError(f"unsupported bump type: {bump_type}")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def _run(command: list[str], *, dry_run: bool) -> None:
    rendered = " ".join(command)
    if dry_run:
        print(f"dry-run: {rendered}")
        return
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _git_output(command: list[str]) -> str:
    return subprocess.check_output(command, cwd=REPO_ROOT, text=True).strip()


def _atomic_write(path: Path, content: str) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    os.replace(temporary_path, path)


def _require_clean_worktree() -> None:
    status = _git_output(["git", "status", "--porcelain"])
    if status:
        raise SystemExit(
            "Working tree is not clean. Commit or stash changes before releasing, "
            "or use --dry-run to preview the release."
        )


def _replace_one(pattern: re.Pattern[str], text: str, version: str, *, path: Path) -> str:
    replacement_count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal replacement_count
        replacement_count += 1
        return f"{match.group(1)}{version}{match.group(3)}"

    new_text = pattern.sub(replace, text)
    if replacement_count != 1:
        relative_path = path.relative_to(REPO_ROOT)
        raise SystemExit(f"Could not find exactly one Wattle version in {relative_path}")
    return new_text


def _read_current_version() -> Version:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = VERSION_RE.search(text)
    if match is None:
        raise SystemExit("Could not find project version in pyproject.toml")
    return Version.parse(match.group("version"))


def _write_version(new_version: Version, *, dry_run: bool) -> None:
    pyproject_text = PYPROJECT.read_text(encoding="utf-8")
    new_pyproject_text = _replace_one(VERSION_RE, pyproject_text, str(new_version), path=PYPROJECT)
    new_lock_text: str | None = None
    if UV_LOCK.exists():
        lock_text = UV_LOCK.read_text(encoding="utf-8")
        new_lock_text = _replace_one(LOCK_WATTLE_RE, lock_text, str(new_version), path=UV_LOCK)

    if dry_run:
        print(f"dry-run: update pyproject.toml to {new_version}")
        if new_lock_text is not None:
            print(f"dry-run: update uv.lock to {new_version}")
        return

    original_pyproject_text = pyproject_text
    original_lock_text = lock_text if UV_LOCK.exists() else None
    try:
        _atomic_write(PYPROJECT, new_pyproject_text)
        if new_lock_text is not None:
            _atomic_write(UV_LOCK, new_lock_text)
    except OSError:
        _atomic_write(PYPROJECT, original_pyproject_text)
        if original_lock_text is not None:
            _atomic_write(UV_LOCK, original_lock_text)
        raise


def _latest_version_payload(new_version: Version) -> dict[str, str | bool]:
    tag = f"v{new_version}"
    return {
        "ok": True,
        "version": str(new_version),
        "tag": tag,
        "repo": REPO_URL,
        "releaseUrl": f"{REPO_URL}/releases/tag/{tag}",
        "installUrl": INSTALL_URL,
    }


def _write_latest_version_api(new_version: Version, *, dry_run: bool) -> None:
    if dry_run:
        print(f"dry-run: update docs/api/latest-version to {new_version}")
        print(f"dry-run: update docs/api/latest-version.json to {new_version}")
        return

    content = json.dumps(_latest_version_payload(new_version), indent=2, sort_keys=True) + "\n"
    LATEST_VERSION_API.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(LATEST_VERSION_API, content)
    _atomic_write(LATEST_VERSION_JSON, content)


def _tag_exists(tag: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}"],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bump Wattle's pyproject/lockfile version and publish a release."
    )
    parser.add_argument("bump", choices=("patch", "minor", "major"), help="Version bump type.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the release actions without changing files, committing, tagging, or pushing.",
    )
    args = parser.parse_args()

    if not args.dry_run:
        _require_clean_worktree()

    current = _read_current_version()
    new_version = current.bump(args.bump)
    tag = f"v{new_version}"
    if _tag_exists(tag):
        raise SystemExit(f"Tag already exists: {tag}")

    print(f"Bumping Wattle version: {current} -> {new_version}")
    _write_version(new_version, dry_run=args.dry_run)
    _write_latest_version_api(new_version, dry_run=args.dry_run)

    _run(["uv", "run", "pytest"], dry_run=args.dry_run)
    _run(
        [
            "git",
            "add",
            "pyproject.toml",
            "uv.lock",
            "docs/api/latest-version",
            "docs/api/latest-version.json",
        ],
        dry_run=args.dry_run,
    )
    _run(["git", "commit", "-m", f"Release {tag}"], dry_run=args.dry_run)
    _run(["git", "tag", tag], dry_run=args.dry_run)
    if args.dry_run:
        print(f"dry-run: would create release commit and tag {tag}")
    else:
        print(f"Created release commit and tag {tag}")

    branch = _git_output(["git", "branch", "--show-current"]) or "HEAD"
    _run(["git", "push", "--atomic", "origin", branch, tag], dry_run=args.dry_run)
    if args.dry_run:
        print(f"dry-run: would push {branch} and {tag} to origin")
    else:
        print(f"Pushed {branch} and {tag} to origin")

    _run(
        [
            "gh",
            "release",
            "create",
            tag,
            "--title",
            f"Wattle {tag}",
            "--generate-notes",
        ],
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"dry-run: would publish GitHub release {tag}")
    else:
        print(f"Published GitHub release {tag}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
