"""Runtime package version helpers for Wattle."""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_PACKAGE_NAME = "wattle"
_UNKNOWN_VERSION = "unknown"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _PROJECT_ROOT / "pyproject.toml"


def get_wattle_version() -> str:
    """Return the installed Wattle package version.

    The version is sourced from the repository ``pyproject.toml`` when running
    from a source tree, otherwise from installed package metadata.
    """

    source_tree_version = _version_from_pyproject()
    if source_tree_version is not None:
        return source_tree_version
    try:
        return version(_PACKAGE_NAME)
    except PackageNotFoundError:
        return _UNKNOWN_VERSION


def _version_from_pyproject() -> str | None:
    if not _PYPROJECT.exists():
        return None
    with _PYPROJECT.open("rb") as handle:
        data = tomllib.load(handle)
    project = data.get("project")
    if not isinstance(project, dict):
        return None
    raw_version = project.get("version")
    return raw_version if isinstance(raw_version, str) else None
