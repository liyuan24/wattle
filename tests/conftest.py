from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from wattle import settings


_REAL_SETTINGS_PATH = Path.home() / ".wattle" / "settings.json"
_REAL_SETTINGS_AUDIT_PATH = Path.home() / ".wattle" / "settings.audit.jsonl"


def _fingerprint(path: Path) -> tuple[bool, int | None, int | None, str | None]:
    try:
        stat = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return (False, None, None, None)
    return (True, stat.st_mtime_ns, stat.st_size, digest)


@pytest.fixture(autouse=True)
def isolate_wattle_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Keep tests from reading or writing the user's real Wattle settings.

    Tests that exercise settings persistence should use the public
    ``WATTLE_SETTINGS_PATH`` override. Defaulting every test to a per-test
    settings file also protects subprocess-based TUI tests because child
    processes inherit the environment unless they deliberately override it.
    """

    real_before = {
        _REAL_SETTINGS_PATH: _fingerprint(_REAL_SETTINGS_PATH),
        _REAL_SETTINGS_AUDIT_PATH: _fingerprint(_REAL_SETTINGS_AUDIT_PATH),
    }
    monkeypatch.setenv(settings.SETTINGS_PATH_ENV, str(tmp_path / "settings.json"))

    yield

    changed_paths = [
        str(path)
        for path, before in real_before.items()
        if _fingerprint(path) != before
    ]
    if changed_paths:
        raise AssertionError(
            "Tests must not modify real Wattle settings files: "
            + ", ".join(changed_paths)
        )
