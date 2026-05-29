"""Tests for Wattle release update helpers."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

from wattle import update


def test_normalize_version_accepts_optional_v_prefix() -> None:
    assert update.normalize_version("0.2.1") == "0.2.1"
    assert update.normalize_version("v0.2.1") == "0.2.1"
    assert update.normalize_version("latest") is None


def test_is_newer_version_compares_semver_parts() -> None:
    assert update.is_newer_version("0.2.1", "0.2.0") is True
    assert update.is_newer_version("0.10.0", "0.9.9") is True
    assert update.is_newer_version("0.2.0", "0.2.0") is False
    assert update.is_newer_version("0.1.9", "0.2.0") is False


def test_fetch_latest_version_reads_api_payload(monkeypatch) -> None:
    payload = {
        "ok": True,
        "version": "0.2.1",
        "tag": "v0.2.1",
        "installUrl": "https://wattleagent.com/install.sh",
        "releaseUrl": "https://github.com/liyuan24/wattle/releases/tag/v0.2.1",
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return json.dumps(payload).encode()

    monkeypatch.setattr(update.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())

    latest = update.fetch_latest_version()

    assert latest == update.LatestVersion(
        version="0.2.1",
        tag="v0.2.1",
        install_url="https://wattleagent.com/install.sh",
        release_url="https://github.com/liyuan24/wattle/releases/tag/v0.2.1",
    )


def test_maybe_latest_update_respects_disable_env(monkeypatch) -> None:
    monkeypatch.setenv(update.DISABLE_UPDATE_CHECK_ENV, "1")
    monkeypatch.setattr(update, "fetch_latest_version", lambda *, timeout: None)

    assert update.maybe_latest_update("0.2.0") is None


def test_run_manual_upgrade_skips_when_current_is_latest(monkeypatch) -> None:
    monkeypatch.setattr(
        update,
        "fetch_latest_version",
        lambda *, timeout: update.LatestVersion(version="0.2.0", tag="v0.2.0"),
    )
    out = io.StringIO()

    assert update.run_manual_upgrade("0.2.0", out=out) == 0
    assert out.getvalue() == "Wattle is already up to date (0.2.0).\n"


def test_run_manual_upgrade_runs_pinned_installer(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        update,
        "fetch_latest_version",
        lambda *, timeout: update.LatestVersion(version="0.2.1", tag="v0.2.1"),
    )

    def fake_run(command: list[str], *, check: bool) -> SimpleNamespace:
        calls.append(command)
        assert check is False
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(update.subprocess, "run", fake_run)
    out = io.StringIO()

    assert update.run_manual_upgrade("0.2.0", out=out) == 0
    assert calls == [
        [
            "bash",
            "-lc",
            "curl -fsSL https://wattleagent.com/install.sh | WATTLE_VERSION=0.2.1 bash",
        ]
    ]
    assert "Updating Wattle from 0.2.0 to 0.2.1" in out.getvalue()


def test_run_installer_returns_process_code(monkeypatch) -> None:
    monkeypatch.setattr(
        update.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=17),
    )

    assert update.run_installer(update.LatestVersion(version="0.2.1", tag="v0.2.1")) == 17
