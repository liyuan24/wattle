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


def test_user_agent_uses_version_when_available() -> None:
    assert update._user_agent("0.4.1") == "Wattle/0.4.1"
    assert update._user_agent("v0.4.1") == "Wattle/0.4.1"
    assert update._user_agent(None) == "Wattle"
    assert update._user_agent("unknown") == "Wattle"


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

    requests: list[object] = []

    def fake_urlopen(request: object, **_kwargs: object) -> Response:
        requests.append(request)
        return Response()

    monkeypatch.setattr(update.urllib.request, "urlopen", fake_urlopen)

    latest = update.fetch_latest_version(current_version="0.4.1")

    assert latest == update.LatestVersion(
        version="0.2.1",
        tag="v0.2.1",
        install_url="https://wattleagent.com/install.sh",
        release_url="https://github.com/liyuan24/wattle/releases/tag/v0.2.1",
    )
    assert len(requests) == 1
    request = requests[0]
    assert isinstance(request, update.urllib.request.Request)
    assert request.get_header("User-agent") == "Wattle/0.4.1"
    assert request.get_header("Accept") == "application/json"


def test_fetch_latest_version_returns_none_for_malformed_url_override(monkeypatch) -> None:
    monkeypatch.setenv(update.LATEST_VERSION_URL_ENV, "not a url")

    assert update.fetch_latest_version(current_version="0.4.1") is None


def test_maybe_latest_update_respects_disable_env(monkeypatch) -> None:
    monkeypatch.setenv(update.DISABLE_UPDATE_CHECK_ENV, "1")
    monkeypatch.setattr(update, "fetch_latest_version", lambda *, timeout, current_version: None)

    assert update.maybe_latest_update("0.2.0") is None


def test_maybe_latest_update_passes_current_version(monkeypatch) -> None:
    calls: list[tuple[float, str | None]] = []

    def fake_fetch(*, timeout: float, current_version: str | None) -> update.LatestVersion:
        calls.append((timeout, current_version))
        return update.LatestVersion(version="0.4.1", tag="v0.4.1")

    monkeypatch.setattr(update, "fetch_latest_version", fake_fetch)

    assert update.maybe_latest_update("0.4.0", timeout=3.0) == update.LatestVersion(
        version="0.4.1",
        tag="v0.4.1",
    )
    assert calls == [(3.0, "0.4.0")]


def test_run_manual_upgrade_skips_when_current_is_latest(monkeypatch) -> None:
    calls: list[tuple[float, str | None]] = []

    def fake_fetch(*, timeout: float, current_version: str | None) -> update.LatestVersion:
        calls.append((timeout, current_version))
        return update.LatestVersion(version="0.2.0", tag="v0.2.0")

    monkeypatch.setattr(update, "fetch_latest_version", fake_fetch)
    out = io.StringIO()

    assert update.run_manual_upgrade("0.2.0", out=out) == 0
    assert calls == [(10.0, "0.2.0")]
    assert out.getvalue() == "Wattle is already up to date (0.2.0).\n"


def test_run_manual_upgrade_runs_pinned_installer(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        update,
        "fetch_latest_version",
        lambda *, timeout, current_version: update.LatestVersion(
            version="0.2.1",
            tag="v0.2.1",
        ),
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
