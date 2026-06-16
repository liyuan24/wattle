"""Update checks and installer dispatch for Wattle releases."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import termios
import tty
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import TextIO

LATEST_VERSION_URL = "https://wattleagent.com/api/latest-version"
INSTALL_URL = "https://wattleagent.com/install.sh"
DISABLE_UPDATE_CHECK_ENV = "WATTLE_DISABLE_UPDATE_CHECK"
LATEST_VERSION_URL_ENV = "WATTLE_LATEST_VERSION_URL"
USER_AGENT_PRODUCT = "Wattle"

_VERSION_RE = re.compile(r"^v?(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$")


@dataclass(frozen=True)
class LatestVersion:
    version: str
    tag: str
    install_url: str = INSTALL_URL
    release_url: str | None = None


def normalize_version(raw: str) -> str | None:
    match = _VERSION_RE.match(raw.strip())
    if match is None:
        return None
    return ".".join(match.group(part) for part in ("major", "minor", "patch"))


def compare_versions(left: str, right: str) -> int:
    left_normalized = normalize_version(left)
    right_normalized = normalize_version(right)
    if left_normalized is None or right_normalized is None:
        return 0
    left_parts = tuple(int(part) for part in left_normalized.split("."))
    right_parts = tuple(int(part) for part in right_normalized.split("."))
    return (left_parts > right_parts) - (left_parts < right_parts)


def is_newer_version(latest: str, current: str) -> bool:
    return compare_versions(latest, current) > 0


def _user_agent(current_version: str | None) -> str:
    version = normalize_version(current_version or "")
    if version is None:
        return USER_AGENT_PRODUCT
    return f"{USER_AGENT_PRODUCT}/{version}"


def fetch_latest_version(
    *,
    timeout: float = 2.0,
    current_version: str | None = None,
) -> LatestVersion | None:
    url = os.environ.get(LATEST_VERSION_URL_ENV, LATEST_VERSION_URL)
    try:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": _user_agent(current_version),
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, ValueError):
        return None

    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return None
    raw_version = payload.get("version")
    if not isinstance(raw_version, str):
        return None
    version = normalize_version(raw_version)
    if version is None:
        return None
    raw_tag = payload.get("tag")
    tag = raw_tag if isinstance(raw_tag, str) and raw_tag else f"v{version}"
    install_url = payload.get("installUrl")
    release_url = payload.get("releaseUrl")
    return LatestVersion(
        version=version,
        tag=tag,
        install_url=install_url if isinstance(install_url, str) else INSTALL_URL,
        release_url=release_url if isinstance(release_url, str) else None,
    )


def install_command(latest: LatestVersion) -> str:
    return f"curl -fsSL {latest.install_url} | WATTLE_VERSION={latest.version} bash"


def run_installer(latest: LatestVersion) -> int:
    command = install_command(latest)
    return subprocess.run(["bash", "-lc", command], check=False).returncode


def maybe_latest_update(current_version: str, *, timeout: float = 2.0) -> LatestVersion | None:
    if os.environ.get(DISABLE_UPDATE_CHECK_ENV):
        return None
    latest = fetch_latest_version(timeout=timeout, current_version=current_version)
    if latest is None or not is_newer_version(latest.version, current_version):
        return None
    return latest


def run_manual_upgrade(
    current_version: str,
    *,
    out: TextIO = sys.stdout,
    err: TextIO = sys.stderr,
) -> int:
    latest = fetch_latest_version(timeout=10.0, current_version=current_version)
    if latest is None:
        err.write("Could not check for Wattle updates.\n")
        err.flush()
        return 1
    if not is_newer_version(latest.version, current_version):
        out.write(f"Wattle is already up to date ({current_version}).\n")
        out.flush()
        return 0

    out.write(f"Updating Wattle from {current_version} to {latest.version}...\n")
    out.write(f"{install_command(latest)}\n")
    out.flush()
    return run_installer(latest)


def prompt_for_tui_update(
    current_version: str,
    latest: LatestVersion,
    *,
    input_stream: TextIO = sys.stdin,
    out: TextIO = sys.stdout,
) -> bool:
    """Return True when the update prompt handled startup and Wattle should exit."""

    if not _is_interactive(input_stream, out):
        return False

    options = (
        f"Update from {current_version} to {latest.version}",
        "Skip update",
    )
    selected = 0
    rendered_lines = 0

    def draw() -> None:
        nonlocal rendered_lines
        if rendered_lines:
            out.write(f"\x1b[{rendered_lines}A\r\x1b[J")
        else:
            out.write("\r\x1b[J")
        lines = [
            f"Wattle {latest.version} is available. You have {current_version}.",
            *(
                f" {'>' if index == selected else ' '} {option}"
                for index, option in enumerate(options)
            ),
            "Use up/down and Enter to select.",
        ]
        out.write(
            "\n".join(lines) + "\n"
        )
        rendered_lines = len(lines)
        out.flush()

    fd = input_stream.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        draw()
        while True:
            data = os.read(fd, 16).decode(errors="ignore")
            if data.startswith(("\x1b[A", "\x1bOA")):
                selected = max(0, selected - 1)
                draw()
            elif data.startswith(("\x1b[B", "\x1bOB")):
                selected = min(len(options) - 1, selected + 1)
                draw()
            elif "\r" in data or "\n" in data:
                out.write("\n")
                out.flush()
                if selected == 0:
                    run_installer(latest)
                    return True
                return False
            elif "\x03" in data or data == "\x1b":
                out.write("\n")
                out.flush()
                return False
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _is_interactive(input_stream: TextIO, out: TextIO) -> bool:
    return (
        hasattr(input_stream, "isatty")
        and input_stream.isatty()
        and hasattr(out, "isatty")
        and out.isatty()
    )
