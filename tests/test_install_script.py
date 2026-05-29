"""Tests for the hosted shell installer."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_hosted_install_script_matches_local_script() -> None:
    assert (REPO_ROOT / "docs" / "install.sh").read_text(encoding="utf-8") == (
        REPO_ROOT / "scripts" / "install.sh"
    ).read_text(encoding="utf-8")


def test_install_scripts_are_valid_bash() -> None:
    subprocess.run(["bash", "-n", "scripts/install.sh"], cwd=REPO_ROOT, check=True)
    subprocess.run(["bash", "-n", "scripts/install-dev.sh"], cwd=REPO_ROOT, check=True)
    subprocess.run(["bash", "-n", "docs/install.sh"], cwd=REPO_ROOT, check=True)


def test_user_installer_is_not_editable() -> None:
    script = (REPO_ROOT / "scripts" / "install.sh").read_text(encoding="utf-8")

    assert "uv tool install --force \"${REPO_DIR}\"" in script
    assert "tool install --force -e" not in script
    assert "WATTLE_EDITABLE" not in script


def test_developer_installer_is_editable() -> None:
    script = (REPO_ROOT / "scripts" / "install-dev.sh").read_text(encoding="utf-8")

    assert 'uv tool install --force -e "${REPO_DIR}"' in script
