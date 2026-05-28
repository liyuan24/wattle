"""Tests for Wattle version helpers."""

from __future__ import annotations

from wattle import version


def test_version_prefers_pyproject_over_package_metadata(
    monkeypatch,
    tmp_path,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "wattle"\nversion = "4.5.6"\n', encoding="utf-8")
    monkeypatch.setattr(version, "_PYPROJECT", pyproject)
    monkeypatch.setattr(version, "version", lambda _package: "1.2.3")

    assert version.get_wattle_version() == "4.5.6"


def test_version_falls_back_to_package_metadata_without_pyproject(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(version, "_PYPROJECT", tmp_path / "missing-pyproject.toml")
    monkeypatch.setattr(version, "version", lambda _package: "1.2.3")

    assert version.get_wattle_version() == "1.2.3"


def test_version_falls_back_to_pyproject_when_package_metadata_missing(
    monkeypatch,
    tmp_path,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "wattle"\nversion = "4.5.6"\n', encoding="utf-8")
    monkeypatch.setattr(version, "_PYPROJECT", pyproject)

    def missing_package(_package: str) -> str:
        raise version.PackageNotFoundError

    monkeypatch.setattr(version, "version", missing_package)

    assert version.get_wattle_version() == "4.5.6"
