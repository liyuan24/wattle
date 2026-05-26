from __future__ import annotations

import base64
import subprocess
from types import SimpleNamespace

from wattle import clipboard


def test_read_clipboard_image_reads_macos_png(monkeypatch) -> None:
    data = b"fake-png"
    monkeypatch.setattr(clipboard.platform, "system", lambda: "Darwin")

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout="public.png\n" + base64.b64encode(data).decode("ascii"),
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    image = clipboard.read_clipboard_image()

    assert image is not None
    assert image.data == data
    assert image.media_type == "image/png"
    assert image.extension == ".png"


def test_read_clipboard_image_returns_none_without_image(monkeypatch) -> None:
    monkeypatch.setattr(clipboard.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=""),
    )

    assert clipboard.read_clipboard_image() is None


def test_read_clipboard_image_is_noop_off_macos(monkeypatch) -> None:
    monkeypatch.setattr(clipboard.platform, "system", lambda: "Linux")

    assert clipboard.read_clipboard_image() is None
