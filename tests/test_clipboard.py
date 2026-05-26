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


def test_read_clipboard_image_reads_wayland_png(monkeypatch) -> None:
    data = b"fake-png"
    monkeypatch.setattr(clipboard.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        clipboard.shutil,
        "which",
        lambda name: "/usr/bin/wl-paste" if name == "wl-paste" else None,
    )

    commands: list[list[str]] = []

    def fake_run(command, *_args, **_kwargs):
        commands.append(command)
        if command == ["wl-paste", "--no-newline", "--type", "public.png"]:
            return SimpleNamespace(returncode=0, stdout=data)
        return SimpleNamespace(returncode=1, stdout=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    image = clipboard.read_clipboard_image()

    assert image is not None
    assert image.data == data
    assert image.media_type == "image/png"
    assert image.extension == ".png"
    assert commands[0] == ["wl-paste", "--no-newline", "--type", "public.png"]


def test_read_clipboard_image_reads_x11_jpeg_after_png_miss(monkeypatch) -> None:
    data = b"fake-jpeg"
    monkeypatch.setattr(clipboard.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        clipboard.shutil,
        "which",
        lambda name: "/usr/bin/xclip" if name == "xclip" else None,
    )

    def fake_run(command, *_args, **_kwargs):
        if command == ["xclip", "-selection", "clipboard", "-t", "public.jpeg", "-o"]:
            return SimpleNamespace(returncode=0, stdout=data)
        return SimpleNamespace(returncode=1, stdout=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    image = clipboard.read_clipboard_image()

    assert image is not None
    assert image.data == data
    assert image.media_type == "image/jpeg"
    assert image.extension == ".jpg"


def test_read_clipboard_image_returns_none_without_unix_clipboard_tool(monkeypatch) -> None:
    monkeypatch.setattr(clipboard.platform, "system", lambda: "Linux")
    monkeypatch.setattr(clipboard.shutil, "which", lambda _name: None)

    assert clipboard.read_clipboard_image() is None


def test_read_clipboard_image_is_noop_on_unknown_platform(monkeypatch) -> None:
    monkeypatch.setattr(clipboard.platform, "system", lambda: "Windows")

    assert clipboard.read_clipboard_image() is None
