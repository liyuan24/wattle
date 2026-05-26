from __future__ import annotations

import base64
import platform
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ClipboardImage:
    data: bytes
    media_type: str
    extension: str


_MACOS_CLIPBOARD_IMAGE_SCRIPT = """
use framework "AppKit"
use framework "Foundation"
use scripting additions

on writeStdout(typeName, encodedText)
    set stdout to current application's NSFileHandle's fileHandleWithStandardOutput()
    set outputText to (typeName & linefeed & encodedText)
    set stdoutData to outputText's dataUsingEncoding:(current application's NSUTF8StringEncoding)
    stdout's writeData:stdoutData
end writeStdout

set pasteboard to current application's NSPasteboard's generalPasteboard()
set imageTypes to {"public.png", "public.jpeg", "com.compuserve.gif", "org.webmproject.webp"}
repeat with imageType in imageTypes
    set imageData to pasteboard's dataForType:imageType
    if imageData is not missing value then
        set encoded to imageData's base64EncodedStringWithOptions:0
        my writeStdout((imageType as text), (encoded as text))
        return
    end if
end repeat

set imageObject to current application's NSImage's alloc()'s initWithPasteboard:pasteboard
if imageObject is not missing value then
    set tiffData to imageObject's TIFFRepresentation()
    if tiffData is not missing value then
        set bitmap to current application's NSBitmapImageRep's imageRepWithData:tiffData
        if bitmap is not missing value then
            set fileType to current application's NSBitmapImageFileTypePNG
            set properties to current application's NSDictionary's dictionary()
            set pngData to bitmap's representationUsingType:fileType |properties|:properties
            if pngData is not missing value then
                set encoded to pngData's base64EncodedStringWithOptions:0
                my writeStdout("public.png", (encoded as text))
            end if
        end if
    end if
end if
"""

_TYPE_INFO = {
    "public.png": ("image/png", ".png"),
    "public.jpeg": ("image/jpeg", ".jpg"),
    "com.compuserve.gif": ("image/gif", ".gif"),
    "org.webmproject.webp": ("image/webp", ".webp"),
}


def read_clipboard_image() -> ClipboardImage | None:
    """Return image bytes from the OS clipboard when supported and available."""

    system = platform.system()
    if system == "Darwin":
        return _read_macos_clipboard_image()
    if system in {"Linux", "FreeBSD", "OpenBSD", "NetBSD"}:
        return _read_unix_clipboard_image()
    return None


def _read_macos_clipboard_image() -> ClipboardImage | None:
    try:
        result = subprocess.run(
            ["osascript", "-e", _MACOS_CLIPBOARD_IMAGE_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    header, separator, payload = result.stdout.partition("\n")
    if not separator or not payload.strip():
        return None
    media_type, extension = _TYPE_INFO.get(header.strip(), ("image/png", ".png"))
    try:
        data = base64.b64decode(payload.strip(), validate=True)
    except ValueError:
        return None
    if not data:
        return None
    return ClipboardImage(data=data, media_type=media_type, extension=extension)


def _read_unix_clipboard_image() -> ClipboardImage | None:
    candidates: list[tuple[list[str], str]] = []
    if shutil.which("wl-paste"):
        candidates.extend(
            (["wl-paste", "--no-newline", "--type", type_name], type_name)
            for type_name in _TYPE_INFO
        )
    if shutil.which("xclip"):
        candidates.extend(
            (["xclip", "-selection", "clipboard", "-t", type_name, "-o"], type_name)
            for type_name in _TYPE_INFO
        )
    if shutil.which("xsel"):
        candidates.extend(
            (["xsel", "--clipboard", "--output", "--mime-type", type_name], type_name)
            for type_name in _TYPE_INFO
        )

    for command, type_name in candidates:
        image = _read_unix_clipboard_type(command, type_name)
        if image is not None:
            return image
    return None


def _read_unix_clipboard_type(command: list[str], type_name: str) -> ClipboardImage | None:
    media_type, extension = _TYPE_INFO[type_name]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=2,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    return ClipboardImage(data=result.stdout, media_type=media_type, extension=extension)
