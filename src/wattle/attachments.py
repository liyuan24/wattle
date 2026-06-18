from __future__ import annotations

import base64
from pathlib import Path

from wattle.providers.base import ImageBlock


class AttachmentUnavailableError(RuntimeError):
    """Raised when a local attachment cannot be read for provider serialization."""

    def __init__(self, block: ImageBlock, reason: str) -> None:
        self.block = block
        self.reason = reason
        super().__init__(unavailable_image_text(block, reason=reason))


def image_unavailable_reason(block: ImageBlock) -> str | None:
    """Return why an image cannot be read, or None when it is readable."""

    try:
        with Path(block.path).open("rb") as handle:
            handle.read(1)
    except (FileNotFoundError, PermissionError, OSError) as exc:
        return _read_error_reason(exc)
    return None


def image_bytes(block: ImageBlock) -> bytes:
    """Read image bytes or raise a Wattle-local attachment error."""

    try:
        return Path(block.path).read_bytes()
    except (FileNotFoundError, PermissionError, OSError) as exc:
        raise AttachmentUnavailableError(block, _read_error_reason(exc)) from exc


def image_base64(block: ImageBlock) -> str:
    """Return base64-encoded image bytes for provider payloads."""

    return base64.b64encode(image_bytes(block)).decode("ascii")


def image_data_url(block: ImageBlock) -> str:
    """Return a data URL for an image attachment."""

    return f"data:{block.media_type};base64,{image_base64(block)}"


def unavailable_image_text(block: ImageBlock, *, reason: str | None = None) -> str:
    """Return model-visible text for an attachment that cannot be sent."""

    details = (
        f"filename={block.filename} media_type={block.media_type} "
        f"size_bytes={block.size_bytes}"
    )
    if reason:
        return (
            f"[image omitted: attached file is no longer available at {block.path}; "
            f"{details}; reason={reason}]"
        )
    return f"[image omitted: attached file is no longer available at {block.path}; {details}]"


def _read_error_reason(error: OSError) -> str:
    text = str(error).strip()
    return text if text else type(error).__name__
