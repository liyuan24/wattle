from __future__ import annotations

import mimetypes
from pathlib import Path

from .base import Tool

SUPPORTED_IMAGE_MEDIA_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/webp", "image/gif"}
)
MAX_IMAGE_BYTES = 20 * 1024 * 1024


class ViewImageTool(Tool):
    name = "view_image"
    supports_parallel_tool_calls = True
    description = (
        "Attach a local image to the next model turn so the model can inspect it. "
        "Use this first for screenshots, debug images, and visual UI issues."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative image path.",
            },
        },
    }

    def __init__(self, cwd: Path | None = None) -> None:
        self.cwd = Path.cwd() if cwd is None else Path(cwd)

    def run(self, path: str):
        from wattle.providers.base import ImageBlock

        resolved = self._resolve_path(path)
        media_type, _encoding = mimetypes.guess_type(resolved.name)
        if media_type not in SUPPORTED_IMAGE_MEDIA_TYPES:
            raise ValueError(f"Unsupported image type: {resolved}")
        size = resolved.stat().st_size
        if size > MAX_IMAGE_BYTES:
            raise ValueError(
                f"Image is too large: {resolved} ({size} bytes; max {MAX_IMAGE_BYTES})"
            )
        return ImageBlock(
            path=str(resolved),
            media_type=media_type,
            filename=resolved.name,
            size_bytes=size,
        )

    def _resolve_path(self, path: str) -> Path:
        candidate = Path(path.strip()).expanduser()
        if not candidate.is_absolute():
            candidate = self.cwd / candidate
        resolved = candidate.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(str(resolved))
        return resolved
