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
    description = (
        "Attach a local image to the next model turn so the model can inspect it. "
        "Use this first for screenshots, debug images, and visual UI issues. "
        "For the latest debug screenshot, call with latest_debug=true."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Absolute or relative image path. Optional when latest_debug is true."
                ),
            },
            "latest_debug": {
                "type": "boolean",
                "description": "Attach the newest image in ./debug_images.",
                "default": False,
            },
        },
    }

    def __init__(self, cwd: Path | None = None) -> None:
        self.cwd = Path.cwd() if cwd is None else Path(cwd)

    def run(
        self,
        path: str | None = None,
        latest_debug: bool = False,
    ):
        from willow.providers.base import ImageBlock

        resolved = self._resolve_path(path, latest_debug=latest_debug)
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

    def _resolve_path(self, path: str | None, *, latest_debug: bool) -> Path:
        if latest_debug or path is None or path.strip() in {"", "latest", "latest_debug"}:
            return self._latest_debug_image()
        candidate = Path(path.strip()).expanduser()
        if not candidate.is_absolute():
            candidate = self.cwd / candidate
        resolved = candidate.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(str(resolved))
        return resolved

    def _latest_debug_image(self) -> Path:
        debug_dir = self.cwd / "debug_images"
        if not debug_dir.is_dir():
            raise FileNotFoundError(str(debug_dir))
        images = [
            path
            for path in debug_dir.iterdir()
            if path.is_file()
            and mimetypes.guess_type(path.name)[0] in SUPPORTED_IMAGE_MEDIA_TYPES
        ]
        if not images:
            raise FileNotFoundError(f"No supported images found in {debug_dir}")
        return max(images, key=lambda path: path.stat().st_mtime_ns).resolve()
