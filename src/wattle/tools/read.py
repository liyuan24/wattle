from __future__ import annotations

import unicodedata
from pathlib import Path

from .base import Tool
from .utils.output import externalize_large_output

MAX_READ_LINES = 2_000
MAX_READ_BYTES = 50 * 1024
UNICODE_SPACES = str.maketrans(
    {
        "\u00a0": " ",
        "\u2000": " ",
        "\u2001": " ",
        "\u2002": " ",
        "\u2003": " ",
        "\u2004": " ",
        "\u2005": " ",
        "\u2006": " ",
        "\u2007": " ",
        "\u2008": " ",
        "\u2009": " ",
        "\u200a": " ",
        "\u202f": " ",
        "\u205f": " ",
        "\u3000": " ",
    }
)


class ReadTool(Tool):
    name = "read"
    supports_parallel_tool_calls = True
    description = (
        "Read a text file from disk. Returns contents prefixed with 1-indexed line numbers. "
        "Use offset/limit for large files. Output is capped with continuation hints."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file.",
            },
            "offset": {
                "type": "integer",
                "description": "1-indexed line to start reading from. Defaults to 1.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to return. Defaults to all.",
            },
        },
        "required": ["path"],
    }

    def __init__(self, cwd: Path | None = None) -> None:
        self.cwd = Path.cwd() if cwd is None else Path(cwd)

    def run(self, path: str, offset: int = 1, limit: int | None = None) -> str:
        resolved = self._resolve_path(path)
        if resolved.is_dir():
            raise IsADirectoryError(f"Path is a directory, not a file: {resolved}")
        if not resolved.is_file():
            raise FileNotFoundError(f"File not found: {resolved}")
        if offset < 1:
            raise ValueError("offset must be 1 or greater")
        if limit is not None and limit < 1:
            raise ValueError("limit must be 1 or greater")

        data = resolved.read_bytes()
        if b"\x00" in data:
            raise ValueError(
                f"File appears to be binary; read supports text files only: {resolved}"
            )
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UnicodeDecodeError(
                exc.encoding,
                exc.object,
                exc.start,
                exc.end,
                f"could not decode {resolved} as UTF-8: {exc.reason}",
            ) from exc

        lines = text.splitlines()
        total_lines = len(lines)
        if total_lines == 0:
            return f"path: {resolved}\nlines: 0 of 0\n[empty file]"
        if offset > total_lines:
            raise ValueError(
                f"Offset {offset} is beyond end of file ({total_lines} lines total)"
            )

        start = offset - 1
        requested_end = start + limit if limit is not None else total_lines
        capped_end = min(requested_end, total_lines, start + MAX_READ_LINES)
        selected: list[str] = []
        used_bytes = 0
        for line in lines[start:capped_end]:
            line_bytes = len(line.encode("utf-8"))
            if not selected and line_bytes > MAX_READ_BYTES:
                raise ValueError(
                    f"Line {start + 1} is {_format_bytes(line_bytes)}, exceeding the "
                    f"{_format_bytes(MAX_READ_BYTES)} read limit. Use a targeted search "
                    "or a narrower inspection method."
                )
            separator_bytes = 1 if selected else 0
            if used_bytes + separator_bytes + line_bytes > MAX_READ_BYTES:
                break
            selected.append(line)
            used_bytes += separator_bytes + line_bytes

        if not selected:
            return f"path: {resolved}\nlines: {offset}-{offset - 1} of {total_lines}\n[empty]"

        shown_start = start + 1
        shown_end = start + len(selected)
        body = "\n".join(
            f"{shown_start + i:6d}\t{line}" for i, line in enumerate(selected)
        )
        parts = [
            f"path: {resolved}",
            f"lines: {shown_start}-{shown_end} of {total_lines}",
            body,
        ]

        next_offset = shown_end + 1
        stopped_by_user_limit = (
            limit is not None
            and shown_end == min(requested_end, total_lines)
            and requested_end < total_lines
        )
        if next_offset <= total_lines:
            if stopped_by_user_limit:
                remaining = total_lines - shown_end
                parts.append(
                    f"[Read incomplete: {remaining} more lines in file. "
                    f"Use offset={next_offset} to continue.]"
                )
            else:
                parts.append(
                    f"[Read incomplete: showing lines {shown_start}-{shown_end} "
                    f"of {total_lines}. Use offset={next_offset} to continue.]"
                )

        output = "\n".join(parts)
        return externalize_large_output(
            output,
            tool_name=self.name,
            max_inline_chars=100_000,
        )

    def _resolve_path(self, path: str) -> Path:
        normalized = path.strip().translate(UNICODE_SPACES)
        if normalized.startswith("@"):
            normalized = normalized[1:]
        candidate = Path(normalized).expanduser()
        if not candidate.is_absolute():
            candidate = self.cwd / candidate
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved

        for fallback in _path_fallbacks(resolved):
            if fallback.exists():
                return fallback
        return resolved


def _path_fallbacks(path: Path) -> list[Path]:
    raw = str(path)
    variants = [
        raw.replace(" AM.", "\u202fAM.").replace(" PM.", "\u202fPM."),
        raw.replace(" am.", "\u202fam.").replace(" pm.", "\u202fpm."),
        unicodedata.normalize("NFD", raw),
        raw.replace("'", "\u2019"),
        unicodedata.normalize("NFD", raw).replace("'", "\u2019"),
    ]
    return [Path(variant) for variant in variants if variant != raw]


def _format_bytes(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"
