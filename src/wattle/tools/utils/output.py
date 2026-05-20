from __future__ import annotations

import time
import uuid
from pathlib import Path

MAX_INLINE_OUTPUT_CHARS = 24_000
MAX_EXCERPT_CHARS = 12_000


def externalize_large_output(
    content: str,
    *,
    root: Path | str | None = None,
    tool_name: str,
    max_inline_chars: int = MAX_INLINE_OUTPUT_CHARS,
    max_excerpt_chars: int = MAX_EXCERPT_CHARS,
) -> str:
    if len(content) <= max_inline_chars:
        return content

    artifact_path = _write_artifact(content, root=root, tool_name=tool_name)
    excerpt, omitted_chars = _head_tail_excerpt(
        content,
        max_excerpt_chars=max_excerpt_chars,
    )

    return "\n".join(
        [
            f"[output truncated: {len(content)} chars]",
            f"full_output_path: {artifact_path}",
            f"full_output_chars: {len(content)}",
            f"excerpt_chars: {len(excerpt)}",
            f"omitted_chars: {omitted_chars}",
            "[excerpt]",
            excerpt,
            "[/excerpt]",
        ]
    )


def _write_artifact(content: str, *, root: Path | str | None, tool_name: str) -> Path:
    base = Path.cwd() if root is None else Path(root)
    artifacts_dir = base / ".wattle" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = (
        artifacts_dir
        / f"{tool_name}-{time.time_ns()}-{uuid.uuid4().hex[:8]}.txt"
    ).resolve()
    path.write_text(content, encoding="utf-8")
    return path


def _head_tail_excerpt(content: str, *, max_excerpt_chars: int) -> tuple[str, int]:
    if len(content) <= max_excerpt_chars:
        return content, 0
    head_chars = max_excerpt_chars // 2
    tail_chars = max_excerpt_chars - head_chars
    omitted = len(content) - max_excerpt_chars
    excerpt = "\n".join(
        [
            content[:head_chars].rstrip("\n"),
            f"[... omitted {omitted} chars; see full_output_path ...]",
            content[-tail_chars:].lstrip("\n"),
        ]
    )
    return excerpt, omitted
