"""Durable Willow session records.

This module intentionally stays independent from the CLI/TUI. It provides the
JSONL primitives needed for future resume support while keeping the saved files
plain enough for users to inspect and edit by hand.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from willow.providers import (
    ContentBlock,
    ImageBlock,
    Message,
    RedactedThinkingBlock,
    Role,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)

SCHEMA_VERSION = 1
SESSION_DIR_ENV = "WILLOW_SESSION_DIR"

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True, slots=True)
class SessionMetadata:
    """Human-inspectable metadata for one saved conversation."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(default_factory=lambda: _utc_now_iso())
    updated_at: str = field(default_factory=lambda: _utc_now_iso())
    title: str | None = None
    cwd: str | None = field(default_factory=lambda: str(Path.cwd()))


@dataclass(frozen=True, slots=True)
class SessionSettings:
    """Provider/model settings needed to reconstruct future requests."""

    provider: str
    model: str
    system: str | None = None
    max_tokens: int = 4096
    max_iterations: int = 20
    thinking: bool = False
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """A complete persisted Willow chat session."""

    metadata: SessionMetadata
    settings: SessionSettings
    messages: list[Message] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class SessionEntry:
    """One discoverable saved session."""

    path: Path
    record: SessionRecord


def new_session(
    *,
    provider: str,
    model: str,
    system: str | None = None,
    max_tokens: int = 4096,
    max_iterations: int = 20,
    thinking: bool = False,
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
    title: str | None = None,
    cwd: str | None = None,
) -> SessionRecord:
    """Create an empty session record with fresh metadata."""
    return SessionRecord(
        metadata=SessionMetadata(title=title, cwd=str(Path.cwd()) if cwd is None else cwd),
        settings=SessionSettings(
            provider=provider,
            model=model,
            system=system,
            max_tokens=max_tokens,
            max_iterations=max_iterations,
            thinking=thinking,
            effort=effort,
        ),
    )


def default_session_dir() -> Path:
    """Return the default directory for persisted sessions.

    ``WILLOW_SESSION_DIR`` can override the location for tests or alternate
    launchers; otherwise Willow follows the existing ``~/.willow`` convention.
    """
    configured = os.environ.get(SESSION_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".willow" / "sessions"


def default_session_path(session_id: str | None = None) -> Path:
    """Return a JSONL file path under :func:`default_session_dir`."""
    sid = session_id or uuid.uuid4().hex
    if not _SESSION_ID_RE.fullmatch(sid):
        raise ValueError(
            "session_id may only contain letters, numbers, dots, underscores, and dashes"
        )
    return default_session_dir() / f"{sid}.jsonl"


def resolve_session_path(selector: str) -> Path:
    """Resolve a user-supplied session id or file path to a session path."""
    candidate = Path(selector).expanduser()
    if (
        candidate.is_absolute()
        or candidate.parent != Path(".")
        or candidate.suffix == ".jsonl"
    ):
        return candidate
    return default_session_path(selector)


def list_sessions(*, limit: int = 20) -> list[SessionEntry]:
    """Return recent loadable sessions, newest ``updated_at`` first."""
    directory = default_session_dir()
    if not directory.exists():
        return []

    entries: list[SessionEntry] = []
    for path in directory.glob("*.jsonl"):
        try:
            record = load_session(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        entries.append(SessionEntry(path=path, record=record))

    entries.sort(
        key=lambda entry: (entry.record.metadata.updated_at, entry.path.name),
        reverse=True,
    )
    return entries[:limit]


def save_session(record: SessionRecord, path: str | Path | None = None) -> Path:
    """Write ``record`` as JSONL, replacing the target file atomically-ish.

    The data is written to a temporary file in the target directory, flushed and
    fsync'd, then moved into place with ``os.replace``. That gives readers either
    the previous complete file or the next complete file on normal filesystems.
    """
    target = Path(path) if path is not None else default_session_path(record.metadata.id)
    target.parent.mkdir(parents=True, exist_ok=True)

    updated = replace(record, metadata=replace(record.metadata, updated_at=_utc_now_iso()))
    temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            for line in session_to_jsonl_lines(updated):
                handle.write(line)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
        _fsync_dir(target.parent)
    finally:
        with suppress(FileNotFoundError):
            temp.unlink()
    return target


def load_session(path: str | Path) -> SessionRecord:
    """Read and validate a session JSONL file."""
    return session_from_jsonl_lines(Path(path).read_text(encoding="utf-8").splitlines())


def session_to_jsonl_lines(record: SessionRecord) -> list[str]:
    """Serialize a :class:`SessionRecord` to JSONL lines."""
    header = {
        "type": "session",
        "schema_version": record.schema_version,
        "metadata": metadata_to_dict(record.metadata),
        "settings": settings_to_dict(record.settings),
    }
    return [
        json.dumps(header, ensure_ascii=False, separators=(",", ":")),
        *(
            json.dumps(
                {"type": "message", "message": message_to_dict(message)},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for message in record.messages
        ),
    ]


def session_from_jsonl_lines(lines: list[str]) -> SessionRecord:
    """Deserialize a :class:`SessionRecord` from JSONL lines."""
    objects = [json.loads(line) for line in lines if line.strip()]
    if not objects:
        raise ValueError("session JSONL file is empty")
    header = objects[0]
    if not isinstance(header, dict) or header.get("type") != "session":
        raise ValueError("session JSONL must start with a session header")
    version = header.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported session schema_version: {version!r}")
    messages: list[Message] = []
    for index, item in enumerate(objects[1:], start=2):
        if not isinstance(item, dict) or item.get("type") != "message":
            raise ValueError(f"session JSONL line {index} must be a message object")
        messages.append(message_from_dict(_require_dict(item, "message")))
    return SessionRecord(
        schema_version=version,
        metadata=metadata_from_dict(_require_dict(header, "metadata")),
        settings=settings_from_dict(_require_dict(header, "settings")),
        messages=messages,
    )


def session_to_dict(record: SessionRecord) -> dict[str, Any]:
    """Serialize a :class:`SessionRecord` to JSON-compatible data."""
    return {
        "schema_version": record.schema_version,
        "metadata": metadata_to_dict(record.metadata),
        "settings": settings_to_dict(record.settings),
        "messages": [message_to_dict(message) for message in record.messages],
    }


def session_from_dict(data: dict[str, Any]) -> SessionRecord:
    """Deserialize a :class:`SessionRecord` from JSON-compatible data."""
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported session schema_version: {version!r}")
    return SessionRecord(
        schema_version=version,
        metadata=metadata_from_dict(_require_dict(data, "metadata")),
        settings=settings_from_dict(_require_dict(data, "settings")),
        messages=[
            message_from_dict(item)
            for item in _require_list(data, "messages")
        ],
    )


def metadata_to_dict(metadata: SessionMetadata) -> dict[str, Any]:
    return {
        "id": metadata.id,
        "created_at": metadata.created_at,
        "updated_at": metadata.updated_at,
        "title": metadata.title,
        "cwd": metadata.cwd,
    }


def metadata_from_dict(data: dict[str, Any]) -> SessionMetadata:
    return SessionMetadata(
        id=_require_str(data, "id"),
        created_at=_require_str(data, "created_at"),
        updated_at=_require_str(data, "updated_at"),
        title=_optional_str(data, "title"),
        cwd=_optional_str(data, "cwd"),
    )


def settings_to_dict(settings: SessionSettings) -> dict[str, Any]:
    return {
        "provider": settings.provider,
        "model": settings.model,
        "system": settings.system,
        "max_tokens": settings.max_tokens,
        "max_iterations": settings.max_iterations,
        "thinking": settings.thinking,
        "effort": settings.effort,
    }


def settings_from_dict(data: dict[str, Any]) -> SessionSettings:
    return SessionSettings(
        provider=_require_str(data, "provider"),
        model=_require_str(data, "model"),
        system=_optional_str(data, "system"),
        max_tokens=_require_int(data, "max_tokens"),
        max_iterations=_require_int(data, "max_iterations"),
        thinking=_optional_bool(data, "thinking", default=False),
        effort=_optional_effort(data, "effort"),
    )


def message_to_dict(message: Message) -> dict[str, Any]:
    """Serialize one provider-normalized message."""
    return {
        "role": message.role,
        "input_tokens": message.input_tokens,
        "output_tokens": message.output_tokens,
        "cached_tokens": message.cached_tokens,
        "content": [content_block_to_dict(block) for block in message.content],
    }


def message_from_dict(data: Any) -> Message:
    """Deserialize one provider-normalized message."""
    if not isinstance(data, dict):
        raise ValueError(f"message must be an object, got {type(data).__name__}")
    role = _require_str(data, "role")
    if role not in ("user", "assistant"):
        raise ValueError(f"unsupported message role: {role!r}")
    return Message(
        role=cast(Role, role),
        content=[
            content_block_from_dict(block)
            for block in _require_list(data, "content")
        ],
        input_tokens=_optional_int(data, "input_tokens", default=0),
        output_tokens=_optional_int(data, "output_tokens", default=0),
        cached_tokens=_optional_int(data, "cached_tokens", default=0),
    )


def content_block_to_dict(block: ContentBlock) -> dict[str, Any]:
    """Serialize the current Willow content block union."""
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ImageBlock):
        return {
            "type": "image",
            "path": block.path,
            "media_type": block.media_type,
            "filename": block.filename,
            "size_bytes": block.size_bytes,
        }
    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if isinstance(block, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
            "is_error": block.is_error,
        }
    if isinstance(block, ThinkingBlock):
        return {
            "type": "thinking",
            "thinking": block.thinking,
            "signature": block.signature,
            "encrypted_content": block.encrypted_content,
        }
    if isinstance(block, RedactedThinkingBlock):
        return {"type": "redacted_thinking", "data": block.data}
    raise TypeError(f"unsupported content block: {type(block).__name__}")


def content_block_from_dict(data: Any) -> ContentBlock:
    """Deserialize one content block from the saved JSON shape."""
    if not isinstance(data, dict):
        raise ValueError(f"content block must be an object, got {type(data).__name__}")
    block_type = _require_str(data, "type")
    if block_type == "text":
        return TextBlock(text=_require_str(data, "text"))
    if block_type == "image":
        return ImageBlock(
            path=_require_str(data, "path"),
            media_type=_require_str(data, "media_type"),
            filename=_require_str(data, "filename"),
            size_bytes=_require_int(data, "size_bytes"),
        )
    if block_type == "tool_use":
        return ToolUseBlock(
            id=_require_str(data, "id"),
            name=_require_str(data, "name"),
            input=_require_dict(data, "input"),
        )
    if block_type == "tool_result":
        return ToolResultBlock(
            tool_use_id=_require_str(data, "tool_use_id"),
            content=_require_str(data, "content"),
            is_error=_require_bool(data, "is_error"),
        )
    if block_type == "thinking":
        return ThinkingBlock(
            thinking=_require_str(data, "thinking"),
            signature=_optional_str(data, "signature"),
            encrypted_content=_optional_str(data, "encrypted_content"),
        )
    if block_type == "redacted_thinking":
        return RedactedThinkingBlock(data=_require_str(data, "data"))
    raise ValueError(f"unsupported content block type: {block_type!r}")


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _require_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key!r} must be an object")
    return value


def _require_list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key!r} must be a list")
    return value


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key!r} must be a string")
    return value


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key!r} must be a string or null")
    return value


def _require_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key!r} must be an integer")
    return value


def _require_bool(data: dict[str, Any], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key!r} must be a boolean")
    return value


def _optional_bool(data: dict[str, Any], key: str, *, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key!r} must be a boolean")
    return value


def _optional_effort(
    data: dict[str, Any], key: str
) -> Literal["low", "medium", "high", "xhigh", "max"] | None:
    value = _optional_str(data, key)
    if value is None:
        return None
    if value not in {"low", "medium", "high", "xhigh", "max"}:
        raise ValueError(f"{key!r} has unsupported effort: {value!r}")
    return cast(Literal["low", "medium", "high", "xhigh", "max"], value)


def _optional_int(data: dict[str, Any], key: str, *, default: int) -> int:
    value = data.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key!r} must be an integer")
    return value


def _fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
