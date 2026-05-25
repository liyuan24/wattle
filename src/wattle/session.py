"""Durable Wattle session records.

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

from wattle.providers import (
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
SESSION_DIR_ENV = "WATTLE_SESSION_DIR"

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True, slots=True)
class SessionMetadata:
    """Human-inspectable metadata for one saved conversation."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: _utc_now_iso())
    updated_at: str = field(default_factory=lambda: _utc_now_iso())
    title: str | None = None
    cwd: str | None = field(default_factory=lambda: str(Path.cwd()))
    parent_session_id: str | None = None


@dataclass(frozen=True, slots=True)
class SessionSettings:
    """Provider/model settings needed to reconstruct future requests."""

    provider: str
    model: str
    system: str | None = None
    max_tokens: int = 4096
    thinking: bool = False
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None


@dataclass(frozen=True, slots=True)
class SessionCompaction:
    """Durable checkpoint describing one compacted request projection."""

    summary: str
    first_kept_message_index: int
    summarized_until_message_index: int
    created_after_message_index: int
    reason: str = "threshold"
    tokens_before: int = 0
    tokens_after: int = 0
    read_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: _utc_now_iso())


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """A complete persisted Wattle chat session."""

    metadata: SessionMetadata
    settings: SessionSettings
    messages: list[Message] = field(default_factory=list)
    compactions: list[SessionCompaction] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class SessionEntry:
    """One discoverable saved session."""

    path: Path
    record: SessionRecord
    preview: str = ""
    search_text: str = ""


def new_session(
    *,
    provider: str,
    model: str,
    system: str | None = None,
    max_tokens: int = 4096,
    thinking: bool = False,
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
    title: str | None = None,
    cwd: str | None = None,
    parent_session_id: str | None = None,
) -> SessionRecord:
    """Create an empty session record with fresh metadata."""
    return SessionRecord(
        metadata=SessionMetadata(
            title=title,
            cwd=str(Path.cwd()) if cwd is None else cwd,
            parent_session_id=parent_session_id,
        ),
        settings=SessionSettings(
            provider=provider,
            model=model,
            system=system,
            max_tokens=max_tokens,
            thinking=thinking,
            effort=effort,
        ),
    )


def default_session_dir() -> Path:
    """Return the default directory for persisted sessions.

    ``WATTLE_SESSION_DIR`` can override the location for tests or alternate
    launchers; otherwise Wattle follows the existing ``~/.wattle`` convention.
    """
    configured = os.environ.get(SESSION_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".wattle" / "sessions"


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


def session_preview(record: SessionRecord, limit: int = 80) -> str:
    """Return a compact human label for search and resume pickers."""
    candidates = [record.metadata.title]
    for role in ("user", "assistant"):
        candidates.append(_first_text_for_role(record, role))
    for candidate in candidates:
        text = _compact_session_text(candidate)
        if text:
            return text if len(text) <= limit else text[: limit - 3] + "..."
    return "(untitled)"


def session_search_text(entry: SessionEntry) -> str:
    """Return normalized fields searched by startup resume selection."""
    record = entry.record
    metadata = record.metadata
    fields = [
        metadata.id,
        metadata.title,
        entry.preview or session_preview(record),
        metadata.cwd,
        record.settings.provider,
        record.settings.model,
        metadata.parent_session_id,
    ]
    return " ".join(_compact_session_text(field) for field in fields if field).lower()


def list_session_entries(*, limit: int | None = None) -> list[SessionEntry]:
    """Return loadable sessions, newest ``updated_at`` first."""
    directory = default_session_dir()
    if not directory.exists():
        return []

    entries: list[SessionEntry] = []
    for path in directory.glob("*.jsonl"):
        try:
            record = load_session(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        preview = session_preview(record)
        entry = SessionEntry(path=path, record=record, preview=preview)
        entries.append(
            SessionEntry(
                path=path,
                record=record,
                preview=preview,
                search_text=session_search_text(entry),
            )
        )

    entries.sort(
        key=lambda entry: (entry.record.metadata.updated_at, entry.path.name),
        reverse=True,
    )
    return entries if limit is None else entries[:limit]


def filter_session_entries(entries: list[SessionEntry], query: str) -> list[SessionEntry]:
    """Filter sessions with case-insensitive all-token matching."""
    tokens = [token.lower() for token in query.split()]
    if not tokens:
        return entries
    return [
        entry
        for entry in entries
        if all(token in (entry.search_text or session_search_text(entry)) for token in tokens)
    ]


def list_sessions(*, limit: int = 20) -> list[SessionEntry]:
    """Return recent loadable sessions, newest ``updated_at`` first."""
    return list_session_entries(limit=limit)


def _first_text_for_role(record: SessionRecord, role: Role) -> str | None:
    for message in record.messages:
        if message.role != role:
            continue
        for block in message.content:
            if isinstance(block, TextBlock) and block.text.strip():
                return block.text
    return None


def _compact_session_text(text: object | None) -> str:
    return " ".join(str(text).split()) if text is not None else ""


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
    lines = [json.dumps(header, ensure_ascii=False, separators=(",", ":"))]
    compactions_by_message_index: dict[int, list[SessionCompaction]] = {}
    for compaction in record.compactions:
        compactions_by_message_index.setdefault(
            compaction.created_after_message_index,
            [],
        ).append(compaction)

    for compaction in compactions_by_message_index.get(0, []):
        lines.append(_compaction_jsonl_line(compaction))

    for index, message in enumerate(record.messages, start=1):
        lines.append(_jsonl_line({"type": "message", "message": message_to_dict(message)}))
        for compaction in compactions_by_message_index.get(index, []):
            lines.append(_compaction_jsonl_line(compaction))

    for index in sorted(key for key in compactions_by_message_index if key > len(record.messages)):
        for compaction in compactions_by_message_index[index]:
            lines.append(_compaction_jsonl_line(compaction))

    return lines


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
    compactions: list[SessionCompaction] = []
    for index, item in enumerate(objects[1:], start=2):
        if not isinstance(item, dict):
            raise ValueError(f"session JSONL line {index} must be an object")
        item_type = item.get("type")
        if item_type == "message":
            messages.append(message_from_dict(_require_dict(item, "message")))
            continue
        if item_type == "compaction":
            compactions.append(compaction_from_dict(_require_dict(item, "compaction")))
            continue
        raise ValueError(f"session JSONL line {index} must be a message or compaction object")
    return SessionRecord(
        schema_version=version,
        metadata=metadata_from_dict(_require_dict(header, "metadata")),
        settings=settings_from_dict(_require_dict(header, "settings")),
        messages=messages,
        compactions=compactions,
    )


def session_to_dict(record: SessionRecord) -> dict[str, Any]:
    """Serialize a :class:`SessionRecord` to JSON-compatible data."""
    return {
        "schema_version": record.schema_version,
        "metadata": metadata_to_dict(record.metadata),
        "settings": settings_to_dict(record.settings),
        "messages": [message_to_dict(message) for message in record.messages],
        "compactions": [
            compaction_to_dict(compaction) for compaction in record.compactions
        ],
    }


def session_from_dict(data: dict[str, Any]) -> SessionRecord:
    """Deserialize a :class:`SessionRecord` from JSON-compatible data."""
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported session schema_version: {version!r}")
    compactions_data = data.get("compactions", [])
    if not isinstance(compactions_data, list):
        raise ValueError("'compactions' must be a list")
    return SessionRecord(
        schema_version=version,
        metadata=metadata_from_dict(_require_dict(data, "metadata")),
        settings=settings_from_dict(_require_dict(data, "settings")),
        messages=[
            message_from_dict(item)
            for item in _require_list(data, "messages")
        ],
        compactions=[
            compaction_from_dict(item)
            for item in cast(list[dict[str, Any]], compactions_data)
        ],
    )


def metadata_to_dict(metadata: SessionMetadata) -> dict[str, Any]:
    return {
        "id": metadata.id,
        "created_at": metadata.created_at,
        "updated_at": metadata.updated_at,
        "title": metadata.title,
        "cwd": metadata.cwd,
        "parent_session_id": metadata.parent_session_id,
    }


def metadata_from_dict(data: dict[str, Any]) -> SessionMetadata:
    return SessionMetadata(
        id=_require_str(data, "id"),
        created_at=_require_str(data, "created_at"),
        updated_at=_require_str(data, "updated_at"),
        title=_optional_str(data, "title"),
        cwd=_optional_str(data, "cwd"),
        parent_session_id=_optional_str(data, "parent_session_id"),
    )


def settings_to_dict(settings: SessionSettings) -> dict[str, Any]:
    return {
        "provider": settings.provider,
        "model": settings.model,
        "system": settings.system,
        "max_tokens": settings.max_tokens,
        "thinking": settings.thinking,
        "effort": settings.effort,
    }


def settings_from_dict(data: dict[str, Any]) -> SessionSettings:
    return SessionSettings(
        provider=_require_str(data, "provider"),
        model=_require_str(data, "model"),
        system=_optional_str(data, "system"),
        max_tokens=_require_int(data, "max_tokens"),
        thinking=_optional_bool(data, "thinking", default=False),
        effort=_optional_effort(data, "effort"),
    )


def compaction_to_dict(compaction: SessionCompaction) -> dict[str, Any]:
    data = {
        "summary": compaction.summary,
        "first_kept_message_index": compaction.first_kept_message_index,
        "summarized_until_message_index": compaction.summarized_until_message_index,
        "created_after_message_index": compaction.created_after_message_index,
        "reason": compaction.reason,
        "tokens_before": compaction.tokens_before,
        "tokens_after": compaction.tokens_after,
        "created_at": compaction.created_at,
    }
    if compaction.read_files:
        data["read_files"] = compaction.read_files
    if compaction.modified_files:
        data["modified_files"] = compaction.modified_files
    return data


def compaction_from_dict(data: dict[str, Any]) -> SessionCompaction:
    return SessionCompaction(
        summary=_require_str(data, "summary"),
        first_kept_message_index=_require_int(data, "first_kept_message_index"),
        summarized_until_message_index=_require_int(
            data,
            "summarized_until_message_index",
        ),
        created_after_message_index=_require_int(data, "created_after_message_index"),
        reason=_optional_str(data, "reason") or "threshold",
        tokens_before=_optional_int(data, "tokens_before", default=0),
        tokens_after=_optional_int(data, "tokens_after", default=0),
        read_files=_optional_str_list(data, "read_files"),
        modified_files=_optional_str_list(data, "modified_files"),
        created_at=_optional_str(data, "created_at") or _utc_now_iso(),
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
    """Serialize the current Wattle content block union."""
    block_type = getattr(block, "type", None)
    if isinstance(block, TextBlock) or block_type == "text":
        return {"type": "text", "text": block.text}
    if isinstance(block, ImageBlock) or block_type == "image":
        return {
            "type": "image",
            "path": block.path,
            "media_type": block.media_type,
            "filename": block.filename,
            "size_bytes": block.size_bytes,
        }
    if isinstance(block, ToolUseBlock) or block_type == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if isinstance(block, ToolResultBlock) or block_type == "tool_result":
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
            "is_error": block.is_error,
        }
    if isinstance(block, ThinkingBlock) or block_type == "thinking":
        return {
            "type": "thinking",
            "thinking": block.thinking,
            "signature": block.signature,
            "encrypted_content": block.encrypted_content,
        }
    if isinstance(block, RedactedThinkingBlock) or block_type == "redacted_thinking":
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


def _jsonl_line(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _compaction_jsonl_line(compaction: SessionCompaction) -> str:
    return _jsonl_line(
        {"type": "compaction", "compaction": compaction_to_dict(compaction)}
    )


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


def _optional_str_list(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key!r} must be a list of strings")
    return list(value)


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
