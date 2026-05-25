"""Tests for Wattle's durable session records."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from wattle import session
from wattle.providers import (
    ImageBlock,
    Message,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)


def _sample_record() -> session.SessionRecord:
    return session.SessionRecord(
        metadata=session.SessionMetadata(
            id="sess_123",
            created_at="2026-05-08T10:00:00Z",
            updated_at="2026-05-08T10:00:00Z",
            title="debug run",
            cwd="/tmp/project",
        ),
        settings=session.SessionSettings(
            provider="openai_responses",
            model="gpt-5.5",
            system="Be concise.",
            max_tokens=2048,
        ),
        messages=[
            Message(
                role="user",
                content=[TextBlock(text="Run the test suite.")],
            ),
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(
                        thinking="I should inspect the project first.",
                        signature="sig_1",
                        encrypted_content="enc_1",
                    ),
                    ToolUseBlock(
                        id="toolu_1",
                        name="bash",
                        input={"command": "pytest", "timeout": 120},
                    ),
                ],
            ),
            Message(
                role="user",
                content=[
                    ToolResultBlock(
                        tool_use_id="toolu_1",
                        content="1 passed",
                        is_error=False,
                    ),
                    TextBlock(text="Now summarize."),
                ],
            ),
            Message(
                role="assistant",
                content=[
                    RedactedThinkingBlock(data="opaque-redacted-payload"),
                    TextBlock(text="Tests passed."),
                ],
                input_tokens=100,
                output_tokens=12,
                cached_tokens=80,
            ),
        ],
        compactions=[
            session.SessionCompaction(
                summary="Earlier work was summarized.",
                first_kept_message_index=2,
                summarized_until_message_index=2,
                created_after_message_index=4,
                reason="threshold",
                tokens_before=1200,
                tokens_after=300,
                created_at="2026-05-08T10:05:00Z",
            )
        ],
    )


def test_session_record_round_trips_through_jsonl_file(tmp_path: Path) -> None:
    record = _sample_record()
    path = tmp_path / "nested" / "session.jsonl"

    saved_path = session.save_session(record, path)
    loaded = session.load_session(saved_path)

    assert saved_path == path
    assert loaded.metadata.id == record.metadata.id
    assert loaded.metadata.created_at == record.metadata.created_at
    assert loaded.metadata.updated_at.endswith("Z")
    assert loaded.settings == record.settings
    assert loaded.messages == record.messages
    assert loaded.compactions == record.compactions

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["type"] == "session"
    assert lines[0]["schema_version"] == session.SCHEMA_VERSION
    assert lines[0]["metadata"]["title"] == "debug run"
    assert lines[0]["settings"] == {
        "provider": "openai_responses",
        "model": "gpt-5.5",
        "system": "Be concise.",
        "max_tokens": 2048,
        "thinking": False,
        "effort": None,
    }
    assert lines[1]["type"] == "message"
    assert lines[1]["message"]["role"] == "user"
    assert lines[2]["message"]["content"][0] == {
        "type": "thinking",
        "thinking": "I should inspect the project first.",
        "signature": "sig_1",
        "encrypted_content": "enc_1",
    }
    assert lines[4]["message"]["input_tokens"] == 100
    assert lines[4]["message"]["output_tokens"] == 12
    assert lines[4]["message"]["cached_tokens"] == 80
    assert lines[4]["message"]["content"][0] == {
        "type": "redacted_thinking",
        "data": "opaque-redacted-payload",
    }
    assert lines[5]["type"] == "compaction"
    assert lines[5]["compaction"] == {
        "summary": "Earlier work was summarized.",
        "first_kept_message_index": 2,
        "summarized_until_message_index": 2,
        "created_after_message_index": 4,
        "reason": "threshold",
        "tokens_before": 1200,
        "tokens_after": 300,
        "created_at": "2026-05-08T10:05:00Z",
    }


def test_message_and_content_block_helpers_are_explicit() -> None:
    message = Message(
        role="assistant",
        content=[
            TextBlock(text="hello"),
            ImageBlock(
                path="/tmp/screenshot.png",
                media_type="image/png",
                filename="screenshot.png",
                size_bytes=123,
            ),
            ToolUseBlock(id="call_1", name="read", input={"path": "README.md"}),
        ],
    )

    encoded = session.message_to_dict(message)

    assert encoded == {
        "role": "assistant",
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "content": [
            {"type": "text", "text": "hello"},
            {
                "type": "image",
                "path": "/tmp/screenshot.png",
                "media_type": "image/png",
                "filename": "screenshot.png",
                "size_bytes": 123,
            },
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "read",
                "input": {"path": "README.md"},
            },
        ],
    }
    assert session.message_from_dict(encoded) == message


def test_save_session_replaces_existing_file_via_same_directory_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _sample_record()
    path = tmp_path / "session.jsonl"
    path.write_text("old data", encoding="utf-8")

    calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def spy_replace(
        src: str | bytes | os.PathLike[str],
        dst: str | bytes | os.PathLike[str],
    ) -> None:
        src_path = Path(src)
        dst_path = Path(dst)
        assert src_path.parent == path.parent
        assert src_path.name.startswith(f".{path.name}.")
        assert src_path.name.endswith(".tmp")
        assert dst_path == path
        first_line = src_path.read_text(encoding="utf-8").splitlines()[0]
        assert json.loads(first_line)["metadata"]["id"] == "sess_123"
        calls.append((src_path, dst_path))
        real_replace(src, dst)

    monkeypatch.setattr(session.os, "replace", spy_replace)

    session.save_session(record, path)

    assert calls
    assert session.load_session(path).metadata.id == "sess_123"
    assert not list(tmp_path.glob("*.tmp"))


def test_default_session_path_uses_env_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(session.SESSION_DIR_ENV, str(tmp_path / "sessions"))

    assert session.default_session_dir() == tmp_path / "sessions"
    assert session.default_session_path("abc-123") == tmp_path / "sessions" / "abc-123.jsonl"


def test_default_session_path_rejects_path_traversal() -> None:
    with pytest.raises(ValueError, match="session_id"):
        session.default_session_path("../outside")


def test_new_session_captures_settings_and_empty_history(tmp_path: Path) -> None:
    record = session.new_session(
        provider="anthropic",
        model="claude-sonnet-4-6",
        system=None,
        max_tokens=512,

        title="investigation",
        cwd=str(tmp_path),
    )

    assert record.metadata.title == "investigation"
    assert record.metadata.cwd == str(tmp_path)
    assert record.settings == session.SessionSettings(
        provider="anthropic",
        model="claude-sonnet-4-6",
        system=None,
        max_tokens=512,

    )
    assert record.messages == []


def _search_record(
    session_id: str,
    *,
    updated_at: str,
    title: str | None = None,
    cwd: str = "/tmp/project",
    provider: str = "openai_responses",
    model: str = "gpt-5.5",
    parent_session_id: str | None = None,
    user_text: str | None = None,
    assistant_text: str | None = None,
) -> session.SessionRecord:
    messages: list[Message] = []
    if user_text is not None:
        messages.append(Message(role="user", content=[TextBlock(text=user_text)]))
    if assistant_text is not None:
        messages.append(Message(role="assistant", content=[TextBlock(text=assistant_text)]))
    return session.SessionRecord(
        metadata=session.SessionMetadata(
            id=session_id,
            created_at="2026-05-01T00:00:00Z",
            updated_at=updated_at,
            title=title,
            cwd=cwd,
            parent_session_id=parent_session_id,
        ),
        settings=session.SessionSettings(provider=provider, model=model),
        messages=messages,
    )


def _write_session_record(record: session.SessionRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(session.session_to_jsonl_lines(record)) + "\n", encoding="utf-8")


def test_list_session_entries_is_uncapped_sorted_and_skips_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(session.SESSION_DIR_ENV, str(tmp_path))
    old = _search_record("old", updated_at="2026-05-01T00:00:00Z")
    newest = _search_record("newest", updated_at="2026-05-03T00:00:00Z")
    middle = _search_record("middle", updated_at="2026-05-02T00:00:00Z")
    _write_session_record(old, tmp_path / "old.jsonl")
    _write_session_record(newest, tmp_path / "newest.jsonl")
    _write_session_record(middle, tmp_path / "middle.jsonl")
    (tmp_path / "broken.jsonl").write_text("not json", encoding="utf-8")

    entries = session.list_session_entries()

    assert [entry.record.metadata.id for entry in entries] == ["newest", "middle", "old"]
    assert [entry.record.metadata.id for entry in session.list_sessions(limit=2)] == [
        "newest",
        "middle",
    ]


def test_session_preview_prefers_title_then_user_then_assistant() -> None:
    titled = _search_record(
        "titled",
        updated_at="2026-05-01T00:00:00Z",
        title="Named Investigation",
        user_text="first user",
    )
    user_only = _search_record(
        "user",
        updated_at="2026-05-01T00:00:00Z",
        user_text="  first\nuser   message  ",
    )
    assistant_only = _search_record(
        "assistant",
        updated_at="2026-05-01T00:00:00Z",
        assistant_text="assistant summary",
    )
    empty = _search_record("empty", updated_at="2026-05-01T00:00:00Z")

    assert session.session_preview(titled) == "Named Investigation"
    assert session.session_preview(user_only) == "first user message"
    assert session.session_preview(assistant_only) == "assistant summary"
    assert session.session_preview(empty) == "(untitled)"


def test_filter_session_entries_searches_all_fields_and_requires_all_tokens() -> None:
    record = _search_record(
        "sess-search-123",
        updated_at="2026-05-01T00:00:00Z",
        title="Quota Debugging",
        cwd="/Users/me/repos/wattle",
        provider="anthropic",
        model="claude-sonnet-4-6",
        parent_session_id="parent-456",
        user_text="investigate weekly limits",
    )
    preview = session.session_preview(record)
    entry = session.SessionEntry(
        path=Path("quota.jsonl"),
        record=record,
        preview=preview,
        search_text=session.session_search_text(
            session.SessionEntry(path=Path("quota.jsonl"), record=record, preview=preview)
        ),
    )
    other = session.SessionEntry(
        path=Path("other.jsonl"),
        record=_search_record("other", updated_at="2026-05-01T00:00:00Z", title="Other"),
    )
    entries = [entry, other]

    assert session.filter_session_entries(entries, "quota") == [entry]
    assert session.filter_session_entries(entries, "WATTLE anthropic") == [entry]
    assert session.filter_session_entries(entries, "sonnet parent-456") == [entry]
    assert session.filter_session_entries(entries, "weekly missing") == []
    assert session.filter_session_entries(entries, "") == entries


def test_load_rejects_unknown_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_text(
        json.dumps({"type": "session", "schema_version": 999, "metadata": {}, "settings": {}})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version"):
        session.load_session(path)


def test_load_rejects_unknown_content_block_type() -> None:
    with pytest.raises(ValueError, match="unsupported content block type"):
        session.content_block_from_dict({"type": "audio", "url": "https://example.test/x.mp3"})


def test_load_rejects_invalid_message_role() -> None:
    with pytest.raises(ValueError, match="unsupported message role"):
        session.message_from_dict({"role": "system", "content": []})
