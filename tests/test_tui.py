"""Tests for Willow's native terminal TUI."""

from __future__ import annotations

import argparse
import io
import os
import re
import threading
import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from willow import auth, session, tui
from willow.models import ModelChoice
from willow.providers import (
    CompletionRequest,
    CompletionResponse,
    ImageBlock,
    Message,
    Provider,
    StreamComplete,
    TextBlock,
    TextDelta,
    ThinkingDelta,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseDelta,
)
from willow.tools import TOOLS_BY_NAME
from willow.tools.base import Tool

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text).replace("\r", "")


class _ScriptedStreamProvider(Provider):
    def __init__(self, scripts: list[list[Any]]) -> None:
        self._scripts: deque[list[Any]] = deque(scripts)
        self.requests: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest) -> CompletionResponse:  # pragma: no cover
        raise NotImplementedError

    def stream(self, request: CompletionRequest) -> Iterator[Any]:
        self.requests.append(request)
        if not self._scripts:
            raise RuntimeError("provider exhausted")
        yield from self._scripts.popleft()


class _ResettableScriptedStreamProvider(_ScriptedStreamProvider):
    def __init__(self, scripts: list[list[Any]]) -> None:
        super().__init__(scripts)
        self.reset_count = 0

    def reset_conversation(self) -> None:
        self.reset_count += 1


class _RaisingStreamProvider(Provider):
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.requests: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest) -> CompletionResponse:  # pragma: no cover
        raise NotImplementedError

    def stream(self, request: CompletionRequest) -> Iterator[Any]:
        self.requests.append(request)
        raise self.error
        yield  # pragma: no cover


class _ParentChildProvider(Provider):
    def __init__(self) -> None:
        self.requests: list[CompletionRequest] = []
        self.child_requests: list[CompletionRequest] = []
        self._stream_turns: deque[list[Any]] = deque(
            [
                [
                    ToolUseDelta(id="spawn_1", name="spawn_agent", partial_json=None),
                    StreamComplete(
                        response=CompletionResponse(
                            content=[
                                ToolUseBlock(
                                    id="spawn_1",
                                    name="spawn_agent",
                                    input={
                                        "task": "inspect child task",
                                        "agent_type": "explorer",
                                    },
                                )
                            ],
                            stop_reason="tool_use",
                        )
                    ),
                ],
                [
                    TextDelta(text="parent done"),
                    StreamComplete(
                        response=CompletionResponse(
                            content=[TextBlock(text="parent done")],
                            stop_reason="end_turn",
                        )
                    ),
                ],
            ]
        )

    def fork(self) -> _ParentChildProvider:
        return self

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.child_requests.append(request)
        return CompletionResponse(
            content=[TextBlock(text="child done")],
            stop_reason="end_turn",
        )

    def stream(self, request: CompletionRequest) -> Iterator[Any]:
        self.requests.append(request)
        if not self._stream_turns:
            raise RuntimeError("provider exhausted")
        yield from self._stream_turns.popleft()


class _RecordingTool(Tool):
    name = "echo"
    description = "Return back the given message."
    input_schema = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return f"echoed: {kwargs.get('message', '')}"


class _PromptObservingTool(Tool):
    name = "observe_prompt"
    description = "Record the terminal while the tool is running."
    input_schema = {"type": "object", "properties": {}}

    def __init__(self, out: io.StringIO) -> None:
        self.out = out
        self.observed = ""

    def run(self, **_kwargs: Any) -> str:
        self.observed = self.out.getvalue()
        return "observed"


class _TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


class _FlushingTTYBuffer(_TTYBuffer):
    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()


def _make_args(**overrides: Any) -> argparse.Namespace:
    defaults = dict(
        provider="openai_responses",
        model="gpt-5.5",
        system=None,
        max_tokens=4096,

        thinking=False,
        effort=None,
        prompt=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _assert_interrupted_retry_content(
    content: list[Any],
    *,
    interrupted: str,
    interrupting: str,
) -> None:
    assert len(content) == 1
    block = content[0]
    assert isinstance(block, TextBlock)
    assert "Answer each active user message below now, in order." in block.text
    assert (
        f"[active user message 1 of 2; interrupted before completion]\n{interrupted}"
        in block.text
    )
    assert (
        f"[active user message 2 of 2; new message after interruption]\n{interrupting}"
        in block.text
    )


def _drive(
    provider: Provider,
    inputs: list[str],
    args: argparse.Namespace | None = None,
) -> tuple[str, tui.WillowApp]:
    args = args or _make_args()
    inputs_iter = iter(inputs)

    def input_func(_prompt: str = "") -> str:
        try:
            return next(inputs_iter)
        except StopIteration as exc:
            raise EOFError from exc

    out = io.StringIO()
    app = tui.WillowApp(args, provider, input_func=input_func, out=out)
    assert app.run() == 0
    return out.getvalue(), app


def test_basic_tui_submits_positional_prompt_before_reading_input() -> None:
    response = CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider(
        [[TextDelta(text="done"), StreamComplete(response=response)]]
    )

    out, app = _drive(provider, ["/exit"], args=_make_args(prompt="run this task"))

    assert len(provider.requests) == 1
    assert provider.requests[0].messages[0].content == [TextBlock(text="run this task")]
    assert [message.role for message in app.messages] == ["user", "assistant"]
    assert "run this task" in out
    assert "done" in out


def test_basic_tui_provider_error_returns_to_prompt() -> None:
    provider = _RaisingStreamProvider(RuntimeError("Codex error: policy warning"))

    out, app = _drive(provider, ["run the task", "/exit"])

    assert len(provider.requests) == 1
    assert [message.role for message in app.messages] == ["user"]
    assert "[error] Codex error: policy warning" in out
    assert "RuntimeError(" not in out
    assert "Goodbye." in out


def _session_record(
    session_id: str,
    *,
    provider: str = "openai_responses",
    model: str = "gpt-5.5",
    updated_at: str = "2026-05-09T10:00:00Z",
    text: str = "hello",
) -> session.SessionRecord:
    return session.SessionRecord(
        metadata=session.SessionMetadata(
            id=session_id,
            created_at="2026-05-09T09:00:00Z",
            updated_at=updated_at,
            title=None,
            cwd="/tmp/project",
        ),
        settings=session.SessionSettings(
            provider=provider,
            model=model,
            system="Be direct.",
            max_tokens=1234,
        ),
        messages=[Message(role="user", content=[TextBlock(text=text)])],
    )


def _text_message(index: int, text: str) -> Message:
    return Message(
        role="user" if index % 2 else "assistant",
        content=[TextBlock(text=text)],
    )


def test_run_tui_builds_provider_and_runs_native_session(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = object()
    monkeypatch.setattr("willow.cli._build_provider", lambda _name: provider)

    instances: list[FakeWillowApp] = []

    class FakeWillowApp:
        def __init__(
            self,
            args: argparse.Namespace,
            built_provider: object,
            *,
            inline_mode: bool,
            state: dict[str, object] | None,
        ) -> None:
            self.args = args
            self.provider = built_provider
            self.inline_mode = inline_mode
            self.state = state
            instances.append(self)

        def run(self) -> int:
            return 42

    monkeypatch.setattr(tui, "WillowApp", FakeWillowApp)

    args = _make_args()
    assert tui.run_tui(args) == 42
    assert args.persist_session is True
    assert len(instances) == 1
    assert instances[0].provider is provider
    assert instances[0].inline_mode is True
    assert instances[0].state is None


def test_run_tui_resumes_session_id_and_restores_saved_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(session.SESSION_DIR_ENV, str(tmp_path / "sessions"))
    record = _session_record(
        "sess_123",
        provider="anthropic",
        model="claude-sonnet-4-6",
        text="continue this",
    )
    path = session.default_session_path(record.metadata.id)
    session.save_session(record, path)
    built: list[str] = []
    provider = object()

    def build_provider(name: str) -> object:
        built.append(name)
        return provider

    monkeypatch.setattr("willow.cli._build_provider", build_provider)

    instances: list[Any] = []

    class FakeWillowApp:
        def __init__(
            self,
            args: argparse.Namespace,
            built_provider: object,
            *,
            inline_mode: bool,
            state: dict[str, object] | None,
        ) -> None:
            self.args = args
            self.provider = built_provider
            self.inline_mode = inline_mode
            self.state = state
            instances.append(self)

        def run(self) -> int:
            return 0

    monkeypatch.setattr(tui, "WillowApp", FakeWillowApp)

    args = _make_args(resume="sess_123")
    assert tui.run_tui(args) == 0

    assert built == ["anthropic"]
    assert args.persist_session is True
    assert args.provider == "anthropic"
    assert args.model == "claude-sonnet-4-6"
    assert args.max_tokens == 1234
    assert args._resume_session_path == path
    assert instances[0].state is not None
    assert instances[0].state["messages"] == record.messages


def test_resumed_session_renders_saved_history_before_prompt() -> None:
    record = _session_record("sess_resume", text="old question")
    record = session.SessionRecord(
        metadata=record.metadata,
        settings=record.settings,
        messages=[
            *record.messages,
            Message(role="assistant", content=[TextBlock(text="old answer")]),
        ],
    )
    args = _make_args(persist_session=True)
    args.provider = record.settings.provider
    args.model = record.settings.model
    args.max_tokens = record.settings.max_tokens
    args._resume_session_record = record
    args._resume_session_path = Path("/tmp/sess_resume.jsonl")

    inputs_iter = iter([])

    def input_func(_prompt: str = "") -> str:
        try:
            return next(inputs_iter)
        except StopIteration as exc:
            raise EOFError from exc

    out_buffer = io.StringIO()
    app = tui.WillowApp(
        args,
        _ScriptedStreamProvider([]),
        state=tui._state_from_session(record),
        input_func=input_func,
        out=out_buffer,
    )
    assert app.run() == 0
    out = out_buffer.getvalue()

    assert "[resumed] Loaded 2 saved message(s)" in out
    assert "old question" in out
    assert "old answer" in out
    assert out.index("old question") < out.index("Goodbye.")


def test_resumed_session_sends_saved_history_on_next_request() -> None:
    record = _session_record("sess_resume", text="old question")
    record = session.SessionRecord(
        metadata=record.metadata,
        settings=record.settings,
        messages=[
            *record.messages,
            Message(role="assistant", content=[TextBlock(text="old answer")]),
        ],
    )
    args = _make_args(persist_session=True)
    args.provider = record.settings.provider
    args.model = record.settings.model
    args.max_tokens = record.settings.max_tokens
    args._resume_session_record = record
    args._resume_session_path = Path("/tmp/sess_resume.jsonl")
    response = CompletionResponse(content=[TextBlock(text="new answer")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider([[StreamComplete(response=response)]])
    inputs_iter = iter(["new question", "/exit"])

    def input_func(_prompt: str = "") -> str:
        try:
            return next(inputs_iter)
        except StopIteration as exc:
            raise EOFError from exc

    app = tui.WillowApp(
        args,
        provider,
        state=tui._state_from_session(record),
        input_func=input_func,
        out=io.StringIO(),
    )
    assert app.run() == 0

    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert [message.role for message in request.messages] == ["user", "assistant", "user"]
    assert request.messages == [
        *record.messages,
        Message(role="user", content=[TextBlock(text="new question")]),
    ]


def test_resumed_session_ending_with_tool_result_continues_turn(tmp_path: Path) -> None:
    record = _session_record("sess_tool_result", text="commit changes")
    record = session.SessionRecord(
        metadata=record.metadata,
        settings=record.settings,
        messages=[
            *record.messages,
            Message(
                role="assistant",
                content=[
                    ToolUseBlock(
                        id="call_1",
                        name="bash",
                        input={"command": "git commit"},
                    )
                ],
            ),
            Message(
                role="user",
                content=[
                    ToolResultBlock(
                        tool_use_id="call_1",
                        content="[stderr]\nfatal: no commits yet",
                    )
                ],
            ),
        ],
    )
    args = _make_args(persist_session=True)
    args.provider = record.settings.provider
    args.model = record.settings.model
    args.max_tokens = record.settings.max_tokens
    args._resume_session_record = record
    args._resume_session_path = tmp_path / "sess_tool_result.jsonl"
    response = CompletionResponse(content=[TextBlock(text="continued")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider(
        [[TextDelta(text="continued"), StreamComplete(response=response)]]
    )
    inputs_iter = iter(["/exit"])

    def input_func(_prompt: str = "") -> str:
        try:
            return next(inputs_iter)
        except StopIteration as exc:
            raise EOFError from exc

    out_buffer = io.StringIO()
    app = tui.WillowApp(
        args,
        provider,
        state=tui._state_from_session(record),
        input_func=input_func,
        out=out_buffer,
    )
    assert app.run() == 0

    assert len(provider.requests) == 1
    assert provider.requests[0].messages == record.messages
    out = out_buffer.getvalue()
    assert "[resumed] Continuing from saved tool result." in out
    saved = session.load_session(app._session_path)
    assert saved.messages[-1] == Message(role="assistant", content=[TextBlock(text="continued")])


def test_run_tui_resume_picker_uses_latest_session_when_not_interactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    older = tui.SessionEntry(
        path=Path("/tmp/older.jsonl"),
        record=_session_record("older", updated_at="2026-05-09T09:00:00Z"),
    )
    latest = tui.SessionEntry(
        path=Path("/tmp/latest.jsonl"),
        record=_session_record(
            "latest",
            provider="anthropic",
            model="claude-sonnet-4-6",
            updated_at="2026-05-09T10:00:00Z",
        ),
    )
    monkeypatch.setattr(tui, "list_sessions", lambda: [latest, older])
    monkeypatch.setattr("willow.cli._build_provider", lambda _name: object())

    instances: list[Any] = []

    class FakeWillowApp:
        def __init__(
            self,
            args: argparse.Namespace,
            built_provider: object,
            *,
            inline_mode: bool,
            state: dict[str, object] | None,
        ) -> None:
            self.args = args
            self.state = state
            instances.append(self)

        def run(self) -> int:
            return 0

    monkeypatch.setattr(tui, "WillowApp", FakeWillowApp)

    args = _make_args(resume="")
    assert tui.run_tui(args) == 0

    assert args._resume_session_path == latest.path
    assert args.provider == "anthropic"
    assert instances[0].state is not None
    assert instances[0].state["current_model"] == "claude-sonnet-4-6"


def test_run_tui_resume_without_sessions_starts_new(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tui, "list_sessions", lambda: [])
    monkeypatch.setattr("willow.cli._build_provider", lambda _name: object())

    instances: list[Any] = []

    class FakeWillowApp:
        def __init__(
            self,
            args: argparse.Namespace,
            built_provider: object,
            *,
            inline_mode: bool,
            state: dict[str, object] | None,
        ) -> None:
            self.args = args
            self.state = state
            instances.append(self)

        def run(self) -> int:
            return 0

    monkeypatch.setattr(tui, "WillowApp", FakeWillowApp)

    args = _make_args(resume="")
    assert tui.run_tui(args) == 0

    assert not hasattr(args, "_resume_session_path")
    assert instances[0].state is None


def test_command_hints_match_slash_prefix() -> None:
    rendered = tui._render_command_hints("/stat")

    assert "/status  show session and status details" in rendered
    assert "/statusline  toggle status snapshots in the prompt" in rendered
    assert "/model" not in rendered
    assert "/clear" not in rendered


def test_skill_hints_match_skill_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    skill_path = tmp_path / "project" / ".willow" / "skills" / "reviewer" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("---\ndescription: Review code.\n---\nbody", encoding="utf-8")

    rendered = tui._render_input_hints("/rev", tmp_path / "project")

    assert "/reviewer  Review code." in rendered


def test_input_hints_show_commands_and_skills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    skill_path = (
        tmp_path / "project" / ".willow" / "skills" / "hello_world" / "SKILL.md"
    )
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text(
        "---\ndescription: Say hello.\n---\nbody",
        encoding="utf-8",
    )

    rendered = tui._render_input_hints("/", tmp_path / "project")

    assert "/clear  reset conversation history" in rendered
    assert "/model  choose what model to use" in rendered
    assert "/hello_world  Say hello." in rendered


def test_input_hints_filter_commands_and_skills_by_same_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    skills_root = tmp_path / "project" / ".willow" / "skills"
    hello_path = skills_root / "hello_world" / "SKILL.md"
    review_path = skills_root / "reviewer" / "SKILL.md"
    hello_path.parent.mkdir(parents=True)
    review_path.parent.mkdir(parents=True)
    hello_path.write_text(
        "---\ndescription: Say hello.\n---\nbody",
        encoding="utf-8",
    )
    review_path.write_text(
        "---\ndescription: Review code.\n---\nbody",
        encoding="utf-8",
    )

    rendered = tui._render_input_hints("/he", tmp_path / "project")

    assert "/help  show commands and settings" in rendered
    assert "/hello_world  Say hello." in rendered
    assert "/reviewer" not in rendered
    assert "/model" not in rendered


def test_input_hints_fuzzy_match_at_file_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "src" / "willow").mkdir(parents=True)
    (project / "src" / "willow" / "tui.py").write_text("print('hi')", encoding="utf-8")
    (project / "README.md").write_text("docs", encoding="utf-8")
    (project / "notes draft.txt").write_text("draft", encoding="utf-8")
    (project / "node_modules").mkdir()
    (project / "node_modules" / "notes.js").write_text("ignored", encoding="utf-8")

    rendered = tui._render_input_hints("please inspect @tui", project)

    assert "@src/willow/tui.py" in rendered
    assert "node_modules" not in rendered

    rendered = tui._render_input_hints("please inspect @draft", project)

    assert "@'notes draft.txt'" in rendered


def test_apply_at_file_hint_replaces_active_token() -> None:
    completed = tui._apply_hint_to_input(
        "check this @rea",
        "@README.md",
        append_space_when_empty=True,
    )

    assert completed == "check this README.md "


def test_render_statusline_includes_context_usage_and_total_tokens() -> None:
    rendered = tui._render_statusline(
        model="gpt-5.5",
        context_tokens=10_500,
        context_window=1_050_000,
        input_tokens=40_000,
        cached_tokens=12_000,
        output_tokens=2_000,
        cwd="~/repos/willow",
    )

    assert (
        rendered
        == "gpt-5.5 | Context 1.0% used (10.5k tok) | window: 1.1M tok | "
        "input: 40.0k tok | cached total: 12.0k tok | output: 2.0k tok | cwd: ~/repos/willow"
    )
    assert " · " not in rendered


def test_render_statusline_shows_zero_before_provider_usage() -> None:
    rendered = tui._render_statusline(
        model="gpt-5.5",
        context_tokens=None,
        context_window=1_050_000,
        input_tokens=0,
        cached_tokens=0,
        output_tokens=0,
        cwd="~/repos/willow",
    )

    assert (
        rendered
        == "gpt-5.5 | Context 0.0% used (0 tok) | window: 1.1M tok | cwd: ~/repos/willow"
    )


def test_styled_statusline_colors_model_and_context_usage() -> None:
    rendered = tui._style_statusline_text(
        "gpt-5.5 | Context 0.0% used (3 tok) | window: 1.1M tok | "
        "input: 1 tok | cached total: 0 tok | output: 2 tok | cwd: ~/repos/willow"
    )

    assert f"{tui.STATUS_MODEL_STYLE}gpt-5.5{tui.STATUS_STYLE}" in rendered
    assert ";1m" not in tui.STATUS_MODEL_STYLE
    assert ";1m" not in tui.STATUS_TOKEN_STYLE
    assert (
        f"{tui.STATUS_TOKEN_STYLE}Context 0.0% used (3 tok) | window: 1.1M tok | "
        f"input: 1 tok | cached total: 0 tok | output: 2 tok"
        f"{tui.STATUS_STYLE}"
    ) in rendered
    assert "cwd: ~/repos/willow" in rendered


def test_status_text_uses_last_provider_input_tokens_not_heuristic() -> None:
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=io.StringIO())
    app.system = None
    app.tool_specs = []
    app.messages = [Message(role="user", content=[TextBlock(text="x" * 42_000)])]
    app._last_context_tokens = 862
    app._total_input_tokens = 900_000
    app._total_cached_tokens = 300_000
    app._total_output_tokens = 100_000

    rendered = app._status_text()

    assert "Context 0.1% used (862 tok) | window: 1.1M tok" in rendered
    assert "input: 900.0k tok" in rendered
    assert "cached total: 300.0k tok" in rendered
    assert "output: 100.0k tok" in rendered


def test_terminal_appends_without_rewriting_scrollback() -> None:
    end_response = CompletionResponse(
        content=[TextBlock(text="hi there")],
        stop_reason="end_turn",
        usage={"input_tokens": 1, "cached_tokens": 1, "output_tokens": 2},
    )
    provider = _ScriptedStreamProvider(
        [[TextDelta(text="hi "), TextDelta(text="there"), StreamComplete(response=end_response)]]
    )

    out, _app = _drive(provider, ["hello", "/exit"])

    assert "Willow Agent" in out
    assert "~~ \\|/ ~~" in out
    assert "model:     gpt-5.5" in out
    assert "directory:" in out
    assert "Session:" not in out
    assert "hi there" in out
    assert "[status] gpt-5.5 | Context " in out
    assert "input: 1 tok | cached total: 1 tok | output: 2 tok" in out
    assert " | cwd: " in out
    assert "Goodbye." in out
    forbidden = ["\x1b[2J", "\x1b[H", "\x1b[1A", "\x1b[K", "\x1b[?1049h"]
    assert all(sequence not in out for sequence in forbidden)


def test_basic_tui_writes_worked_duration_when_turn_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([10.0, 75.0])
    monkeypatch.setattr(tui.time, "monotonic", lambda: next(ticks))
    response = CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider([[StreamComplete(response=response)]])

    out, _app = _drive(provider, ["hello", "/exit"])

    assert "Worked for 1m 5s" in out
    assert "[status] Worked" not in out


def test_tty_worked_duration_uses_muted_foreground(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tui.time, "monotonic", lambda: 75.0)
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False

    app._write_worked_duration(10.0)

    rendered = out.getvalue()
    assert rendered == f"{tui.WORKED_DURATION_STYLE}Worked for 1m 5s{tui.RESET}\n"


def test_tui_attaches_local_image_from_user_text(tmp_path: Path) -> None:
    image = tmp_path / "debug shot.png"
    image.write_bytes(b"fake-png")
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)

    assert app._submit_user_text(f'check image at "{image}"', render=True)

    message = app.messages[-1]
    assert message.role == "user"
    assert isinstance(message.content[0], TextBlock)
    assert isinstance(message.content[1], ImageBlock)
    assert message.content[0].text == "check image at [image#1]"
    assert message.content[1].path == str(image.resolve())
    assert message.content[1].media_type == "image/png"
    rendered = out.getvalue()
    assert "check image at [image#1]" in rendered
    assert str(image) not in rendered
    assert "[image] debug shot.png" not in rendered


def test_tui_numbers_multiple_local_images_in_user_text(tmp_path: Path) -> None:
    first = tmp_path / "left.png"
    second = tmp_path / "right.png"
    first.write_bytes(b"fake-png-left")
    second.write_bytes(b"fake-png-right")
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)

    assert app._submit_user_text(f"compare {first} with {second}", render=True)

    message = app.messages[-1]
    assert message.role == "user"
    assert message.content[0] == TextBlock(text="compare [image#1] with [image#2]")
    assert [
        block.filename for block in message.content if isinstance(block, ImageBlock)
    ] == ["left.png", "right.png"]
    rendered = out.getvalue()
    assert "compare [image#1] with [image#2]" in rendered
    assert str(first) not in rendered
    assert str(second) not in rendered


def test_tui_sends_text_file_path_as_text_context(tmp_path: Path) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.write_text("secret file content", encoding="utf-8")
    cwd = Path.cwd()
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)

    assert app._submit_user_text(f'check file "{text_file}"', render=True)

    message = app.messages[-1]
    assert message.role == "user"
    assert message.content == [
        TextBlock(text=f'check file "{text_file}"'),
        TextBlock(text=Path(os.path.relpath(text_file.resolve(), cwd)).as_posix()),
    ]
    assert "secret file content" not in str(message.content)
    rendered = out.getvalue()
    assert "[file]" not in rendered
    assert "(text/plain" not in rendered


def test_tui_supports_at_prefixed_image_and_file_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "shot.png"
    image.write_bytes(b"fake-png")
    text_file = tmp_path / "notes.txt"
    text_file.write_text("secret file content", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)

    assert app._submit_user_text("check @shot.png and @notes.txt", render=True)

    message = app.messages[-1]
    assert message.content[0] == TextBlock(text="check [image#1] and @notes.txt")
    assert any(
        isinstance(block, ImageBlock) and block.filename == "shot.png"
        for block in message.content
    )
    assert TextBlock(text="notes.txt") in message.content
    assert "secret file content" not in str(message.content)


def test_absolute_dragged_image_path_is_not_treated_as_slash_command(
    tmp_path: Path,
) -> None:
    image = tmp_path / "Screenshot 2026-05-18.png"
    image.write_bytes(b"fake-png")
    dragged_path = str(image).replace(" ", "\\ ")
    response = CompletionResponse(content=[TextBlock(text="ok")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider([[StreamComplete(response=response)]])

    out, app = _drive(provider, [dragged_path, "/exit"])

    assert "Unknown command" not in out
    assert len(provider.requests) == 1
    message = provider.requests[0].messages[0]
    assert message.role == "user"
    assert message.content[0] == TextBlock(text="[image#1]")
    assert any(
        isinstance(block, ImageBlock) and block.filename == image.name
        for block in message.content
    )
    assert app.messages[0].content == message.content


def test_unknown_slash_text_still_routes_to_command_error() -> None:
    out, _app = _drive(_ScriptedStreamProvider([]), ["/definitely-not-a-command", "/exit"])

    assert "Unknown command: /definitely-not-a-command" in out


def test_assistant_text_uses_terminal_background() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False

    app._write_block("hello", tui.USER_STYLE)
    app._write_block("hi", tui.ASSISTANT_STYLE)
    app._write_panel("tool result", "ok", tui.TOOL_STYLE)

    rendered = out.getvalue()
    assert tui.USER_STYLE in rendered
    assert tui.ASSISTANT_STYLE in rendered
    assert tui.TOOL_STYLE in rendered
    assert "48;5" not in tui.ASSISTANT_STYLE
    assert "48;5" not in tui.THINKING_STYLE
    assert "48;5" in tui.PROMPT_STYLE
    assert "48;5;235" in tui.USER_STYLE
    assert tui.USER_STYLE == tui.PROMPT_STYLE
    assert "you" not in rendered
    assert "assistant" not in rendered
    assert rendered.count(tui.RESET) >= 3


def test_chat_text_blocks_keep_single_row_vertical_padding() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._terminal_width = lambda: 10  # type: ignore[method-assign]

    app._write_block("hello", tui.USER_STYLE)
    app._write_block("hi", tui.ASSISTANT_STYLE)

    rendered_lines = _strip_ansi(out.getvalue()).splitlines()
    assert rendered_lines == [
        "          ",
        " hello    ",
        "          ",
        "          ",
        " hi       ",
        "          ",
    ]


def test_user_history_block_clears_full_terminal_rows() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._terminal_width = lambda: 10  # type: ignore[method-assign]

    app._write_block("hello", tui.USER_STYLE)

    rendered = out.getvalue()
    assert rendered.count("\x1b[?7l") == 3
    assert rendered.count("\x1b[2K") == 3
    assert rendered.count("\x1b[?7h") == 3


def test_styled_transcript_block_returns_to_column_zero_before_painting() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._terminal_width = lambda: 10  # type: ignore[method-assign]
    app._write("stale cursor column")

    app._write_block("hello", tui.ASSISTANT_STYLE)

    rendered = out.getvalue()
    assert "stale cursor column\r" in rendered
    assert f"\r\r\x1b[?7l{tui.ASSISTANT_STYLE}\x1b[2K" in rendered


def test_styled_terminal_line_clears_before_applying_background() -> None:
    rendered = tui._styled_terminal_line("hi", tui.PROMPT_STYLE, 8)

    assert rendered.startswith("\r")
    assert f"{tui.PROMPT_STYLE}\x1b[2K" in rendered
    assert f"\x1b[2K{tui.PROMPT_STYLE}" not in rendered
    assert _strip_ansi(rendered) == "hi      "


def test_terminal_width_allows_zoomed_terminals_below_forty_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    monkeypatch.setattr(
        tui.shutil,
        "get_terminal_size",
        lambda _fallback: os.terminal_size((28, 24)),
    )

    assert app._terminal_width() == 28


def test_styled_transcript_wraps_long_lines_without_dropping_text() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._terminal_width = lambda: 12  # type: ignore[method-assign]

    app._write_block("abcdefghijklmnopqrstuvwxyz", tui.ASSISTANT_STYLE)
    app._write_panel("status", "0123456789abcdefghijklmnopqrstuvwxyz", tui.STATUS_STYLE)

    rendered = out.getvalue()
    assert "mnopqrstuvw" in rendered
    assert "xyz" in rendered
    assert "z           " in rendered


def test_assistant_markdown_renders_as_terminal_text() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._terminal_width = lambda: 40  # type: ignore[method-assign]

    app._write_block(
        "# Plan\n\n- **Build** it\n- Use `pytest`\n\n```python\nprint('ok')\n```",
        tui.ASSISTANT_STYLE,
    )

    rendered = out.getvalue()
    visible = _strip_ansi(rendered)
    assert "Plan" in visible
    assert "• Build it" in visible
    assert "• Use pytest" in visible
    assert "print('ok')" in visible
    assert "# Plan" not in visible
    assert "**Build**" not in visible
    assert "`pytest`" not in visible
    assert "```" not in visible
    assert tui.TOOL_PREVIEW_STYLE in rendered


def test_running_terminal_line_animates_without_changing_text() -> None:
    first = tui._running_terminal_line(" running bash - pytest", 28, frame=0)
    second = tui._running_terminal_line(" running bash - pytest", 28, frame=6)
    bright = tui._running_terminal_line(" running bash - pytest", 28, frame=4)

    assert first != second
    assert "\x1b[40;" in first
    assert "\x1b[40;38;5;51;1m" in bright
    assert "\x1b[40;38;5;255m" in first
    assert tui.STATUS_STYLE not in first
    assert _strip_ansi(first) == " running bash - pytest      "
    assert _strip_ansi(second) == " running bash - pytest      "


def test_flower_working_status_renders_shape_with_gradient() -> None:
    flower, text = tui._flower_working_status(0)

    rendered = tui._running_terminal_line(f" {text}", 30, frame=0, flower=flower)

    assert flower.name == "wattle"
    assert text == "✿ wattling..."
    assert "\x1b[40;38;5;227;1m✿" in rendered
    assert _strip_ansi(rendered) == " ✿ wattling...                "


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "0s"),
        (12, "12s"),
        (75, "1m 15s"),
        (3600, "1h"),
        (3725, "1h 2m"),
    ],
)
def test_format_elapsed_compact(seconds: int, expected: str) -> None:
    assert tui._format_elapsed_compact(seconds) == expected


def test_worked_duration_text_uses_compact_elapsed_format() -> None:
    assert tui._worked_duration_text(10.0, ended_at=85.0) == "Worked for 1m 15s"


def test_tool_running_title_collapses_multiline_bash_command() -> None:
    block = ToolUseBlock(
        id="call_1",
        name="bash",
        input={
            "command": (
                "python - <<'PY'\n"
                "from pathlib import Path\n"
                "from transformers import AutoTokenizer\n"
                "print('ready')\n"
                "PY"
            )
        },
    )

    title = tui._tool_running_title(block)

    assert title.startswith("running bash - python - <<'PY' from pathlib import Path")
    assert "\n" not in title
    assert "\r" not in title


def test_tool_running_title_describes_wait_agent() -> None:
    block = ToolUseBlock(
        id="call_1",
        name="wait_agent",
        input={"subagent_id": "subagent-123"},
    )

    assert tui._tool_running_title(block) == "Waiting for subagent"


def test_live_prompt_box_shows_working_when_streaming_without_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tui.time, "monotonic", lambda: 100.0)
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.working_started_at = 88.0

    live._draw_prompt()

    rendered = out.getvalue()
    assert tui.PROMPT_STYLE in rendered
    assert "\x1b[40;" in rendered
    assert tui.STATUS_STYLE not in rendered[: rendered.index(tui.PROMPT_STYLE)]
    visible = _strip_ansi(rendered)
    assert "irising... (12s, press esc to interrupt)" in visible
    assert " > " in rendered
    assert live.prompt_lines == 7
    assert live.prompt_cursor_offset_from_bottom == 2
    assert "\r\x1b[3C\x1b[?25h" in rendered
    visible_lines = visible.splitlines()
    status_line_index = next(
        index for index, line in enumerate(visible_lines) if "press esc to interrupt" in line
    )
    assert status_line_index > 0
    assert visible_lines[status_line_index - 1].strip() == ""
    assert visible_lines[status_line_index + 1].strip() == ""
    assert visible.index("press esc to interrupt") < visible.index(" > ")
    assert "streaming" not in rendered
    assert "ready" not in rendered


def test_live_active_tool_status_has_gap_before_input_box() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.active_tool_status = "running bash - pytest"

    live._draw_prompt()

    rendered = _strip_ansi(out.getvalue())
    assert "running bash - pytest" in rendered
    assert "running bash - pytest" in rendered[: rendered.index(" > ")]
    visible_lines = rendered.splitlines()
    running_line_index = next(
        index for index, line in enumerate(visible_lines) if "running bash - pytest" in line
    )
    assert running_line_index > 0
    assert visible_lines[running_line_index - 1].strip() == ""
    assert visible_lines[running_line_index + 1].strip() == ""
    assert live.prompt_lines == 7
    assert live.prompt_cursor_offset_from_bottom == 2


def test_live_running_status_redraw_does_not_repaint_input_box() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.active_tool_status = "running bash - pytest"

    live._draw_prompt()
    out.seek(0)
    out.truncate(0)
    live._redraw_running_status_line()

    rendered = out.getvalue()
    assert "running bash - pytest" in _strip_ansi(rendered)
    assert " > " not in rendered
    assert tui.PROMPT_STYLE not in rendered
    assert "\x1b[2K" in rendered


def test_live_running_status_width_change_repaints_prompt_box() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    widths = [24]
    app._terminal_width = lambda: widths[-1]  # type: ignore[method-assign]
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.working_started_at = time.monotonic() - 2
    live.buffer = "queued text"
    live.cursor = len(live.buffer)
    live._draw_prompt()

    out.seek(0)
    out.truncate(0)
    widths.append(10)
    live._redraw_running_status_line()

    rendered = out.getvalue()
    assert tui.PROMPT_STYLE in rendered
    assert " > queued" in rendered
    assert "   text" in rendered
    assert "\x1b[?7l" in rendered
    assert "\x1b[40;" not in rendered[rendered.index(tui.PROMPT_STYLE) :]
    assert live.prompt_width == 10


def test_live_running_status_zoom_in_clears_below_prompt_bottom() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._statusline_enabled = False
    widths = [28]
    app._terminal_width = lambda: widths[-1]  # type: ignore[method-assign]
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.working_started_at = time.monotonic() - 2
    live.buffer = "queued input that wraps after zoom"
    live.cursor = len(live.buffer)
    live._draw_prompt()

    out.seek(0)
    out.truncate(0)
    widths.append(9)
    live._redraw_running_status_line()

    rendered = out.getvalue()
    assert "\x1b[1B\x1b[?25l\r\x1b[J" in rendered
    assert "\x1b[1A\r\x1b[2K" in rendered
    visible_lines = _strip_ansi(rendered).splitlines()
    input_line_index = next(
        index for index, line in enumerate(visible_lines) if line.startswith(" > ")
    )
    assert visible_lines[input_line_index + 1].startswith("   ")
    assert live.prompt_width == 9


def test_live_prompt_survives_repeated_zoom_in_and_out_without_black_input_gap() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    widths = [34]
    app._terminal_width = lambda: widths[-1]  # type: ignore[method-assign]
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.working_started_at = time.monotonic() - 2
    live.buffer = "queued input that should stay in the teal input box"
    live.cursor = len(live.buffer)
    live._draw_prompt()

    out.seek(0)
    out.truncate(0)
    for width in (12, 42, 9, 36):
        widths.append(width)
        live._redraw_running_status_line()

    rendered = out.getvalue()
    assert live.prompt_width == 36
    assert "\x1b[J" in rendered
    final_prompt = rendered[rendered.rfind(" > queued input") :]
    input_box = final_prompt[final_prompt.index(tui.PROMPT_STYLE) :]
    assert "\x1b[40;" not in input_box
    visible_lines = _strip_ansi(final_prompt).splitlines()
    input_line_index = next(
        index for index, line in enumerate(visible_lines) if line.startswith(" > ")
    )
    assert visible_lines[input_line_index + 1].startswith("   ")


def test_live_prompt_box_switches_to_type_box_when_streaming_with_input() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.buffer = "queued text"
    live.cursor = len(live.buffer)

    live._draw_prompt()

    rendered = out.getvalue()
    assert " > queued text" in rendered
    assert "\r\x1b[14C\x1b[?25h" in rendered
    visible = _strip_ansi(rendered)
    assert "press esc to interrupt" in visible
    assert visible.index("press esc to interrupt") < visible.index(" > queued text")


def test_live_prompt_cursor_overlays_character_without_shifting_text() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "abcdef"
    live.cursor = 2

    live._draw_prompt()

    rendered = out.getvalue()
    assert " > abcdef" in rendered
    assert tui.PROMPT_MARKER_STYLE not in rendered
    assert "\r\x1b[5C\x1b[?25h" in rendered
    assert " > ab▌cdef" not in rendered


def test_live_prompt_wraps_long_input_across_lines() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._statusline_enabled = False
    app._terminal_width = lambda: 10  # type: ignore[method-assign]
    live = tui._LiveTerminal(app)
    live.buffer = "abcdefghijkl"
    live.cursor = len(live.buffer)

    live._draw_prompt()

    rendered = out.getvalue()
    assert " > abcdefg" in rendered
    assert "\n" in rendered
    assert "   hijkl" in rendered
    assert " > hijkl" not in rendered
    assert live.prompt_lines == 4
    assert "\x1b[8C\x1b[?25h" in rendered


def test_live_prompt_growing_input_overwrites_without_full_clear() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._statusline_enabled = False
    app._terminal_width = lambda: 10  # type: ignore[method-assign]
    live = tui._LiveTerminal(app)
    live.buffer = "abcdefg"
    live.cursor = len(live.buffer)
    live._draw_prompt()

    out.seek(0)
    out.truncate(0)
    live.buffer = "abcdefghijkl"
    live.cursor = len(live.buffer)
    live._draw_prompt()

    rendered = out.getvalue()
    assert " > abcdefg" in rendered
    assert "   hijkl" in rendered
    assert "\x1b[1A\r\x1b[2K" not in rendered
    assert live.prompt_lines == 4


def test_live_prompt_collapses_pasted_content_placeholder() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._statusline_enabled = False
    live = tui._LiveTerminal(app)
    pasted = "Instruction\n" + ("Set up service.\n" * 20)

    live._insert_pasted_text(pasted)
    live._draw_prompt()

    rendered = out.getvalue()
    assert f"[Pasted Content {len(pasted)} chars]" in rendered
    assert "Set up service." not in rendered
    assert live.buffer == pasted
    assert live.cursor == len(pasted)


def test_live_prompt_clear_accounts_for_terminal_zoom_in() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    widths = [20]
    app._terminal_width = lambda: widths[-1]  # type: ignore[method-assign]
    live = tui._LiveTerminal(app)

    live._draw_prompt()
    out.seek(0)
    out.truncate(0)
    widths.append(10)
    live._draw_prompt()

    rendered = out.getvalue()
    assert "\x1b[2B\x1b[?25l\r\x1b[J" in rendered
    assert "\x1b[1A\r\x1b[2K" in rendered
    assert " > " in rendered
    assert live.prompt_width == 10


def test_live_prompt_clear_accounts_for_cursor_line_reflow_after_zoom_in() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._statusline_enabled = False
    widths = [24]
    app._terminal_width = lambda: widths[-1]  # type: ignore[method-assign]
    live = tui._LiveTerminal(app)
    live.buffer = "abc"
    live.cursor = len(live.buffer)

    live._draw_prompt()
    out.seek(0)
    out.truncate(0)
    widths.append(8)
    live._draw_prompt()

    rendered = out.getvalue()
    assert "\x1b[1B\x1b[?25l\r\x1b[J" in rendered
    assert "\x1b[1A\r\x1b[2K" in rendered
    assert " > abc" in rendered
    assert live.prompt_width == 8


def test_live_prompt_redraw_flushes_one_frame() -> None:
    out = _FlushingTTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)

    live._draw_prompt()
    out.flush_count = 0
    live.buffer = "a"
    live.cursor = len(live.buffer)
    live._draw_prompt()

    assert out.flush_count == 1
    rendered = out.getvalue()
    assert "\x1b[2K" in rendered
    assert " > a" in rendered


def test_welcome_card_keeps_compact_width() -> None:
    out = io.StringIO()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._terminal_width = lambda: 120  # type: ignore[method-assign]

    app._write_welcome_card()

    first_line = out.getvalue().splitlines()[0]
    assert len(first_line) == 36


def test_live_prompt_shows_slash_command_hints_for_prefix() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/mo"
    live.cursor = len(live.buffer)

    live._draw_prompt()

    rendered = out.getvalue()
    assert "/model  choose what model to use" in rendered
    assert "/clear" not in rendered
    assert rendered.index(" > /mo") < rendered.index("/model")
    assert "tokens:" not in rendered


def test_live_input_hints_move_highlight_with_up_and_down(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    skill_path = project / ".willow" / "skills" / "hello_world" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text(
        "---\ndescription: Say hello.\n---\nbody",
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.chdir(project)
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/he"
    live.cursor = len(live.buffer)

    live._draw_prompt()

    rendered = out.getvalue()
    assert f"{tui.SELECTED_ROW_STYLE} /help" in rendered

    out.seek(0)
    out.truncate(0)
    live._move_picker_selection(1)
    live._draw_prompt()

    rendered = out.getvalue()
    assert f"{tui.SELECTED_ROW_STYLE} /hello_world" in rendered


def test_live_input_hint_enter_executes_selected_skill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    skill_path = project / ".willow" / "skills" / "hello_world" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("Say hello before doing the task.", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.chdir(project)
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/he write docs"
    live.cursor = len(live.buffer)
    started: list[bool] = []

    def fake_start_worker() -> None:
        started.append(True)

    live._start_worker = fake_start_worker  # type: ignore[method-assign]

    live._move_picker_selection(1)
    live._submit_buffer()

    assert started == [True]
    assert len(app.messages) == 1
    block = app.messages[0].content[0]
    assert isinstance(block, TextBlock)
    assert "Use the Willow skill 'hello_world'" in block.text
    assert "User task:\nwrite docs" in block.text


def test_live_tab_completes_selected_skill_without_submitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    skill_path = project / ".willow" / "skills" / "hello_world" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("Say hello before doing the task.", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.chdir(project)
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/he write docs"
    live.cursor = len(live.buffer)

    live._move_picker_selection(1)
    completed = live._complete_selected_hint()

    assert completed is True
    assert live.buffer == "/hello_world write docs"
    assert live.cursor == len(live.buffer)
    assert app.messages == []


def test_live_tab_completes_selected_command_and_adds_space() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/he"
    live.cursor = len(live.buffer)

    completed = live._complete_selected_hint()

    assert completed is True
    assert live.buffer == "/help "
    assert live.cursor == len(live.buffer)
    assert app.messages == []


def test_live_tab_completes_selected_at_file_without_submitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "notes.txt").write_text("secret", encoding="utf-8")
    monkeypatch.chdir(project)
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "check @not"
    live.cursor = len(live.buffer)

    completed = live._complete_selected_hint()

    assert completed is True
    assert live.buffer == "check notes.txt "
    assert live.cursor == len(live.buffer)
    assert app.messages == []


def test_live_input_hint_enter_selects_at_file_without_submitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    notes = project / "notes.txt"
    notes.write_text("secret", encoding="utf-8")
    monkeypatch.chdir(project)
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "check @not"
    live.cursor = len(live.buffer)
    live._submit_buffer()

    assert live.buffer == "check notes.txt "
    assert live.cursor == len(live.buffer)
    assert app.messages == []


def test_live_input_hint_enter_executes_selected_command() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/he"
    live.cursor = len(live.buffer)

    live._submit_buffer()

    rendered = out.getvalue()
    assert "Commands:" in rendered
    assert "/model [name|#]" in rendered
    assert app.messages == []


def test_live_prompt_shows_model_picker_for_model_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    choices = [
        ModelChoice(
            model="gpt-5.5",
            provider="openai_responses",
            vendor="openai",
            description="Frontier model.",
        ),
        ModelChoice(
            model="claude-sonnet-4-6",
            provider="anthropic",
            vendor="anthropic",
            description="Balanced model.",
        ),
    ]
    monkeypatch.setattr(tui, "available_model_choices", lambda: choices)
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/model"
    live.cursor = len(live.buffer)

    live._draw_prompt()

    rendered = out.getvalue()
    assert " > /model" in rendered
    assert "gpt-5.5" in rendered
    assert "claude-sonnet-4-6" in rendered
    assert tui.SELECTED_ROW_STYLE in rendered
    assert rendered.index(" > /model") < rendered.index("gpt-5.5")
    assert "/model  choose what model to use" not in rendered
    assert "tokens:" not in rendered


def test_live_prompt_shows_login_picker_for_login_command() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/login"
    live.cursor = len(live.buffer)

    live._draw_prompt()

    rendered = out.getvalue()
    assert " > /login" in rendered
    assert "openai-codex" in rendered
    assert "ChatGPT Plus/Pro Codex OAuth" in rendered
    assert tui.SELECTED_ROW_STYLE in rendered
    assert rendered.index(" > /login") < rendered.index("openai-codex")
    assert "/login  authenticate OpenAI Codex with OAuth" not in rendered


def test_live_prompt_shows_auto_compacting_status() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.compacting = True

    live._draw_prompt()

    rendered = out.getvalue()
    assert "Auto-compacting..." in rendered
    assert tui.COMPACTION_STYLE in rendered


def test_live_login_picker_enter_logs_in_selected_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_handle_login(provider: str) -> None:
        calls.append(provider)

    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._handle_login = fake_handle_login  # type: ignore[method-assign]
    live = tui._LiveTerminal(app)
    live.buffer = "/login"
    live.cursor = len(live.buffer)

    live._submit_buffer()

    assert calls == ["openai-codex"]
    assert live.buffer == ""
    assert app.messages == []


def test_live_tab_completes_selected_login_provider_without_submitting() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/login"
    live.cursor = len(live.buffer)

    completed = live._complete_selected_hint()

    assert completed is True
    assert live.buffer == "/login openai-codex"
    assert live.cursor == len(live.buffer)
    assert app.messages == []


def test_live_model_picker_moves_highlight_with_up_and_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    choices = [
        ModelChoice(
            model="gpt-5.5",
            provider="openai_responses",
            vendor="openai",
            description="Frontier model.",
        ),
        ModelChoice(
            model="claude-sonnet-4-6",
            provider="anthropic",
            vendor="anthropic",
            description="Balanced model.",
        ),
    ]
    monkeypatch.setattr(tui, "available_model_choices", lambda: choices)
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/model"
    live.cursor = len(live.buffer)

    live._move_model_picker_selection(1)
    live._draw_prompt()

    rendered = out.getvalue()
    assert f"{tui.SELECTED_ROW_STYLE} > claude-sonnet-4-6" in rendered

    out.seek(0)
    out.truncate(0)
    live._move_model_picker_selection(-1)
    live._draw_prompt()

    rendered = out.getvalue()
    assert f"{tui.SELECTED_ROW_STYLE} > gpt-5.5" in rendered


def test_live_model_picker_enter_applies_selected_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anthropic_provider = _ScriptedStreamProvider([])
    choices = [
        ModelChoice(
            model="gpt-5.5",
            provider="openai_responses",
            vendor="openai",
            description="Frontier model.",
        ),
        ModelChoice(
            model="claude-sonnet-4-6",
            provider="anthropic",
            vendor="anthropic",
            description="Balanced model.",
        ),
    ]

    monkeypatch.setattr(tui, "available_model_choices", lambda: choices)
    monkeypatch.setattr("willow.cli._build_provider", lambda _name: anthropic_provider)
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/model"
    live.cursor = len(live.buffer)

    live._move_model_picker_selection(1)
    live._submit_buffer()

    assert app.current_model == "claude-sonnet-4-6"
    assert app.current_provider_name == "anthropic"
    assert app.provider is anthropic_provider


def test_live_tab_completes_selected_model_without_applying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    choices = [
        ModelChoice(
            model="gpt-5.5",
            provider="openai_responses",
            vendor="openai",
            description="Frontier model.",
        ),
        ModelChoice(
            model="claude-sonnet-4-6",
            provider="anthropic",
            vendor="anthropic",
            description="Balanced model.",
        ),
    ]
    monkeypatch.setattr(tui, "available_model_choices", lambda: choices)
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/model"
    live.cursor = len(live.buffer)

    live._move_model_picker_selection(1)
    completed = live._complete_selected_hint()

    assert completed is True
    assert live.buffer == "/model claude-sonnet-4-6"
    assert live.cursor == len(live.buffer)
    assert app.current_model == "gpt-5.5"


def test_live_prompt_restores_statusline_when_command_prefix_is_removed() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/mo"
    live.cursor = len(live.buffer)

    live._draw_prompt()
    out.seek(0)
    out.truncate(0)
    live.buffer = ""
    live.cursor = 0
    live._draw_prompt()

    rendered = out.getvalue()
    assert "/model  choose what model to use" not in rendered
    assert "Context " in rendered


def test_live_stream_chunk_keeps_prompt_stable_until_completion() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live._draw_prompt()
    out.seek(0)
    out.truncate(0)

    live.events.put((live.active_turn_id, "stream", TextDelta(text="Hello")))
    live._drain_events()

    rendered = out.getvalue()
    assert rendered == ""
    assert live.stream_text == ["Hello"]
    assert "tokens:" not in rendered
    assert "streaming" not in rendered

    response = CompletionResponse(
        content=[TextBlock(text="Hello")],
        stop_reason="end_turn",
    )
    live.events.put((live.active_turn_id, "complete", response))
    live._drain_events()

    rendered = out.getvalue()
    assert "Hello" in rendered
    assert " > " in rendered


def test_live_provider_error_flushes_partial_output_and_keeps_prompt_usable() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.active_turn_id = 3
    live.streaming = True
    live.worker = threading.Thread(target=lambda: None)
    live.active_tool_status = "running bash - pytest"

    live.events.put((3, "stream", TextDelta(text="partial output")))
    live.events.put((3, "error", RuntimeError("Codex error: policy warning")))
    live._drain_events()

    rendered = out.getvalue()
    assert "partial output" in rendered
    visible = _strip_ansi(rendered)
    assert "error" in visible
    assert "Codex error: policy warning" in visible
    assert "RuntimeError(" not in rendered
    assert " > " in rendered
    assert live.streaming is False
    assert live.worker is None
    assert live.active_tool_status is None


def test_live_prompt_shows_queued_messages_with_interrupt_hint() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.pending_user_inputs = ["first", "second"]

    live._draw_prompt()

    rendered = out.getvalue()
    assert "Messages to be submitted after next tool call" in rendered
    assert "press esc to interrupt and send immediately" in rendered
    assert "↳ first" in rendered
    assert "↳ second" in rendered


def test_live_prompt_shows_active_subagent_waiting_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True

    class FakeSubagents:
        def snapshots(self) -> list[dict[str, object]]:
            return [
                {
                    "subagent_id": "subagent-123",
                    "display_name": "Hopper",
                    "role": "explorer",
                    "status": "running",
                    "task": "Inspect the prompt state",
                },
                {
                    "subagent_id": "subagent-456",
                    "display_name": "Grace",
                    "role": "explorer",
                    "status": "pending",
                    "task": "Inspect tool flows",
                },
                {
                    "subagent_id": "subagent-789",
                    "display_name": "Ada",
                    "role": "worker",
                    "status": "closing",
                    "task": "Patch tests",
                },
            ]

    monkeypatch.setattr(app.runtime, "_subagents", FakeSubagents())

    live._draw_prompt()

    rendered = out.getvalue()
    assert f"{tui.SUBAGENT_WAIT_TITLE_STYLE}\x1b[2K Waiting for 3 subagents" in rendered
    assert "Waiting for subagent(s)" not in rendered
    assert "Waiting for 3 subagents" in rendered
    assert "Hopper [explorer] Inspect the prompt state" in rendered
    assert "Grace [explorer] Inspect tool flows" in rendered
    assert "Ada [worker] Patch tests" in rendered


def test_live_prompt_suppresses_generic_wait_agent_status_for_active_subagents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.active_tool_status = "Waiting for subagent"

    class FakeSubagents:
        def snapshots(self) -> list[dict[str, object]]:
            return [
                {
                    "subagent_id": "subagent-123",
                    "display_name": "Hopper",
                    "role": "explorer",
                    "status": "running",
                    "task": "Inspect the prompt state",
                }
            ]

    monkeypatch.setattr(app.runtime, "_subagents", FakeSubagents())

    live._draw_prompt()

    rendered = _strip_ansi(out.getvalue())
    assert "Waiting for 1 subagent" in rendered
    assert "Waiting for subagent\n" not in rendered


def test_live_prompt_shows_queued_image_messages_as_anchors(tmp_path: Path) -> None:
    image = tmp_path / "dragged image.png"
    image.write_bytes(b"fake-png")
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    escaped_path = str(image).replace(" ", "\\ ")
    live.pending_user_inputs = [f"check {escaped_path}"]

    live._draw_prompt()

    rendered = out.getvalue()
    assert "↳ check [image#1]" in rendered
    assert str(image) not in rendered


def test_live_prompt_hides_pending_monitor_events() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.pending_monitor_inputs = ["Monitor event: service ready"]

    live._draw_prompt()

    rendered = out.getvalue()
    assert "Monitor event:" not in rendered
    assert "service ready" not in rendered
    assert "Messages to be submitted after next tool call" not in rendered


def test_live_queue_monitor_event_uses_hidden_queue() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True

    live._queue_monitor_event(
        {
            "event_type": "command_output",
            "severity": "info",
            "summary": "service ready",
        }
    )

    assert live.pending_user_inputs == []
    assert live.pending_monitor_inputs == [
        "Monitor event: service ready"
    ]
    assert "Monitor event:" not in out.getvalue()


def test_live_subagent_event_renders_notification_without_queueing() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    started: list[bool] = []
    live._start_queued_turn = lambda: started.append(True)  # type: ignore[method-assign]

    live._queue_monitor_event(
        {
            "event_type": "subagent",
            "subagent_id": "subagent-123",
            "name": "Hopper",
            "role": "explorer",
            "status": "completed",
            "task": "Inspect the prompt state",
        }
    )

    rendered = _strip_ansi(out.getvalue())
    assert "Hopper [explorer] completed" in rendered
    assert "Inspect the prompt state" in rendered
    assert live.pending_monitor_inputs == []
    assert started == []


def test_live_finish_response_sends_monitor_events_without_rendering_them() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.pending_monitor_inputs = ["Monitor event: service ready"]
    started: list[bool] = []

    def fake_start_worker() -> None:
        started.append(True)

    live._start_worker = fake_start_worker  # type: ignore[method-assign]

    live._finish_response(
        CompletionResponse(content=[TextBlock(text="ack")], stop_reason="end_turn")
    )

    rendered = out.getvalue()
    assert started
    assert "Monitor event:" not in rendered
    assert "service ready" not in rendered
    assert [message.role for message in app.messages] == ["assistant", "user"]
    assert app.messages[1].content == [TextBlock(text="Monitor event: service ready")]
    assert live.pending_monitor_inputs == []


def test_live_submit_while_streaming_shows_queue_only_in_prompt() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.buffer = "queued followup"
    live.cursor = len(live.buffer)

    live._submit_buffer()

    rendered = out.getvalue()
    assert live.pending_user_inputs == ["queued followup"]
    assert "Queued input while a turn is streaming" not in rendered
    assert "press Esc to interrupt and send now" not in rendered
    assert "Messages to be submitted after next tool call" in rendered
    assert "\n > queued followup" not in _strip_ansi(rendered)


def test_live_queued_image_inputs_keep_placeholder_order(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"fake-png-first")
    second.write_bytes(b"fake-png-second")
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.pending_user_inputs = [f"first {first}", f"second {second}"]

    live._append_pending_user_message(render=True)

    message = app.messages[-1]
    assert message.content[0] == TextBlock(
        text="[queued user message 1 of 2]\nfirst [image#1]"
    )
    assert message.content[1] == TextBlock(
        text="[queued user message 2 of 2]\nsecond [image#2]"
    )
    assert [
        block.filename for block in message.content if isinstance(block, ImageBlock)
    ] == ["first.png", "second.png"]
    rendered = out.getvalue()
    assert "first [image#1]" in rendered
    assert "second [image#2]" in rendered
    assert str(first) not in rendered
    assert str(second) not in rendered


def test_live_finish_response_anchors_pending_image_input(tmp_path: Path) -> None:
    image = tmp_path / "dragged image.png"
    image.write_bytes(b"fake-png")
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    escaped_path = str(image).replace(" ", "\\ ")
    live.pending_user_inputs = [f"check {escaped_path}"]

    live._finish_response(
        CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")
    )

    rendered = out.getvalue()
    assert "check [image#1]" in rendered
    assert str(image) not in rendered
    assert [message.role for message in app.messages] == ["assistant", "user"]
    followup = app.messages[1]
    assert followup.content[0] == TextBlock(text="check [image#1]")
    assert any(
        isinstance(block, ImageBlock) and block.filename == image.name
        for block in followup.content
    )


def test_live_esc_interrupt_keeps_user_and_queued_messages_without_partial_assistant() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.pending_user_inputs = ["queued followup"]
    live.stream_text = ["partial assistant text"]
    app.messages.append(Message(role="user", content=[TextBlock(text="refactor the TUI")]))
    started: list[int] = []

    def fake_reset_provider() -> None:
        return None

    def fake_start_worker() -> None:
        live.streaming = True
        live.active_turn_id += 1
        started.append(live.active_turn_id)

    live._reset_provider_for_interrupt = fake_reset_provider  # type: ignore[method-assign]
    live._start_worker = fake_start_worker  # type: ignore[method-assign]

    live._interrupt_and_send_queued()

    assert started
    assert [message.role for message in app.messages] == ["user"]
    _assert_interrupted_retry_content(
        app.messages[0].content,
        interrupted="refactor the TUI",
        interrupting="queued followup",
    )
    assert live.stream_text == []
    rendered = out.getvalue()
    assert "Interrupted current turn; sending queued input now." in rendered
    assert "partial assistant text" not in rendered


def test_live_esc_interrupt_sends_queued_message_to_provider_request() -> None:
    out = _TTYBuffer()
    response = CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider([[StreamComplete(response=response)]])
    app = tui.WillowApp(_make_args(), provider, out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.pending_user_inputs = ["surprise me"]
    app.messages.append(Message(role="user", content=[TextBlock(text="refactor the TUI")]))

    live._reset_provider_for_interrupt = lambda: None  # type: ignore[method-assign]

    live._interrupt_and_send_queued()
    assert live.worker is not None
    live.worker.join(timeout=1)

    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert [message.role for message in request.messages] == ["user"]
    _assert_interrupted_retry_content(
        request.messages[0].content,
        interrupted="refactor the TUI",
        interrupting="surprise me",
    )


def test_live_esc_interrupt_replaces_provider_before_sending_queued_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = _TTYBuffer()
    old_provider = _ScriptedStreamProvider([])
    response = CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")
    new_provider = _ScriptedStreamProvider([[StreamComplete(response=response)]])
    app = tui.WillowApp(_make_args(), old_provider, out=out)
    app._force_plain = False
    app.messages.append(Message(role="user", content=[TextBlock(text="previous")]))
    app.messages.append(Message(role="assistant", content=[TextBlock(text="previous answer")]))
    app.messages.append(Message(role="user", content=[TextBlock(text="refactor the TUI")]))
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.pending_user_inputs = ["steer now"]

    monkeypatch.setattr("willow.cli._build_provider", lambda _name: new_provider)

    live._interrupt_and_send_queued()
    assert live.worker is not None
    live.worker.join(timeout=1)

    assert app.provider is new_provider
    assert old_provider.requests == []
    assert len(new_provider.requests) == 1
    request = new_provider.requests[0]
    assert [message.role for message in request.messages] == ["user", "assistant", "user"]
    assert request.messages[0].content == [TextBlock(text="previous")]
    assert request.messages[1].content == [TextBlock(text="previous answer")]
    _assert_interrupted_retry_content(
        request.messages[2].content,
        interrupted="refactor the TUI",
        interrupting="steer now",
    )


def test_live_esc_interrupt_sends_current_buffer_to_provider_request() -> None:
    out = _TTYBuffer()
    response = CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider([[StreamComplete(response=response)]])
    app = tui.WillowApp(_make_args(), provider, out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.buffer = "surprise me"
    app.messages.append(Message(role="user", content=[TextBlock(text="refactor the TUI")]))

    live._reset_provider_for_interrupt = lambda: None  # type: ignore[method-assign]

    live._interrupt_with_buffer_if_possible()
    assert live.worker is not None
    live.worker.join(timeout=1)

    assert live.buffer == ""
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert [message.role for message in request.messages] == ["user"]
    _assert_interrupted_retry_content(
        request.messages[0].content,
        interrupted="refactor the TUI",
        interrupting="surprise me",
    )


def test_live_esc_without_buffer_defers_interrupted_message_until_next_submit() -> None:
    out = _TTYBuffer()
    response = CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider([[StreamComplete(response=response)]])
    app = tui.WillowApp(_make_args(), provider, out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.stream_text = ["partial assistant text"]
    app.messages.append(Message(role="user", content=[TextBlock(text="first message")]))
    live._reset_provider_for_interrupt = lambda: None  # type: ignore[method-assign]

    live._interrupt_with_buffer_if_possible()

    assert live.streaming is False
    assert app.messages == []
    assert live.interrupted_user_inputs == ["first message"]
    assert live.stream_text == []
    assert "partial assistant text" not in out.getvalue()

    live.buffer = "second message"
    live.cursor = len(live.buffer)
    live._submit_buffer()
    assert live.worker is not None
    live.worker.join(timeout=1)

    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert [message.role for message in request.messages] == ["user"]
    _assert_interrupted_retry_content(
        request.messages[0].content,
        interrupted="first message",
        interrupting="second message",
    )


def test_live_left_arrow_moves_cursor_for_mid_buffer_edit() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    read_fd, write_fd = os.pipe()
    try:
        live.fd = read_fd
        os.write(write_fd, b"abc\x1b[D\x1b[DX")
        live._read_available_input()
    finally:
        os.close(read_fd)
        os.close(write_fd)

    assert live.buffer == "aXbc"
    assert live.cursor == 2


def test_live_up_and_down_arrows_navigate_input_history() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.input_history = ["first prompt", "second prompt"]
    read_fd, write_fd = os.pipe()
    try:
        live.fd = read_fd
        os.write(write_fd, b"draft\x1b[A")
        live._read_available_input()
        assert live.buffer == "second prompt"
        assert live.cursor == len("second prompt")

        os.write(write_fd, b"\x1b[A")
        live._read_available_input()
        assert live.buffer == "first prompt"

        os.write(write_fd, b"\x1b[B")
        live._read_available_input()
        assert live.buffer == "second prompt"

        os.write(write_fd, b"\x1b[B")
        live._read_available_input()
        assert live.buffer == "draft"
        assert live.cursor == len("draft")
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_live_split_arrow_escape_navigates_input_history() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.input_history = ["first prompt", "second prompt"]
    read_fd, write_fd = os.pipe()
    try:
        live.fd = read_fd
        os.write(write_fd, b"\x1b")
        live._read_available_input()
        assert live.pending_escape_sequence == "\x1b"
        assert live.buffer == ""

        os.write(write_fd, b"[A")
        live._read_available_input()
        assert live.pending_escape_sequence is None
        assert live.buffer == "second prompt"
        assert live.cursor == len("second prompt")
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_live_split_csi_arrow_escape_navigates_input_history() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.input_history = ["first prompt", "second prompt"]
    read_fd, write_fd = os.pipe()
    try:
        live.fd = read_fd
        os.write(write_fd, b"\x1b[")
        live._read_available_input()
        assert live.pending_escape_sequence == "\x1b["
        assert live.buffer == ""

        os.write(write_fd, b"A")
        live._read_available_input()
        assert live.pending_escape_sequence is None
        assert live.buffer == "second prompt"
        assert live.cursor == len("second prompt")
    finally:
        os.close(read_fd)
        os.close(write_fd)


@pytest.mark.parametrize(
    "sequence",
    ["\x1b[13;2u", "\x1b[13;2~", "\x1b[27;2;13~"],
)
def test_live_shift_enter_inserts_newline_without_submitting(sequence: str) -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    read_fd, write_fd = os.pipe()
    try:
        live.fd = read_fd
        os.write(write_fd, f"first{sequence}second".encode())
        live._read_available_input()

        assert live.buffer == "first\nsecond"
        assert live.cursor == len("first\nsecond")
        assert app.messages == []

        out.seek(0)
        out.truncate(0)
        live._draw_prompt()
        rendered = out.getvalue()
        assert " > first" in rendered
        assert "   second" in rendered
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_live_split_shift_enter_escape_inserts_newline() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    read_fd, write_fd = os.pipe()
    try:
        live.fd = read_fd
        os.write(write_fd, b"first\x1b[13")
        live._read_available_input()
        assert live.pending_escape_sequence == "\x1b[13"
        assert live.buffer == "first"

        os.write(write_fd, b";2usecond")
        live._read_available_input()
        assert live.pending_escape_sequence is None
        assert live.buffer == "first\nsecond"
        assert live.cursor == len("first\nsecond")
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_live_raw_terminal_enables_modified_key_reporting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    live = tui._LiveTerminal(app)
    live.fd = 123
    old_attrs = ["old"]
    monkeypatch.setattr(tui.termios, "tcgetattr", lambda fd: old_attrs)
    monkeypatch.setattr(tui.termios, "tcsetattr", lambda fd, when, attrs: None)
    monkeypatch.setattr(tui.tty, "setcbreak", lambda fd: None)

    with live._raw_terminal():
        pass

    rendered = out.getvalue()
    assert "\x1b[>1u" in rendered
    assert "\x1b[>4;2m" in rendered
    assert "\x1b[<u" in rendered
    assert "\x1b[>4m" in rendered


def test_live_submit_records_prompt_for_input_history() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "remember this"
    live.cursor = len(live.buffer)
    live._start_worker = lambda: None  # type: ignore[method-assign]

    live._submit_buffer()
    live._move_input_history(-1)

    assert live.buffer == "remember this"
    assert live.cursor == len("remember this")


def test_live_bracketed_paste_split_across_reads_preserves_newlines() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    read_fd, write_fd = os.pipe()
    try:
        live.fd = read_fd
        os.write(write_fd, b"\x1b[200~line one\nline")
        live._read_available_input()
        os.write(write_fd, b" two\n\x1b[201~")
        live._read_available_input()
    finally:
        os.close(read_fd)
        os.close(write_fd)

    assert live.buffer == "line one\nline two\n"
    assert live.cursor == len(live.buffer)
    assert live.pasted_ranges == [(0, len(live.buffer))]
    assert app.messages == []


def test_live_option_arrow_moves_cursor_by_word_for_mid_buffer_edit() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    read_fd, write_fd = os.pipe()
    try:
        live.fd = read_fd
        os.write(write_fd, b"alpha beta gamma\x1bb\x1bbX\x1b[1;3C!")
        live._read_available_input()
    finally:
        os.close(read_fd)
        os.close(write_fd)

    assert live.buffer == "alpha Xbeta! gamma"
    assert live.cursor == len("alpha Xbeta!")


def test_live_ignores_stale_stream_and_complete_events_after_interrupt() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.active_turn_id = 2
    app.messages.append(Message(role="user", content=[TextBlock(text="first")]))
    stale_response = CompletionResponse(
        content=[TextBlock(text="stale final")],
        stop_reason="end_turn",
    )

    live.events.put((1, "stream", TextDelta(text="stale partial")))
    live.events.put((1, "complete", stale_response))
    live._drain_events()

    assert [message.role for message in app.messages] == ["user"]
    assert live.stream_text == []
    assert "stale partial" not in out.getvalue()
    assert "stale final" not in out.getvalue()


def test_terminal_streams_text_thinking_and_tool_markers_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _RecordingTool()
    monkeypatch.setitem(TOOLS_BY_NAME, tool.name, tool)
    tool_use_response = CompletionResponse(
        content=[ToolUseBlock(id="call_1", name="echo", input={"message": "hi"})],
        stop_reason="tool_use",
    )
    end_response = CompletionResponse(
        content=[TextBlock(text="done")],
        stop_reason="end_turn",
    )
    provider = _ScriptedStreamProvider(
        [
            [
                ThinkingDelta(thinking="reasoning"),
                TextDelta(text="answer"),
                ToolUseDelta(id="call_1", name="echo", partial_json=None),
                StreamComplete(response=tool_use_response),
            ],
            [TextDelta(text="done"), StreamComplete(response=end_response)],
        ]
    )

    out, _app = _drive(provider, ["use tool", "/exit"])

    assert tool.calls == [{"message": "hi"}]
    assert out.index("thinking") < out.index("reasoning")
    assert out.index("reasoning") < out.index("answer")
    assert out.index("answer") < out.index("| echo ok")
    assert out.index("| echo ok") < out.index("echoed: hi")
    assert out.index("echoed: hi") < out.index("done")
    assert "---" in out


def test_live_terminal_configures_subagents_before_tool_dispatch() -> None:
    out = _TTYBuffer()
    provider = _ParentChildProvider()
    app = tui.WillowApp(_make_args(), provider, out=out)
    app._force_plain = False
    app.messages.append(Message(role="user", content=[TextBlock(text="delegate")]))
    live = tui._LiveTerminal(app)
    read_fd, write_fd = os.pipe()
    try:
        live.fd = read_fd
        live._start_worker()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and (live.worker is not None or live.streaming):
            live._drain_events()
            time.sleep(0.01)
        live._drain_events()
    finally:
        os.close(read_fd)
        os.close(write_fd)

    rendered = _strip_ansi(out.getvalue())
    assert provider.child_requests
    assert "Spawned Hopper [explorer] (gpt-5.5)" in rendered
    assert "Workspace:" in rendered
    assert "inspect child task" in rendered
    assert "subagent runtime is not configured" not in rendered
    assert "spawn_agent error" not in rendered


def test_spawn_agent_success_renders_friendly_summary() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    block = ToolUseBlock(
        id="call_1",
        name="spawn_agent",
        input={"task": "Investigate reproduction surfaces for issue 3 only"},
    )
    result = ToolResultBlock(
        tool_use_id="call_1",
        content=(
            "subagent_id: subagent-123\n"
            "name: Hopper\n"
            "role: explorer\n"
            "status: running\n"
            "model: gpt-5.5\n"
            "effort: xhigh\n"
            "workspace: /Users/LiyuanLiu/repos/enterprise-rag\n"
            "task: Investigate reproduction surfaces for issue 3 only\n"
            "turns: 0"
        ),
    )

    app._write_tool_result(block, result)

    rendered = _strip_ansi(out.getvalue())
    assert "Spawned Hopper [explorer] (gpt-5.5 xhigh)" in rendered
    assert (
        "Workspace: /Users/LiyuanLiu/repos/enterprise-rag. "
        "Investigate reproduction surfaces for issue 3 only"
    ) in rendered
    assert "subagent_id:" not in rendered


@pytest.mark.parametrize("tool_name", ["wait_agent", "send_input", "close_agent"])
def test_subagent_housekeeping_successes_are_hidden_in_transcript(tool_name: str) -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    block = ToolUseBlock(
        id="call_1",
        name=tool_name,
        input={"subagent_id": "subagent-123"},
    )
    result = ToolResultBlock(
        tool_use_id="call_1",
        content="subagent_id: subagent-123\nstatus: completed",
    )

    app._write_tool_result(block, result)

    assert out.getvalue() == ""


def test_subagent_housekeeping_errors_are_visible_in_transcript() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    block = ToolUseBlock(
        id="call_1",
        name="wait_agent",
        input={"subagent_id": "subagent-123"},
    )
    result = ToolResultBlock(
        tool_use_id="call_1",
        content="RuntimeError: failed",
        is_error=True,
    )

    app._write_tool_result(block, result)

    rendered = _strip_ansi(out.getvalue())
    assert "wait_agent error" in rendered
    assert "RuntimeError: failed" in rendered


def test_terminal_read_only_blocks_non_read_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _RecordingTool()
    monkeypatch.setitem(TOOLS_BY_NAME, tool.name, tool)
    tool_use_response = CompletionResponse(
        content=[ToolUseBlock(id="call_1", name="echo", input={"message": "hi"})],
        stop_reason="tool_use",
    )
    end_response = CompletionResponse(
        content=[TextBlock(text="blocked")],
        stop_reason="end_turn",
    )
    provider = _ScriptedStreamProvider(
        [
            [
                ToolUseDelta(id="call_1", name="echo", partial_json=None),
                StreamComplete(response=tool_use_response),
            ],
            [TextDelta(text="blocked"), StreamComplete(response=end_response)],
        ]
    )

    out, _app = _drive(
        provider,
        ["use tool", "/exit"],
        args=_make_args(permission_mode=tui.PermissionMode.READ_ONLY),
    )

    assert tool.calls == []
    assert "| echo error" in out
    assert "read-only mode" in out


def test_terminal_ask_permission_denies_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _RecordingTool()
    monkeypatch.setitem(TOOLS_BY_NAME, tool.name, tool)
    tool_use_response = CompletionResponse(
        content=[ToolUseBlock(id="call_1", name="echo", input={"message": "hi"})],
        stop_reason="tool_use",
    )
    end_response = CompletionResponse(content=[TextBlock(text="denied")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider(
        [
            [
                ToolUseDelta(id="call_1", name="echo", partial_json=None),
                StreamComplete(response=tool_use_response),
            ],
            [TextDelta(text="denied"), StreamComplete(response=end_response)],
        ]
    )

    out, _app = _drive(
        provider,
        ["use tool", "n", "/exit"],
        args=_make_args(permission_mode=tui.PermissionMode.ASK),
    )

    assert tool.calls == []
    assert "[permission] Allow echo" in out
    assert "Permission denied by user" in out


def test_terminal_ask_permission_allows_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _RecordingTool()
    monkeypatch.setitem(TOOLS_BY_NAME, tool.name, tool)
    tool_use_response = CompletionResponse(
        content=[ToolUseBlock(id="call_1", name="echo", input={"message": "hi"})],
        stop_reason="tool_use",
    )
    end_response = CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider(
        [
            [
                ToolUseDelta(id="call_1", name="echo", partial_json=None),
                StreamComplete(response=tool_use_response),
            ],
            [TextDelta(text="done"), StreamComplete(response=end_response)],
        ]
    )

    out, _app = _drive(
        provider,
        ["use tool", "y", "/exit"],
        args=_make_args(permission_mode=tui.PermissionMode.ASK),
    )

    assert tool.calls == [{"message": "hi"}]
    assert "[permission] Allow echo" in out
    assert "| echo ok" in out


def test_terminal_bash_tool_renders_command_and_concise_output() -> None:
    tool_use_response = CompletionResponse(
        content=[
            ToolUseBlock(
                id="call_1",
                name="bash",
                input={"command": "seq 1 5"},
            )
        ],
        stop_reason="tool_use",
    )
    end_response = CompletionResponse(
        content=[TextBlock(text="done")],
        stop_reason="end_turn",
    )
    provider = _ScriptedStreamProvider(
        [
            [
                ToolUseDelta(id="call_1", name="bash", partial_json=None),
                StreamComplete(response=tool_use_response),
            ],
            [TextDelta(text="done"), StreamComplete(response=end_response)],
        ]
    )

    out, _app = _drive(provider, ["run command", "/exit"])

    assert "| bash ok - ran seq 1 5" in out
    assert "  1" in out
    assert "  4" in out
    assert "... +1 lines" in out
    assert "\n  5\n" not in out


def test_live_tool_execution_keeps_working_prompt_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = _TTYBuffer()
    tool = _PromptObservingTool(out)
    monkeypatch.setitem(TOOLS_BY_NAME, tool.name, tool)
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    response = CompletionResponse(
        content=[ToolUseBlock(id="call_1", name=tool.name, input={})],
        stop_reason="tool_use",
    )

    live._finish_response(response)

    observed = _strip_ansi(tool.observed)
    assert "running observe_prompt" in observed
    assert " > " in observed
    assert live.active_tool_status is None
    assert "observe_prompt ok" in out.getvalue()


def test_terminal_skill_command_expands_user_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    skill_path = tmp_path / ".willow" / "skills" / "writer" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("Write tersely.", encoding="utf-8")
    response = CompletionResponse(
        content=[TextBlock(text="done")],
        stop_reason="end_turn",
    )
    provider = _ScriptedStreamProvider([[StreamComplete(response=response)]])

    out, _app = _drive(provider, ["/writer draft notes", "/exit"])

    sent_text = provider.requests[0].messages[0].content[0].text
    assert "Write tersely." in sent_text
    assert "User task:\ndraft notes" in sent_text
    assert "/writer draft notes" in out


def test_terminal_project_hello_world_skill_works(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_path = tmp_path / ".willow" / "skills" / "hello_world" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text(
        "---\n"
        "name: hello_world\n"
        "description: Respond with a concise hello world confirmation.\n"
        "---\n\n"
        "# Hello World\n\n"
        "When invoked, respond with a concise confirmation.",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    response = CompletionResponse(
        content=[TextBlock(text="done")],
        stop_reason="end_turn",
    )
    provider = _ScriptedStreamProvider([[StreamComplete(response=response)]])

    out, _app = _drive(provider, ["/hello_world confirm skill wiring", "/exit"])

    sent_text = provider.requests[0].messages[0].content[0].text
    assert "Hello World" in sent_text
    assert "User task:\nconfirm skill wiring" in sent_text
    assert "/hello_world confirm skill wiring" in out


def test_terminal_tool_error_renders_tool_name_state_and_clean_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingTool(Tool):
        name = "explode"
        description = "Raise an error."
        input_schema = {"type": "object", "properties": {}}

        def run(self, **_kwargs: Any) -> str:
            raise ValueError("bad input")

    monkeypatch.setitem(TOOLS_BY_NAME, "explode", FailingTool())
    tool_use_response = CompletionResponse(
        content=[ToolUseBlock(id="call_1", name="explode", input={})],
        stop_reason="tool_use",
    )
    end_response = CompletionResponse(
        content=[TextBlock(text="done")],
        stop_reason="end_turn",
    )
    provider = _ScriptedStreamProvider(
        [
            [StreamComplete(response=tool_use_response)],
            [TextDelta(text="done"), StreamComplete(response=end_response)],
        ]
    )

    out, _app = _drive(provider, ["run tool", "/exit"])

    assert "| explode error" in out
    assert "ValueError: bad input" in out
    assert "ValueError('bad input')" not in out


def test_terminal_edit_error_title_includes_path() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    block = ToolUseBlock(
        id="call_1",
        name="edit",
        input={"path": "src/willow/loop.py"},
    )
    result = ToolResultBlock(
        tool_use_id="call_1",
        content="ValueError: old_text not found in src/willow/loop.py",
        is_error=True,
    )

    app._write_tool_result(block, result)

    rendered = out.getvalue()
    assert "edit error - src/willow/loop.py" in rendered
    assert "old_text not found" in rendered


def test_tui_persists_tool_use_before_tool_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(session.SESSION_DIR_ENV, str(tmp_path / "sessions"))
    app_box: dict[str, tui.WillowApp] = {}
    saw_persisted_tool_use = False

    class InspectingTool(Tool):
        name = "inspect_persisted_tool_use"
        description = "Inspect the saved session while the tool is running."
        input_schema = {"type": "object", "properties": {}}

        def run(self, **_kwargs: Any) -> str:
            nonlocal saw_persisted_tool_use
            app = app_box["app"]
            assert app._session_path is not None
            saved = session.load_session(app._session_path)
            last = saved.messages[-1]
            saw_persisted_tool_use = (
                last.role == "assistant"
                and len(last.content) == 1
                and isinstance(last.content[0], ToolUseBlock)
                and last.content[0].name == self.name
            )
            return "ok"

    monkeypatch.setitem(TOOLS_BY_NAME, "inspect_persisted_tool_use", InspectingTool())
    tool_use_response = CompletionResponse(
        content=[ToolUseBlock(id="call_1", name="inspect_persisted_tool_use", input={})],
        stop_reason="tool_use",
    )
    end_response = CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider(
        [
            [StreamComplete(response=tool_use_response)],
            [StreamComplete(response=end_response)],
        ]
    )
    inputs = iter(["use tool", "/exit"])

    def input_func(_prompt: str = "") -> str:
        try:
            return next(inputs)
        except StopIteration as exc:
            raise EOFError from exc

    app = tui.WillowApp(
        _make_args(persist_session=True),
        provider,
        input_func=input_func,
        out=io.StringIO(),
    )
    app_box["app"] = app

    assert app.run() == 0

    assert saw_persisted_tool_use is True
    assert app._session_path is not None
    saved = session.load_session(app._session_path)
    assert [message.role for message in saved.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert isinstance(saved.messages[1].content[0], ToolUseBlock)
    assert isinstance(saved.messages[2].content[0], ToolResultBlock)


def test_edit_tool_result_renders_diff_review_style() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    block = ToolUseBlock(
        id="call_1",
        name="write",
        input={"path": "codex_tui_test_haha1/wobbly_pickle_dispatch.txt"},
    )
    result = ToolResultBlock(
        tool_use_id="call_1",
        content=(
            "Wrote 12 bytes to codex_tui_test_haha1/wobbly_pickle_dispatch.txt\n"
            "--- codex_tui_test_haha1/wobbly_pickle_dispatch.txt (before)\n"
            "+++ codex_tui_test_haha1/wobbly_pickle_dispatch.txt (after)\n"
            "@@ -0,0 +1,2 @@\n"
            "+hello\n"
            "+world"
        ),
    )

    app._write_tool_result(block, result)

    rendered = out.getvalue()
    assert (
        "write ok - added codex_tui_test_haha1/wobbly_pickle_dispatch.txt (+2 -0)"
        in rendered
    )
    assert "    1 +hello" in rendered
    assert "    2 +world" in rendered
    assert tui.DIFF_ADD_STYLE in rendered


def test_write_added_file_renders_full_diff_preview() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    path = "/app/kv-store.proto"
    added_lines = [f"+line {index}" for index in range(1, 24)]
    block = ToolUseBlock(id="call_1", name="write", input={"path": path})
    result = ToolResultBlock(
        tool_use_id="call_1",
        content="\n".join(
            [
                f"Wrote 200 bytes to {path}",
                f"--- {path} (before)",
                f"+++ {path} (after)",
                "@@ -0,0 +1,23 @@",
                *added_lines,
            ]
        ),
    )

    app._write_tool_result(block, result)

    rendered = out.getvalue()
    assert f"write ok - added {path} (+23 -0)" in rendered
    assert "   23 +line 23" in rendered
    assert "changed lines" not in rendered


def test_terminal_deduplicates_separators_between_consecutive_tool_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _RecordingTool()
    monkeypatch.setitem(TOOLS_BY_NAME, tool.name, tool)

    def tool_turn(idx: int) -> list[Any]:
        response = CompletionResponse(
            content=[ToolUseBlock(id=f"call_{idx}", name="echo", input={"message": str(idx)})],
            stop_reason="tool_use",
        )
        return [
            ToolUseDelta(id=f"call_{idx}", name="echo", partial_json=None),
            StreamComplete(response=response),
        ]

    end_response = CompletionResponse(
        content=[TextBlock(text="done")],
        stop_reason="end_turn",
    )
    provider = _ScriptedStreamProvider(
        [
            tool_turn(1),
            tool_turn(2),
            [TextDelta(text="done"), StreamComplete(response=end_response)],
        ]
    )

    out, _app = _drive(provider, ["use tools", "/exit"])

    assert out.count("---") == 3
    assert "---\n---" not in out
    assert out.index("| echo ok") < out.index("echoed: 1")
    assert out.rindex("| echo ok") < out.index("echoed: 2")


def test_tui_persists_session_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(session.SESSION_DIR_ENV, str(tmp_path / "sessions"))
    response = CompletionResponse(
        content=[TextBlock(text="assistant done")],
        stop_reason="end_turn",
    )
    provider = _ScriptedStreamProvider([[StreamComplete(response=response)]])

    out, app = _drive(
        provider,
        ["hello", "/exit"],
        args=_make_args(persist_session=True),
    )

    assert app._session_path is not None
    assert str(app._session_path) not in out
    saved = session.load_session(app._session_path)
    assert saved.settings.provider == "openai_responses"
    assert saved.settings.model == "gpt-5.5"
    assert [message.role for message in saved.messages] == ["user", "assistant"]
    assert saved.messages[0].content == [TextBlock(text="hello")]
    assert saved.messages[1].content == [TextBlock(text="assistant done")]


def test_terminal_session_command_shows_persistence_and_saved_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(session.SESSION_DIR_ENV, str(tmp_path / "sessions"))
    provider = _ScriptedStreamProvider([])

    out, app = _drive(
        provider,
        ["/session", "/exit"],
        args=_make_args(persist_session=True),
    )

    assert app._session_path is not None
    assert "[status] persistence: enabled" in out
    assert f"session: {app._session_path}" in out
    assert "statusline: on" in out


def test_terminal_status_command_shows_disabled_persistence() -> None:
    provider = _ScriptedStreamProvider([])

    out, _app = _drive(provider, ["/status", "/exit"])

    assert "[status] persistence: disabled" in out
    assert "session: (not saved)" in out


def test_terminal_login_openai_codex(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_login(**kwargs: object) -> auth.AuthCredential:
        calls.append(kwargs)
        on_auth = kwargs["on_auth"]
        assert callable(on_auth)
        on_auth("https://auth.example/login")
        return auth.AuthCredential(
            kind="oauth",
            bearer_token="access-token",
            source="/tmp/auth.json openai.oauth",
            expires_at=2_000_000_000,
        )

    monkeypatch.setattr(tui, "login_openai_codex", fake_login)

    out, _app = _drive(_ScriptedStreamProvider([]), ["/login", "/exit"])

    assert len(calls) == 1
    assert calls[0]["originator"] == "willow"
    assert "Open this URL to authenticate OpenAI Codex" in out
    assert "https://auth.example/login" in out
    assert "OpenAI Codex OAuth saved to /tmp/auth.json openai.oauth" in out


def test_terminal_compaction_uses_projection_but_persists_full_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(session.SESSION_DIR_ENV, str(tmp_path / "sessions"))
    monkeypatch.setattr(tui, "_context_window_for_model", lambda _model: 100)
    prior_messages = [_text_message(index, "history " + ("x" * 80)) for index in range(24)]
    summary_response = CompletionResponse(
        content=[TextBlock(text="middle summary")],
        stop_reason="end_turn",
    )
    final_response = CompletionResponse(
        content=[TextBlock(text="final answer")],
        stop_reason="end_turn",
    )
    provider = _ResettableScriptedStreamProvider(
        [
            [StreamComplete(response=summary_response)],
            [TextDelta(text="final answer"), StreamComplete(response=final_response)],
        ]
    )
    args = _make_args(persist_session=True)
    inputs = iter(["new user request", "/exit"])

    def input_func(_prompt: str = "") -> str:
        try:
            return next(inputs)
        except StopIteration as exc:
            raise EOFError from exc

    out = io.StringIO()
    app = tui.WillowApp(
        args,
        provider,
        state={"messages": list(prior_messages)},
        input_func=input_func,
        out=out,
    )

    assert app.run() == 0

    assert "[status] Auto-compacting..." in out.getvalue()
    assert len(provider.requests) == 2
    compacted_request = provider.requests[1]
    assert "middle summary" in _message_text(compacted_request.messages[0])
    assert compacted_request.messages[1:] == app.messages[15:25]
    assert len(compacted_request.messages) == 11
    assert provider.reset_count >= 3
    assert len(app.messages) == 26
    assert all("middle summary" not in _message_text(message) for message in app.messages)
    assert app._session_path is not None
    saved = session.load_session(app._session_path)
    assert saved.messages == app.messages
    assert len(saved.compactions) == 1
    assert saved.compactions[0].first_kept_message_index == 15
    assert saved.compactions[0].summarized_until_message_index == 15
    assert saved.compactions[0].created_after_message_index == 25


def test_resumed_compaction_rebuilds_projected_context() -> None:
    record = _session_record("sess_compacted", text="old question")
    messages = [
        _text_message(index, f"message {index}")
        for index in range(14)
    ]
    record = session.SessionRecord(
        metadata=record.metadata,
        settings=record.settings,
        messages=messages,
        compactions=[
            session.SessionCompaction(
                summary="durable summary",
                first_kept_message_index=10,
                summarized_until_message_index=10,
                created_after_message_index=14,
            )
        ],
    )
    args = _make_args(persist_session=True)
    args.provider = record.settings.provider
    args.model = record.settings.model
    args.max_tokens = record.settings.max_tokens
    args._resume_session_record = record
    args._resume_session_path = Path("/tmp/sess_compacted.jsonl")
    response = CompletionResponse(content=[TextBlock(text="new answer")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider([[StreamComplete(response=response)]])
    inputs_iter = iter(["new question", "/exit"])

    def input_func(_prompt: str = "") -> str:
        try:
            return next(inputs_iter)
        except StopIteration as exc:
            raise EOFError from exc

    app = tui.WillowApp(
        args,
        provider,
        state=tui._state_from_session(record),
        input_func=input_func,
        out=io.StringIO(),
    )
    assert app.run() == 0

    request = provider.requests[0]
    assert "durable summary" in _message_text(request.messages[0])
    assert request.messages[1:] == [
        *messages[10:],
        Message(role="user", content=[TextBlock(text="new question")]),
    ]


def test_branch_copies_history_and_compaction_into_new_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(session.SESSION_DIR_ENV, str(tmp_path / "sessions"))
    parent = _session_record("parent_session", text="old question")
    messages = [_text_message(index, f"message {index}") for index in range(14)]
    parent = session.SessionRecord(
        metadata=parent.metadata,
        settings=parent.settings,
        messages=messages,
        compactions=[
            session.SessionCompaction(
                summary="parent summary",
                first_kept_message_index=10,
                summarized_until_message_index=10,
                created_after_message_index=14,
            )
        ],
    )
    parent_path = session.save_session(parent, session.default_session_path(parent.metadata.id))
    response = CompletionResponse(content=[TextBlock(text="branch answer")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider([[StreamComplete(response=response)]])
    args = _make_args(persist_session=True)
    args.provider = parent.settings.provider
    args.model = parent.settings.model
    args.max_tokens = parent.settings.max_tokens
    args._resume_session_record = parent
    args._resume_session_path = parent_path
    inputs_iter = iter(["/branch", "branch question", "/exit"])

    def input_func(_prompt: str = "") -> str:
        try:
            return next(inputs_iter)
        except StopIteration as exc:
            raise EOFError from exc

    out = io.StringIO()
    app = tui.WillowApp(
        args,
        provider,
        state=tui._state_from_session(parent),
        input_func=input_func,
        out=out,
    )
    assert app.run() == 0

    assert app._session_record is not None
    branch = app._session_record
    assert branch.metadata.id != parent.metadata.id
    assert branch.metadata.parent_session_id == parent.metadata.id
    assert branch.compactions == parent.compactions
    assert session.load_session(parent_path).messages == parent.messages
    assert app._session_path is not None
    saved_branch = session.load_session(app._session_path)
    assert saved_branch.metadata.parent_session_id == parent.metadata.id
    assert saved_branch.messages[:14] == parent.messages
    assert [block.text for block in saved_branch.messages[-2].content] == ["branch question"]

    rendered = out.getvalue()
    assert "Branched conversation" in rendered
    assert f"session {branch.metadata.id}" in rendered
    assert f"/resume {parent.metadata.id}" in rendered
    assert f"willow -r {parent.metadata.id}" in rendered

    request = provider.requests[0]
    assert "parent summary" in _message_text(request.messages[0])
    assert request.messages[1:] == [
        *messages[10:],
        Message(role="user", content=[TextBlock(text="branch question")]),
    ]


def test_tui_resume_command_switches_sessions_and_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(session.SESSION_DIR_ENV, str(tmp_path / "sessions"))
    parent = _session_record("parent_session", text="old question")
    parent_messages = [_text_message(index, f"parent {index}") for index in range(12)]
    parent = session.SessionRecord(
        metadata=parent.metadata,
        settings=parent.settings,
        messages=parent_messages,
        compactions=[
            session.SessionCompaction(
                summary="parent durable summary",
                first_kept_message_index=8,
                summarized_until_message_index=8,
                created_after_message_index=12,
            )
        ],
    )
    session.save_session(parent, session.default_session_path(parent.metadata.id))
    branch = _session_record("branch_session", text="branch question")
    branch = session.SessionRecord(
        metadata=branch.metadata,
        settings=branch.settings,
        messages=[Message(role="user", content=[TextBlock(text="branch only")])],
    )
    branch_path = session.save_session(branch, session.default_session_path(branch.metadata.id))
    response = CompletionResponse(content=[TextBlock(text="parent answer")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider([[StreamComplete(response=response)]])
    args = _make_args(persist_session=True)
    args.provider = branch.settings.provider
    args.model = branch.settings.model
    args.max_tokens = branch.settings.max_tokens
    args._resume_session_record = branch
    args._resume_session_path = branch_path
    inputs_iter = iter([f"/resume {parent.metadata.id}", "parent followup", "/exit"])

    def input_func(_prompt: str = "") -> str:
        try:
            return next(inputs_iter)
        except StopIteration as exc:
            raise EOFError from exc

    out = io.StringIO()
    app = tui.WillowApp(
        args,
        provider,
        state=tui._state_from_session(branch),
        input_func=input_func,
        out=out,
    )
    assert app.run() == 0

    assert app._session_record is not None
    assert app._session_record.metadata.id == parent.metadata.id
    assert "Resumed session parent_session" in out.getvalue()
    request = provider.requests[0]
    assert "parent durable summary" in _message_text(request.messages[0])
    assert request.messages[1:] == [
        *parent_messages[8:],
        Message(role="user", content=[TextBlock(text="parent followup")]),
    ]


def _message_text(message: Message) -> str:
    return "\n".join(
        block.text for block in message.content if isinstance(block, TextBlock)
    )


def test_terminal_clear_is_append_only_and_resets_history() -> None:
    end1 = CompletionResponse(content=[TextBlock(text="first done")], stop_reason="end_turn")
    end2 = CompletionResponse(content=[TextBlock(text="second done")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider(
        [
            [TextDelta(text="first done"), StreamComplete(response=end1)],
            [TextDelta(text="second done"), StreamComplete(response=end2)],
        ]
    )

    out, _app = _drive(provider, ["first", "/clear", "second", "/exit"])

    assert "first done" in out
    assert "Conversation cleared." in out
    assert "second done" in out
    assert len(provider.requests) == 2
    assert len(provider.requests[1].messages) == 1
    block = provider.requests[1].messages[0].content[0]
    assert isinstance(block, TextBlock)
    assert block.text == "second"


def test_terminal_model_command_lists_models_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _ScriptedStreamProvider([])
    monkeypatch.setattr(
        tui,
        "available_model_choices",
        lambda: [
            ModelChoice(
                model="gpt-5.5",
                provider="openai_responses",
                vendor="openai",
                description="Frontier model.",
            )
        ],
    )

    out, _app = _drive(provider, ["/model", "/exit"])

    assert "Select Model" in out
    assert "gpt-5.5" in out
    assert provider.requests == []


def test_terminal_model_selection_switches_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    anthropic_provider = _ScriptedStreamProvider([])
    openai_response = CompletionResponse(content=[TextBlock(text="ok")], stop_reason="end_turn")
    openai_provider = _ScriptedStreamProvider(
        [[TextDelta(text="ok"), StreamComplete(response=openai_response)]]
    )
    choices = [
        ModelChoice(
            model="gpt-5.5",
            provider="openai_responses",
            vendor="openai",
            description="Frontier model.",
        )
    ]

    def build_provider(name: str) -> Provider:
        return {
            "anthropic": anthropic_provider,
            "openai_responses": openai_provider,
        }[name]

    monkeypatch.setattr(tui, "available_model_choices", lambda: choices)
    monkeypatch.setattr("willow.cli._build_provider", build_provider)

    out, _app = _drive(
        anthropic_provider,
        ["/model 1", "go", "/exit"],
        args=_make_args(provider="anthropic"),
    )

    assert "using provider 'openai_responses'" in out
    assert anthropic_provider.requests == []
    assert len(openai_provider.requests) == 1
    assert openai_provider.requests[0].model == "gpt-5.5"


def test_terminal_statusline_can_be_disabled() -> None:
    end_response = CompletionResponse(
        content=[TextBlock(text="ok")],
        stop_reason="end_turn",
        usage={"input_tokens": 10, "output_tokens": 5},
    )
    provider = _ScriptedStreamProvider(
        [[TextDelta(text="ok"), StreamComplete(response=end_response)]]
    )

    out, app = _drive(provider, ["/statusline off", "go", "/exit"])

    assert "Status snapshots disabled." in out
    assert "[status]" not in out
    assert app._statusline_enabled is False
