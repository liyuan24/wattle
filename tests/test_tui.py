"""Tests for Wattle's native terminal TUI."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import re
import shlex
import sys
import threading
import time
from collections import deque
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from wattle import auth, cli, request_preparation, session, settings, tui
from wattle.command_summary import CommandSummary, CommandSummaryKind
from wattle.goal import create_goal, set_goal_status
from wattle.hooks import FINAL_AUDIT_REMINDER
from wattle.models import ModelChoice
from wattle.providers import (
    CompletionRequest,
    CompletionResponse,
    ImageBlock,
    Message,
    OpenAICodexResponsesProvider,
    Provider,
    StreamComplete,
    TextBlock,
    TextDelta,
    ThinkingDelta,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseDelta,
    TransientProviderError,
)
from wattle.runtime import TaskStatus, WattleRuntime
from wattle.tools import TOOLS_BY_NAME
from wattle.tools.base import Tool
from wattle.version import get_wattle_version

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-png"
JPEG_BYTES = b"\xff\xd8\xff\xe0fake-jpeg"
GIF_BYTES = b"GIF89afake-gif"
WEBP_BYTES = b"RIFF\x04\x00\x00\x00WEBPfake-webp"
IMAGE_FIXTURES = (
    ("png", "image/png", ".png", PNG_BYTES),
    ("jpeg", "image/jpeg", ".jpg", JPEG_BYTES),
    ("gif", "image/gif", ".gif", GIF_BYTES),
    ("webp", "image/webp", ".webp", WEBP_BYTES),
)


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text).replace("\r", "")


def _complete_goal_input(evidence: str = "All requested checks passed.") -> dict[str, str]:
    return {
        "status": "complete",
        "evidence": evidence,
    }


class _ScriptedStreamProvider(Provider):
    def __init__(self, scripts: list[list[Any]]) -> None:
        self._scripts: deque[list[Any]] = deque(scripts)
        self.requests: list[CompletionRequest] = []

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:  # pragma: no cover
        raise NotImplementedError

    async def astream(self, request: CompletionRequest) -> AsyncIterator[Any]:
        self.requests.append(request)
        if not self._scripts:
            raise RuntimeError("provider exhausted")
        for event in self._scripts.popleft():
            yield event


class _ResettableScriptedStreamProvider(_ScriptedStreamProvider):
    def __init__(self, scripts: list[list[Any]]) -> None:
        super().__init__(scripts)
        self.reset_count = 0

    async def areset_conversation(self) -> None:
        self.reset_count += 1


class _RaisingStreamProvider(Provider):
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.requests: list[CompletionRequest] = []

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:  # pragma: no cover
        raise NotImplementedError

    async def astream(self, request: CompletionRequest) -> AsyncIterator[Any]:
        self.requests.append(request)
        raise self.error
        yield  # pragma: no cover


class _IncompleteThenSuccessStreamProvider(Provider):
    def __init__(self) -> None:
        self.requests: list[CompletionRequest] = []

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:  # pragma: no cover
        raise NotImplementedError

    async def astream(self, request: CompletionRequest) -> AsyncIterator[Any]:
        self.requests.append(request)
        if len(self.requests) == 1:
            yield TextDelta(text="partial")
            raise request_preparation.IncompleteStreamError(
                "stream closed before response.completed"
            )
        response = CompletionResponse(content=[TextBlock(text="final")], stop_reason="end_turn")
        yield TextDelta(text="final")
        yield StreamComplete(response=response)


class _TransientThenSuccessStreamProvider(Provider):
    def __init__(self) -> None:
        self.requests: list[CompletionRequest] = []

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:  # pragma: no cover
        raise NotImplementedError

    async def astream(self, request: CompletionRequest) -> AsyncIterator[Any]:
        self.requests.append(request)
        if len(self.requests) == 1:
            raise TransientProviderError(
                "Codex request failed with HTTP 503: upstream reset",
                status_code=503,
            )
        response = CompletionResponse(content=[TextBlock(text="final")], stop_reason="end_turn")
        yield TextDelta(text="final")
        yield StreamComplete(response=response)


class _IdleThenSuccessStreamProvider(Provider):
    def __init__(self) -> None:
        self.requests: list[CompletionRequest] = []

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:  # pragma: no cover
        raise NotImplementedError

    async def astream(self, request: CompletionRequest) -> AsyncIterator[Any]:
        self.requests.append(request)
        if len(self.requests) == 1:
            yield TextDelta(text="partial")
            await asyncio.sleep(60)
        response = CompletionResponse(content=[TextBlock(text="final")], stop_reason="end_turn")
        yield TextDelta(text="final")
        yield StreamComplete(response=response)


class _FakeJsonResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _FakeJsonResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _size: int = -1) -> bytes:
        return self._payload


def _codex_token() -> str:
    def encode(data: dict[str, Any]) -> str:
        raw = json.dumps(data).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return (
        f"{encode({'alg': 'none'})}."
        f"{encode({'https://api.openai.com/auth': {'chatgpt_account_id': 'acct_123'}})}."
        "sig"
    )


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

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        self.child_requests.append(request)
        return CompletionResponse(
            content=[TextBlock(text="child done")],
            stop_reason="end_turn",
        )

    async def astream(self, request: CompletionRequest) -> AsyncIterator[Any]:
        self.requests.append(request)
        if not self._stream_turns:
            raise RuntimeError("provider exhausted")
        for event in self._stream_turns.popleft():
            yield event


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


class _ParallelRecordingTool(Tool):
    name = "parallel_echo"
    supports_parallel_tool_calls = True
    description = "Return back the given message with a concurrency probe."
    input_schema = {
        "type": "object",
        "properties": {
            "message": {"type": "string"},
            "wait_for_peer": {"type": "boolean"},
        },
        "required": ["message"],
    }

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.started = threading.Condition(self.lock)
        self.started_count = 0
        self.active = 0
        self.max_active = 0

    def run(self, message: str, wait_for_peer: bool = True) -> str:
        with self.started:
            self.started_count += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.started.notify_all()
            if wait_for_peer:
                deadline = time.monotonic() + 1.0
                while self.started_count < 2:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self.started.wait(timeout=remaining)
        time.sleep(0.03)
        with self.started:
            self.active -= 1
        return f"echoed: {message}"


class _DiffEditTool(Tool):
    name = "edit"
    description = "Return a focused edit diff."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "line": {"type": "integer"},
            "before": {"type": "string"},
            "after": {"type": "string"},
            "fail": {"type": "boolean"},
        },
        "required": ["path"],
    }

    def run(self, **kwargs: Any) -> str:
        path = str(kwargs["path"])
        if kwargs.get("fail"):
            raise ValueError(f"old_text not found in {path}")
        line = int(kwargs.get("line", 1))
        before = str(kwargs.get("before", "old"))
        after = str(kwargs.get("after", "new"))
        return "\n".join(
            [
                f"Edited {path}",
                f"--- {path} (before)",
                f"+++ {path} (after)",
                f"@@ -{line},1 +{line},1 @@",
                f"-{before}",
                f"+{after}",
            ]
        )


class _DiffWriteTool(Tool):
    name = "write"
    description = "Return a focused write diff."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "line": {"type": "integer"},
            "before": {"type": "string"},
            "after": {"type": "string"},
        },
        "required": ["path"],
    }

    def run(self, **kwargs: Any) -> str:
        path = str(kwargs["path"])
        line = int(kwargs.get("line", 1))
        after = str(kwargs.get("after", "new"))
        diff_lines = [
            f"Wrote {len(after)} bytes to {path}",
            f"--- {path} (before)",
            f"+++ {path} (after)",
        ]
        if "before" in kwargs:
            before = str(kwargs["before"])
            diff_lines.extend(
                [
                    f"@@ -{line},1 +{line},1 @@",
                    f"-{before}",
                    f"+{after}",
                ]
            )
        else:
            diff_lines.extend([f"@@ -0,0 +{line},1 @@", f"+{after}"])
        return "\n".join(diff_lines)


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


def _extensionless_screenshot_path(tmp_path: Path) -> Path:
    image = tmp_path / "TemporaryItems" / "NSIRD_screencaptureui_BRymlQ" / "Screenshot"
    image.parent.mkdir(parents=True)
    image.write_bytes(PNG_BYTES)
    return image


def _extensionless_temp_image_path(tmp_path: Path, image_name: str, data: bytes) -> Path:
    image = (
        tmp_path
        / "TemporaryItems"
        / f"NSIRD_screencaptureui_{image_name}"
        / "Screenshot"
    )
    image.parent.mkdir(parents=True)
    image.write_bytes(data)
    return image


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
) -> tuple[str, tui.WattleApp]:
    args = args or _make_args()
    inputs_iter = iter(inputs)

    def input_func(_prompt: str = "") -> str:
        try:
            return next(inputs_iter)
        except StopIteration as exc:
            raise EOFError from exc

    out = io.StringIO()
    app = tui.WattleApp(args, provider, input_func=input_func, out=out)
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


def test_basic_tui_goal_starts_continuation_and_update_goal_completes() -> None:
    tool_use_response = CompletionResponse(
        content=[
            ToolUseBlock(
                id="goal_1",
                name="update_goal",
                input=_complete_goal_input("All requested checks passed."),
            )
        ],
        stop_reason="tool_use",
    )
    final_response = CompletionResponse(
        content=[TextBlock(text="Goal complete.")],
        stop_reason="end_turn",
    )
    audit_response = CompletionResponse(
        content=[TextBlock(text="Audit complete.")],
        stop_reason="end_turn",
    )
    provider = _ScriptedStreamProvider(
        [
            [StreamComplete(response=tool_use_response)],
            [TextDelta(text="Goal complete."), StreamComplete(response=final_response)],
            [StreamComplete(response=audit_response)],
        ]
    )

    out, app = _drive(provider, ["/goal Finish the hook", "/exit"])

    assert len(provider.requests) == 3
    assert _message_text(provider.requests[0].messages[0]) == "Finish the hook"
    assert _message_text(provider.requests[2].messages[-1]) == FINAL_AUDIT_REMINDER
    assert app.goal is not None
    assert app.goal.status == "complete"
    assert isinstance(app.messages[1].content[0], ToolUseBlock)
    assert app.messages[1].content[0].name == "update_goal"
    assert isinstance(app.messages[2].content[0], ToolResultBlock)
    assert app.messages[2].content[0].tool_use_id == "goal_1"
    assert "Goal active." in out
    assert "Goal complete." in out
    assert "update_goal" not in out
    assert "Status: complete" not in out


def test_tui_goal_continues_with_template_after_incomplete_turn() -> None:
    first_response = CompletionResponse(
        content=[TextBlock(text="Need another turn.")],
        stop_reason="end_turn",
    )
    tool_use_response = CompletionResponse(
        content=[
            ToolUseBlock(
                id="goal_1",
                name="update_goal",
                input=_complete_goal_input("The hook behavior was verified."),
            )
        ],
        stop_reason="tool_use",
    )
    final_response = CompletionResponse(
        content=[TextBlock(text="Goal complete.")],
        stop_reason="end_turn",
    )
    audit_response = CompletionResponse(
        content=[TextBlock(text="Audit complete.")],
        stop_reason="end_turn",
    )
    provider = _ScriptedStreamProvider(
        [
            [TextDelta(text="Need another turn."), StreamComplete(response=first_response)],
            [StreamComplete(response=tool_use_response)],
            [TextDelta(text="Goal complete."), StreamComplete(response=final_response)],
            [StreamComplete(response=audit_response)],
        ]
    )

    _out, app = _drive(provider, ["/goal Finish the hook", "/exit"])

    assert len(provider.requests) == 4
    assert _message_text(provider.requests[0].messages[0]) == "Finish the hook"
    continuation_text = _message_text(provider.requests[1].messages[-1])
    assert "Continue working toward the active Wattle goal." in continuation_text
    assert "<objective>\nFinish the hook\n</objective>" in continuation_text
    assert _message_text(provider.requests[3].messages[-1]) == FINAL_AUDIT_REMINDER
    assert app.goal is not None
    assert app.goal.status == "complete"


def test_goal_clear_clears_state_and_disables_pending_goal_start() -> None:
    out = io.StringIO()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)

    app._handle_goal("Finish the hook")
    assert app.goal is not None
    assert app._goal_start_content is not None

    app._handle_goal("clear")

    assert app.goal is None
    assert app._goal_start_content is None
    assert app._append_requested_goal_start() is False
    assert "Goal cleared." in out.getvalue()


def test_goal_resume_requests_continuation_template() -> None:
    out = io.StringIO()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app.goal = set_goal_status(create_goal("Finish the hook"), "paused").goal

    app._handle_goal("resume")

    assert app.goal is not None
    assert app.goal.status == "active"
    assert app._append_requested_goal_start() is True
    assert len(app.messages) == 1
    text = _message_text(app.messages[0])
    assert "Continue working toward the active Wattle goal." in text
    assert "<objective>\nFinish the hook\n</objective>" in text


def test_new_goal_replaces_previous_goal_and_pending_start() -> None:
    out = io.StringIO()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)

    app._handle_goal("Old objective")
    app._handle_goal("New objective")

    assert app.goal is not None
    assert app.goal.objective == "New objective"
    assert app._append_requested_goal_start() is True
    assert len(app.messages) == 1
    assert _message_text(app.messages[0]) == "New objective"


def test_update_goal_tool_result_is_hidden_in_history_rendering() -> None:
    out = io.StringIO()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    messages = [
        Message(
            role="assistant",
            content=[
                ToolUseBlock(
                    id="goal_1",
                    name="update_goal",
                    input=_complete_goal_input("Done."),
                )
            ],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="goal_1", content="Goal complete.")],
        ),
    ]

    app._write_history_messages(messages, with_separators=True)

    assert out.getvalue() == ""


def test_live_update_goal_tool_runs_without_visible_status_or_output() -> None:
    out = io.StringIO()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app.goal = create_goal("Finish silently")
    live = tui._LiveTerminal(app)
    block = ToolUseBlock(
        id="goal_1",
        name="update_goal",
        input=_complete_goal_input("Verified."),
    )

    try:
        blocks = live._dispatch_tool_with_animated_prompt(block)
    finally:
        live._unsubscribe_monitor_events()

    assert app.goal is not None
    assert app.goal.status == "complete"
    assert isinstance(blocks[0], ToolResultBlock)
    assert blocks[0].tool_use_id == "goal_1"
    assert live.active_tool_status is None
    assert out.getvalue() == ""


def test_live_goal_continuation_cap_pauses_goal() -> None:
    out = io.StringIO()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app.goal = create_goal("Never stop")
    live = tui._LiveTerminal(app)
    response = CompletionResponse(content=[TextBlock(text="still working")], stop_reason="end_turn")

    try:
        for _ in range(tui.MAX_GOAL_CONTINUATIONS_PER_TURN):
            assert live._append_turn_stop_continuation_if_allowed(response) is True
        assert live._append_turn_stop_continuation_if_allowed(response) is False
    finally:
        live._unsubscribe_monitor_events()

    assert app.goal is not None
    assert app.goal.status == "paused"
    assert "Stopped automatic goal continuation after too many consecutive turns" in out.getvalue()


def test_basic_tui_retries_incomplete_stream_and_reports_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(request_preparation, "_stream_retry_delay", lambda _attempt: 0.0)
    provider = _IncompleteThenSuccessStreamProvider()

    out, app = _drive(provider, ["run the task", "/exit"])

    assert len(provider.requests) == 2
    assert provider.requests[0].messages == provider.requests[1].messages
    assert [message.role for message in app.messages] == ["user", "assistant"]
    assert app.messages[-1].content == [TextBlock(text="final")]
    assert "partial" not in out
    assert "[status] Reconnecting... 1/3" in out
    assert "final" in out
    assert "Codex stream ended without" not in out


def test_basic_tui_retries_transient_error_and_reports_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(request_preparation, "_stream_retry_delay", lambda _attempt: 0.0)
    provider = _TransientThenSuccessStreamProvider()

    out, app = _drive(provider, ["run the task", "/exit"])

    assert len(provider.requests) == 2
    assert provider.requests[0].messages == provider.requests[1].messages
    assert [message.role for message in app.messages] == ["user", "assistant"]
    assert app.messages[-1].content == [TextBlock(text="final")]
    assert "[status] Reconnecting... 1/3" in out
    assert "final" in out
    assert "upstream reset" not in out


def test_basic_tui_retries_idle_stream_and_reports_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(request_preparation, "_stream_retry_delay", lambda _attempt: 0.0)
    monkeypatch.setattr(
        request_preparation,
        "stream_idle_timeout_seconds_from_env",
        lambda: 0.01,
    )
    provider = _IdleThenSuccessStreamProvider()

    out, app = _drive(provider, ["run the task", "/exit"])

    assert len(provider.requests) == 2
    assert provider.requests[0].messages == provider.requests[1].messages
    assert [message.role for message in app.messages] == ["user", "assistant"]
    assert app.messages[-1].content == [TextBlock(text="final")]
    assert "partial" not in out
    assert "[status] Reconnecting... 1/3" in out
    assert "final" in out


def test_basic_tui_exhausted_transient_error_keeps_prompt_readable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(request_preparation, "_stream_retry_delay", lambda _attempt: 0.0)
    provider = _RaisingStreamProvider(
        TransientProviderError(
            "Codex request failed with HTTP 503: upstream reset",
            status_code=503,
        )
    )

    out, app = _drive(provider, ["run the task", "/exit"])

    assert len(provider.requests) == 4
    assert [message.role for message in app.messages] == ["user"]
    assert out.count("[status] Reconnecting...") == 3
    assert "[error] Temporary provider error after retries:" in out
    assert "Conversation history and completed tool results were kept." in out
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
    monkeypatch.setattr("wattle.cli._build_provider", lambda _name: provider)

    instances: list[FakeWattleApp] = []

    class FakeWattleApp:
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

    monkeypatch.setattr(tui, "WattleApp", FakeWattleApp)

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

    monkeypatch.setattr("wattle.cli._build_provider", build_provider)

    instances: list[Any] = []

    class FakeWattleApp:
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

    monkeypatch.setattr(tui, "WattleApp", FakeWattleApp)

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
    app = tui.WattleApp(
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


def test_resumed_session_renders_tool_history_semantically() -> None:
    record = _session_record("sess_tool_history", text="check feature request md")
    record = session.SessionRecord(
        metadata=record.metadata,
        settings=record.settings,
        messages=[
            *record.messages,
            Message(
                role="assistant",
                content=[
                    ToolUseBlock(
                        id="read_1",
                        name="read",
                        input={"path": "feature_requests.md"},
                    )
                ],
            ),
            Message(
                role="user",
                content=[
                    ToolResultBlock(
                        tool_use_id="read_1",
                        content=(
                            "path: /tmp/project/feature_requests.md\n"
                            "lines: 1-1 of 1\n"
                            "     1\tsearch sessions based on query"
                        ),
                    )
                ],
            ),
            Message(role="assistant", content=[TextBlock(text="feature_requests.md contains:")]),
        ],
    )
    args = _make_args(persist_session=True)
    args.provider = record.settings.provider
    args.model = record.settings.model
    args.max_tokens = record.settings.max_tokens
    args._resume_session_record = record
    args._resume_session_path = Path("/tmp/sess_tool_history.jsonl")

    def input_func(_prompt: str = "") -> str:
        raise EOFError

    out_buffer = io.StringIO()
    app = tui.WattleApp(
        args,
        _ScriptedStreamProvider([]),
        state=tui._state_from_session(record),
        input_func=input_func,
        out=out_buffer,
    )
    assert app.run() == 0
    out = out_buffer.getvalue()

    assert "[resumed] Loaded 4 saved message(s)" in out
    assert "Researched" in out
    assert "Read feature_requests.md" in out
    assert "feature_requests.md contains:" in out
    assert "tool use" not in out
    assert "tool result" not in out
    assert "path: /tmp/project/feature_requests.md" not in out


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

    app = tui.WattleApp(
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


def test_resumed_legacy_read_only_session_rebuilds_yolo_system_prompt() -> None:
    record = _session_record("sess_legacy_permissions", text="old question")
    record = session.SessionRecord(
        metadata=record.metadata,
        settings=session.SessionSettings(
            provider=record.settings.provider,
            model=record.settings.model,
            system="Be direct.\nRead-only mode is active. Do not write.",
            max_tokens=record.settings.max_tokens,
        ),
        messages=record.messages,
    )
    args = _make_args(persist_session=True)
    args.provider = record.settings.provider
    args.model = record.settings.model
    args.max_tokens = record.settings.max_tokens
    args._resume_session_record = record
    args._resume_session_path = Path("/tmp/sess_legacy_permissions.jsonl")
    response = CompletionResponse(content=[TextBlock(text="new answer")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider([[StreamComplete(response=response)]])
    inputs_iter = iter(["new question", "/exit"])

    def input_func(_prompt: str = "") -> str:
        try:
            return next(inputs_iter)
        except StopIteration as exc:
            raise EOFError from exc

    app = tui.WattleApp(
        args,
        provider,
        state=tui._state_from_session(record),
        input_func=input_func,
        out=io.StringIO(),
    )
    assert app.run() == 0

    assert len(provider.requests) == 1
    assert provider.requests[0].system is not None
    assert "Read-only mode is active" not in provider.requests[0].system
    assert "Available tools:" in provider.requests[0].system


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
    audit_response = CompletionResponse(
        content=[TextBlock(text="Audit complete.")],
        stop_reason="end_turn",
    )
    provider = _ScriptedStreamProvider(
        [
            [TextDelta(text="continued"), StreamComplete(response=response)],
            [StreamComplete(response=audit_response)],
        ]
    )
    inputs_iter = iter(["/exit"])

    def input_func(_prompt: str = "") -> str:
        try:
            return next(inputs_iter)
        except StopIteration as exc:
            raise EOFError from exc

    out_buffer = io.StringIO()
    app = tui.WattleApp(
        args,
        provider,
        state=tui._state_from_session(record),
        input_func=input_func,
        out=out_buffer,
    )
    assert app.run() == 0

    assert len(provider.requests) == 2
    assert provider.requests[0].messages == record.messages
    assert _message_text(provider.requests[1].messages[-1]) == FINAL_AUDIT_REMINDER
    out = out_buffer.getvalue()
    assert "[resumed] Continuing from saved tool result." in out
    saved = session.load_session(app._session_path)
    assert saved.messages[: len(record.messages)] == record.messages
    assert saved.messages[-3] == Message(role="assistant", content=[TextBlock(text="continued")])
    assert saved.messages[-2] == Message(
        role="user",
        content=[TextBlock(text=FINAL_AUDIT_REMINDER)],
    )
    assert saved.messages[-1] == Message(
        role="assistant",
        content=[TextBlock(text="Audit complete.")],
    )


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
    monkeypatch.setattr(tui, "list_session_entries", lambda: [latest, older])
    monkeypatch.setattr("wattle.cli._build_provider", lambda _name: object())

    instances: list[Any] = []

    class FakeWattleApp:
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

    monkeypatch.setattr(tui, "WattleApp", FakeWattleApp)

    args = _make_args(resume="")
    assert tui.run_tui(args) == 0

    assert args._resume_session_path == latest.path
    assert args.provider == "anthropic"
    assert instances[0].state is not None
    assert instances[0].state["current_model"] == "claude-sonnet-4-6"


def test_resume_picker_state_filters_and_clears_query() -> None:
    alpha = tui.SessionEntry(
        path=Path("/tmp/alpha.jsonl"),
        record=_session_record("alpha", text="inspect quota usage"),
        preview="inspect quota usage",
        search_text="alpha inspect quota usage openai_responses gpt-5.5 /tmp/project",
    )
    beta = tui.SessionEntry(
        path=Path("/tmp/beta.jsonl"),
        record=_session_record("beta", text="fix tui resize"),
        preview="fix tui resize",
        search_text="beta fix tui resize openai_responses gpt-5.5 /tmp/project",
    )

    state = tui._ResumePickerState(
        all_entries=[alpha, beta],
        filtered_entries=[alpha, beta],
        selected=1,
        query="quota",
        rows=10,
        width=100,
    )
    tui._apply_resume_picker_filter(state)

    assert state.filtered_entries == [alpha]
    assert state.selected == 0
    state.query = ""
    tui._apply_resume_picker_filter(state)
    assert state.filtered_entries == [alpha, beta]


def test_run_tui_resume_exact_title_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(session.SESSION_DIR_ENV, str(tmp_path / "sessions"))
    record = _session_record("sess_named", text="old question")
    record = session.SessionRecord(
        metadata=session.SessionMetadata(
            id=record.metadata.id,
            created_at=record.metadata.created_at,
            updated_at=record.metadata.updated_at,
            title="Named Debug Session",
            cwd=record.metadata.cwd,
        ),
        settings=record.settings,
        messages=record.messages,
    )
    path = session.default_session_path(record.metadata.id)
    session.save_session(record, path)
    monkeypatch.setattr("wattle.cli._build_provider", lambda _name: object())

    instances: list[Any] = []

    class FakeWattleApp:
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

    monkeypatch.setattr(tui, "WattleApp", FakeWattleApp)

    args = _make_args(resume="Named Debug Session")
    assert tui.run_tui(args) == 0

    assert args._resume_session_path == path
    assert instances[0].state is not None


def test_run_tui_resume_ambiguous_search_fails_non_tty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(session.SESSION_DIR_ENV, str(tmp_path / "sessions"))
    first = _session_record("sess_a", text="debug alpha")
    second = _session_record("sess_b", text="debug beta")
    session.save_session(first, session.default_session_path(first.metadata.id))
    session.save_session(second, session.default_session_path(second.metadata.id))

    args = _make_args(resume="debug")
    assert tui.run_tui(args) == 1

    captured = capsys.readouterr()
    assert "ambiguous resume selector 'debug'" in captured.err
    assert "sess_a" in captured.err
    assert "sess_b" in captured.err


def test_run_tui_resume_without_sessions_starts_new(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tui, "list_session_entries", lambda: [])
    monkeypatch.setattr("wattle.cli._build_provider", lambda _name: object())

    instances: list[Any] = []

    class FakeWattleApp:
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

    monkeypatch.setattr(tui, "WattleApp", FakeWattleApp)

    args = _make_args(resume="")
    assert tui.run_tui(args) == 0

    assert not hasattr(args, "_resume_session_path")
    assert instances[0].state is None


def test_command_hints_match_slash_prefix() -> None:
    rendered = tui._render_command_hints("/stat")

    assert "/status  show session and status details" in rendered
    assert "/statusline  choose fields in status line at bottom" in rendered
    assert "/model" not in rendered
    assert "/clear" not in rendered


def test_skill_hints_match_skill_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    skill_path = tmp_path / "project" / ".wattle" / "skills" / "reviewer" / "SKILL.md"
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
        tmp_path / "project" / ".wattle" / "skills" / "hello_world" / "SKILL.md"
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
    skills_root = tmp_path / "project" / ".wattle" / "skills"
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
    (project / "src" / "wattle").mkdir(parents=True)
    (project / "src" / "wattle" / "tui.py").write_text("print('hi')", encoding="utf-8")
    (project / "README.md").write_text("docs", encoding="utf-8")
    (project / "notes draft.txt").write_text("draft", encoding="utf-8")
    (project / "node_modules").mkdir()
    (project / "node_modules" / "notes.js").write_text("ignored", encoding="utf-8")

    rendered = tui._render_input_hints("please inspect @tui", project)

    assert "@src/wattle/tui.py" in rendered
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


def test_render_statusline_defaults_to_model_thinking_and_cwd() -> None:
    rendered = tui._render_statusline(
        model="gpt-5.5",
        context_tokens=10_500,
        context_window=1_050_000,
        input_tokens=40_000,
        cached_tokens=12_000,
        output_tokens=2_000,
        cwd="~/repos/wattle",
    )

    assert rendered == "gpt-5.5 | thinking: off | ~/repos/wattle"
    assert " · " not in rendered


def test_render_statusline_can_show_context_without_provider_usage() -> None:
    rendered = tui._render_statusline(
        model="gpt-5.5",
        context_tokens=None,
        context_window=1_050_000,
        input_tokens=0,
        cached_tokens=0,
        output_tokens=0,
        cwd="~/repos/wattle",
        fields=("model", "context_used", "context_size", "cwd"),
    )

    assert rendered == (
        "gpt-5.5 | Context 0.0% used | window: 1.1M tok | ~/repos/wattle"
    )


def test_render_statusline_uses_configured_fields() -> None:
    rendered = tui._render_statusline(
        model="gpt-5.5",
        context_tokens=100,
        context_window=1000,
        input_tokens=40,
        cached_tokens=12,
        output_tokens=2,
        cwd="~/repos/wattle",
        thinking=True,
        effort="high",
        fields=(
            "model",
            "thinking_level",
            "context_remaining",
            "total_input_token_limit",
            "current_input_tokens",
            "output_tokens",
            "cached_tokens",
            "current_working_directory",
            "5_hour_quota_limit",
            "1_week_limit",
        ),
    )

    assert rendered == (
        "gpt-5.5 | thinking: high | remaining: 900 tok | window: 1.0k tok | "
        "input: 40 tok | output: 2 tok | cached total: 12 tok | ~/repos/wattle | "
        "5h quota: unknown | 1w limit: unknown"
    )


def test_render_statusline_shows_provider_quota_percentages() -> None:
    rendered = tui._render_statusline(
        model="gpt-5.5",
        context_tokens=100,
        context_window=1000,
        input_tokens=40,
        cached_tokens=12,
        output_tokens=2,
        cwd="~/repos/wattle",
        quota_5h_remaining_percent=60,
        quota_1w_remaining_percent=6,
        fields=("model", "quota_5h", "quota_1w", "cwd"),
    )

    assert rendered == "gpt-5.5 | 5h 60% | weekly 6% | ~/repos/wattle"


def test_status_text_uses_last_provider_input_tokens_not_heuristic() -> None:
    app = tui.WattleApp(
        _make_args(
            statusline_fields=(
                "model",
                "context_used",
                "context_size",
                "input_tokens",
                "cached_tokens",
                "output_tokens",
                "cwd",
            )
        ),
        _ScriptedStreamProvider([]),
        out=io.StringIO(),
    )
    app.system = None
    app.tool_specs = []
    app.messages = [Message(role="user", content=[TextBlock(text="x" * 42_000)])]
    app._last_context_tokens = 862
    app._total_input_tokens = 900_000
    app._total_cached_tokens = 300_000
    app._total_output_tokens = 100_000

    rendered = app._status_text()

    assert "Context 0.3% used | window: 272.0k tok" in rendered
    assert "input: 900.0k tok" in rendered
    assert "cached total: 300.0k tok" in rendered
    assert "output: 100.0k tok" in rendered


def test_status_text_uses_last_provider_quota_percentages() -> None:
    app = tui.WattleApp(
        _make_args(statusline_fields=("model", "quota_5h", "quota_1w", "cwd")),
        _ScriptedStreamProvider([]),
        out=io.StringIO(),
    )

    app._record_usage(
        CompletionResponse(
            content=[TextBlock(text="done")],
            stop_reason="end_turn",
            usage={
                "input_tokens": 1,
                "output_tokens": 2,
                "quota_5h_remaining_percent": 72,
                "quota_1w_remaining_percent": 90,
            },
        )
    )

    assert app._status_text().startswith("gpt-5.5 | 5h 72% | weekly 90% | ")


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

    assert f"Wattle Agent v{get_wattle_version()}" in out
    assert "model:     gpt-5.5" in out
    assert "directory:" in out
    assert "Session:" not in out
    assert "hi there" in out
    assert "[status] gpt-5.5 | " in out
    assert "thinking: off" in out
    assert "cwd: " not in out
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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False

    app._write_worked_duration(10.0)

    rendered = out.getvalue()
    assert rendered == f"{tui.WORKED_DURATION_STYLE}Worked for 1m 5s{tui.RESET}\n"


def test_tui_attaches_local_image_from_user_text(tmp_path: Path) -> None:
    image = tmp_path / "debug shot.png"
    image.write_bytes(b"fake-png")
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)

    assert app._submit_user_text(f'check image at "{image}"', render=True)

    message = app.messages[-1]
    assert message.role == "user"
    assert isinstance(message.content[0], TextBlock)
    assert isinstance(message.content[1], ImageBlock)
    assert message.content[0].text == "check image at [IMAGE #1]"
    assert message.content[1].path == str(image.resolve())
    assert message.content[1].media_type == "image/png"
    rendered = out.getvalue()
    assert "check image at [IMAGE #1]" in rendered
    assert str(image) not in rendered
    assert "[image] debug shot.png" not in rendered


def test_text_only_model_omits_submitted_images_with_notice(tmp_path: Path) -> None:
    image = tmp_path / "debug shot.png"
    image.write_bytes(b"fake-png")
    out = _TTYBuffer()
    app = tui.WattleApp(
        _make_args(model="deepseek-v4-flash"),
        _ScriptedStreamProvider([]),
        out=out,
    )

    assert app._submit_user_text(f'check image at "{image}"', render=True)

    message = app.messages[-1]
    assert message.role == "user"
    assert message.content == [
        TextBlock(text="check image at [IMAGE #1]"),
        TextBlock(
            text=(
                "[image omitted: model deepseek-v4-flash "
                "does not support image inputs]"
            )
        ),
    ]
    rendered = out.getvalue()
    assert "check image at [IMAGE #1]" in rendered
    assert "Images were not sent because model deepseek-v4-flash" in rendered
    assert str(image) not in rendered


def test_text_only_model_hides_view_image_tool() -> None:
    app = tui.WattleApp(
        _make_args(model="deepseek-v4-flash"),
        _ScriptedStreamProvider([]),
        out=io.StringIO(),
    )

    assert "view_image" not in {spec["name"] for spec in app.tool_specs}
    assert app.system is not None
    assert "view_image" not in app.system


def test_tui_subagents_full_tool_set_includes_custom_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _RecordingTool()
    monkeypatch.setitem(TOOLS_BY_NAME, tool.name, tool)
    app = tui.WattleApp(
        _make_args(model="deepseek-v4-flash"),
        _ParentChildProvider(),
        out=io.StringIO(),
    )

    app._configure_subagents()

    config = app.runtime.subagents._config
    assert config is not None
    assert "echo" in config.full_tools_by_name


def test_raw_model_command_refreshes_model_dependent_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tui,
        "available_model_choices",
        lambda: [
            ModelChoice(
                model="deepseek-v4-flash",
                provider="openai_responses",
                vendor="deepseek",
                description="Fast DeepSeek model.",
                supported_modalities=("text",),
            )
        ],
    )
    app = tui.WattleApp(
        _make_args(model="gpt-5.5"),
        _ScriptedStreamProvider([]),
        out=io.StringIO(),
    )
    assert "view_image" in {spec["name"] for spec in app.tool_specs}

    app._handle_model("deepseek-v4-flash")

    assert app.current_model == "deepseek-v4-flash"
    assert "view_image" not in {spec["name"] for spec in app.tool_specs}
    assert app.system is not None
    assert "view_image" not in app.system


def test_raw_model_command_rejects_unknown_model() -> None:
    out = io.StringIO()
    app = tui.WattleApp(
        _make_args(model="gpt-5.5"),
        _ScriptedStreamProvider([]),
        out=out,
    )

    app._handle_model("vendor-model-name")

    assert app.current_model == "gpt-5.5"
    assert "Unknown model: vendor-model-name" in out.getvalue()


def test_removed_model_subcommands_do_not_change_model() -> None:
    out = io.StringIO()
    app = tui.WattleApp(
        _make_args(model="gpt-5.5"),
        _ScriptedStreamProvider([]),
        out=out,
    )

    app._handle_model("next")
    app._handle_model("enable deepseek-v4-flash")

    assert app.current_model == "gpt-5.5"
    rendered = out.getvalue()
    assert "Unsupported model command: /model next" in rendered
    assert "Unsupported model command: /model enable" in rendered


def test_history_replay_keeps_user_image_anchor_without_summary(tmp_path: Path) -> None:
    image = tmp_path / "debug shot.png"
    image.write_bytes(b"fake-png")
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)

    content = app._user_content_blocks(f'check image at "{image}"')
    message = Message(role="user", content=content)
    app._write_history_message(message)

    rendered = out.getvalue()
    assert "check image at [IMAGE #1]" in rendered
    assert "[image] debug shot.png" not in rendered
    assert str(image) not in rendered


def test_history_replay_renders_image_only_user_message_summary(tmp_path: Path) -> None:
    image = tmp_path / "debug shot.png"
    image.write_bytes(b"fake-png")
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    message = Message(
        role="user",
        content=[
            ImageBlock(
                path=str(image.resolve()),
                media_type="image/png",
                size_bytes=image.stat().st_size,
                filename=image.name,
            )
        ],
    )

    app._write_history_message(message)

    rendered = out.getvalue()
    assert "[image] debug shot.png" in rendered


def test_tui_numbers_multiple_local_images_in_user_text(tmp_path: Path) -> None:
    first = tmp_path / "left.png"
    second = tmp_path / "right.png"
    first.write_bytes(b"fake-png-left")
    second.write_bytes(b"fake-png-right")
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)

    assert app._submit_user_text(f"compare {first} with {second}", render=True)

    message = app.messages[-1]
    assert message.role == "user"
    assert message.content[0] == TextBlock(text="compare [IMAGE #1] with [IMAGE #2]")
    assert [
        block.filename for block in message.content if isinstance(block, ImageBlock)
    ] == ["left.png", "right.png"]
    rendered = out.getvalue()
    assert "compare [IMAGE #1] with [IMAGE #2]" in rendered
    assert str(first) not in rendered
    assert str(second) not in rendered


def test_tui_sends_text_file_path_as_text_context(tmp_path: Path) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.write_text("secret file content", encoding="utf-8")
    cwd = Path.cwd()
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)

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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)

    assert app._submit_user_text("check @shot.png and @notes.txt", render=True)

    message = app.messages[-1]
    assert message.content[0] == TextBlock(text="check [IMAGE #1] and @notes.txt")
    assert any(
        isinstance(block, ImageBlock) and block.filename == "shot.png"
        for block in message.content
    )
    assert TextBlock(text="notes.txt") in message.content
    assert "secret file content" not in str(message.content)


def test_live_ctrl_v_pastes_clipboard_image_as_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = io.StringIO()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    monkeypatch.setattr(app, "cwd", tmp_path)
    monkeypatch.setattr(
        tui,
        "read_clipboard_image",
        lambda: tui.ClipboardImage(b"fake-png", "image/png", ".png"),
    )
    live = tui._LiveTerminal(app)

    live._paste_clipboard_image_or_insert_literal("\x16")

    assert "[IMAGE #1]" in tui._image_placeholder_prompt_render(
        tui._render_prompt_input(live.buffer, live.pasted_ranges, live.cursor)
    ).text
    assert "\x16" not in live.buffer
    assert (tmp_path / ".wattle" / "clipboard-images").is_dir()
    content = app._user_content_blocks(live.buffer)
    assert content[0] == TextBlock(text="[IMAGE #1]")
    assert any(isinstance(block, ImageBlock) for block in content)


def test_live_ctrl_v_falls_back_to_literal_when_no_clipboard_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = io.StringIO()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    monkeypatch.setattr(tui, "read_clipboard_image", lambda: None)
    live = tui._LiveTerminal(app)

    live._paste_clipboard_image_or_insert_literal("\x16")

    assert live.buffer == "\x16"


def test_live_xterm_modified_ctrl_v_pastes_clipboard_image_as_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    monkeypatch.setattr(app, "cwd", tmp_path)
    monkeypatch.setattr(
        tui,
        "read_clipboard_image",
        lambda: tui.ClipboardImage(b"fake-png", "image/png", ".png"),
    )
    live = tui._LiveTerminal(app)
    read_fd, write_fd = os.pipe()
    try:
        live.fd = read_fd
        os.write(write_fd, b"\x1b[27;5;118~")
        live._read_available_input()
    finally:
        os.close(read_fd)
        os.close(write_fd)

    assert "[IMAGE #1]" in tui._image_placeholder_prompt_render(
        tui._render_prompt_input(live.buffer, live.pasted_ranges, live.cursor)
    ).text
    assert "\x16" not in live.buffer
    assert app._user_content_blocks(live.buffer)[0] == TextBlock(text="[IMAGE #1]")


def test_live_ctrl_v_inserts_spaces_around_clipboard_image_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = io.StringIO()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    monkeypatch.setattr(app, "cwd", tmp_path)
    monkeypatch.setattr(
        tui,
        "read_clipboard_image",
        lambda: tui.ClipboardImage(b"fake-png", "image/png", ".png"),
    )
    live = tui._LiveTerminal(app)

    live.buffer = "beforeafter"
    live.cursor = len("before")
    live._paste_clipboard_image_or_insert_literal("\x16")

    rendered = tui._image_placeholder_prompt_render(
        tui._render_prompt_input(live.buffer, live.pasted_ranges, live.cursor)
    ).text
    assert rendered == "before [IMAGE #1] after"
    assert app._user_content_blocks(live.buffer)[0] == TextBlock(
        text="before [IMAGE #1] after"
    )


def test_live_backspace_on_image_anchor_deletes_entire_attachment_path(
    tmp_path: Path,
) -> None:
    image = tmp_path / "dragged image.png"
    image.write_bytes(PNG_BYTES)
    escaped_path = str(image).replace(" ", "\\ ")
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    read_fd, write_fd = os.pipe()
    try:
        live.fd = read_fd
        os.write(write_fd, f"check {escaped_path}\x7f".encode())
        live._read_available_input()
    finally:
        os.close(read_fd)
        os.close(write_fd)

    rendered = tui._image_placeholder_prompt_render(
        tui._render_prompt_input(live.buffer, live.pasted_ranges, live.cursor)
    ).text
    assert live.buffer == "check "
    assert live.cursor == len("check ")
    assert rendered == "check "
    assert str(image) not in rendered
    assert escaped_path not in rendered


def test_live_option_arrow_treats_image_anchor_as_single_word(
    tmp_path: Path,
) -> None:
    image = tmp_path / "dragged image with spaces.png"
    image.write_bytes(PNG_BYTES)
    escaped_path = str(image).replace(" ", "\\ ")
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    read_fd, write_fd = os.pipe()
    try:
        live.fd = read_fd
        os.write(
            write_fd,
            f"check {escaped_path} after\x1bb\x1bbX \x1b[1;3C !".encode(),
        )
        live._read_available_input()
    finally:
        os.close(read_fd)
        os.close(write_fd)

    rendered = tui._image_placeholder_prompt_render(
        tui._render_prompt_input(live.buffer, live.pasted_ranges, live.cursor)
    )
    assert live.buffer == f"check X {escaped_path} ! after"
    assert live.cursor == len(f"check X {escaped_path} !")
    assert rendered.text == "check X [IMAGE #1] ! after"
    assert rendered.cursor == len("check X [IMAGE #1] !")
    assert str(image) not in rendered.text
    assert escaped_path not in rendered.text


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
    assert message.content[0] == TextBlock(text="[IMAGE #1]")
    assert any(
        isinstance(block, ImageBlock) and block.filename == image.name
        for block in message.content
    )
    assert app.messages[0].content == message.content


def test_unescaped_dragged_image_path_with_spaces_uses_anchor(tmp_path: Path) -> None:
    image = tmp_path / "Screenshot 2026-05-18.png"
    image.write_bytes(b"fake-png")
    dragged_path = str(image)
    response = CompletionResponse(content=[TextBlock(text="ok")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider([[StreamComplete(response=response)]])

    out, app = _drive(provider, [dragged_path, "/exit"])

    assert str(image) not in out
    assert len(provider.requests) == 1
    message = provider.requests[0].messages[0]
    assert message.role == "user"
    assert message.content[0] == TextBlock(text="[IMAGE #1]")
    assert any(
        isinstance(block, ImageBlock) and block.filename == image.name
        for block in message.content
    )
    assert app.messages[0].content == message.content


@pytest.mark.parametrize(
    "format_path",
    [
        pytest.param(lambda path: str(path).replace(" ", "\\ "), id="escaped-spaces"),
        pytest.param(lambda path: str(path), id="unescaped-spaces"),
        pytest.param(lambda path: shlex.quote(str(path)), id="single-quoted"),
        pytest.param(lambda path: path.as_uri(), id="file-uri"),
    ],
)
def test_dragged_extensionless_screenshot_path_variants_use_anchor(
    tmp_path: Path,
    format_path: Any,
) -> None:
    image = tmp_path / "TemporaryItems" / "NSIRD_screencaptureui_BRymlQ" / "Screenshot 1"
    image.parent.mkdir(parents=True)
    image.write_bytes(PNG_BYTES)
    text = f"check {format_path(image)}"
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)

    prompt = tui._image_placeholder_prompt_render(
        tui._PromptInputRender(text=text, cursor=len(text))
    )
    content = app._user_content_blocks(text)

    assert prompt.text == "check [IMAGE #1]"
    assert content[0] == TextBlock(text="check [IMAGE #1]")
    image_blocks = [block for block in content if isinstance(block, ImageBlock)]
    assert len(image_blocks) == 1
    assert image_blocks[0].filename == image.name
    assert image_blocks[0].media_type == "image/png"


@pytest.mark.parametrize(
    ("image_name", "media_type", "extension", "data"),
    IMAGE_FIXTURES,
    ids=[fixture[0] for fixture in IMAGE_FIXTURES],
)
def test_extensionless_dragged_supported_image_types_use_anchor(
    tmp_path: Path,
    image_name: str,
    media_type: str,
    extension: str,
    data: bytes,
) -> None:
    del extension
    image = _extensionless_temp_image_path(tmp_path, image_name, data)
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)

    prompt = tui._image_placeholder_prompt_render(
        tui._PromptInputRender(text=str(image), cursor=len(str(image)))
    )
    content = app._user_content_blocks(str(image))

    assert prompt.text == "[IMAGE #1]"
    assert content[0] == TextBlock(text="[IMAGE #1]")
    image_blocks = [block for block in content if isinstance(block, ImageBlock)]
    assert len(image_blocks) == 1
    assert image_blocks[0].filename == "Screenshot"
    assert image_blocks[0].media_type == media_type


@pytest.mark.parametrize(
    ("image_name", "media_type", "extension", "data"),
    IMAGE_FIXTURES,
    ids=[fixture[0] for fixture in IMAGE_FIXTURES],
)
def test_dragged_supported_image_extensions_use_anchor_without_sniffing(
    tmp_path: Path,
    image_name: str,
    media_type: str,
    extension: str,
    data: bytes,
) -> None:
    image = tmp_path / f"dragged {image_name}{extension}"
    image.write_bytes(data)
    text = str(image).replace(" ", "\\ ")
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)

    prompt = tui._image_placeholder_prompt_render(
        tui._PromptInputRender(text=text, cursor=len(text))
    )
    content = app._user_content_blocks(text)

    assert prompt.text == "[IMAGE #1]"
    assert content[0] == TextBlock(text="[IMAGE #1]")
    image_blocks = [block for block in content if isinstance(block, ImageBlock)]
    assert len(image_blocks) == 1
    assert image_blocks[0].filename == image.name
    assert image_blocks[0].media_type == media_type


def test_extensionless_dragged_screenshot_uses_anchor(tmp_path: Path) -> None:
    image = _extensionless_screenshot_path(tmp_path)
    response = CompletionResponse(content=[TextBlock(text="ok")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider([[StreamComplete(response=response)]])

    out, app = _drive(provider, [str(image), "/exit"])

    assert str(image) not in out
    assert len(provider.requests) == 1
    message = provider.requests[0].messages[0]
    assert message.role == "user"
    assert message.content[0] == TextBlock(text="[IMAGE #1]")
    image_blocks = [block for block in message.content if isinstance(block, ImageBlock)]
    assert len(image_blocks) == 1
    assert image_blocks[0].filename == "Screenshot"
    assert image_blocks[0].media_type == "image/png"
    assert app.messages[0].content == message.content


def test_dragged_macos_screenshot_with_narrow_no_break_space_uses_anchor(
    tmp_path: Path,
) -> None:
    image = (
        tmp_path
        / "TemporaryItems"
        / "NSIRD_screencaptureui_xwxUjd"
        / "Screenshot 2026-06-22 at 9.55.18\u202fAM.png"
    )
    image.parent.mkdir(parents=True)
    image.write_bytes(PNG_BYTES)
    dragged_path = str(image).replace(" ", "\\ ")
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)

    prompt = tui._image_placeholder_prompt_render(
        tui._PromptInputRender(text=dragged_path, cursor=len(dragged_path))
    )
    content = app._user_content_blocks(dragged_path)

    assert prompt.text == "[IMAGE #1]"
    assert content[0] == TextBlock(text="[IMAGE #1]")
    image_blocks = [block for block in content if isinstance(block, ImageBlock)]
    assert len(image_blocks) == 1
    assert image_blocks[0].filename == image.name
    assert image_blocks[0].media_type == "image/png"

    normalized_dragged_path = dragged_path.replace("\u202f", "")
    prompt = tui._image_placeholder_prompt_render(
        tui._PromptInputRender(
            text=normalized_dragged_path,
            cursor=len(normalized_dragged_path),
        )
    )
    content = app._user_content_blocks(normalized_dragged_path)

    assert prompt.text == "[IMAGE #1]"
    assert content[0] == TextBlock(text="[IMAGE #1]")
    image_blocks = [block for block in content if isinstance(block, ImageBlock)]
    assert len(image_blocks) == 1
    assert image_blocks[0].filename == image.name
    assert image_blocks[0].media_type == "image/png"


@pytest.mark.parametrize(
    ("image_name", "media_type", "extension", "data"),
    IMAGE_FIXTURES,
    ids=[fixture[0] for fixture in IMAGE_FIXTURES],
)
def test_persisted_extensionless_screenshot_asset_gets_image_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image_name: str,
    media_type: str,
    extension: str,
    data: bytes,
) -> None:
    image = _extensionless_temp_image_path(tmp_path, image_name, data)
    monkeypatch.setenv(session.SESSION_DIR_ENV, str(tmp_path / "sessions"))
    out = _TTYBuffer()
    app = tui.WattleApp(
        _make_args(persist_session=True),
        _ScriptedStreamProvider([]),
        out=out,
    )

    assert app._submit_user_text(str(image), render=True)

    message = app.messages[-1]
    image_blocks = [block for block in message.content if isinstance(block, ImageBlock)]
    assert len(image_blocks) == 1
    stored_path = Path(image_blocks[0].path)
    assert stored_path.exists()
    assert image_blocks[0].media_type == media_type
    assert stored_path.suffix == extension
    assert stored_path.read_bytes() == data
    assert stored_path != image


def test_tui_ignores_pasted_plan_prose_when_scanning_file_references() -> None:
    text = (
        "# Improve `wattle --resume` Session Search\n"
        "      5 +Narrow the gap between Wattle's `--resume` flow and "
        "Codex's resume picker.\n"
        "     68 +Important nuance: Codex's interactive picker does not "
        "currently pass the typed picker query into `thread/list.search_term`; "
        "it loads backend-filtered pages by cwd/provider/source/sort and "
        "applies typed query locally over rows.\n"
        "    144 +Wattle's JSONL store is simpler than Codex's app-server-backed "
        "thread store, so an eager full scan is acceptable for now.\n"
        + " ".join(f"{index:04d}" for index in range(400))
    )
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)

    assert app._submit_user_text(text, render=True)

    message = app.messages[-1]
    assert message.role == "user"
    assert message.content == [TextBlock(text=text)]


def test_unknown_slash_text_still_routes_to_command_error() -> None:
    out, _app = _drive(_ScriptedStreamProvider([]), ["/definitely-not-a-command", "/exit"])

    assert "Unknown command: /definitely-not-a-command" in out


def test_transcript_user_text_keeps_distinct_prompt_background() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False

    app._write_block("hello", tui.USER_STYLE)
    app._write_block("hi", tui.ASSISTANT_STYLE)
    app._write_panel("tool result", "ok", tui.TOOL_STYLE)

    rendered = out.getvalue()
    assert tui.USER_STYLE in rendered
    assert tui.ASSISTANT_STYLE in rendered
    assert tui.TOOL_STYLE in rendered
    assert "48;5;235" in tui.USER_STYLE
    assert "48;5" not in tui.ASSISTANT_STYLE
    assert "48;5" not in tui.THINKING_STYLE
    assert ";3" not in tui.THINKING_STYLE
    assert "48;5" in tui.PROMPT_STYLE
    assert tui.USER_STYLE == tui.PROMPT_STYLE
    assert "you" not in rendered
    assert "assistant" not in rendered
    assert rendered.count(tui.RESET) >= 3


def test_chat_text_blocks_keep_three_row_shape_without_assistant_fill() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._terminal_width = lambda: 10  # type: ignore[method-assign]

    app._write_block("hello", tui.USER_STYLE)
    app._write_block("hi", tui.ASSISTANT_STYLE)

    rendered_lines = _strip_ansi(out.getvalue()).splitlines()
    assert rendered_lines == [
        "",
        " hello",
        "",
        "",
        " hi",
        "",
    ]
    assert all(line == line.rstrip(" ") for line in rendered_lines)


def test_user_history_block_clears_full_terminal_rows() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._terminal_width = lambda: 10  # type: ignore[method-assign]

    app._write_block("hello", tui.USER_STYLE)

    rendered = out.getvalue()
    assert rendered.count("\x1b[?7l") == 3
    assert rendered.count("\x1b[2K") == 3
    assert rendered.count("\x1b[?7h") == 3


def test_assistant_transcript_block_does_not_clear_terminal_background() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._terminal_width = lambda: 10  # type: ignore[method-assign]
    app._write("stale cursor column")

    app._write_block("hello", tui.ASSISTANT_STYLE)

    rendered = out.getvalue()
    assert "stale cursor column\r" in rendered
    assistant_start = rendered.index("stale cursor column") + len("stale cursor column")
    assistant_render = rendered[assistant_start:]
    assert f"\r{tui.ASSISTANT_STYLE} hello{tui.RESET}\n" in assistant_render
    assert "\x1b[2K" not in assistant_render
    assert "\x1b[K" not in assistant_render


def test_styled_terminal_line_clears_then_fills_styled_background() -> None:
    rendered = tui._styled_terminal_line("hi", tui.PROMPT_STYLE, 8)

    assert rendered.startswith("\r")
    assert f"\x1b[0m\x1b[2K{tui.PROMPT_STYLE}" in rendered
    assert f"{tui.PROMPT_STYLE}\x1b[2K" not in rendered
    assert "\x1b[K" in rendered
    assert _strip_ansi(rendered) == "hi"


def test_prompt_clear_resets_styles_before_erasing_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tui.shutil,
        "get_terminal_size",
        lambda _fallback: os.terminal_size((96, 10)),
    )
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=_TTYBuffer())
    app._terminal_width = lambda: 24  # type: ignore[method-assign]
    live = tui._LiveTerminal(app)
    live.prompt_lines = 3
    live.prompt_width = 48
    live.prompt_cursor_offset_from_bottom = 1

    sequence = live._clear_prompt_sequence(force_reflow_clear=True)

    assert sequence.startswith("\x1b[1B")
    assert "\x1b[0m\x1b[J" in sequence
    assert sequence.count("\x1b[0m\x1b[2K") == 2
    assert live.prompt_lines == 0


def test_terminal_width_allows_zoomed_terminals_below_forty_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    monkeypatch.setattr(
        tui.shutil,
        "get_terminal_size",
        lambda _fallback: os.terminal_size((28, 24)),
    )

    assert app._terminal_width() == 28


def test_styled_transcript_wraps_long_lines_without_dropping_text() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._terminal_width = lambda: 12  # type: ignore[method-assign]

    app._write_block("abcdefghijklmnopqrstuvwxyz", tui.ASSISTANT_STYLE)
    app._write_panel("status", "0123456789abcdefghijklmnopqrstuvwxyz", tui.STATUS_STYLE)

    rendered = out.getvalue()
    assert "abcdefghijk" in rendered
    assert "lmnopqrstuv" in rendered
    assert "wxyz" in rendered
    assert "nopqrstuvwxy" in rendered
    assert "xyz" in rendered
    assert "z\n" in _strip_ansi(rendered)


def test_styled_transcript_uses_soft_wrapping_for_resize_reflow() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._terminal_width = lambda: 12  # type: ignore[method-assign]

    app._write_block("abcdefghijklmnopqrstuvwxyz", tui.USER_STYLE)

    rendered = _strip_ansi(out.getvalue())
    assert " abcdefghijklmnopqrstuvwxyz\n" in rendered
    assert "abcdefghijkl\nmnopqrstuvwxyz" not in rendered


def test_transcript_rows_avoid_trailing_fill_spaces_that_reflow_on_zoom() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._terminal_width = lambda: 120  # type: ignore[method-assign]

    app._write_block("HELLO", tui.USER_STYLE)
    app._write_block("HELLO! How can I help?", tui.ASSISTANT_STYLE)

    rendered_lines = _strip_ansi(out.getvalue()).splitlines()
    assert rendered_lines == [
        "",
        " HELLO",
        "",
        "",
        " HELLO! How can I help?",
        "",
    ]
    assert all(line == line.rstrip(" ") for line in rendered_lines)
    assert " " * 20 not in _strip_ansi(out.getvalue())
    assert "\x1b[2K" in out.getvalue()


def test_assistant_markdown_renders_as_terminal_text() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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


def test_assistant_markdown_preserves_underscores_in_code_spans() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False

    app._write_block(
        "Copy to `search_resume_plan_test.md` under `design_log`.",
        tui.ASSISTANT_STYLE,
    )

    visible = _strip_ansi(out.getvalue())
    assert "search_resume_plan_test.md" in visible
    assert "design_log" in visible


def test_assistant_markdown_wraps_lists_with_continuation_indent() -> None:
    rows = tui._render_markdown_text(
        "- outer item with several words to wrap\n  - inner item that also needs wrapping",
        width=19,
    )

    assert [row.text for row in rows] == [
        "• outer item with",
        "  several words to",
        "  wrap",
        "  • inner item",
        "    that also",
        "    needs wrapping",
    ]


def test_assistant_markdown_wraps_quotes_with_quote_indent() -> None:
    rows = tui._render_markdown_text(
        "> quoted text that should wrap neatly across terminal lines",
        width=25,
    )

    assert [row.text for row in rows] == [
        "│ quoted text that",
        "│ should wrap neatly",
        "│ across terminal lines",
    ]
    assert {row.style for row in rows} == {tui.THINKING_STYLE}


def test_assistant_markdown_table_wraps_cells_without_losing_columns() -> None:
    rows = tui._render_markdown_text(
        "| User query | Time window | Matching semantics |\n"
        "|---|---:|---|\n"
        "| find my meetings at 5pm | point: 5pm | overlap/contains: "
        "meeting_start <= 5pm <= meeting_end |\n"
        "| what's my next meeting today | now to today end | start_after + top=1 "
        "+ sort asc |",
        width=60,
    )

    assert [row.text for row in rows] == [
        "User query            │ Time      │ Matching semantics",
        "                      │ window    │",
        "──────────────────────┼───────────┼────────────────────────",
        "find my meetings at   │ point:    │ overlap/contains:",
        "5pm                   │ 5pm       │ meeting_start <= 5pm <=",
        "                      │           │ meeting_end",
        "what's my next        │ now to    │ start_after + top=1 +",
        "meeting today         │ today end │ sort asc",
    ]
    assert all(
        "│" in row.text or "┼" in row.text
        for row in rows
    )


def test_assistant_markdown_table_remains_a_table_at_narrow_width() -> None:
    rows = tui._render_markdown_text(
        "| Query | Window | Match |\n"
        "|---|---|---|\n"
        "| next meeting today | now to today end | start_after top one |",
        width=32,
    )

    assert [row.text for row in rows] == [
        "Query    │ Window   │ Match",
        "─────────┼──────────┼──────────",
        "next     │ now to   │ start_aft",
        "meeting  │ today    │ er top",
        "today    │ end      │ one",
    ]


def test_assistant_markdown_highlights_fenced_code() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False

    app._write_block("```python\ndef hello():\n    return 'ok'\n```", tui.ASSISTANT_STYLE)

    rendered = out.getvalue()
    visible = _strip_ansi(rendered)
    assert "def hello():" in visible
    assert "return 'ok'" in visible
    assert tui.SYNTAX_KEYWORD_STYLE in rendered
    assert tui.SYNTAX_STRING_STYLE in rendered


def test_assistant_markdown_fence_renders_as_raw_markdown_text() -> None:
    rows = tui._render_markdown_text(
        "```markdown\n# Plan\n\n- Keep `raw_markdown` copyable\n```",
        width=40,
    )

    assert [row.text for row in rows] == [
        "# Plan",
        "",
        "- Keep `raw_markdown` copyable",
    ]
    assert {row.style for row in rows} == {tui.ASSISTANT_STYLE}
    assert all(row.ansi_text is None for row in rows)


def test_assistant_md_fence_renders_as_raw_markdown_text() -> None:
    rows = tui._render_markdown_text(
        "```md\n## Heading\n\n| A | B |\n|---|---|\n| 1 | 2 |\n```",
        width=40,
    )

    assert [row.text for row in rows] == [
        "## Heading",
        "",
        "| A | B |",
        "|---|---|",
        "| 1 | 2 |",
    ]


def test_assistant_markdown_requires_matching_code_fence_length() -> None:
    rows = tui._render_markdown_text("````python\n```\nprint(1)\n````", width=40)

    assert [row.text for row in rows] == [
        "    ```",
        "    print(1)",
    ]


def test_assistant_markdown_keeps_nonclosing_fence_text_in_code_block() -> None:
    rows = tui._render_markdown_text("````\n````not close\nprint(1)\n````", width=40)

    assert [row.text for row in rows] == [
        "    ````not close",
        "    print(1)",
    ]


def test_assistant_markdown_syntax_spans_reset_bold_between_tokens() -> None:
    row = tui._render_code_line('return "ok"', language="python")

    assert row.ansi_text is not None
    assert f"{tui.SYNTAX_KEYWORD_STYLE}return{tui.RESET}{tui.TOOL_PREVIEW_STYLE} " in (
        row.ansi_text
    )


def test_assistant_markdown_wraps_long_unbroken_words() -> None:
    rows = tui._render_markdown_text(
        "- pneumonoultramicroscopicsilicovolcanoconiosis",
        width=20,
    )

    assert [row.text for row in rows] == [
        "• pneumonoultramicr",
        "  oscopicsilicovolc",
        "  anoconiosis",
    ]


def test_user_transcript_wraps_words_with_leading_message_margin() -> None:
    rows = tui._render_user_transcript_rows("hello world", width=10)

    assert [row.text for row in rows] == ["hello", "world"]
    assert {row.style for row in rows} == {tui.USER_STYLE}


def test_user_transcript_preserves_leading_whitespace_when_wrapping() -> None:
    rows = tui._render_user_transcript_rows("  hello world", width=10)

    assert [row.text for row in rows] == ["  hello", "world"]


def test_user_transcript_keeps_long_unbroken_words_for_terminal_soft_wrap() -> None:
    rows = tui._render_user_transcript_rows("abcdefghijklmnopqrstuvwxyz", width=12)

    assert [row.text for row in rows] == ["abcdefghijklmnopqrstuvwxyz"]


def test_prompt_path_scanner_ignores_unknown_tilde_user() -> None:
    text = "You can explore codex in ~Traceback"

    assert tui._path_references_from_text(text) == []

    rendered = tui._image_placeholder_prompt_render(
        tui._PromptInputRender(text=text, cursor=len(text))
    )
    assert rendered.text == text
    assert rendered.cursor == len(text)


def test_running_terminal_line_animates_without_changing_text() -> None:
    first = tui._running_terminal_line(" running bash - pytest", 28, frame=0)
    second = tui._running_terminal_line(" running bash - pytest", 28, frame=6)
    bright = tui._running_terminal_line(" running bash - pytest", 28, frame=4)

    assert first != second
    assert "\x1b[40;" not in first
    assert "\x1b[38;5;51;1m" in bright
    assert "\x1b[38;5;255m" in first
    assert tui.STATUS_STYLE not in first
    assert _strip_ansi(first) == " running bash - pytest      "
    assert _strip_ansi(second) == " running bash - pytest      "


def test_flower_working_status_renders_shape_with_gradient() -> None:
    flower, text = tui._flower_working_status(0)

    rendered = tui._running_terminal_line(f" {text}", 30, frame=0, flower=flower)

    assert flower.name == "wattle"
    assert text == "✿ wattling..."
    assert "\x1b[38;5;227;1m✿" in rendered
    assert _strip_ansi(rendered) == " ✿ wattling...                "


def test_active_tool_terminal_line_animates_only_marker() -> None:
    first = tui._active_tool_terminal_line("running bash - pytest", 32, frame=0)
    second = tui._active_tool_terminal_line("running bash - pytest", 32, frame=2)

    assert first != second
    assert _strip_ansi(first) == " • running bash - pytest        "
    assert _strip_ansi(second) == " • running bash - pytest        "
    first_marker = first.index("•")
    second_marker = second.index("•")
    assert first[:first_marker] != second[:second_marker]
    assert first[first_marker + 1 :] == second[second_marker + 1 :]


def test_append_terminal_output_replaces_carriage_return_line() -> None:
    output, at_line_start = tui._append_terminal_output("", "Progress: 21.9% ETA: 48s\r")
    assert output == "Progress: 21.9% ETA: 48s"
    assert at_line_start

    output, at_line_start = tui._append_terminal_output(
        output,
        "Progress: 22.0% ETA: 47s\r",
        cursor_at_line_start=at_line_start,
    )
    assert output == "Progress: 22.0% ETA: 47s"
    assert at_line_start

    output, at_line_start = tui._append_terminal_output(
        output,
        "\ndone\n",
        cursor_at_line_start=at_line_start,
    )
    assert output == "Progress: 22.0% ETA: 47s\ndone\n"
    assert not at_line_start


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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.working_started_at = 88.0

    live._draw_prompt()

    rendered = out.getvalue()
    assert tui.PROMPT_STYLE in rendered
    assert "\x1b[40;" not in rendered[: rendered.index(tui.PROMPT_STYLE)]
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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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


def test_live_prompt_reminds_stop_for_running_background_tasks(tmp_path: Path) -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app.runtime = WattleRuntime(root=tmp_path)
    log_path = app.runtime.tasks.jobs_dir / "background.log"
    log_path.write_text("")
    app.runtime.tasks.register_shell_task(
        command="sleep 30",
        pid=12345,
        pgid=12345,
        log_path=log_path,
    )
    live = tui._LiveTerminal(app)

    live._draw_prompt()

    visible = _strip_ansi(out.getvalue())
    assert "Background · 1 running task; type /stop to stop all" in visible
    assert "/stop stops background tasks" in visible


def test_live_stop_command_stops_background_tasks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app.runtime = WattleRuntime(root=tmp_path)
    log_path = app.runtime.tasks.jobs_dir / "background.log"
    log_path.write_text("")
    task = app.runtime.tasks.register_shell_task(
        command="sleep 30",
        pid=12345,
        pgid=12345,
        log_path=log_path,
    )
    killed_pgids: list[int] = []

    def fake_killpg(pgid: int, sig: int) -> None:
        if sig == 0:
            raise ProcessLookupError
        killed_pgids.append(pgid)

    monkeypatch.setattr("wattle.runtime.os.killpg", fake_killpg)
    live = tui._LiveTerminal(app)
    live.buffer = "/stop"
    live.cursor = len(live.buffer)

    live._submit_buffer()

    assert killed_pgids == [12345]
    assert app.runtime.tasks.snapshot(task.task_id)["status"] == TaskStatus.KILLED.value  # type: ignore[index]
    assert "Stopped 1 background task(s)." in _strip_ansi(out.getvalue())
    assert app.messages == []


def test_live_running_status_redraw_does_not_repaint_input_box() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    assert "\x1b[1B\x1b[?25l\r\x1b[0m\x1b[J" in rendered
    assert "\x1b[1A\r\x1b[0m\x1b[2K" in rendered
    visible_lines = _strip_ansi(rendered).splitlines()
    input_line_index = next(
        index for index, line in enumerate(visible_lines) if line.startswith(" > ")
    )
    assert visible_lines[input_line_index + 1].startswith("   ")
    assert live.prompt_width == 9


def test_live_prompt_survives_repeated_zoom_in_and_out_without_black_input_gap() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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


def test_live_resize_repaints_screen_once_with_last_worked_duration() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    widths = [34]
    app._terminal_width = lambda: widths[-1]  # type: ignore[method-assign]
    live = tui._LiveTerminal(app)
    live.last_worked_duration_text = "Worked for 1s"
    live._draw_prompt()

    out.seek(0)
    out.truncate(0)
    widths.append(52)
    live._draw_prompt(force_reflow_clear=True)

    rendered = out.getvalue()
    visible = _strip_ansi(rendered)
    assert tui.VISIBLE_SCREEN_CLEAR in rendered
    assert tui.TERMINAL_HISTORY_CLEAR in rendered
    assert visible.count("Worked for 1s") == 1
    assert live.prompt_width == 52


def test_live_new_worker_replaces_previous_worked_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeThread:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(tui.threading, "Thread", FakeThread)
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.last_worked_duration_text = "Worked for 1s"

    live._start_worker()

    assert live.last_worked_duration_text is None
    assert live.streaming is True


def test_live_prompt_box_switches_to_type_box_when_streaming_with_input() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    assert "press Enter to queue after next tool call" in visible
    assert "Tab for next turn" in visible
    assert visible.index(" > queued text") < visible.index(
        "press Enter to queue after next tool call"
    )
    assert visible.index("press Enter to queue after next tool call") > visible.index(
        " > queued text"
    )
    assert "gpt-5.5 |" not in visible


def test_live_prompt_cursor_overlays_character_without_shifting_text() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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


def test_live_prompt_wraps_whole_words_to_next_input_line() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._statusline_enabled = False
    app._terminal_width = lambda: 10  # type: ignore[method-assign]
    live = tui._LiveTerminal(app)
    live.buffer = "hello world"
    live.cursor = len(live.buffer)

    live._draw_prompt()

    rendered = out.getvalue()
    assert " > hello " in rendered
    assert "   world" in rendered
    assert " > hello w" not in rendered
    assert "   orld" not in rendered
    assert live.prompt_lines == 4
    assert "\x1b[8C\x1b[?25h" in rendered


def test_live_prompt_growing_input_overwrites_without_full_clear() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._statusline_enabled = False
    live = tui._LiveTerminal(app)
    pasted = "Instruction\n" + ("Set up service.\n" * 40)

    live._insert_pasted_text(pasted)
    live._draw_prompt()

    rendered = out.getvalue()
    assert f"[Pasted Content {len(pasted)} chars]" in rendered
    assert "Set up service." not in rendered
    assert live.buffer == pasted
    assert live.cursor == len(pasted)


def test_live_prompt_keeps_pasted_placeholder_after_space_backspace() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._statusline_enabled = False
    live = tui._LiveTerminal(app)
    read_fd, write_fd = os.pipe()
    pasted = "Instruction\n" + ("Set up service.\n" * 40)
    try:
        live.fd = read_fd
        os.write(write_fd, f"\x1b[200~{pasted}\x1b[201~ \x7f".encode())
        live._read_available_input()
    finally:
        os.close(read_fd)
        os.close(write_fd)

    rendered = out.getvalue()
    assert f"[Pasted Content {len(pasted)} chars]" in rendered
    assert "Set up service." not in rendered
    assert live.buffer == pasted
    assert live.cursor == len(pasted)
    assert live.pasted_ranges == [(0, len(pasted))]


def test_live_prompt_keeps_pasted_placeholder_after_newline_backspace() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._statusline_enabled = False
    live = tui._LiveTerminal(app)
    pasted = "Instruction\n" + ("Set up service.\n" * 40)

    live._insert_pasted_text(pasted)
    live._insert_text("\n")
    live.buffer = live.buffer[: live.cursor - 1] + live.buffer[live.cursor :]
    live.pasted_ranges = tui._delete_pasted_ranges(
        live.pasted_ranges,
        start=live.cursor - 1,
        deleted_length=1,
    )
    live.cursor -= 1
    live._draw_prompt()

    rendered = out.getvalue()
    assert f"[Pasted Content {len(pasted)} chars]" in rendered
    assert "Set up service." not in rendered
    assert live.buffer == pasted
    assert live.cursor == len(pasted)
    assert live.pasted_ranges == [(0, len(pasted))]


def test_live_prompt_shows_short_multiline_pasted_content() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._statusline_enabled = False
    live = tui._LiveTerminal(app)
    pasted = "Instruction\nSet up service."

    live._insert_pasted_text(pasted)
    live._draw_prompt()

    rendered = out.getvalue()
    assert "[Pasted Content" not in rendered
    assert "Instruction" in rendered
    assert "Set up service." in rendered
    assert live.buffer == pasted
    assert live.cursor == len(pasted)


def test_live_prompt_clear_accounts_for_terminal_zoom_in() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    assert "\x1b[2B\x1b[?25l\r\x1b[0m\x1b[J" in rendered
    assert "\x1b[1A\r\x1b[0m\x1b[2K" in rendered
    assert " > " in rendered
    assert live.prompt_width == 10


def test_live_prompt_clear_accounts_for_cursor_line_reflow_after_zoom_in() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    assert "\x1b[1B\x1b[?25l\r\x1b[0m\x1b[J" in rendered
    assert "\x1b[1A\r\x1b[0m\x1b[2K" in rendered
    assert " > abc" in rendered
    assert live.prompt_width == 8


def test_live_prompt_redraw_flushes_one_frame() -> None:
    out = _FlushingTTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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


def test_welcome_card_includes_version_after_title() -> None:
    out = io.StringIO()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)

    app._write_welcome_card()

    rendered = out.getvalue()
    assert f"Wattle Agent v{get_wattle_version()}" in rendered


def test_welcome_card_keeps_compact_width() -> None:
    out = io.StringIO()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._terminal_width = lambda: 120  # type: ignore[method-assign]

    app._write_welcome_card()

    first_line = out.getvalue().splitlines()[0]
    assert len(first_line) == 36


@pytest.mark.parametrize("styled", [False, True])
@pytest.mark.parametrize("terminal_width", [2, 4, 5, 14])
def test_welcome_card_truncates_title_to_narrow_terminal(
    styled: bool,
    terminal_width: int,
) -> None:
    out: io.StringIO | _TTYBuffer = _TTYBuffer() if styled else io.StringIO()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = not styled
    app._terminal_width = lambda: terminal_width  # type: ignore[method-assign]

    app._write_welcome_card()

    lines = [ANSI_RE.sub("", line) for line in out.getvalue().splitlines()]
    if terminal_width <= 5:
        assert len(lines) == 1
        assert len(lines[0]) == terminal_width
    else:
        assert all(len(line) == terminal_width - 2 for line in lines)
        assert "Wattle" in lines[1]


def test_live_prompt_shows_slash_command_hints_for_prefix() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    skill_path = project / ".wattle" / "skills" / "hello_world" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text(
        "---\ndescription: Say hello.\n---\nbody",
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.chdir(project)
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    skill_path = project / ".wattle" / "skills" / "hello_world" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("Say hello before doing the task.", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.chdir(project)
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    assert "Use the Wattle skill 'hello_world'" in block.text
    assert "User task:\nwrite docs" in block.text


def test_live_tab_completes_selected_skill_without_submitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    skill_path = project / ".wattle" / "skills" / "hello_world" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("Say hello before doing the task.", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.chdir(project)
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/he"
    live.cursor = len(live.buffer)

    live._submit_buffer()

    rendered = out.getvalue()
    assert "Commands:" in rendered
    assert "/model [name]" in rendered
    assert app.messages == []


def test_live_voice_command_toggles_dictation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tui, "resolve_voice_dictation_config", lambda: object())
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)

    app._handle_slash("/voice")

    assert app._voice_dictation_enabled is True
    assert "Voice dictation enabled" in out.getvalue()

    app._handle_slash("/voice off")

    assert app._voice_dictation_enabled is False
    assert "Voice dictation disabled" in out.getvalue()


def test_live_voice_command_requires_voice_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_config() -> object:
        raise tui.VoiceDictationError(
            "Set WATTLE_VOICE_DICTATION_API_KEY to a non-empty OpenAI API key."
        )

    monkeypatch.setattr(tui, "resolve_voice_dictation_config", missing_config)
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)

    app._handle_slash("/voice")

    assert app._voice_dictation_enabled is False
    assert "WATTLE_VOICE_DICTATION_API_KEY" in out.getvalue()


def test_live_voice_idle_hint_uses_existing_statusline_row() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    live = tui._LiveTerminal(app)
    normal_frame = live._build_prompt_frame()

    app._voice_dictation_enabled = True
    voice_frame = live._build_prompt_frame()

    assert len(voice_frame.rows) == len(normal_frame.rows)
    assert "Voice · hold Space to dictate" in "\n".join(voice_frame.rows)
    assert " > " in voice_frame.rows[1]


def test_live_voice_space_tap_preserves_normal_space() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._voice_dictation_enabled = True
    live = tui._LiveTerminal(app)
    live.buffer = "hello"
    live.cursor = len(live.buffer)

    live._handle_voice_space_key()
    live._finalize_voice_space_before_non_space_key()
    live._insert_text("world")

    assert live.buffer == "hello world"
    assert live.cursor == len(live.buffer)


def test_live_voice_hold_transcribes_into_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamps = iter([0.0, 0.3, 0.6])
    monkeypatch.setattr(tui.time, "monotonic", lambda: next(timestamps))
    monkeypatch.setattr(tui, "resolve_voice_dictation_config", lambda: object())
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"wav")

    class FakeRecorder:
        def start(self) -> None:
            pass

        def stop_to_wav(self) -> Path:
            return audio_path

        def discard(self) -> None:
            pass

    class ImmediateThread:
        def __init__(self, *, target: Any, args: tuple[Any, ...], daemon: bool) -> None:
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self) -> None:
            self.target(*self.args)

    monkeypatch.setattr(tui, "MicrophoneRecorder", FakeRecorder)
    monkeypatch.setattr(tui, "transcribe_audio_file", lambda _path: "dictated text")
    monkeypatch.setattr(tui.threading, "Thread", ImmediateThread)
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._voice_dictation_enabled = True
    live = tui._LiveTerminal(app)
    live.buffer = "prefix"
    live.cursor = len(live.buffer)

    live._handle_voice_space_key()
    live._handle_voice_space_key()
    assert live._voice_recording is True

    live._update_voice_hold_state()
    live._drain_events()

    assert live._voice_recording is False
    assert live._voice_transcribing is False
    assert live.buffer == "prefix dictated text"
    assert live.cursor == len(live.buffer)
    assert not audio_path.exists()


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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    assert "/login  authenticate a provider" not in rendered


def test_live_prompt_shows_statusline_picker_for_statusline_command() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/statusline"
    live.cursor = len(live.buffer)

    live._draw_prompt()

    rendered = _strip_ansi(out.getvalue())
    assert " > /statusline" in rendered
    assert "Use ↑/↓ to move, x to select/deselect, Enter to confirm." in rendered
    assert "> [x] model" in rendered
    assert "  [x] thinking" in rendered
    assert "  [x] cwd" in rendered
    assert "  [ ] context_used" in rendered
    assert "/statusline  choose fields in status line at bottom" not in rendered


def test_live_prompt_shows_auto_compacting_status() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/login"
    live.cursor = len(live.buffer)

    completed = live._complete_selected_hint()

    assert completed is True
    assert live.buffer == "/login openai-codex"
    assert live.cursor == len(live.buffer)
    assert app.messages == []


def test_live_statusline_picker_x_toggles_selected_field() -> None:
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=_TTYBuffer())
    live = tui._LiveTerminal(app)
    live.buffer = "/statusline"
    live.cursor = len(live.buffer)

    live._ensure_statusline_picker_fields()
    live._toggle_statusline_picker_selection()
    live._move_statusline_picker_selection(3)
    live._toggle_statusline_picker_selection()

    assert live.statusline_picker_fields == ("thinking", "context_remaining", "cwd")


def test_live_statusline_picker_enter_applies_and_persists_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(settings.SETTINGS_PATH_ENV, str(tmp_path / "settings.json"))
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/statusline"
    live.cursor = len(live.buffer)
    live._ensure_statusline_picker_fields()
    live._toggle_statusline_picker_selection()
    live._move_statusline_picker_selection(3)
    live._toggle_statusline_picker_selection()

    live._submit_buffer()

    assert live.buffer == ""
    assert app._statusline_fields == ("thinking", "context_remaining", "cwd")
    assert settings.load_settings().tui.statusline == (
        "thinking",
        "context_remaining",
        "cwd",
    )
    assert "Statusline fields: thinking, context_remaining, cwd" in out.getvalue()


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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    monkeypatch.setattr("wattle.cli._build_provider", lambda _name: anthropic_provider)
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    assert "gpt-5.5" in rendered


def test_live_stream_chunk_keeps_prompt_stable_until_completion() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    live.events.put((live.active_turn_id, "complete", (response, None)))
    live._drain_events()

    rendered = out.getvalue()
    assert "Hello" in rendered
    assert " > " in rendered


def test_live_provider_error_flushes_partial_output_and_keeps_prompt_usable() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.active_turn_id = 3
    live.streaming = True
    live.worker = threading.Thread(target=lambda: None)
    live.active_tool_status = "running bash - pytest"
    live.inflight_tool_results = [
        ToolResultBlock(tool_use_id="call_1", content="completed before failure")
    ]

    live.events.put((3, "stream", TextDelta(text="partial output")))
    live.events.put((3, "error", (RuntimeError("Codex error: policy warning"), None)))
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
    assert live.inflight_tool_results == []


def test_live_finished_turn_clears_stale_running_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tui.time, "monotonic", lambda: 20.0)
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.turn_started_at = 10.0
    live.working_started_at = 10.0
    live.active_tool_status = "running bash - pytest"

    live._finish_response(
        CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")
    )
    out.seek(0)
    out.truncate(0)
    live._draw_prompt()

    rendered = _strip_ansi(out.getvalue())
    assert live.streaming is False
    assert live.active_tool_status is None
    assert live.working_started_at is None
    assert "Worked for 10s" in rendered
    assert "running bash - pytest" not in rendered
    assert "wattling..." not in rendered
    assert "press esc to interrupt" not in rendered


def test_live_complete_applies_compaction_state_before_finishing_response() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    state = tui.RuntimeCompaction(
        summary="summary",
        summarized_until=10,
        first_kept_index=10,
    )
    response = CompletionResponse(
        content=[TextBlock(text="done")],
        stop_reason="end_turn",
    )
    seen_state: list[object] = []

    def finish_response(_response: CompletionResponse) -> None:
        seen_state.append(app._compaction_state)
        live.streaming = False

    live._finish_response = finish_response  # type: ignore[method-assign]

    live.events.put((live.active_turn_id, "complete", (response, state)))
    live._drain_events()

    assert seen_state == [state]
    assert app._compaction_state == state


def test_live_prompt_shows_queued_messages_with_interrupt_hint() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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


def test_live_prompt_shows_end_turn_queue_in_separate_panel(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"fake-png-first")
    second.write_bytes(b"fake-png-second")
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.pending_user_inputs = [f"after tool {first}"]
    live.turn_followup_user_inputs = [f"after turn {second}"]

    live._draw_prompt()

    rendered = out.getvalue()
    assert "Messages to be submitted after next tool call" in rendered
    assert "Messages to be submitted after assistant turn completes" in rendered
    assert rendered.index("Messages to be submitted after next tool call") < rendered.index(
        "Messages to be submitted after assistant turn completes"
    )
    assert "↳ after tool [IMAGE #1]" in rendered
    assert "↳ after turn [IMAGE #2]" in rendered
    assert str(first) not in rendered
    assert str(second) not in rendered


def test_subagent_selector_rows_render_required_vertical_contract() -> None:
    rows = tui._agent_selector_row_texts(
        active_focus_id="subagent-456",
        visible_subagents=[
            {
                "subagent_id": "subagent-123",
                "display_name": "Hopper",
                "role": "explorer",
                "status": "running",
            },
            {
                "subagent_id": "subagent-456",
                "display_name": "Grace",
                "role": "worker",
                "status": "pending",
            },
        ],
        width=80,
    )

    assert rows == [
        "○ main",
        "○ Hopper explorer running",
        "▸ Grace worker pending",
    ]
    assert all("Agents:" not in row for row in rows)
    assert all("input" not in row for row in rows)


def test_subagent_selector_rows_truncate_long_rows() -> None:
    rows = tui._agent_selector_row_texts(
        active_focus_id="main",
        visible_subagents=[
            {
                "subagent_id": "subagent-123",
                "display_name": "VeryLongSubagentName",
                "role": "explorer",
                "status": "running",
            }
        ],
        width=12,
    )

    assert rows == ["▸ main", "○ VeryLon..."]


def test_visible_subagent_snapshots_filter_terminal_and_keep_launch_order() -> None:
    snapshots = [
        {"subagent_id": "closed", "status": "closed", "launch_index": 0},
        {"subagent_id": "second", "status": "running", "launch_index": 2},
        {"subagent_id": "first", "status": "pending", "launch_index": 1},
        {"subagent_id": "failed", "status": "failed", "launch_index": 3},
    ]

    visible = tui._visible_subagent_snapshots(snapshots)

    assert [snapshot["subagent_id"] for snapshot in visible] == ["first", "second"]


def test_subagent_view_header_matches_contract() -> None:
    assert (
        tui._subagent_view_header(
            {"display_name": "Hopper", "role": "explorer", "subagent_id": "subagent-123"}
        )
        == "── subagent: Hopper explorer ──"
    )


def test_live_prompt_shows_vertical_subagent_selector_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
                    "status": "completed",
                    "task": "Inspect the prompt state",
                    "result": "TUI input path identified",
                },
                {
                    "subagent_id": "subagent-234",
                    "display_name": "Katherine",
                    "role": "explorer",
                    "status": "failed",
                    "task": "Inspect failing review setup",
                    "error": "PermissionError: local codex denied",
                },
                {
                    "subagent_id": "subagent-456",
                    "display_name": "Grace",
                    "role": "explorer",
                    "status": "running",
                    "task": "Inspect tool flows",
                },
                {
                    "subagent_id": "subagent-789",
                    "display_name": "Ada",
                    "role": "worker",
                    "status": "running",
                    "task": "Patch tests",
                },
            ]

    monkeypatch.setattr(app.runtime, "_subagents", FakeSubagents())

    live._draw_prompt()

    rendered = _strip_ansi(out.getvalue())
    assert "Agents:" not in rendered
    assert "Subagents ·" not in rendered
    assert "Waiting for subagent(s)" not in rendered
    assert "▸ input" not in rendered
    assert "▸ main" in rendered
    assert "○ Grace explorer running" in rendered
    assert "○ Ada worker running" in rendered
    assert rendered.index("▸ main") < rendered.index("○ Grace explorer running")
    assert rendered.index("○ Grace explorer running") < rendered.index("○ Ada worker running")
    assert any(line.rstrip().endswith("↑↓") for line in rendered.splitlines())
    assert "Hopper explorer" not in rendered
    assert "TUI input path identified" not in rendered
    assert "Katherine explorer" not in rendered
    assert "PermissionError: local codex denied" not in rendered


def test_live_arrows_select_agent_and_enter_switches_active_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app.messages = [Message(role="user", content=[TextBlock(text="main prompt")])]
    live = tui._LiveTerminal(app)

    class FakeSubagents:
        def snapshots(self) -> list[dict[str, object]]:
            return [
                {
                    "subagent_id": "subagent-123",
                    "display_name": "Hopper",
                    "role": "explorer",
                    "status": "running",
                    "messages": (
                        Message(role="user", content=[TextBlock(text="child task")]),
                    ),
                }
            ]

    monkeypatch.setattr(app.runtime, "_subagents", FakeSubagents())
    read_fd, write_fd = os.pipe()
    try:
        live.fd = read_fd
        live._draw_prompt()
        assert app._active_focus_id == tui.MAIN_AGENT_ID
        assert app._active_agent_id == tui.MAIN_AGENT_ID
        rendered = _strip_ansi(out.getvalue())
        assert "▸ input" not in rendered
        assert "▸ main" in rendered
        assert "○ Hopper explorer running" in rendered

        out.seek(0)
        out.truncate(0)
        os.write(write_fd, b"\x1b[B")
        live._read_available_input()
        assert app._active_focus_id == "subagent-123"
        assert app._active_agent_id == tui.MAIN_AGENT_ID
        rendered = _strip_ansi(out.getvalue())
        assert "input" not in rendered
        assert "○ main" in rendered
        assert "▸ Hopper explorer running" in rendered
        assert "── subagent: Hopper explorer ──" not in rendered
        assert "child task" not in rendered

        out.seek(0)
        out.truncate(0)
        os.write(write_fd, b"\n")
        live._read_available_input()
        assert app._active_focus_id == "subagent-123"
        assert app._active_agent_id == "subagent-123"
        rendered = _strip_ansi(out.getvalue())
        assert "○ main" in rendered
        assert "▸ Hopper explorer running" in rendered
        assert "── subagent: Hopper explorer ──" in rendered
        assert "child task" in rendered

        out.seek(0)
        out.truncate(0)
        os.write(write_fd, b"\x1b[A")
        live._read_available_input()
        assert app._active_focus_id == tui.MAIN_AGENT_ID
        assert app._active_agent_id == "subagent-123"
        rendered = _strip_ansi(out.getvalue())
        assert "▸ main" in rendered
        assert "○ Hopper explorer running" in rendered
        assert "── subagent: Hopper explorer ──" not in rendered

        out.seek(0)
        out.truncate(0)
        os.write(write_fd, b"\n")
        live._read_available_input()
        assert app._active_focus_id == tui.MAIN_AGENT_ID
        assert app._active_agent_id == tui.MAIN_AGENT_ID
        rendered = _strip_ansi(out.getvalue())
        assert "▸ main" in rendered
        assert "○ Hopper explorer running" in rendered
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_live_subagent_running_input_is_rejected_without_queueing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._active_agent_id = "subagent-123"
    live = tui._LiveTerminal(app)
    live.buffer = "please change course"
    live.cursor = len(live.buffer)

    class FakeSubagents:
        def snapshots(self) -> list[dict[str, object]]:
            return [
                {
                    "subagent_id": "subagent-123",
                    "display_name": "Hopper",
                    "role": "explorer",
                    "status": "running",
                }
            ]

        def send_input(self, subagent_id: str, message: str) -> object:  # pragma: no cover
            raise AssertionError("send_input should not be called for running subagents")

    monkeypatch.setattr(app.runtime, "_subagents", FakeSubagents())

    live._submit_buffer()

    rendered = _strip_ansi(out.getvalue())
    assert "subagent Hopper is running; wait until it is idle before sending input." in rendered
    assert live.pending_user_inputs == []
    assert app.messages == []
    assert app._active_agent_id == "subagent-123"


def test_live_prompt_suppresses_generic_wait_agent_status_for_active_subagents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    assert "▸ input" not in rendered
    assert "▸ main" in rendered
    assert "○ Hopper explorer running" in rendered
    assert "Subagents ·" not in rendered
    assert "Waiting for subagent\n" not in rendered


def test_live_prompt_shows_queued_image_messages_as_anchors(tmp_path: Path) -> None:
    image = tmp_path / "dragged image.png"
    image.write_bytes(b"fake-png")
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    escaped_path = str(image).replace(" ", "\\ ")
    live.pending_user_inputs = [f"check {escaped_path}"]

    live._draw_prompt()

    rendered = out.getvalue()
    assert "↳ check [IMAGE #1]" in rendered
    assert str(image) not in rendered


def test_live_prompt_shows_queued_extensionless_screenshot_as_anchor(
    tmp_path: Path,
) -> None:
    image = _extensionless_screenshot_path(tmp_path)
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.pending_user_inputs = [f"check {image}"]

    live._draw_prompt()

    rendered = out.getvalue()
    assert "↳ check [IMAGE #1]" in rendered
    assert "NSIRD_screencaptureui_BRymlQ" not in rendered
    assert str(image) not in rendered


def test_live_prompt_hides_pending_monitor_events() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.pending_monitor_inputs = ["Monitor event: service ready"]

    live._draw_prompt()

    rendered = out.getvalue()
    assert "Monitor event:" not in rendered
    assert "service ready" not in rendered
    assert "Messages to be submitted after assistant turn completes" not in rendered


def test_live_queue_monitor_event_uses_hidden_queue() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    assert "Hopper [explorer] complete" in rendered
    assert "Inspect the prompt state" in rendered
    assert live.pending_monitor_inputs == []
    assert started == []


def test_live_finish_response_sends_monitor_events_without_rendering_them() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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


def test_live_queue_command_while_streaming_uses_end_turn_panel() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.buffer = "/queue after the full turn"
    live.cursor = len(live.buffer)

    live._submit_buffer()

    rendered = out.getvalue()
    assert live.pending_user_inputs == []
    assert live.turn_followup_user_inputs == ["after the full turn"]
    assert "Messages to be submitted after next tool call" not in rendered
    assert "Messages to be submitted after assistant turn completes" in rendered
    assert "↳ after the full turn" in rendered


def test_live_tab_while_streaming_queues_buffer_for_end_of_turn() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.buffer = "after the full turn"
    live.cursor = len(live.buffer)
    read_fd, write_fd = os.pipe()
    try:
        live.fd = read_fd
        os.write(write_fd, b"\t")
        live._read_available_input()
    finally:
        os.close(read_fd)
        os.close(write_fd)

    rendered = out.getvalue()
    assert live.buffer == ""
    assert live.pending_user_inputs == []
    assert live.turn_followup_user_inputs == ["after the full turn"]
    assert live.input_history[-1] == "after the full turn"
    assert "Messages to be submitted after assistant turn completes" in rendered
    assert "↳ after the full turn" in rendered


def test_live_queued_image_inputs_keep_placeholder_order(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"fake-png-first")
    second.write_bytes(b"fake-png-second")
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.pending_user_inputs = [f"first {first}", f"second {second}"]

    live._append_pending_user_message(render=True)

    message = app.messages[-1]
    assert message.content[0] == TextBlock(
        text="[queued user message 1 of 2]\nfirst [IMAGE #1]"
    )
    assert message.content[1] == TextBlock(
        text="[queued user message 2 of 2]\nsecond [IMAGE #2]"
    )
    assert [
        block.filename for block in message.content if isinstance(block, ImageBlock)
    ] == ["first.png", "second.png"]
    rendered = out.getvalue()
    assert "first [IMAGE #1]" in rendered
    assert "second [IMAGE #2]" in rendered
    assert str(first) not in rendered
    assert str(second) not in rendered


def test_live_finish_response_anchors_pending_image_input(tmp_path: Path) -> None:
    image = tmp_path / "dragged image.png"
    image.write_bytes(b"fake-png")
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    escaped_path = str(image).replace(" ", "\\ ")
    live.pending_user_inputs = [f"check {escaped_path}"]

    live._finish_response(
        CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")
    )

    rendered = out.getvalue()
    assert "check [IMAGE #1]" in rendered
    assert str(image) not in rendered
    assert [message.role for message in app.messages] == ["assistant", "user"]
    followup = app.messages[1]
    assert followup.content[0] == TextBlock(text="check [IMAGE #1]")
    assert any(
        isinstance(block, ImageBlock) and block.filename == image.name
        for block in followup.content
    )


def test_live_finish_response_frames_mid_tool_input_as_active_task_guidance() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.pending_user_inputs = ["hello"]
    started: list[bool] = []

    def fake_start_worker() -> None:
        started.append(True)

    live._start_worker = fake_start_worker  # type: ignore[method-assign]

    live._finish_response(
        CompletionResponse(
            content=[ToolUseBlock(id="call_1", name="missing_tool", input={})],
            stop_reason="tool_use",
        )
    )

    assert started == [True]
    assert live.pending_user_inputs == []
    assert [message.role for message in app.messages] == ["assistant", "user"]
    followup = app.messages[1]
    assert isinstance(followup.content[0], ToolResultBlock)
    assert isinstance(followup.content[1], TextBlock)
    assert "additional guidance for the active task" in followup.content[1].text
    assert "Continue the active task" in followup.content[1].text
    assert "[guidance message 1 of 1]\nhello" in followup.content[1].text


def test_live_end_turn_queued_input_waits_for_assistant_turn_to_finish() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.turn_followup_user_inputs = ["after the full turn"]
    starts: list[bool] = []

    def fake_start_worker() -> None:
        live.streaming = True
        starts.append(True)

    live._start_worker = fake_start_worker  # type: ignore[method-assign]
    tool_use_response = CompletionResponse(
        content=[ToolUseBlock(id="call_1", name="missing_tool", input={})],
        stop_reason="tool_use",
    )
    final_response = CompletionResponse(
        content=[TextBlock(text="done")],
        stop_reason="end_turn",
    )

    live._finish_response(tool_use_response)

    assert starts == [True]
    assert live.pending_user_inputs == []
    assert live.turn_followup_user_inputs == ["after the full turn"]
    assert [message.role for message in app.messages] == ["assistant", "user"]
    tool_followup = app.messages[1]
    assert any(isinstance(block, ToolResultBlock) for block in tool_followup.content)
    assert not any(
        isinstance(block, TextBlock) and block.text == "after the full turn"
        for block in tool_followup.content
    )

    live._finish_response(final_response)

    assert starts == [True, True]
    assert live.pending_user_inputs == []
    assert live.turn_followup_user_inputs == []
    assert [message.role for message in app.messages] == [
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert app.messages[3].content == [TextBlock(text="after the full turn")]
    assert "after the full turn" in out.getvalue()


def test_live_interrupt_current_turn_clears_transient_inflight_tool_results() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.active_turn_id = 4
    cancel_event = threading.Event()
    live.active_turn_cancel_event = cancel_event
    live.inflight_tool_results = [
        ToolResultBlock(tool_use_id="call_1", content="completed before interrupt")
    ]
    app.messages.append(Message(role="user", content=[TextBlock(text="refactor the TUI")]))

    live._reset_provider_for_interrupt = lambda: None  # type: ignore[method-assign]
    live._interrupt_current_turn()

    assert live.streaming is False
    assert live.inflight_tool_results == []
    assert live.active_turn_id == 5
    assert cancel_event.is_set()
    assert live.active_turn_cancel_event is None


def test_live_dispatch_uses_inflight_tools_after_model_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _RecordingTool()
    monkeypatch.setitem(TOOLS_BY_NAME, tool.name, tool)
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(model="gpt-5.5"), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.inflight_tools_by_name = app._available_tools()
    app.current_model = "deepseek-v4-flash"
    app._refresh_model_dependent_context()

    blocks = live._dispatch_tool_with_animated_prompt(
        ToolUseBlock(id="call_1", name="echo", input={"message": "hi"})
    )

    assert tool.calls == [{"message": "hi"}]
    assert any(
        isinstance(block, ToolResultBlock) and block.tool_use_id == "call_1" and not block.is_error
        for block in blocks
    )


def test_live_parallel_safe_dispatch_preserves_order_and_inflight_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _ParallelRecordingTool()
    monkeypatch.setitem(TOOLS_BY_NAME, tool.name, tool)
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    response = CompletionResponse(
        content=[
            ToolUseBlock(
                id="call_1",
                name=tool.name,
                input={"message": "one", "wait_for_peer": True},
            ),
            ToolUseBlock(
                id="call_2",
                name=tool.name,
                input={"message": "two", "wait_for_peer": True},
            ),
        ],
        stop_reason="tool_use",
    )

    returned = live._dispatch_tools_with_prompt(response)

    assert tool.max_active == 2
    assert [
        block.tool_use_id for block in returned if isinstance(block, ToolResultBlock)
    ] == ["call_1", "call_2"]
    assert [block.tool_use_id for block in live.inflight_tool_results] == [
        "call_1",
        "call_2",
    ]
    assert "running 2 tools" not in _strip_ansi(out.getvalue()).lower()


def test_live_inflight_resize_replay_tracks_only_tool_results() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    tool_result = ToolResultBlock(tool_use_id="call_1", content="image attached")
    image = ImageBlock(
        path="/tmp/image.png",
        media_type="image/png",
        filename="image.png",
        size_bytes=12,
    )
    response = CompletionResponse(
        content=[ToolUseBlock(id="call_1", name="image", input={})],
        stop_reason="tool_use",
    )

    live._dispatch_tool_with_animated_prompt = lambda _block: [  # type: ignore[method-assign]
        tool_result,
        image,
    ]
    returned = live._dispatch_tools_with_prompt(response)

    assert returned == [tool_result, image]
    assert live.inflight_tool_results == [tool_result]


def test_live_esc_interrupt_keeps_user_and_queued_messages_without_partial_assistant() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.pending_user_inputs = ["queued followup"]
    live.stream_text = ["partial assistant text"]
    live.inflight_tool_results = [
        ToolResultBlock(tool_use_id="call_1", content="completed before interrupt")
    ]
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
    assert live.inflight_tool_results == []
    rendered = out.getvalue()
    assert "Interrupted current turn; sending queued input now." in rendered
    assert "partial assistant text" not in rendered


def test_live_esc_interrupt_sends_queued_message_to_provider_request() -> None:
    out = _TTYBuffer()
    response = CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider([[StreamComplete(response=response)]])
    app = tui.WattleApp(_make_args(), provider, out=out)
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
    app = tui.WattleApp(_make_args(), old_provider, out=out)
    app._force_plain = False
    app.messages.append(Message(role="user", content=[TextBlock(text="previous")]))
    app.messages.append(Message(role="assistant", content=[TextBlock(text="previous answer")]))
    app.messages.append(Message(role="user", content=[TextBlock(text="refactor the TUI")]))
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.pending_user_inputs = ["steer now"]

    monkeypatch.setattr("wattle.cli._build_provider", lambda _name: new_provider)

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
    app = tui.WattleApp(_make_args(), provider, out=out)
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
    app = tui.WattleApp(_make_args(), provider, out=out)
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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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


def test_live_up_and_down_arrows_move_cursor_across_wrapped_input_rows() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._terminal_width = lambda: 10  # type: ignore[method-assign]
    live = tui._LiveTerminal(app)
    live.buffer = "abcdefghijkl"
    live.cursor = len(live.buffer)

    live._move_picker_or_history(-1)

    assert live.buffer == "abcdefghijkl"
    assert live.cursor == len("abcde")

    live._move_picker_or_history(1)

    assert live.buffer == "abcdefghijkl"
    assert live.cursor == len("abcdefghijkl")


def test_live_up_and_down_arrows_preserve_column_across_shorter_input_rows() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._terminal_width = lambda: 80  # type: ignore[method-assign]
    live = tui._LiveTerminal(app)
    live.buffer = "abcdef\nxy\n123456"
    live.cursor = len("abcdef")

    live._move_picker_or_history(1)

    assert live.cursor == len("abcdef\nxy")

    live._move_picker_or_history(1)

    assert live.cursor == len("abcdef\nxy\n123456")


def test_live_down_arrow_on_last_input_row_does_not_restore_history_draft() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._terminal_width = lambda: 10  # type: ignore[method-assign]
    live = tui._LiveTerminal(app)
    live.input_history = ["old prompt"]
    live.input_history_index = 0
    live.input_history_draft = "draft"
    live.buffer = "abcdefghijkl"
    live.cursor = len(live.buffer)

    live._move_picker_or_history(1)

    assert live.buffer == "abcdefghijkl"
    assert live.cursor == len("abcdefghijkl")
    assert live.input_history_index == 0


def test_live_up_arrow_on_first_input_row_still_navigates_history() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._terminal_width = lambda: 10  # type: ignore[method-assign]
    live = tui._LiveTerminal(app)
    live.input_history = ["old prompt"]
    live.buffer = "abcdefghijkl"
    live.cursor = len(live.buffer)

    live._move_picker_or_history(-1)
    assert live.cursor == len("abcde")

    live._move_picker_or_history(-1)

    assert live.buffer == "old prompt"
    assert live.cursor == len("old prompt")


def test_live_split_arrow_escape_navigates_input_history() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    live = tui._LiveTerminal(app)
    live.fd = 123
    old_attrs = [0, 0, 0, tui.termios.IEXTEN, 0, 0, 0]
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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "remember this"
    live.cursor = len(live.buffer)
    live._start_worker = lambda: None  # type: ignore[method-assign]

    live._submit_buffer()
    live._move_input_history(-1)

    assert live.buffer == "remember this"
    assert live.cursor == len("remember this")


def test_live_submit_exit_command_while_streaming_exits_instead_of_queueing() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.buffer = "/exit"
    live.cursor = len(live.buffer)

    live._submit_buffer()

    assert live.running is False
    assert live.pending_user_inputs == []
    assert app.messages == []


def test_live_submit_builtin_command_while_streaming_runs_instead_of_queueing() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.buffer = "/help"
    live.cursor = len(live.buffer)

    live._submit_buffer()

    assert live.running is True
    assert live.pending_user_inputs == []
    assert app.messages == []
    assert "Commands:" in out.getvalue()


def test_live_bracketed_paste_split_across_reads_preserves_newlines() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    assert live.pasted_ranges == []
    assert app.messages == []


def test_live_option_arrow_moves_cursor_by_word_for_mid_buffer_edit() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.active_turn_id = 2
    app.messages.append(Message(role="user", content=[TextBlock(text="first")]))
    stale_response = CompletionResponse(
        content=[TextBlock(text="stale final")],
        stop_reason="end_turn",
    )

    live.events.put((1, "stream", TextDelta(text="stale partial")))
    live.events.put((1, "complete", (stale_response, None)))
    live._drain_events()

    assert [message.role for message in app.messages] == ["user"]
    assert live.stream_text == []
    assert "stale partial" not in out.getvalue()
    assert "stale final" not in out.getvalue()


def test_terminal_hides_thinking_content_by_default(
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
    assert "reasoning" not in out
    assert "answer" in out
    assert "| echo ok" in out
    assert "echoed: hi" in out
    assert "done" in out


def test_terminal_streams_text_thinking_and_tool_markers_in_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
    settings_path = tmp_path / "settings.json"
    settings_path.write_text('{"tui": {"show_thinking": true}}', encoding="utf-8")
    monkeypatch.setenv(settings.SETTINGS_PATH_ENV, str(settings_path))

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
    app = tui.WattleApp(_make_args(), provider, out=out)
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
    assert "inspect child task" in rendered
    assert "subagent runtime is not configured" not in rendered
    assert "spawn_agent error" not in rendered


def test_spawn_agent_success_renders_friendly_summary() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    assert "└ Investigate reproduction surfaces for issue 3 only" in rendered
    assert "subagent_id:" not in rendered


def test_wait_agent_success_renders_finished_waiting_block() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    block = ToolUseBlock(
        id="call_1",
        name="wait_agent",
        input={"subagent_id": "subagent-123"},
    )
    result = ToolResultBlock(
        tool_use_id="call_1",
        content=(
            "subagent_id: subagent-123\n"
            "name: Hopper\n"
            "role: explorer\n"
            "status: completed\n"
            "result:\n"
            "inspected TUI rendering"
        ),
    )

    app._write_tool_result(block, result)

    rendered = _strip_ansi(out.getvalue())
    assert "Finished waiting" in rendered
    assert "Hopper [explorer]: Complete - inspected TUI rendering" in rendered
    assert "subagent_id:" not in rendered


def test_wait_agent_group_result_uses_visible_subagent_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False

    class FakeSubagents:
        def snapshots(self) -> list[dict[str, object]]:
            return [
                {
                    "subagent_id": "subagent-123",
                    "display_name": "Grace",
                    "role": "explorer",
                    "status": "completed",
                    "result": "inspected TUI rendering",
                },
                {
                    "subagent_id": "subagent-456",
                    "display_name": "Ada",
                    "role": "explorer",
                    "status": "running",
                    "task": "checking PTY coverage",
                },
                {
                    "subagent_id": "subagent-789",
                    "display_name": "Hopper",
                    "role": "worker",
                    "status": "failed",
                    "error": "failed to apply patch",
                },
            ]

    monkeypatch.setattr(app.runtime, "_subagents", FakeSubagents())
    block = ToolUseBlock(
        id="call_1",
        name="wait_agent",
        input={"subagent_id": "subagent-123"},
    )
    result = ToolResultBlock(tool_use_id="call_1", content="status: completed")

    app._write_tool_result(block, result)

    rendered = _strip_ansi(out.getvalue())
    assert "Finished waiting" in rendered
    assert "Grace [explorer]: Complete - inspected TUI rendering" in rendered
    assert "Ada [explorer]: Running - checking PTY coverage" in rendered
    assert "Hopper [worker]: Error - failed to apply patch" in rendered
    assert "subagent_id:" not in rendered


def test_wait_agent_begin_renders_visible_subagent_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False

    class FakeSubagents:
        def snapshots(self) -> list[dict[str, object]]:
            return [
                {
                    "subagent_id": "subagent-123",
                    "display_name": "Grace",
                    "role": "explorer",
                    "status": "running",
                },
                {
                    "subagent_id": "subagent-456",
                    "display_name": "Ada",
                    "role": "worker",
                    "status": "pending",
                },
            ]

    monkeypatch.setattr(app.runtime, "_subagents", FakeSubagents())

    app._write_wait_agent_begin(
        ToolUseBlock(id="call_1", name="wait_agent", input={"subagent_id": "subagent-123"})
    )

    rendered = _strip_ansi(out.getvalue())
    assert "Waiting for 2 agents" in rendered
    assert "Grace [explorer]" in rendered
    assert "Ada [worker]" in rendered


def test_update_plan_success_renders_semantic_cell_and_suppresses_tool_output() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    block = ToolUseBlock(
        id="call_1",
        name="update_plan",
        input={
            "explanation": "Switching to TUI coverage.",
            "plan": [
                {"step": "Inspect current flow", "status": "completed"},
                {"step": "Add plan renderer", "status": "in_progress"},
                {"step": "Add PTY regression", "status": "pending"},
            ],
        },
    )
    result = ToolResultBlock(tool_use_id="call_1", content="Plan updated")

    app._write_tool_result(block, result)

    rendered = out.getvalue()
    visible = _strip_ansi(rendered)
    assert "| Updated Plan" in visible
    assert "  Switching to TUI coverage." in visible
    assert "  - [x] Inspect current flow" in visible
    assert "  - [>] Add plan renderer" in visible
    assert "  - [ ] Add PTY regression" in visible
    assert "Plan updated" not in visible
    assert "update_plan ok" not in visible
    assert tui.PLAN_COMPLETED_STYLE in rendered
    assert tui.PLAN_IN_PROGRESS_STYLE in rendered
    assert tui.PLAN_PENDING_STYLE in rendered


def test_update_plan_cell_wraps_long_explanation_and_steps() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._terminal_width = lambda: 34  # type: ignore[method-assign]
    block = ToolUseBlock(
        id="call_1",
        name="update_plan",
        input={
            "explanation": "A concise note that wraps across rows",
            "plan": [
                {
                    "step": "Add semantic TUI rendering coverage",
                    "status": "in_progress",
                }
            ],
        },
    )

    app._write_tool_result(block, ToolResultBlock(tool_use_id="call_1", content="Plan updated"))

    rendered_lines = _strip_ansi(out.getvalue()).splitlines()
    assert "  A concise note that wraps across" in rendered_lines
    assert " rows" in rendered_lines
    assert "  - [>] Add semantic TUI rendering" in rendered_lines
    assert " coverage" in rendered_lines


def test_update_plan_history_replay_uses_tool_input_and_skips_generic_result() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app.messages = [
        Message(role="user", content=[TextBlock(text="do the feature")]),
        Message(
            role="assistant",
            content=[
                ToolUseBlock(
                    id="call_1",
                    name="update_plan",
                    input={
                        "plan": [
                            {"step": "Inspect flow", "status": "completed"},
                            {"step": "Render history", "status": "in_progress"},
                        ]
                    },
                )
            ],
        ),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="call_1", content="Plan updated")],
        ),
    ]

    app._write_history_transcript()

    rendered = _strip_ansi(out.getvalue())
    assert "Updated Plan" in rendered
    assert "  - [x] Inspect flow" in rendered
    assert "  - [>] Render history" in rendered
    assert "[tool use]" not in rendered
    assert "[tool result]" not in rendered
    assert "Plan updated" not in rendered


def test_unmatched_tool_use_history_replay_uses_compact_display_summary() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app.messages = [
        Message(role="user", content=[TextBlock(text="write a file")]),
        Message(
            role="assistant",
            content=[
                ToolUseBlock(
                    id="call_1",
                    name="write",
                    input={"path": "notes.txt", "content": "one\ntwo\nthree"},
                )
            ],
        ),
    ]

    app._write_history_transcript()

    rendered = _strip_ansi(out.getvalue())
    assert "write: notes.txt (3 lines)" in rendered
    assert "one" not in rendered
    assert "two" not in rendered
    assert "three" not in rendered


@pytest.mark.parametrize(
    ("tool_name", "expected"),
    [
        ("send_input", "Sent input to Hopper [explorer]"),
        ("close_agent", "Closed Hopper [explorer]"),
    ],
)
def test_subagent_housekeeping_successes_render_lifecycle_rows(
    tool_name: str,
    expected: str,
) -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    block = ToolUseBlock(
        id="call_1",
        name=tool_name,
        input={"subagent_id": "subagent-123"},
    )
    result = ToolResultBlock(
        tool_use_id="call_1",
        content="subagent_id: subagent-123\nname: Hopper\nrole: explorer\nstatus: completed",
    )

    app._write_tool_result(block, result)

    rendered = _strip_ansi(out.getvalue())
    assert expected in rendered
    assert "subagent_id:" not in rendered


def test_subagent_housekeeping_errors_remain_visible() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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


def test_terminal_yolo_allows_tool(monkeypatch: pytest.MonkeyPatch) -> None:
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

    out, _app = _drive(provider, ["use tool", "/exit"])

    assert tool.calls == [{"message": "hi"}]
    assert "[permission]" not in out
    assert "| echo ok" in out


def test_research_tool_results_aggregate_consecutive_passive_calls() -> None:
    tool_use_response = CompletionResponse(
        content=[
            ToolUseBlock(id="call_1", name="bash", input={"command": "rg Wattle README.md"}),
            ToolUseBlock(id="call_2", name="bash", input={"command": "ls src/wattle"}),
            ToolUseBlock(id="call_3", name="read", input={"path": "pyproject.toml", "limit": 2}),
        ],
        stop_reason="tool_use",
    )
    end_response = CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider(
        [
            [StreamComplete(response=tool_use_response)],
            [TextDelta(text="done"), StreamComplete(response=end_response)],
        ]
    )

    out, _app = _drive(provider, ["inspect", "/exit"])
    rendered = _strip_ansi(out)

    assert "| Researched" in rendered
    assert "└ Search Wattle" in rendered
    assert "List src/wattle" in rendered
    assert "Read pyproject.toml" in rendered
    assert "| Ran rg Wattle README.md" not in rendered
    assert "read ok - pyproject.toml" not in rendered


def test_research_results_dedupe_repeated_adjacent_summaries() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    summary = CommandSummary(CommandSummaryKind.READ_FILE, "src/wattle/tui/__init__.py")

    app._write_research_result([summary])
    app._write_research_result([summary])

    rendered = _strip_ansi(out.getvalue())
    assert rendered.count("Researched") == 1
    assert rendered.count("Read src/wattle/tui/__init__.py") == 1


def test_research_dedupe_resets_after_non_research_tool_result() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    summary = CommandSummary(CommandSummaryKind.READ_FILE, "src/wattle/tui/__init__.py")

    app._write_research_result([summary])
    app._write_tool_result(
        ToolUseBlock(id="call_1", name="bash", input={"command": "python --version"}),
        ToolResultBlock(tool_use_id="call_1", content="Python 3.12.11"),
    )
    app._write_research_result([summary])

    rendered = _strip_ansi(out.getvalue())
    assert rendered.count("Researched") == 2
    assert rendered.count("Read src/wattle/tui/__init__.py") == 2
    assert "Ran python --version" in rendered


def test_research_aggregate_flushes_before_unknown_command() -> None:
    tool_use_response = CompletionResponse(
        content=[
            ToolUseBlock(id="call_1", name="bash", input={"command": "ls src/wattle"}),
            ToolUseBlock(id="call_2", name="bash", input={"command": "python --version"}),
        ],
        stop_reason="tool_use",
    )
    end_response = CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider(
        [
            [StreamComplete(response=tool_use_response)],
            [TextDelta(text="done"), StreamComplete(response=end_response)],
        ]
    )

    out, _app = _drive(provider, ["inspect", "/exit"])
    rendered = _strip_ansi(out)

    assert "| Researched" in rendered
    assert "List src/wattle" in rendered
    assert "| Ran python --version" in rendered


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

    assert "| Ran seq 1 5" in out
    assert "  └ 1" in out
    assert "    2" in out
    assert "... +1 lines" in out
    assert "    4" in out
    assert "    5" in out


def test_terminal_bash_exec_cell_prompt_rows_show_running_output() -> None:
    cell = tui._BashExecCell(
        tool_use_id="call_1",
        command="uv run pytest tests/test_tui.py -q",
        output="line1\nline2\nline3\nline4\nline5\nline6",
    )

    rendered = "\n".join(
        tui._bash_exec_cell_prompt_rows(cell, width=80, styles_enabled=False)
    )
    plain = _strip_ansi(rendered)

    assert "| Running uv run pytest tests/test_tui.py -q" in plain
    assert "  └ line2" in plain
    assert "    line6" in plain
    assert "line1" not in plain


def test_terminal_bash_live_progress_replaces_carriage_return_line() -> None:
    cell = tui._BashExecCell(
        tool_use_id="call_1",
        command="python train.py",
    )
    for chunk in [
        "Progress: 21.9% Trials: 2 ETA: 48s\r",
        "Progress: 22.0% Trials: 2 ETA: 47s\r",
        "Progress: 22.1% Trials: 2 ETA: 46s\r",
    ]:
        output, cursor_at_line_start = tui._append_terminal_output(
            cell.output,
            chunk,
            cursor_at_line_start=cell.cursor_at_line_start,
        )
        cell.output = output
        cell.cursor_at_line_start = cursor_at_line_start

    rendered = "\n".join(
        tui._bash_exec_cell_prompt_rows(cell, width=80, styles_enabled=False)
    )
    plain = _strip_ansi(rendered)

    assert "Progress: 22.1% Trials: 2 ETA: 46s" in plain
    assert "Progress: 21.9%" not in plain
    assert "Progress: 22.0%" not in plain


def test_terminal_bash_tool_externalized_output_hides_metadata() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = True
    block = ToolUseBlock(id="call_1", name="bash", input={"command": "uv run pytest"})
    result = ToolResultBlock(
        tool_use_id="call_1",
        content=(
            "[output truncated: 12000 chars]\n"
            "full_output_path: /tmp/wattle-output.txt\n"
            "full_output_chars: 12000\n"
            "excerpt_chars: 1000\n"
            "omitted_chars: 11000\n"
            "[excerpt]\n"
            "============================= test session starts ==============================\n"
            "platform darwin -- Python 3.12.11\n"
            "[... omitted 11000 chars; see full_output_path ...]\n"
            "tests/test_tui.py::test_example PASSED\n"
            "============================= 437 passed in 28.06s =============================\n"
            "[/excerpt]"
        ),
    )

    app._write_tool_result(block, result)

    rendered = _strip_ansi(out.getvalue())
    assert "| Ran uv run pytest" in rendered
    assert "test session starts" in rendered
    assert "437 passed in 28.06s" in rendered
    assert "full_output_path:" not in rendered
    assert "excerpt_chars:" not in rendered


def test_terminal_bash_timeout_card_includes_metadata_and_chain_hint() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = True
    command = "pwd && git status --short --branch --untracked-files=all && git remote -v"
    block = ToolUseBlock(
        id="call_1",
        name="bash",
        input={"command": command, "workdir": "/repo", "timeout": 30},
    )
    result = ToolResultBlock(
        tool_use_id="call_1",
        content="Status: timed out\nWall time: 1.22s\nRequested timeout: 30s\nExit code: unknown",
        metadata={
            "kind": "command",
            "command": command,
            "workdir": "/repo",
            "status": "timed_out",
            "exit_code": None,
            "elapsed_seconds": 1.22,
            "timeout_seconds": 30.0,
            "output_capture_stopped": True,
            "is_shell_chain": True,
        },
    )

    app._write_tool_result(block, result)

    rendered = _strip_ansi(out.getvalue())
    assert "| Ran 3 shell commands in /repo" in rendered
    assert "Status: timed out" in rendered
    assert "Wall time: 1.22s" in rendered
    assert "Requested timeout: 30s" in rendered
    assert "Exit code: unknown" in rendered
    assert "Output capture stopped: descendant process kept stdout/stderr open" in rendered
    assert "This was a chained command" in rendered


def test_terminal_bash_tool_title_uses_token_styles() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    block = ToolUseBlock(
        id="call_1",
        name="bash",
        input={
            "command": "git diff -- src/wattle/tui/__init__.py",
        },
    )
    result = ToolResultBlock(
        tool_use_id="call_1",
        content="diff --git a/src/wattle/tui/__init__.py b/src/wattle/tui/__init__.py",
    )

    app._write_tool_result(block, result)

    rendered = out.getvalue()
    plain = _strip_ansi(rendered)
    assert "Ran git diff -- src/wattle/tui/__init__.py" in plain
    assert f"{tui.COMMAND_EXEC_STYLE}git{tui.RESET}" in rendered
    assert f"{tui.COMMAND_OPTION_STYLE}--{tui.RESET}" in rendered
    assert f"{tui.COMMAND_PATH_STYLE}src/wattle/tui/__init__.py{tui.RESET}" in rendered


def test_terminal_bash_chain_title_uses_summary_without_shell_token_styles() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    block = ToolUseBlock(
        id="call_1",
        name="bash",
        input={
            "command": "git diff -- src/wattle/tui/__init__.py | sed -n '1,20p'",
        },
    )
    result = ToolResultBlock(
        tool_use_id="call_1",
        content="diff --git a/src/wattle/tui/__init__.py b/src/wattle/tui/__init__.py",
    )

    app._write_tool_result(block, result)

    rendered = out.getvalue()
    plain = _strip_ansi(rendered)
    assert "Ran 2 shell commands" in plain
    assert "Ran git diff -- src/wattle/tui/__init__.py | sed -n '1,20p'" not in plain
    assert f"{tui.TOOL_TITLE_STYLE}Ran{tui.RESET} 2 shell commands" in rendered
    assert f"{tui.COMMAND_EXEC_STYLE}2{tui.RESET}" not in rendered


def test_terminal_bash_live_events_update_prompt_frame() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = True
    live = tui._LiveTerminal(app)
    live.streaming = True

    live._tool_event_queue.put(
        tui.ToolRunEvent("call_1", "bash", "started", "python -c 'print(1)'")
    )
    live._tool_event_queue.put(tui.ToolRunEvent("call_1", "bash", "output", "hello\n"))
    assert live._drain_tool_events()
    frame = live._build_prompt_frame()
    plain = _strip_ansi("\n".join(frame.rows))

    assert "| Running python -c 'print(1)'" in plain
    assert "  └ hello" in plain
    assert " > " in plain


def test_live_shell_mode_prompt_noops_bare_bang_and_runs_without_provider() -> None:
    out = _TTYBuffer()
    provider = _ScriptedStreamProvider([])
    app = tui.WattleApp(_make_args(), provider, out=out)
    app._force_plain = True
    live = tui._LiveTerminal(app)

    live.buffer = "!echo shell-mode"
    live.cursor = len(live.buffer)
    frame = live._build_prompt_frame()
    plain_frame = _strip_ansi("\n".join(frame.rows))
    assert " ! echo shell-mode" in plain_frame
    assert " ! !echo shell-mode" not in plain_frame
    assert "! shell mode" in plain_frame

    live.buffer = "!"
    live.cursor = 1
    live._submit_buffer()
    assert live.buffer == "!"
    assert app.messages == []
    assert provider.requests == []
    assert "Ran" not in _strip_ansi(out.getvalue())

    live.buffer = "!printf shell-mode"
    live.cursor = len(live.buffer)
    live._submit_buffer()
    rendered = _strip_ansi(out.getvalue())
    assert "Ran printf shell-mode" in rendered
    assert "shell-mode" in rendered
    assert app.messages == []
    assert provider.requests == []

    live.buffer = "!seq 1 6"
    live.cursor = len(live.buffer)
    live._submit_buffer()
    rendered = _strip_ansi(out.getvalue())
    assert "Ran seq 1 6" in rendered
    for line in range(1, 7):
        assert f"    {line}" in rendered or f"  └ {line}" in rendered
    assert "... +" not in rendered

    large_output = "z" * 25_050
    live.buffer = (
        f"!{shlex.quote(sys.executable)} -c "
        f"{shlex.quote('import sys; sys.stdout.write(chr(122) * 25050)')}"
    )
    live.cursor = len(live.buffer)
    live._submit_buffer()
    rendered = _strip_ansi(out.getvalue())
    assert "[output truncated:" not in rendered
    assert rendered.count("z") >= len(large_output)


def test_live_tool_execution_keeps_working_prompt_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = _TTYBuffer()
    tool = _PromptObservingTool(out)
    monkeypatch.setitem(TOOLS_BY_NAME, tool.name, tool)
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    skill_path = tmp_path / ".wattle" / "skills" / "writer" / "SKILL.md"
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
    skill_path = tmp_path / ".wattle" / "skills" / "hello_world" / "SKILL.md"
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


def test_terminal_tool_error_is_hidden_but_kept_for_model(
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

    out, app = _drive(provider, ["run tool", "/exit"])

    assert "| explode error" not in out
    assert "ValueError: bad input" not in out
    assert "ValueError('bad input')" not in out
    followup = app.messages[2]
    assert followup.role == "user"
    assert any(
        isinstance(block, ToolResultBlock)
        and block.is_error
        and block.content == "ValueError: bad input"
        for block in followup.content
    )


def test_terminal_edit_error_is_hidden_from_transcript() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    block = ToolUseBlock(
        id="call_1",
        name="edit",
        input={"path": "src/wattle/loop.py"},
    )
    result = ToolResultBlock(
        tool_use_id="call_1",
        content="ValueError: old_text not found in src/wattle/loop.py",
        is_error=True,
    )

    app._write_tool_result(block, result)

    rendered = out.getvalue()
    assert "edit error - src/wattle/loop.py" not in rendered
    assert "old_text not found" not in rendered


def test_tui_persists_tool_use_before_tool_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(session.SESSION_DIR_ENV, str(tmp_path / "sessions"))
    app_box: dict[str, tui.WattleApp] = {}
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
    audit_response = CompletionResponse(
        content=[TextBlock(text="Audit complete.")],
        stop_reason="end_turn",
    )
    provider = _ScriptedStreamProvider(
        [
            [StreamComplete(response=tool_use_response)],
            [StreamComplete(response=end_response)],
            [StreamComplete(response=audit_response)],
        ]
    )
    inputs = iter(["use tool", "/exit"])

    def input_func(_prompt: str = "") -> str:
        try:
            return next(inputs)
        except StopIteration as exc:
            raise EOFError from exc

    app = tui.WattleApp(
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
        "user",
        "assistant",
    ]
    assert isinstance(saved.messages[1].content[0], ToolUseBlock)
    assert isinstance(saved.messages[2].content[0], ToolResultBlock)
    assert saved.messages[4] == Message(
        role="user",
        content=[TextBlock(text=FINAL_AUDIT_REMINDER)],
    )


def test_edit_tool_result_renders_diff_review_style() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    plain = _strip_ansi(rendered)
    assert "Added codex_tui_test_haha1/wobbly_pickle_dispatch.txt (+2 -0)" in plain
    assert "write ok -" not in rendered
    assert f"{tui.DIFF_ADD_COUNT_STYLE}+2{tui.RESET}" in rendered
    assert f"{tui.DIFF_DELETE_COUNT_STYLE}-0{tui.RESET}" in rendered
    assert "48;5" not in tui.DIFF_ADD_COUNT_STYLE
    assert "48;5" not in tui.DIFF_DELETE_COUNT_STYLE
    assert "48;5;22" in tui.DIFF_ADD_LINE_NUMBER_STYLE
    assert "48;5;22" in tui.DIFF_ADD_MARKER_STYLE
    assert "48;5;22" in tui.DIFF_ADD_CODE_STYLE
    assert "48;5;52" in tui.DIFF_DELETE_LINE_NUMBER_STYLE
    assert "48;5;52" in tui.DIFF_DELETE_MARKER_STYLE
    assert "48;5;52" in tui.DIFF_DELETE_CODE_STYLE
    assert "    1 +hello" in plain
    assert "    2 +world" in plain
    assert tui.DIFF_ADD_LINE_NUMBER_STYLE in rendered
    assert tui.DIFF_ADD_MARKER_STYLE in rendered


def test_edit_tool_result_diff_rows_disable_autowrap() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._terminal_width = lambda: 40  # type: ignore[method-assign]
    block = ToolUseBlock(id="call_1", name="write", input={"path": "long.txt"})
    result = ToolResultBlock(
        tool_use_id="call_1",
        content=(
            "Wrote 80 bytes to long.txt\n"
            "--- long.txt (before)\n"
            "+++ long.txt (after)\n"
            "@@ -0,0 +1,1 @@\n"
            "+abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz"
        ),
    )

    app._write_tool_result(block, result)

    rendered = out.getvalue()
    plain = _strip_ansi(rendered)
    assert "\x1b[?7l" in rendered
    assert "\x1b[K" in rendered
    assert "    1 +abcdefghijklmnopqrstuvwxyzabcdefg" in plain
    assert "       hijklmnopqrstuvwxyz" in plain
    assert "..." not in plain


def test_write_added_file_renders_full_diff_preview() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
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
    plain = _strip_ansi(rendered)
    assert f"Added {path} (+23 -0)" in plain
    assert "write ok -" not in rendered
    assert "   23 +line 23" in plain
    assert "changed lines" not in rendered


def test_write_added_file_title_counts_full_large_diff_preview() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    path = "design_log/search_resume_plan.md"
    added_lines = [f"+line {index}" for index in range(1, 214)]
    block = ToolUseBlock(id="call_1", name="write", input={"path": path})
    result = ToolResultBlock(
        tool_use_id="call_1",
        content="\n".join(
            [
                f"Wrote 200 bytes to {path}",
                f"--- {path} (before)",
                f"+++ {path} (after)",
                "@@ -0,0 +1,213 @@",
                *added_lines,
            ]
        ),
    )

    app._write_tool_result(block, result)

    plain = _strip_ansi(out.getvalue())
    assert f"Added {path} (+213 -0)" in plain
    assert f"Added {path} (+117 -0)" not in plain
    assert "  213 +line 213" in plain
    assert "diff lines" not in plain
    assert "changed lines" not in plain


def test_write_added_markdown_file_uses_pygments_diff_syntax() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    path = "design_log/search_resume_plan.md"
    block = ToolUseBlock(id="call_1", name="write", input={"path": path})
    result = ToolResultBlock(
        tool_use_id="call_1",
        content="\n".join(
            [
                f"Wrote 80 bytes to {path}",
                f"--- {path} (before)",
                f"+++ {path} (after)",
                "@@ -0,0 +1,2 @@",
                "+# Resume Search",
                "+Plain body text",
            ]
        ),
    )

    app._write_tool_result(block, result)

    rendered = out.getvalue()
    plain = _strip_ansi(rendered)
    assert "    1 +# Resume Search" in plain
    assert "    2 +Plain body text" in plain
    assert f"{tui.DIFF_ADD_SYNTAX_HEADING_STYLE}# Resume Search" in rendered


def test_write_added_cpp_file_uses_pygments_diff_syntax() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    path = "src/demo.cpp"
    block = ToolUseBlock(id="call_1", name="write", input={"path": path})
    result = ToolResultBlock(
        tool_use_id="call_1",
        content="\n".join(
            [
                f"Wrote 80 bytes to {path}",
                f"--- {path} (before)",
                f"+++ {path} (after)",
                "@@ -0,0 +1,1 @@",
                "+int main() { return 0; }",
            ]
        ),
    )

    app._write_tool_result(block, result)

    rendered = out.getvalue()
    plain = _strip_ansi(rendered)
    assert "    1 +int main() { return 0; }" in plain
    assert f"{tui.DIFF_ADD_SYNTAX_KEYWORD_STYLE}int" in rendered
    assert f"{tui.DIFF_ADD_SYNTAX_KEYWORD_STYLE}return" in rendered


def test_diff_preview_simple_replacement_keeps_changed_line_rows() -> None:
    rows = tui._diff_preview_lines(
        [
            "@@ -100,1 +100,1 @@",
            "-old",
            "+new",
        ],
        max_changes=None,
    )

    assert rows == [("delete", "  100 -old"), ("add", "  100 +new")]


def test_diff_preview_renders_context_and_old_new_line_numbers() -> None:
    rows = tui._diff_preview_lines(
        [
            "@@ -10,4 +10,8 @@",
            " context",
            "+added 1",
            "+added 2",
            "+added 3",
            "+added 4",
            "-removed",
            " after",
        ],
        max_changes=None,
    )

    assert rows == [
        ("context", "   10  context"),
        ("add", "   11 +added 1"),
        ("add", "   12 +added 2"),
        ("add", "   13 +added 3"),
        ("add", "   14 +added 4"),
        ("delete", "   11 -removed"),
        ("context", "   15  after"),
    ]


def test_diff_preview_keeps_intra_hunk_context_visible_between_change_blocks() -> None:
    rows = tui._diff_preview_lines(
        [
            "@@ -229,30 +256,36 @@",
            "+async def arun_agent_with_history(",
            "+    provider_name: str,",
            "+    model: str,",
            "+    user_input: str,",
            "+    *,",
            "+    max_tokens: int = 4096,",
            "+    permission_mode: PermissionMode = PermissionMode.YOLO,",
            "+    thinking: bool = False,",
            "+    effort: Literal[\"low\", \"medium\", \"high\", \"xhigh\", \"max\"] | None = None,",  # noqa: E501
            "+) -> AgentRunResult:",
            "+    \"\"\"Async headless agent runner returning the full transcript.\"\"\"",
            "     provider, built_system = _build_provider_and_system(provider_name, permission_mode)",  # noqa: E501
            "     messages: list[Message] = []",
            "-",
            "-    if permission_mode == PermissionMode.YOLO:",
            "-        response = loop.run(",
            "-            provider,",
            "-            TOOLS_BY_NAME,",
            "-            built_system,",
            "-            user_input,",
            "-            model,",
            "-            max_tokens,",
            "-            thinking=thinking,",
            "-            effort=effort,",
            "-            messages_out=messages,",
            "-        )",
            "-    else:",
            "-        response = loop.run(",
            "-            provider,",
            "-            TOOLS_BY_NAME,",
            "-            built_system,",
            "-            user_input,",
            "-            model,",
            "-            max_tokens,",
            "-            permission_gate=PermissionGate(permission_mode),",
            "-            thinking=thinking,",
            "-            effort=effort,",
            "-            messages_out=messages,",
            "-        )",
            "-",
            "+    response = await loop.arun(",
            "+        provider,",
        ],
        max_changes=None,
    )

    assert (
        "add",
        "  266 +    \"\"\"Async headless agent runner returning the full transcript.\"\"\"",
    ) in rows
    assert (
        "context",
        "  267      provider, built_system = _build_provider_and_system("
        "provider_name, permission_mode)",
    ) in rows
    assert ("context", "  268      messages: list[Message] = []") in rows
    assert ("delete", "  232 -    if permission_mode == PermissionMode.YOLO:") in rows
    assert ("add", "  269 +    response = await loop.arun(") in rows


def test_diff_preview_marks_non_contiguous_hunk_boundary() -> None:
    rows = tui._diff_preview_lines(
        [
            "@@ -158,8 +160,36 @@",
            "+def session_preview(record):",
            "+    return 'preview'",
            "+def session_search_text(entry):",
            "+    return entry.search_text",
            "+def list_session_entries():",
            "+    return []",
            "@@ -170,13 +200,53 @@",
            "-        entries.append(SessionEntry(path=path, record=record))",
            "+        preview = session_preview(record)",
            "+        entries.append(SessionEntry(path=path, record=record, preview=preview))",
        ],
        max_changes=None,
    )

    assert rows == [
        ("add", "  160 +def session_preview(record):"),
        ("add", "  161 +    return 'preview'"),
        ("add", "  162 +def session_search_text(entry):"),
        ("add", "  163 +    return entry.search_text"),
        ("add", "  164 +def list_session_entries():"),
        ("add", "  165 +    return []"),
        ("meta", tui._DIFF_CONTEXT_MARKER_ROW),
        ("delete", "  170 -        entries.append(SessionEntry(path=path, record=record))"),
        ("add", "  200 +        preview = session_preview(record)"),
        (
            "add",
            "  201 +        entries.append("
            "SessionEntry(path=path, record=record, preview=preview))",
        ),
    ]


def test_diff_preview_added_and_deleted_files_do_not_show_old_new_marker() -> None:
    added_rows = tui._diff_preview_lines(
        [
            "@@ -0,0 +1,2 @@",
            "+new 1",
            "+new 2",
        ],
        max_changes=None,
    )
    deleted_rows = tui._diff_preview_lines(
        [
            "@@ -1,2 +0,0 @@",
            "-old 1",
            "-old 2",
        ],
        max_changes=None,
    )

    assert added_rows == [("add", "    1 +new 1"), ("add", "    2 +new 2")]
    assert deleted_rows == [("delete", "    1 -old 1"), ("delete", "    2 -old 2")]
    assert ("meta", tui._DIFF_CONTEXT_MARKER_ROW) not in added_rows
    assert ("meta", tui._DIFF_CONTEXT_MARKER_ROW) not in deleted_rows


def test_edit_tool_result_renders_full_diff_preview() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    path = "/app/service.py"
    diff_lines = []
    for index in range(1, 18):
        diff_lines.extend([f"-old {index}", f"+new {index}"])
    block = ToolUseBlock(id="call_1", name="edit", input={"path": path})
    result = ToolResultBlock(
        tool_use_id="call_1",
        content="\n".join(
            [
                f"Edited {path}",
                f"--- {path} (before)",
                f"+++ {path} (after)",
                "@@ -1,17 +1,17 @@",
                *diff_lines,
            ]
        ),
    )

    app._write_tool_result(block, result)

    rendered = out.getvalue()
    plain = _strip_ansi(rendered)
    assert f"Edited {path} (+17 -17)" in plain
    assert "edit ok -" not in rendered
    assert f"{tui.DIFF_ADD_COUNT_STYLE}+17{tui.RESET}" in rendered
    assert f"{tui.DIFF_DELETE_COUNT_STYLE}-17{tui.RESET}" in rendered
    assert "48;5" not in tui.DIFF_ADD_COUNT_STYLE
    assert "48;5" not in tui.DIFF_DELETE_COUNT_STYLE
    assert "   17 -old 17" in plain
    assert "   17 +new 17" in plain
    assert tui.DIFF_DELETE_LINE_NUMBER_STYLE in rendered
    assert tui.DIFF_DELETE_MARKER_STYLE in rendered
    assert tui.SYNTAX_NAME_STYLE not in rendered
    assert "changed lines" not in rendered


def test_adjacent_same_file_edits_render_as_one_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(TOOLS_BY_NAME, "edit", _DiffEditTool())
    response = CompletionResponse(
        content=[
            ToolUseBlock(
                id="call_1",
                name="edit",
                input={"path": "src/app.py", "line": 1, "before": "a", "after": "b"},
            ),
            ToolUseBlock(
                id="call_2",
                name="edit",
                input={"path": "src/app.py", "line": 5, "before": "c", "after": "d"},
            ),
            ToolUseBlock(
                id="call_3",
                name="edit",
                input={"path": "src/app.py", "line": 9, "before": "e", "after": "f"},
            ),
        ],
        stop_reason="tool_use",
    )
    end_response = CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider(
        [[StreamComplete(response=response)], [StreamComplete(response=end_response)]]
    )

    out, _app = _drive(provider, ["edit files", "/exit"])
    rendered = _strip_ansi(out)

    assert rendered.count("Edited src/app.py") == 1
    assert "Edited src/app.py (+3 -3)" in rendered
    assert rendered.count(tui._EDIT_HUNK_SEPARATOR_ROW) == 2
    assert rendered.index("    1 -a") < rendered.index("    5 -c")
    assert rendered.index("    5 -c") < rendered.index("    9 -e")


def test_same_file_edit_groups_do_not_cross_order_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(TOOLS_BY_NAME, "edit", _DiffEditTool())
    echo_tool = _RecordingTool()
    monkeypatch.setitem(TOOLS_BY_NAME, "echo", echo_tool)
    response = CompletionResponse(
        content=[
            ToolUseBlock(
                id="call_1",
                name="edit",
                input={"path": "src/app.py", "line": 1, "before": "a", "after": "b"},
            ),
            ToolUseBlock(id="call_2", name="echo", input={"message": "middle"}),
            ToolUseBlock(
                id="call_3",
                name="edit",
                input={"path": "src/app.py", "line": 5, "before": "c", "after": "d"},
            ),
        ],
        stop_reason="tool_use",
    )
    end_response = CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider(
        [[StreamComplete(response=response)], [StreamComplete(response=end_response)]]
    )

    out, _app = _drive(provider, ["edit files", "/exit"])
    rendered = _strip_ansi(out)

    assert rendered.count("Edited src/app.py (+1 -1)") == 2
    assert "echo ok" in rendered
    assert rendered.index("    1 -a") < rendered.index("echo ok")
    assert rendered.index("echo ok") < rendered.index("    5 -c")


def test_edit_groups_split_on_different_file_and_failed_edit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(TOOLS_BY_NAME, "edit", _DiffEditTool())
    response = CompletionResponse(
        content=[
            ToolUseBlock(
                id="call_1",
                name="edit",
                input={"path": "src/app.py", "line": 1, "before": "a", "after": "b"},
            ),
            ToolUseBlock(
                id="call_2",
                name="edit",
                input={"path": "src/other.py", "line": 3, "before": "x", "after": "y"},
            ),
            ToolUseBlock(
                id="call_3",
                name="edit",
                input={"path": "src/app.py", "fail": True},
            ),
            ToolUseBlock(
                id="call_4",
                name="edit",
                input={"path": "src/app.py", "line": 8, "before": "c", "after": "d"},
            ),
        ],
        stop_reason="tool_use",
    )
    end_response = CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider(
        [[StreamComplete(response=response)], [StreamComplete(response=end_response)]]
    )

    out, _app = _drive(provider, ["edit files", "/exit"])
    rendered = _strip_ansi(out)

    assert rendered.count("Edited src/app.py (+1 -1)") == 2
    assert rendered.count("Edited src/other.py (+1 -1)") == 1
    assert "old_text not found" not in rendered
    assert rendered.index("    1 -a") < rendered.index("Edited src/other.py")
    assert rendered.index("Edited src/other.py") < rendered.index("    8 -c")


def test_mixed_write_edit_group_uses_updated_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(TOOLS_BY_NAME, "write", _DiffWriteTool())
    monkeypatch.setitem(TOOLS_BY_NAME, "edit", _DiffEditTool())
    response = CompletionResponse(
        content=[
            ToolUseBlock(
                id="call_1",
                name="write",
                input={"path": "src/app.py", "line": 1, "after": "created"},
            ),
            ToolUseBlock(
                id="call_2",
                name="edit",
                input={"path": "src/app.py", "line": 2, "before": "old", "after": "new"},
            ),
        ],
        stop_reason="tool_use",
    )
    end_response = CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider(
        [[StreamComplete(response=response)], [StreamComplete(response=end_response)]]
    )

    out, _app = _drive(provider, ["edit files", "/exit"])
    rendered = _strip_ansi(out)

    assert "Updated src/app.py (+2 -1)" in rendered
    assert "Added src/app.py" not in rendered
    assert "Edited src/app.py" not in rendered
    assert rendered.index("    1 +created") < rendered.index("    2 -old")


def test_grouped_edit_separator_is_dim_and_keeps_count_styles() -> None:
    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    first = tui._edit_render_item(
        ToolUseBlock(id="call_1", name="edit", input={"path": "src/app.py"}),
        ToolResultBlock(
            tool_use_id="call_1",
            content=(
                "Edited src/app.py\n"
                "--- src/app.py (before)\n"
                "+++ src/app.py (after)\n"
                "@@ -1,1 +1,1 @@\n"
                "-old\n"
                "+new"
            ),
        ),
    )
    second = tui._edit_render_item(
        ToolUseBlock(id="call_2", name="edit", input={"path": "src/app.py"}),
        ToolResultBlock(
            tool_use_id="call_2",
            content=(
                "Edited src/app.py\n"
                "--- src/app.py (before)\n"
                "+++ src/app.py (after)\n"
                "@@ -5,1 +5,1 @@\n"
                "-before\n"
                "+after"
            ),
        ),
    )
    assert first is not None
    assert second is not None

    app._write_edit_result_group(tui._EditRenderGroup(path="src/app.py", items=[first, second]))

    rendered = out.getvalue()
    assert f"{tui.DIFF_ADD_COUNT_STYLE}+2{tui.RESET}" in rendered
    assert f"{tui.DIFF_DELETE_COUNT_STYLE}-2{tui.RESET}" in rendered
    assert f"{tui.DIFF_META_STYLE}{tui._EDIT_HUNK_SEPARATOR_ROW}{tui.RESET}" in rendered
    assert f"{tui.DIFF_ADD_STYLE}{tui._EDIT_HUNK_SEPARATOR_ROW}" not in rendered
    assert f"{tui.DIFF_DELETE_STYLE}{tui._EDIT_HUNK_SEPARATOR_ROW}" not in rendered


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


def test_terminal_requires_login_before_task() -> None:
    provider = _ScriptedStreamProvider([])

    out, app = _drive(
        provider,
        ["start work", "/exit"],
        args=_make_args(provider=None, model=None),
    )

    assert "Authenticate before starting a task. Run /login" in out
    assert app.current_provider_name == ""
    assert app.current_model == ""
    assert provider.requests == []
    assert app.messages == []


def test_terminal_login_openai_codex(monkeypatch: pytest.MonkeyPatch) -> None:
    from wattle import cli

    for name in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"):
        monkeypatch.delenv(name, raising=False)

    calls: list[dict[str, object]] = []
    provider_names: list[str] = []
    reloaded_provider = _ScriptedStreamProvider([])

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
    def fake_build_provider(provider: str) -> _ScriptedStreamProvider:
        provider_names.append(provider)
        return reloaded_provider

    monkeypatch.setattr(cli, "_build_provider", fake_build_provider)

    out, app = _drive(_ScriptedStreamProvider([]), ["/login", "/exit"])

    assert len(calls) == 1
    assert calls[0]["originator"] == "wattle"
    assert calls[0]["callback_timeout_seconds"] == tui.DEFAULT_LOGIN_CALLBACK_TIMEOUT_SECONDS
    assert "Open this URL to authenticate OpenAI Codex" in out
    assert "https://auth.example/login" in out
    assert "OpenAI Codex OAuth saved to /tmp/auth.json openai.oauth" in out
    assert provider_names == ["openai_codex"]
    assert app.current_provider_name == "openai_codex"
    assert app.provider is reloaded_provider


def test_terminal_login_openai_codex_uses_ssh_manual_callback_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from wattle import cli

    monkeypatch.setenv("SSH_CONNECTION", "client 123 server 22")
    monkeypatch.delenv("SSH_CLIENT", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)

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
    monkeypatch.setattr(cli, "_build_provider", lambda _provider: _ScriptedStreamProvider([]))

    out, _app = _drive(_ScriptedStreamProvider([]), ["/login", "/exit"])

    assert calls[0]["callback_timeout_seconds"] == tui.SSH_LOGIN_CALLBACK_TIMEOUT_SECONDS
    assert "SSH detected" in out
    assert "copy the full callback URL from the browser address bar" in out


def test_terminal_login_openai_codex_styles_ssh_manual_callback_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from wattle import cli

    monkeypatch.setenv("SSH_CONNECTION", "client 123 server 22")
    inputs = iter(["/login", "/exit"])

    def input_func(_prompt: str = "") -> str:
        return next(inputs)

    def fake_login(**kwargs: object) -> auth.AuthCredential:
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
    monkeypatch.setattr(cli, "_build_provider", lambda _provider: _ScriptedStreamProvider([]))

    out = _TTYBuffer()
    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), input_func=input_func, out=out)
    app._force_plain = False

    assert app.run() == 0
    rendered = out.getvalue()
    assert tui.SSH_LOGIN_HINT_STYLE in rendered
    assert "SSH detected" in rendered
    assert "callback URL" in rendered


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
    app = tui.WattleApp(
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
    assert compacted_request.messages[1:] == app.messages[24:25]
    assert len(compacted_request.messages) == 2
    assert provider.reset_count >= 3
    assert len(app.messages) == 26
    assert all("middle summary" not in _message_text(message) for message in app.messages)
    assert app._session_path is not None
    saved = session.load_session(app._session_path)
    assert saved.messages == app.messages
    assert len(saved.compactions) == 1
    assert saved.compactions[0].first_kept_message_index == 24
    assert saved.compactions[0].summarized_until_message_index == 24
    assert saved.compactions[0].created_after_message_index == 25


def test_terminal_compaction_retriggers_when_provider_usage_crosses_trigger_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tui, "_context_window_for_model", lambda _model: 9_500)
    prior_messages = [_text_message(index, "history " + ("x" * 80)) for index in range(90)]
    summary_response = CompletionResponse(
        content=[TextBlock(text="middle summary")],
        stop_reason="end_turn",
    )
    first_response = CompletionResponse(
        content=[TextBlock(text="first answer")],
        stop_reason="end_turn",
        usage={"input_tokens": 8_000},
    )
    refresh_summary_response = CompletionResponse(
        content=[TextBlock(text="refreshed summary")],
        stop_reason="end_turn",
    )
    second_response = CompletionResponse(
        content=[TextBlock(text="second answer")],
        stop_reason="end_turn",
    )
    provider = _ResettableScriptedStreamProvider(
        [
            [StreamComplete(response=summary_response)],
            [StreamComplete(response=first_response)],
            [StreamComplete(response=refresh_summary_response)],
            [StreamComplete(response=second_response)],
        ]
    )
    inputs = iter(["first request", "second request", "/exit"])

    def input_func(_prompt: str = "") -> str:
        try:
            return next(inputs)
        except StopIteration as exc:
            raise EOFError from exc

    out = io.StringIO()
    app = tui.WattleApp(
        _make_args(max_tokens=10),
        provider,
        state={"messages": list(prior_messages)},
        input_func=input_func,
        out=out,
    )

    assert app.run() == 0

    rendered = out.getvalue()
    assert rendered.count("[status] Auto-compacting...") == 2
    assert len(provider.requests) == 4
    second_request = provider.requests[3]
    assert "refreshed summary" in _message_text(second_request.messages[0])
    assert any("second request" in _message_text(message) for message in second_request.messages)


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

    app = tui.WattleApp(
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
    app = tui.WattleApp(
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
    assert f"wattle -r {parent.metadata.id}" in rendered

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
    app = tui.WattleApp(
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


def test_terminal_clear_redraws_card_and_resets_history() -> None:
    end1 = CompletionResponse(
        content=[TextBlock(text="first done")],
        stop_reason="end_turn",
        usage={"input_tokens": 12, "cached_tokens": 3, "output_tokens": 4},
    )
    end2 = CompletionResponse(
        content=[TextBlock(text="second done")],
        stop_reason="end_turn",
        usage={"input_tokens": 2, "output_tokens": 1},
    )
    provider = _ScriptedStreamProvider(
        [
            [TextDelta(text="first done"), StreamComplete(response=end1)],
            [TextDelta(text="second done"), StreamComplete(response=end2)],
        ]
    )

    out, app = _drive(provider, ["first", "/clear", "second", "/exit"])

    assert "first done" in out
    assert tui.VISIBLE_SCREEN_CLEAR in out
    assert tui.TERMINAL_HISTORY_CLEAR in out
    assert "Last session usage: last context: 12 tok | input: 12 tok" in out
    assert "Conversation cleared." not in out
    assert "second done" in out
    assert len(provider.requests) == 2
    assert len(provider.requests[1].messages) == 1
    block = provider.requests[1].messages[0].content[0]
    assert isinstance(block, TextBlock)
    assert block.text == "second"
    assert app._total_input_tokens == 2
    assert app._total_cached_tokens == 0
    assert app._total_output_tokens == 1


def test_terminal_clear_starts_new_persisted_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(session.SESSION_DIR_ENV, str(tmp_path / "sessions"))
    end1 = CompletionResponse(content=[TextBlock(text="first done")], stop_reason="end_turn")
    end2 = CompletionResponse(content=[TextBlock(text="second done")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider(
        [
            [TextDelta(text="first done"), StreamComplete(response=end1)],
            [TextDelta(text="second done"), StreamComplete(response=end2)],
        ]
    )

    _out, app = _drive(
        provider,
        ["first", "/clear", "second", "/exit"],
        args=_make_args(persist_session=True),
    )

    session_paths = sorted((tmp_path / "sessions").glob("*.jsonl"))
    assert len(session_paths) == 2
    records = [session.load_session(path) for path in session_paths]
    histories = [[_message_text(message) for message in record.messages] for record in records]
    assert ["first", "first done"] in histories
    assert ["second", "second done"] in histories
    assert app._session_path in session_paths
    active_record = session.load_session(app._session_path)
    assert [_message_text(message) for message in active_record.messages] == [
        "second",
        "second done",
    ]


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
    monkeypatch.setattr("wattle.cli._build_provider", build_provider)

    out, _app = _drive(
        anthropic_provider,
        ["/model gpt-5.5", "go", "/exit"],
        args=_make_args(provider="anthropic"),
    )

    assert "using provider 'openai_responses'" in out
    assert anthropic_provider.requests == []
    assert len(openai_provider.requests) == 1
    assert openai_provider.requests[0].model == "gpt-5.5"


def test_settings_change_event_is_persisted_outside_messages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(settings.SETTINGS_PATH_ENV, str(tmp_path / "settings.json"))
    monkeypatch.setenv(session.SESSION_DIR_ENV, str(tmp_path / "sessions"))
    settings.save_settings(
        settings.WattleSettings(provider="openai_responses", model="gpt-5.4"),
        tmp_path / "settings.json",
    )
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

    _out, app = _drive(
        _ScriptedStreamProvider([]),
        ["/model gpt-5.5", "/exit"],
        args=_make_args(model="gpt-5.4", persist_session=True),
    )

    assert app._session_path is not None
    record = session.load_session(app._session_path)
    assert record.messages == []
    assert len(record.events) == 1
    event = record.events[0]
    assert event.type == "settings_change"
    assert event.source == {"kind": "slash_command", "name": "/model"}
    assert event.data["path"] == str(tmp_path / "settings.json")
    assert event.data["changes"]["model"] == {"old": "gpt-5.4", "new": "gpt-5.5"}


def test_settings_change_event_created_before_writing_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(settings.SETTINGS_PATH_ENV, str(tmp_path / "settings.json"))
    monkeypatch.setenv(session.SESSION_DIR_ENV, str(tmp_path / "sessions"))

    _out, app = _drive(
        _ScriptedStreamProvider([]),
        ["/effort high", "/exit"],
        args=_make_args(persist_session=True),
    )

    assert app._session_path is not None
    record = session.load_session(app._session_path)
    assert len(record.events) == 1
    event = record.events[0]
    assert event.source == {"kind": "slash_command", "name": "/effort"}
    assert event.data["changes"]["thinking"] == {"old": False, "new": True}
    assert event.data["changes"]["effort"] == {"old": None, "new": "high"}


def test_settings_change_event_logs_before_auth_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(settings.SETTINGS_PATH_ENV, str(tmp_path / "settings.json"))
    monkeypatch.setenv(session.SESSION_DIR_ENV, str(tmp_path / "sessions"))

    _out, app = _drive(
        _ScriptedStreamProvider([]),
        ["/effort high", "/exit"],
        args=_make_args(provider="", model="", persist_session=True),
    )

    assert app._session_path is not None
    record = session.load_session(app._session_path)
    assert record.settings.provider == "(not authenticated)"
    assert record.settings.model == "(not authenticated)"
    assert len(record.events) == 1
    assert record.events[0].source == {"kind": "slash_command", "name": "/effort"}
    assert record.events[0].data["changes"]["thinking"] == {"old": False, "new": True}
    assert record.events[0].data["changes"]["effort"] == {"old": None, "new": "high"}


def test_terminal_model_selection_rejects_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
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

    out, app = _drive(provider, ["/model 1", "/exit"])

    assert "Model numbers are not supported. Use a model name." in out
    assert app.current_model == "gpt-5.5"
    assert provider.requests == []


def test_terminal_login_xiaomi_token_plan_uses_provider_default_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from wattle import cli

    monkeypatch.setenv(settings.SETTINGS_PATH_ENV, str(tmp_path / "settings.json"))
    monkeypatch.setattr(auth, "AUTH_PATH", tmp_path / "auth.json")
    reloaded_provider = _ScriptedStreamProvider([])
    provider_names: list[str] = []

    def fake_build_provider(provider: str) -> _ScriptedStreamProvider:
        provider_names.append(provider)
        return reloaded_provider

    monkeypatch.setattr(cli, "_build_provider", fake_build_provider)

    out, app = _drive(
        _ScriptedStreamProvider([]),
        ["/login xiaomi-token-plan-sgp", "tp-test-key", "/exit"],
        args=_make_args(),
    )

    saved_settings = settings.load_settings(tmp_path / "settings.json")
    assert "Xiaomi Token Plan SGP API key saved" in out
    assert provider_names == ["xiaomi-token-plan-sgp"]
    assert app.current_provider_name == "xiaomi-token-plan-sgp"
    assert app.current_model == "mimo-v2.5-pro"
    assert app.provider is reloaded_provider
    assert saved_settings.provider == "xiaomi-token-plan-sgp"
    assert saved_settings.model == "mimo-v2.5-pro"


def test_tui_thinking_content_visibility_loads_from_settings_section(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(settings.SETTINGS_PATH_ENV, str(tmp_path / "settings.json"))
    (tmp_path / "settings.json").write_text(
        '{"tui": {"show_thinking": true}}',
        encoding="utf-8",
    )

    app = tui.WattleApp(_make_args(), _ScriptedStreamProvider([]), out=io.StringIO())
    saved = settings.load_settings()

    assert saved.tui.show_thinking is True
    assert app.show_thinking_content is True


def test_tui_statusline_defaults_to_model_thinking_and_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(settings.SETTINGS_PATH_ENV, str(tmp_path / "settings.json"))

    parser = cli._build_parser()
    args = parser.parse_args([])
    cli._apply_settings_defaults(args, [], settings.load_settings())

    assert args.statusline is True
    assert args.statusline_fields == ("model", "thinking", "cwd")


def test_tui_statusline_empty_settings_section_disables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(settings.SETTINGS_PATH_ENV, str(tmp_path / "settings.json"))
    (tmp_path / "settings.json").write_text(
        '{"tui": {"statusline": []}}',
        encoding="utf-8",
    )

    parser = cli._build_parser()
    args = parser.parse_args([])
    cli._apply_settings_defaults(args, [], settings.load_settings())
    app = tui.WattleApp(args, _ScriptedStreamProvider([]), out=io.StringIO())

    assert args.statusline is False
    assert args.statusline_fields == ()
    assert app._statusline_enabled is False


def test_tui_statusline_fields_load_from_settings_section(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(settings.SETTINGS_PATH_ENV, str(tmp_path / "settings.json"))
    (tmp_path / "settings.json").write_text(
        '{"tui": {"statusline": ["model", "thinking", "context_remaining", "quota_5h"]}}',
        encoding="utf-8",
    )

    parser = cli._build_parser()
    args = parser.parse_args([])
    cli._apply_settings_defaults(args, [], settings.load_settings())
    app = tui.WattleApp(args, _ScriptedStreamProvider([]), out=io.StringIO())
    app.thinking = True
    app.effort = "medium"
    app._last_context_tokens = 50

    assert args.statusline is True
    assert args.statusline_fields == ("model", "thinking", "context_remaining", "quota_5h")
    assert app._status_text() == (
        "gpt-5.5 | thinking: medium | remaining: 271.9k tok | 5h quota: unknown"
    )


def test_statusline_shows_thinking_off_when_disabled() -> None:
    rendered = tui._render_statusline(
        model="mimo-v2.5-pro",
        context_tokens=None,
        context_window=1_000_000,
        input_tokens=0,
        cached_tokens=0,
        output_tokens=0,
        cwd="~/repos/wattle",
        thinking=False,
        effort=None,
        fields=("model", "thinking", "cwd"),
    )

    assert rendered == "mimo-v2.5-pro | thinking: off | ~/repos/wattle"


def test_effort_choices_are_limited_by_current_model() -> None:
    out = io.StringIO()
    app = tui.WattleApp(
        _make_args(provider="xiaomi-token-plan-sgp", model="mimo-v2.5-pro"),
        _ScriptedStreamProvider([]),
        out=out,
    )

    app._handle_effort("")
    app._handle_effort("xhigh")
    app._handle_effort("high")

    rendered = out.getvalue()
    assert "Choices for mimo-v2.5-pro: low, medium, high, off" in rendered
    assert "Usage for mimo-v2.5-pro: /effort low|medium|high|off" in rendered
    assert app.thinking is True
    assert app.effort == "high"


def test_shift_tab_cycles_current_model_effort_levels() -> None:
    app = tui.WattleApp(
        _make_args(provider="xiaomi-token-plan-sgp", model="mimo-v2.5-pro"),
        _ScriptedStreamProvider([]),
        out=io.StringIO(),
    )

    observed: list[str | None] = []
    for _ in range(4):
        app._cycle_thinking_level()
        observed.append(app.effort)

    assert observed == ["low", "medium", "high", None]
    assert app.thinking is False


def test_model_switch_coerces_unsupported_effort() -> None:
    app = tui.WattleApp(
        _make_args(provider="xiaomi-token-plan-sgp"),
        _ScriptedStreamProvider([]),
        out=io.StringIO(),
    )
    app.thinking = True
    app.effort = "xhigh"

    app._apply_model_choice(
        ModelChoice(
            model="mimo-v2.5-pro",
            provider="xiaomi-token-plan-sgp",
            vendor="xiaomi-token-plan-sgp",
            description="Xiaomi MiMo V2.5 Pro model.",
            effort_levels=("low", "medium", "high"),
        )
    )

    assert app.current_model == "mimo-v2.5-pro"
    assert app.thinking is True
    assert app.effort == "high"


def test_tui_prefetches_codex_quota_for_startup_statusline() -> None:
    captured: list[Any] = []

    def urlopen(req: Any) -> _FakeJsonResponse:
        captured.append(req)
        return _FakeJsonResponse(
            {
                "plan_type": "pro",
                "rate_limit": {
                    "allowed": True,
                    "limit_reached": False,
                    "primary_window": {
                        "used_percent": 28,
                        "limit_window_seconds": 18_000,
                        "reset_after_seconds": 10,
                        "reset_at": 1_700_000_000,
                    },
                    "secondary_window": {
                        "used_percent": 7,
                        "limit_window_seconds": 604_800,
                        "reset_after_seconds": 20,
                        "reset_at": 1_700_000_001,
                    },
                },
            }
        )

    provider = OpenAICodexResponsesProvider(
        bearer_token=_codex_token(),
        urlopen=urlopen,
    )
    app = tui.WattleApp(
        _make_args(
            provider="openai_codex",
            statusline_fields=("model", "quota_5h", "quota_1w"),
        ),
        provider,
        out=io.StringIO(),
    )

    assert app._status_text() == "gpt-5.5 | 5h 72% | weekly 93%"
    assert captured[0].full_url == "https://chatgpt.com/backend-api/wham/usage"


def test_terminal_statusline_command_rejects_typed_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(settings.SETTINGS_PATH_ENV, str(tmp_path / "settings.json"))
    end_response = CompletionResponse(
        content=[TextBlock(text="ok")],
        stop_reason="end_turn",
        usage={"input_tokens": 10, "output_tokens": 5},
    )
    provider = _ScriptedStreamProvider(
        [[TextDelta(text="ok"), StreamComplete(response=end_response)]]
    )

    out, app = _drive(provider, ["/statusline off", "go", "/exit"])

    assert "Use /statusline to choose fields from the picker." in out
    assert app._statusline_enabled is True
    assert settings.load_settings().tui.statusline == ("model", "thinking", "cwd")


def test_terminal_effort_updates_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(settings.SETTINGS_PATH_ENV, str(tmp_path / "settings.json"))

    out, app = _drive(
        _ScriptedStreamProvider([]),
        ["/effort high", "/exit"],
    )

    saved = settings.load_settings()
    assert app.thinking is True
    assert app.effort == "high"
    assert app.permission_mode == tui.PermissionMode.YOLO
    assert saved.thinking is True
    assert saved.effort == "high"
    assert saved.permission_mode == tui.PermissionMode.YOLO
    assert saved.tui.show_thinking is False
    assert "Effort set to high" in out
