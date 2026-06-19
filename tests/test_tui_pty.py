"""End-to-end PTY tests for Wattle's live TUI."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from pty_harness import PtySession

from wattle import tui


def _slow_wattle_child_code(
    *,
    first_delay: float = 1.0,
    later_delay: float = 0.2,
    prompt: str | None = "start",
) -> str:
    return textwrap.dedent(
        f"""
        import argparse
        import time

        from wattle.permissions import PermissionMode
        from wattle.providers import (
            CompletionResponse,
            Provider,
            StreamComplete,
            TextBlock,
            TextDelta,
        )
        from wattle.tui import WattleApp


        class SlowProvider(Provider):
            def __init__(self):
                self.calls = 0

            async def acomplete(self, request):
                self.calls += 1
                return CompletionResponse(
                    content=[TextBlock(text=f"done {{self.calls}}")],
                    stop_reason="end_turn",
                    usage={{}},
                )

            async def astream(self, request):
                self.calls += 1
                time.sleep({first_delay!r} if self.calls == 1 else {later_delay!r})
                yield TextDelta(text=f"done {{self.calls}}")
                yield StreamComplete(
                    CompletionResponse(
                        content=[TextBlock(text=f"done {{self.calls}}")],
                        stop_reason="end_turn",
                        usage={{}},
                    )
                )


        args = argparse.Namespace(
            provider="openai_responses",
            model="gpt-5.5",
            max_tokens=4096,
            thinking=False,
            effort=None,
            prompt={prompt!r},
            persist_session=False,
            permission_mode=PermissionMode.YOLO,
        )
        raise SystemExit(WattleApp(args, SlowProvider()).run())
        """
    )


def _statusline_picker_child_code(settings_path: Path) -> str:
    return textwrap.dedent(
        f"""
        import argparse
        import os

        from wattle.permissions import PermissionMode
        from wattle.providers import CompletionResponse, Provider, TextBlock
        from wattle.tui import WattleApp


        class StaticProvider(Provider):
            async def acomplete(self, request):
                return CompletionResponse(
                    content=[TextBlock(text="ok")],
                    stop_reason="end_turn",
                    usage={{}},
                )

            async def astream(self, request):
                raise NotImplementedError
                yield


        os.environ["WATTLE_SETTINGS_PATH"] = {str(settings_path)!r}
        args = argparse.Namespace(
            provider="openai_responses",
            model="gpt-5.5",
            max_tokens=4096,
            thinking=False,
            effort=None,
            prompt=None,
            persist_session=False,
            permission_mode=PermissionMode.YOLO,
            statusline_fields=("model", "thinking", "cwd"),
        )
        raise SystemExit(WattleApp(args, StaticProvider()).run())
        """
    )


def _echo_prompt_child_code() -> str:
    return textwrap.dedent(
        """
        import argparse

        from wattle.permissions import PermissionMode
        from wattle.providers import (
            CompletionResponse,
            Provider,
            StreamComplete,
            TextBlock,
            TextDelta,
        )
        from wattle.tui import WattleApp


        class EchoProvider(Provider):
            async def acomplete(self, request):
                text = request.messages[-1].content[0].text
                return CompletionResponse(
                    content=[TextBlock(text=text)],
                    stop_reason="end_turn",
                    usage={},
                )

            async def astream(self, request):
                text = request.messages[-1].content[0].text
                yield TextDelta(text=text)
                yield StreamComplete(
                    CompletionResponse(
                        content=[TextBlock(text=text)],
                        stop_reason="end_turn",
                        usage={},
                    )
                )


        args = argparse.Namespace(
            provider="openai_responses",
            model="gpt-5.5",
            max_tokens=4096,
            thinking=False,
            effort=None,
            prompt=None,
            persist_session=False,
            permission_mode=PermissionMode.YOLO,
            statusline_fields=(),
        )
        raise SystemExit(WattleApp(args, EchoProvider()).run())
        """
    )


def _update_prompt_child_code() -> str:
    return textwrap.dedent(
        """
        from wattle import update

        update.run_installer = lambda _latest: 0
        handled = update.prompt_for_tui_update(
            "0.2.0",
            update.LatestVersion(version="0.2.1", tag="v0.2.1"),
        )
        print(f"handled={handled}")
        """
    )


def _resume_search_child_code() -> str:
    return textwrap.dedent(
        """
        import argparse
        from pathlib import Path

        from wattle import session
        from wattle.permissions import PermissionMode
        from wattle.providers import Message, TextBlock
        from wattle.tui import run_tui


        def write_record(session_id, *, title, text, updated_at):
            record = session.SessionRecord(
                metadata=session.SessionMetadata(
                    id=session_id,
                    created_at="2026-05-09T09:00:00Z",
                    updated_at=updated_at,
                    title=title,
                    cwd=str(Path.cwd()),
                ),
                settings=session.SessionSettings(
                    provider="openai_responses",
                    model="gpt-5.5",
                    max_tokens=4096,
                ),
                messages=[Message(role="user", content=[TextBlock(text=text)])],
            )
            path = session.default_session_path(session_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "\\n".join(session.session_to_jsonl_lines(record)) + "\\n",
                encoding="utf-8",
            )


        write_record(
            "newest",
            title="Newest Session",
            text="recent unrelated work",
            updated_at="2026-05-09T12:00:00Z",
        )
        write_record(
            "older_quota",
            title="Quota Investigation",
            text="inspect weekly limits",
            updated_at="2026-05-09T10:00:00Z",
        )
        write_record(
            "oldest",
            title="Old Resize",
            text="fix prompt redraw",
            updated_at="2026-05-09T08:00:00Z",
        )


        args = argparse.Namespace(
            provider="openai_responses",
            model="gpt-5.5",
            max_tokens=4096,
            thinking=False,
            effort=None,
            prompt=None,
            resume="",
            system=None,
            permission_mode=PermissionMode.YOLO,
        )
        raise SystemExit(run_tui(args))
        """
    )


def _history_tool_child_code() -> str:
    return textwrap.dedent(
        """
        import argparse
        from pathlib import Path

        from wattle import session
        from wattle.permissions import PermissionMode
        from wattle.providers import Message, Provider, TextBlock, ToolResultBlock, ToolUseBlock
        from wattle.tui import WattleApp, _state_from_session


        class IdleProvider(Provider):
            async def acomplete(self, request):
                raise RuntimeError("provider should not be called")

            async def astream(self, request):
                raise RuntimeError("provider should not be called")
                yield


        record = session.SessionRecord(
            metadata=session.SessionMetadata(
                id="sess_tool_history_pty",
                created_at="2026-05-09T09:00:00Z",
                updated_at="2026-05-09T10:00:00Z",
                cwd=str(Path.cwd()),
            ),
            settings=session.SessionSettings(
                provider="openai_responses",
                model="gpt-5.5",
                system=None,
                max_tokens=4096,
            ),
            messages=[
                Message(role="user", content=[TextBlock(text="check feature request md")]),
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
                                "path: /tmp/project/feature_requests.md\\n"
                                "lines: 1-1 of 1\\n"
                                "     1\\tsearch sessions based on query"
                            ),
                        )
                    ],
                ),
                Message(
                    role="assistant",
                    content=[TextBlock(text="feature_requests.md contains:")],
                ),
            ],
        )
        args = argparse.Namespace(
            provider="openai_responses",
            model="gpt-5.5",
            max_tokens=4096,
            thinking=False,
            effort=None,
            prompt=None,
            persist_session=True,
            permission_mode=PermissionMode.YOLO,
            _resume_session_record=record,
            _resume_session_path=Path("/tmp/sess_tool_history_pty.jsonl"),
        )
        raise SystemExit(WattleApp(args, IdleProvider(), state=_state_from_session(record)).run())
        """
    )


def _clear_wattle_child_code() -> str:
    return textwrap.dedent(
        """
        import argparse

        from wattle.permissions import PermissionMode
        from wattle.providers import (
            CompletionResponse,
            Provider,
            StreamComplete,
            TextBlock,
            TextDelta,
        )
        from wattle.tui import WattleApp


        class ClearProvider(Provider):
            def __init__(self):
                self.calls = 0

            async def acomplete(self, request):
                self.calls += 1
                return CompletionResponse(
                    content=[TextBlock(text=f"done {self.calls}")],
                    stop_reason="end_turn",
                    usage={"input_tokens": 10 * self.calls, "output_tokens": self.calls},
                )

            async def astream(self, request):
                self.calls += 1
                yield TextDelta(text=f"done {self.calls}")
                yield StreamComplete(
                    CompletionResponse(
                        content=[TextBlock(text=f"done {self.calls}")],
                        stop_reason="end_turn",
                        usage={
                            "input_tokens": 10 * self.calls,
                            "output_tokens": self.calls,
                        },
                    )
                )


        args = argparse.Namespace(
            provider="openai_responses",
            model="gpt-5.5",
            max_tokens=4096,
            thinking=False,
            effort=None,
            prompt=None,
            persist_session=False,
            permission_mode=PermissionMode.YOLO,
        )
        raise SystemExit(WattleApp(args, ClearProvider()).run())
        """
    )


def _tool_rendering_child_code() -> str:
    return textwrap.dedent(
        """
        import argparse

        from wattle.permissions import PermissionMode
        from wattle.providers import (
            CompletionResponse,
            Provider,
            StreamComplete,
            TextBlock,
            TextDelta,
            ToolUseBlock,
            ToolUseDelta,
        )
        from wattle.tools import TOOLS_BY_NAME
        from wattle.tools.base import Tool
        from wattle.tui import WattleApp


        class FastBashTool(Tool):
            name = "bash"
            description = "Fast test bash."
            input_schema = {"type": "object", "properties": {"command": {"type": "string"}}}

            def run(self, command):
                return "hello"


        class FastWriteTool(Tool):
            name = "write"
            description = "Fast test write."
            input_schema = {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
            }

            def run(self, path, content):
                return "\\n".join(
                    [
                        f"Wrote {len(content)} bytes to {path}",
                        f"--- {path} (before)",
                        f"+++ {path} (after)",
                        "@@ -0,0 +1,2 @@",
                        "+def hello():",
                        "+    return 'world'",
                    ]
                )


        TOOLS_BY_NAME["bash"] = FastBashTool()
        TOOLS_BY_NAME["write"] = FastWriteTool()


        class ToolRenderingProvider(Provider):
            def __init__(self):
                self.calls = 0

            async def acomplete(self, request):
                return CompletionResponse(
                    content=[TextBlock(text="done")],
                    stop_reason="end_turn",
                    usage={},
                )

            async def astream(self, request):
                self.calls += 1
                if self.calls == 1:
                    yield ToolUseDelta(id="bash_1", name="bash", partial_json=None)
                    yield ToolUseDelta(id="write_1", name="write", partial_json=None)
                    yield StreamComplete(
                        CompletionResponse(
                            content=[
                                ToolUseBlock(
                                    id="bash_1",
                                    name="bash",
                                    input={
                                        "command": "echo hello | sed -n 1p",
                                    },
                                ),
                                ToolUseBlock(
                                    id="write_1",
                                    name="write",
                                    input={
                                        "path": "src/demo.py",
                                        "content": "def hello():\\\\n    return 'world'\\\\n",
                                    },
                                ),
                            ],
                            stop_reason="tool_use",
                            usage={},
                        )
                    )
                    return
                yield TextDelta(text="done")
                yield StreamComplete(
                    CompletionResponse(
                        content=[TextBlock(text="done")],
                        stop_reason="end_turn",
                        usage={},
                    )
                )


        args = argparse.Namespace(
            provider="openai_responses",
            model="gpt-5.5",
            max_tokens=4096,
            thinking=False,
            effort=None,
            prompt="render tools",
            persist_session=False,
            permission_mode=PermissionMode.YOLO,
        )
        raise SystemExit(WattleApp(args, ToolRenderingProvider()).run())
        """
    )


def _slow_second_write_child_code() -> str:
    return textwrap.dedent(
        """
        import argparse
        import time

        from wattle.permissions import PermissionMode
        from wattle.providers import (
            CompletionResponse,
            Provider,
            StreamComplete,
            TextBlock,
            TextDelta,
            ToolUseBlock,
            ToolUseDelta,
        )
        from wattle.tools import TOOLS_BY_NAME
        from wattle.tools.base import Tool
        from wattle.tui import WattleApp


        class SlowSecondWriteTool(Tool):
            name = "write"
            description = "Fast first write, slow second write."
            input_schema = {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
            }

            def run(self, path, content):
                if path == "src/slow.py":
                    time.sleep(1.0)
                return "\\n".join(
                    [
                        f"Wrote {len(content)} bytes to {path}",
                        f"--- {path} (before)",
                        f"+++ {path} (after)",
                        "@@ -0,0 +1,1 @@",
                        f"+{content.rstrip()}",
                    ]
                )


        TOOLS_BY_NAME["write"] = SlowSecondWriteTool()


        class TwoWriteProvider(Provider):
            def __init__(self):
                self.calls = 0

            async def acomplete(self, request):
                return CompletionResponse(
                    content=[TextBlock(text="done")],
                    stop_reason="end_turn",
                    usage={},
                )

            async def astream(self, request):
                self.calls += 1
                if self.calls == 1:
                    yield ToolUseDelta(id="write_fast", name="write", partial_json=None)
                    yield ToolUseDelta(id="write_slow", name="write", partial_json=None)
                    yield StreamComplete(
                        CompletionResponse(
                            content=[
                                ToolUseBlock(
                                    id="write_fast",
                                    name="write",
                                    input={"path": "src/fast.py", "content": "FAST_SENTINEL\\n"},
                                ),
                                ToolUseBlock(
                                    id="write_slow",
                                    name="write",
                                    input={"path": "src/slow.py", "content": "SLOW_SENTINEL\\n"},
                                ),
                            ],
                            stop_reason="tool_use",
                            usage={},
                        )
                    )
                    return
                yield TextDelta(text="done")
                yield StreamComplete(
                    CompletionResponse(
                        content=[TextBlock(text="done")],
                        stop_reason="end_turn",
                        usage={},
                    )
                )


        args = argparse.Namespace(
            provider="openai_responses",
            model="gpt-5.5",
            max_tokens=4096,
            thinking=False,
            effort=None,
            prompt="render slow writes",
            persist_session=False,
            permission_mode=PermissionMode.YOLO,
        )
        raise SystemExit(WattleApp(args, TwoWriteProvider()).run())
        """
    )


def _grouped_edit_child_code() -> str:
    return textwrap.dedent(
        """
        import argparse

        from wattle.permissions import PermissionMode
        from wattle.providers import (
            CompletionResponse,
            Provider,
            StreamComplete,
            TextBlock,
            TextDelta,
            ToolUseBlock,
            ToolUseDelta,
        )
        from wattle.tools import TOOLS_BY_NAME
        from wattle.tools.base import Tool
        from wattle.tui import WattleApp


        class FastEditTool(Tool):
            name = "edit"
            description = "Fast test edit."
            input_schema = {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "line": {"type": "integer"},
                    "before": {"type": "string"},
                    "after": {"type": "string"},
                },
            }

            def run(self, path, line, before, after):
                return "\\n".join(
                    [
                        f"Edited {path}",
                        f"--- {path} (before)",
                        f"+++ {path} (after)",
                        f"@@ -{line},1 +{line},1 @@",
                        f"-{before}",
                        f"+{after}",
                    ]
                )


        TOOLS_BY_NAME["edit"] = FastEditTool()


        class GroupedEditProvider(Provider):
            def __init__(self):
                self.calls = 0

            async def acomplete(self, request):
                return CompletionResponse(
                    content=[TextBlock(text="done")],
                    stop_reason="end_turn",
                    usage={},
                )

            async def astream(self, request):
                self.calls += 1
                if self.calls == 1:
                    yield ToolUseDelta(id="edit_1", name="edit", partial_json=None)
                    yield ToolUseDelta(id="edit_2", name="edit", partial_json=None)
                    yield StreamComplete(
                        CompletionResponse(
                            content=[
                                ToolUseBlock(
                                    id="edit_1",
                                    name="edit",
                                    input={
                                        "path": "src/demo.py",
                                        "line": 1,
                                        "before": "old_one",
                                        "after": "new_one",
                                    },
                                ),
                                ToolUseBlock(
                                    id="edit_2",
                                    name="edit",
                                    input={
                                        "path": "src/demo.py",
                                        "line": 7,
                                        "before": "old_two",
                                        "after": "new_two",
                                    },
                                ),
                            ],
                            stop_reason="tool_use",
                            usage={},
                        )
                    )
                    return
                yield TextDelta(text="done")
                yield StreamComplete(
                    CompletionResponse(
                        content=[TextBlock(text="done")],
                        stop_reason="end_turn",
                        usage={},
                    )
                )


        args = argparse.Namespace(
            provider="openai_responses",
            model="gpt-5.5",
            max_tokens=4096,
            thinking=False,
            effort=None,
            prompt="group edits",
            persist_session=False,
            permission_mode=PermissionMode.YOLO,
        )
        raise SystemExit(WattleApp(args, GroupedEditProvider()).run())
        """
    )


def _subagent_wait_child_code() -> str:
    return textwrap.dedent(
        """
        import argparse
        import time

        from wattle.permissions import PermissionMode
        from wattle.providers import (
            CompletionResponse,
            Provider,
            StreamComplete,
            TextBlock,
            TextDelta,
            ToolUseBlock,
            ToolUseDelta,
        )
        from wattle.tui import WattleApp


        def find_subagent_id(messages):
            for message in messages:
                for block in message.content:
                    content = getattr(block, "content", "")
                    if not isinstance(content, str):
                        continue
                    for line in content.splitlines():
                        if line.startswith("subagent_id: "):
                            return line.removeprefix("subagent_id: ")
            return "missing-subagent"


        class ChildProvider(Provider):
            async def acomplete(self, request):
                time.sleep(0.8)
                return CompletionResponse(
                    content=[TextBlock(text="child result")],
                    stop_reason="end_turn",
                    usage={},
                )

            async def astream(self, request):
                response = await self.acomplete(request)
                yield TextDelta(text="child result")
                yield StreamComplete(response)


        class ParentProvider(Provider):
            def __init__(self):
                self.calls = 0

            def fork(self):
                return ChildProvider()

            async def acomplete(self, request):
                return CompletionResponse(
                    content=[TextBlock(text="done")],
                    stop_reason="end_turn",
                    usage={},
                )

            async def astream(self, request):
                self.calls += 1
                if self.calls == 1:
                    yield ToolUseDelta(id="spawn_1", name="spawn_agent", partial_json=None)
                    yield StreamComplete(
                        CompletionResponse(
                            content=[
                                ToolUseBlock(
                                    id="spawn_1",
                                    name="spawn_agent",
                                    input={
                                        "task": "inspect prompt waiting state",
                                        "agent_type": "explorer",
                                    },
                                )
                            ],
                            stop_reason="tool_use",
                            usage={},
                        )
                    )
                    return
                if self.calls == 2:
                    subagent_id = find_subagent_id(request.messages)
                    yield ToolUseDelta(id="wait_1", name="wait_agent", partial_json=None)
                    yield StreamComplete(
                        CompletionResponse(
                            content=[
                                ToolUseBlock(
                                    id="wait_1",
                                    name="wait_agent",
                                    input={
                                        "subagent_id": subagent_id,
                                        "timeout_seconds": 3,
                                    },
                                )
                            ],
                            stop_reason="tool_use",
                            usage={},
                        )
                    )
                    return
                yield TextDelta(text="done")
                yield StreamComplete(
                    CompletionResponse(
                        content=[TextBlock(text="done")],
                        stop_reason="end_turn",
                        usage={},
                    )
                )


        args = argparse.Namespace(
            provider="openai_responses",
            model="gpt-5.5",
            max_tokens=4096,
            thinking=False,
            effort="xhigh",
            prompt="delegate",
            persist_session=False,
            permission_mode=PermissionMode.YOLO,
        )
        raise SystemExit(WattleApp(args, ParentProvider()).run())
        """
    )


def _research_child_code() -> str:
    return textwrap.dedent(
        """
        import argparse

        from wattle.permissions import PermissionMode
        from wattle.providers import (
            CompletionResponse,
            Provider,
            StreamComplete,
            TextBlock,
            TextDelta,
            ToolUseBlock,
            ToolUseDelta,
        )
        from wattle.tui import WattleApp


        class ResearchProvider(Provider):
            def __init__(self):
                self.calls = 0

            async def acomplete(self, request):
                return CompletionResponse(
                    content=[TextBlock(text="done")],
                    stop_reason="end_turn",
                    usage={},
                )

            async def astream(self, request):
                self.calls += 1
                if self.calls == 1:
                    yield ToolUseDelta(id="read_1", name="read", partial_json=None)
                    yield ToolUseDelta(id="read_2", name="read", partial_json=None)
                    yield ToolUseDelta(id="read_3", name="read", partial_json=None)
                    yield ToolUseDelta(id="read_4", name="read", partial_json=None)
                    yield StreamComplete(
                        CompletionResponse(
                            content=[
                                ToolUseBlock(
                                    id="read_1",
                                    name="read",
                                    input={"path": "notes.txt"},
                                ),
                                ToolUseBlock(
                                    id="read_2",
                                    name="read",
                                    input={"path": "other.txt"},
                                ),
                                ToolUseBlock(
                                    id="read_3",
                                    name="read",
                                    input={"path": "third.txt"},
                                ),
                                ToolUseBlock(
                                    id="read_4",
                                    name="read",
                                    input={"path": "notes.txt"},
                                ),
                            ],
                            stop_reason="tool_use",
                            usage={},
                        )
                    )
                    return
                if self.calls == 2:
                    yield ToolUseDelta(id="read_5", name="read", partial_json=None)
                    yield StreamComplete(
                        CompletionResponse(
                            content=[
                                ToolUseBlock(
                                    id="read_5",
                                    name="read",
                                    input={"path": "notes.txt"},
                                ),
                            ],
                            stop_reason="tool_use",
                            usage={},
                        )
                    )
                    return
                yield TextDelta(text="done")
                yield StreamComplete(
                    CompletionResponse(
                        content=[TextBlock(text="done")],
                        stop_reason="end_turn",
                        usage={},
                    )
                )


        args = argparse.Namespace(
            provider="openai_responses",
            model="gpt-5.5",
            max_tokens=4096,
            thinking=False,
            effort=None,
            prompt="research",
            persist_session=False,
            permission_mode=PermissionMode.YOLO,
        )
        raise SystemExit(WattleApp(args, ResearchProvider()).run())
        """
    )


def _plan_update_child_code() -> str:
    return textwrap.dedent(
        """
        import argparse

        from wattle.permissions import PermissionMode
        from wattle.providers import (
            CompletionResponse,
            Provider,
            StreamComplete,
            TextBlock,
            TextDelta,
            ToolUseBlock,
            ToolUseDelta,
        )
        from wattle.tui import WattleApp


        class PlanUpdateProvider(Provider):
            def __init__(self):
                self.calls = 0

            async def acomplete(self, request):
                return CompletionResponse(
                    content=[TextBlock(text="done")],
                    stop_reason="end_turn",
                    usage={},
                )

            async def astream(self, request):
                self.calls += 1
                if self.calls == 1:
                    yield ToolUseDelta(id="plan_1", name="update_plan", partial_json=None)
                    yield StreamComplete(
                        CompletionResponse(
                            content=[
                                ToolUseBlock(
                                    id="plan_1",
                                    name="update_plan",
                                    input={
                                        "explanation": "Moving into TUI checks.",
                                        "plan": [
                                            {
                                                "step": "Inspect current flow",
                                                "status": "completed",
                                            },
                                            {
                                                "step": "Add semantic plan cell",
                                                "status": "in_progress",
                                            },
                                            {
                                                "step": "Run PTY coverage",
                                                "status": "pending",
                                            },
                                        ],
                                    },
                                )
                            ],
                            stop_reason="tool_use",
                            usage={},
                        )
                    )
                    return
                yield TextDelta(text="done")
                yield StreamComplete(
                    CompletionResponse(
                        content=[TextBlock(text="done")],
                        stop_reason="end_turn",
                        usage={},
                    )
                )


        args = argparse.Namespace(
            provider="openai_responses",
            model="gpt-5.5",
            max_tokens=4096,
            thinking=False,
            effort=None,
            prompt="plan",
            persist_session=False,
            permission_mode=PermissionMode.YOLO,
        )
        raise SystemExit(WattleApp(args, PlanUpdateProvider()).run())
        """
    )


def test_pty_update_prompt_redraws_without_duplicate_options(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _update_prompt_child_code(),
        cwd=tmp_path,
        cols=100,
        rows=12,
    ) as session:
        session.read_until("Update from 0.2.0 to 0.2.1")
        assert "curl -fsSL" not in session.screen.text()

        session.write("\x1b[B")
        session.read_for(0.1)
        screen_text = session.screen.text()
        assert screen_text.count("Update from 0.2.0 to 0.2.1") == 1
        assert screen_text.count("Skip update") == 1
        assert " > Skip update" in screen_text

        session.write("\x1b[A")
        session.read_for(0.1)
        screen_text = session.screen.text()
        assert screen_text.count("Update from 0.2.0 to 0.2.1") == 1
        assert screen_text.count("Skip update") == 1
        assert " > Update from 0.2.0 to 0.2.1" in screen_text

        session.write("\x1b[B")
        session.read_for(0.1)
        session.write("\r")
        session.read_until("handled=False")
        assert session.process.wait(timeout=3) == 0


def test_pty_update_plan_renders_semantic_cell_without_tool_result(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _plan_update_child_code(),
        cwd=tmp_path,
        cols=100,
        rows=30,
    ) as session:
        session.read_until("Updated Plan", timeout=4)
        session.read_until("Add semantic plan cell", timeout=4)
        session.read_until("done", timeout=4)

        screen_text = session.screen.text()
        assert "Updated Plan" in screen_text
        assert "Moving into TUI checks." in screen_text
        assert "- [x] Inspect current flow" in screen_text
        assert "- [>] Add semantic plan cell" in screen_text
        assert "- [ ] Run PTY coverage" in screen_text
        assert "Plan updated" not in screen_text
        assert "update_plan ok" not in screen_text


def test_pty_startup_resume_picker_filters_query_and_resumes_match(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _resume_search_child_code(),
        cwd=tmp_path,
        cols=120,
        rows=36,
        env={"WATTLE_SESSION_DIR": str(tmp_path / "sessions")},
    ) as session:
        session.read_until("Resume Wattle Session", timeout=4)
        session.read_until("Newest Session", timeout=4)
        session.write("quota")
        session.read_until("Search: quota", timeout=4)
        session.read_until("Quota Investigation", timeout=4)

        screen_text = session.screen.text()
        assert "Quota Investigation" in screen_text
        assert "Newest Session" not in screen_text

        session.write("\x7f")
        session.read_until("Search: quot", timeout=4)
        session.write("a")
        session.read_until("Search: quota", timeout=4)
        session.write("\x1b")
        session.read_until("Search:", timeout=4)
        session.read_until("Newest Session", timeout=4)
        session.write("quota")
        session.read_until("Quota Investigation", timeout=4)
        session.write("\n")
        session.read_until("Loaded 1 saved message(s)", timeout=4)
        session.read_until("inspect weekly limits", timeout=4)

        final_text = session.screen.text()
        assert "inspect weekly limits" in final_text
        assert "recent unrelated work" not in final_text


def test_pty_history_tool_replay_and_resize_use_semantic_rendering(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _history_tool_child_code(),
        cwd=tmp_path,
        cols=120,
        rows=36,
    ) as session:
        session.read_until("Read feature_requests.md", timeout=4)
        session.read_until("feature_requests.md contains:", timeout=4)

        screen_text = session.screen.text()
        assert "Researched" in screen_text
        assert "Read feature_requests.md" in screen_text
        assert "feature_requests.md contains:" in screen_text
        assert "tool use" not in screen_text
        assert "tool result" not in screen_text
        assert "path: /tmp/project/feature_requests.md" not in screen_text

        session.resize(cols=80, rows=36)
        session.read_for(0.35)

        resized_screen_text = session.screen.text()
        assert "Read feature_requests.md" in resized_screen_text
        assert "feature_requests.md contains:" in resized_screen_text
        assert "tool use" not in resized_screen_text
        assert "tool result" not in resized_screen_text
        assert "path: /tmp/project/feature_requests.md" not in resized_screen_text


def test_pty_research_tool_calls_render_aggregate(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "other.txt").write_text("more\n", encoding="utf-8")
    (tmp_path / "third.txt").write_text("final\n", encoding="utf-8")

    with PtySession.spawn_python(
        _research_child_code(),
        cwd=tmp_path,
        cols=100,
        rows=30,
    ) as session:
        session.read_until("Researched", timeout=4)
        session.read_until("done", timeout=4)

        screen_text = session.screen.text()
        assert screen_text.count("Read notes.txt") == 1
        assert "Read notes.txt" in screen_text
        assert "Read other.txt" in screen_text
        assert "Read third.txt" in screen_text
        assert "read ok - notes.txt" not in screen_text


def test_pty_resize_preserves_completed_inflight_diff_rendering(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _slow_second_write_child_code(),
        cwd=tmp_path,
        cols=120,
        rows=36,
    ) as session:
        session.read_until("Added src/fast.py", timeout=4)

        session.resize(cols=80, rows=36)
        session.read_for(0.35)

        resized_screen_text = session.screen.text()
        assert "Added src/fast.py (+1 -0)" in resized_screen_text
        assert "    1 +FAST_SENTINEL" in resized_screen_text
        assert "tool result" not in resized_screen_text
        assert "Wrote 14 bytes to src/fast.py" not in resized_screen_text


def test_pty_bash_exec_cell_finalizes_with_output(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _tool_rendering_child_code(),
        cwd=tmp_path,
        cols=120,
        rows=36,
    ) as session:
        session.read_until("Ran echo", timeout=4)
        session.read_until("Added src/demo.py", timeout=4)

        screen_text = session.screen.text()
        assert "Ran echo hello | sed -n 1p" in screen_text
        assert "  └ hello" in screen_text
        assert screen_text.count("Ran echo hello | sed -n 1p") == 1
        assert "  hello" not in screen_text


def test_pty_tool_rendering_uses_distinct_command_and_diff_styles(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _tool_rendering_child_code(),
        cwd=tmp_path,
        cols=120,
        rows=36,
    ) as session:
        session.read_until("Ran echo", timeout=4)
        session.read_until("Added src/demo.py", timeout=4)
        session.read_until("done", timeout=4)

        screen_text = session.screen.text()
        assert "Ran echo hello | sed -n 1p" in screen_text
        assert "Added src/demo.py (+2 -0)" in screen_text
        assert "    1 +def hello():" in screen_text
        assert "    2 +    return 'world'" in screen_text

        raw = session.raw_output
        assert "\x1b[38;5;75;1mecho\x1b[0m" in raw
        assert "\x1b[38;5;80;1m|\x1b[0m" in raw
        assert "\x1b[38;5;159;1msrc/demo.py\x1b[0m" in raw
        assert "\x1b[48;5;22;38;5;72m    1 " in raw
        assert "\x1b[48;5;22;38;5;40m+" in raw
        assert f"{tui.DIFF_ADD_SYNTAX_KEYWORD_STYLE}def" in raw
        assert f"{tui.DIFF_ADD_SYNTAX_NAME_STYLE}hello" in raw


def test_pty_groups_adjacent_same_file_edit_results(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _grouped_edit_child_code(),
        cwd=tmp_path,
        cols=120,
        rows=36,
    ) as session:
        session.read_until("Edited src/demo.py", timeout=4)
        session.read_until("done", timeout=4)

        screen_text = session.screen.text()
        assert screen_text.count("Edited src/demo.py") == 1
        assert "Edited src/demo.py (+2 -2)" in screen_text
        assert "    1 -old_one" in screen_text
        assert "    1 +new_one" in screen_text
        assert "      ..." in screen_text
        assert "    7 -old_two" in screen_text
        assert "    7 +new_two" in screen_text
        assert screen_text.index("    1 -old_one") < screen_text.index("    7 -old_two")


def test_pty_subagent_waiting_and_completion_notifications(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _subagent_wait_child_code(),
        cwd=tmp_path,
        cols=120,
        rows=36,
    ) as session:
        session.read_until("Spawned Hopper [explorer] (gpt-5.5 xhigh)", timeout=4)
        session.read_until("Waiting for 1 agent", timeout=4)
        session.read_until("▸ input", timeout=4)
        session.read_until("○ main", timeout=4)
        session.read_until("○ Hopper explorer running", timeout=4)
        session.read_until("Hopper [explorer] complete", timeout=6)

        screen_text = session.screen.text()
        assert "Waiting for subagent(s)" not in screen_text
        assert "Agents:" not in screen_text
        assert "Subagents ·" not in screen_text
        assert "inspect prompt waiting state" in screen_text
        assert "subagent_id:" not in screen_text


def test_pty_dragged_image_uses_anchor_while_queued_and_after_finish(tmp_path: Path) -> None:
    image = tmp_path / "dragged image.png"
    image.write_bytes(b"fake-png")
    escaped_path = str(image).replace(" ", "\\ ")

    with PtySession.spawn_python(
        _slow_wattle_child_code(first_delay=2.0, later_delay=0.1),
        cwd=tmp_path,
        cols=100,
        rows=30,
    ) as session:
        session.read_until("press esc to interrupt", timeout=3)
        session.write(f"check {escaped_path}\n")
        session.read_until("Messages to be submitted after next tool call", timeout=3)
        session.read_until("check [image#1]", timeout=3)

        screen_text = session.screen.text()
        assert "check [image#1]" in screen_text
        assert "dragged\\ image.png" not in screen_text
        assert str(image) not in screen_text

        session.read_until("Worked for", timeout=5)
        screen_text = session.screen.text()
        assert "check [image#1]" in screen_text
        assert str(image) not in screen_text


def test_pty_resize_keeps_user_image_history_as_anchor(tmp_path: Path) -> None:
    image = tmp_path / "dragged image.png"
    image.write_bytes(b"fake-png")
    escaped_path = str(image).replace(" ", "\\ ")

    with PtySession.spawn_python(
        _slow_wattle_child_code(first_delay=0.1, later_delay=0.1, prompt=None),
        cwd=tmp_path,
        cols=100,
        rows=30,
    ) as session:
        session.read_until("gpt-5.5 |", timeout=3)
        session.write(f"check {escaped_path}\n")
        session.read_until("check [image#1]", timeout=3)
        session.read_until("done 1", timeout=3)

        session.resize(cols=80, rows=36)
        session.read_until("check [image#1]", timeout=3)

        screen_text = session.screen.text()
        assert "check [image#1]" in screen_text
        assert "[image] dragged image.png" not in screen_text
        assert str(image) not in screen_text


def test_pty_statusline_picker_toggles_and_persists_fields(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    with PtySession.spawn_python(
        _statusline_picker_child_code(settings_path),
        cwd=tmp_path,
        cols=100,
        rows=30,
    ) as session:
        session.read_until("gpt-5.5 | thinking: off", timeout=3)
        session.write("/statusline")
        session.read_until("x to select/deselect", timeout=3)
        session.read_until("> [x] model", timeout=3)
        session.write("x")
        session.read_until("> [ ] model", timeout=3)
        session.write("\x1b[B\x1b[B\x1b[B")
        session.read_until("> [ ] context_remaining", timeout=3)
        session.write("x\n")
        session.read_until("Statusline fields: thinking, context_remaining, cwd", timeout=3)
        session.write("/exit\n")

    saved = json.loads(settings_path.read_text(encoding="utf-8"))
    assert saved["tui"]["statusline"] == ["thinking", "context_remaining", "cwd"]


def test_pty_up_arrow_moves_cursor_to_previous_wrapped_input_row(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _echo_prompt_child_code(),
        cwd=tmp_path,
        cols=10,
        rows=20,
    ) as session:
        session.read_until(">", timeout=3)
        session.write("abcdefghijkl")
        session.read_until("hijkl", timeout=3)
        session.write("\x1b[AX\n")
        session.read_until("abcdeXfghijkl", timeout=3)
        session.write("/exit\n")


def test_pty_queue_command_renders_end_turn_followup_panel(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _slow_wattle_child_code(first_delay=1.5, later_delay=0.1),
        cwd=tmp_path,
        cols=100,
        rows=30,
    ) as session:
        session.read_until("press esc to interrupt", timeout=3)
        session.write("/queue after the full turn\n")
        session.read_until(
            "Messages to be submitted after assistant turn completes",
            timeout=3,
        )
        session.read_until("after the full turn", timeout=3)

        screen_text = session.screen.text()
        assert "Messages to be submitted after next tool call" not in screen_text
        assert "after the full turn" in screen_text


def test_pty_tab_queues_input_for_end_turn_followup(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _slow_wattle_child_code(first_delay=1.5, later_delay=0.1),
        cwd=tmp_path,
        cols=100,
        rows=30,
    ) as session:
        session.read_until("press esc to interrupt", timeout=3)
        session.write("after the full turn")
        session.read_until("press Enter to queue after next tool call", timeout=3)
        session.read_until("Tab for next turn", timeout=3)
        session.write("\t")
        session.read_until(
            "Messages to be submitted after assistant turn completes",
            timeout=3,
        )
        session.read_until("after the full turn", timeout=3)

        screen_text = session.screen.text()
        assert "Messages to be submitted after next tool call" not in screen_text
        assert "Messages to be submitted after assistant turn completes" in screen_text
        assert "after the full turn" in screen_text


def test_pty_enter_during_tool_is_guidance_for_active_task(tmp_path: Path) -> None:
    child_code = textwrap.dedent(
        """
        import argparse
        import time

        from wattle.permissions import PermissionMode
        from wattle.providers import (
            CompletionResponse,
            Provider,
            StreamComplete,
            TextBlock,
            TextDelta,
            ToolUseBlock,
        )
        from wattle.tools import TOOLS_BY_NAME
        from wattle.tools.base import Tool
        from wattle.tui import WattleApp


        class SlowGuidanceTool(Tool):
            name = "slow_guidance"
            description = "Wait briefly."
            input_schema = {"type": "object", "properties": {}}

            def run(self, **_kwargs):
                time.sleep(1.2)
                return "tool complete"


        class GuidanceProvider(Provider):
            def __init__(self):
                self.calls = 0

            async def acomplete(self, request):
                raise NotImplementedError

            async def astream(self, request):
                self.calls += 1
                if self.calls == 1:
                    yield StreamComplete(
                        CompletionResponse(
                            content=[
                                ToolUseBlock(
                                    id="call_1",
                                    name="slow_guidance",
                                    input={},
                                )
                            ],
                            stop_reason="tool_use",
                            usage={},
                        )
                    )
                    return

                texts = [
                    block.text
                    for block in request.messages[-1].content
                    if isinstance(block, TextBlock)
                ]
                joined = "\\n".join(texts)
                if (
                    "additional guidance for the active task" in joined
                    and "hello" in joined
                ):
                    text = "continued active task"
                else:
                    text = "stopped on hello"
                yield TextDelta(text=text)
                yield StreamComplete(
                    CompletionResponse(
                        content=[TextBlock(text=text)],
                        stop_reason="end_turn",
                        usage={},
                    )
                )


        TOOLS_BY_NAME["slow_guidance"] = SlowGuidanceTool()
        args = argparse.Namespace(
            provider="openai_responses",
            model="gpt-5.5",
            max_tokens=4096,
            thinking=False,
            effort=None,
            prompt="start",
            persist_session=False,
            permission_mode=PermissionMode.YOLO,
        )
        raise SystemExit(WattleApp(args, GuidanceProvider()).run())
        """
    )

    with PtySession.spawn_python(child_code, cwd=tmp_path, cols=100, rows=30) as session:
        session.read_until("running slow_guidance", timeout=3)
        session.write("hello\n")
        session.read_until("Messages to be submitted after next tool call", timeout=3)
        session.read_until("continued active task", timeout=5)

        screen_text = session.screen.text()
        assert "continued active task" in screen_text
        assert "stopped on hello" not in screen_text
        session.write("/exit\n")


def test_pty_shift_tab_cycles_thinking_level(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _slow_wattle_child_code(first_delay=0.1, later_delay=0.1, prompt=None),
        cwd=tmp_path,
        cols=100,
        rows=30,
    ) as session:
        session.read_until("gpt-5.5 |", timeout=3)
        session.write("\x1b[Z")
        session.read_until("thinking: low", timeout=3)
        session.write("\x1b[Z")
        session.read_until("thinking: medium", timeout=3)

        screen_text = session.screen.text()
        assert "thinking: medium" in screen_text
        assert "thinking: low" not in screen_text


def test_pty_ctrl_v_pastes_clipboard_image_as_anchor(tmp_path: Path) -> None:
    child_code = textwrap.dedent(
        """
        import argparse

        from wattle.permissions import PermissionMode
        from wattle.providers import CompletionResponse, Provider, StreamComplete, TextBlock
        from wattle.tui import ClipboardImage, WattleApp
        import wattle.tui as tui

        tui.read_clipboard_image = lambda: ClipboardImage(b"fake-png", "image/png", ".png")

        class ProviderStub(Provider):
            async def acomplete(self, request):
                return CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")

            async def astream(self, request):
                yield StreamComplete(
                    CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")
                )

        args = argparse.Namespace(
            provider="openai_responses",
            model="gpt-5.5",
            max_tokens=4096,
            thinking=False,
            effort=None,
            prompt=None,
            persist_session=False,
            permission_mode=PermissionMode.YOLO,
        )
        raise SystemExit(WattleApp(args, ProviderStub()).run())
        """
    )

    with PtySession.spawn_python(child_code, cwd=tmp_path, cols=60, rows=30) as session:
        session.read_until("gpt-5.5 |", timeout=3)
        session.write("\x16")
        session.read_until("[image#1]", timeout=3)
        session.write("\x1b[27;5;118~")
        session.read_until("[image#2]", timeout=3)

        screen_text = session.screen.text()
        assert "[image#1]" in screen_text
        assert "[image#2]" in screen_text
        assert "^V" not in screen_text
        assert "\x16" not in screen_text


def test_pty_dragged_image_uses_anchor_in_active_input(tmp_path: Path) -> None:
    image = tmp_path / "dragged image.png"
    image.write_bytes(b"fake-png")
    escaped_path = str(image).replace(" ", "\\ ")

    with PtySession.spawn_python(
        _slow_wattle_child_code(first_delay=2.0, later_delay=0.1),
        cwd=tmp_path,
        cols=60,
        rows=30,
    ) as session:
        session.read_until("press esc to interrupt", timeout=3)
        session.write(f"check {escaped_path}")
        session.read_until("check [image#1]", timeout=3)

        screen_text = session.screen.text()
        assert "check [image#1]" in screen_text
        assert "dragged\\ image.png" not in screen_text
        assert str(image) not in screen_text


def _input_box_rows(screen: object) -> tuple[int, int, int]:
    row = screen.find_row_containing(" > ")
    return row - 1, row, row + 1


def _assert_single_three_row_input_box(screen: object) -> None:
    rows = _input_box_rows(screen)
    for row_index in rows:
        backgrounds = screen.row_backgrounds(row_index)
        assert "black" not in backgrounds
        assert all(background == "ansi-235" for background in backgrounds)
    prompt_backgrounds = screen.row_backgrounds(rows[1])
    assert any(background == "ansi-235" for background in prompt_backgrounds)
    before_row = rows[0] - 1
    after_row = rows[2] + 1
    if before_row >= 0:
        assert any(background != "ansi-235" for background in screen.row_backgrounds(before_row))
    if after_row < screen.rows:
        assert any(background != "ansi-235" for background in screen.row_backgrounds(after_row))


def _assert_no_right_side_prompt_or_status_stripes(screen: object) -> None:
    for row_index in range(screen.rows):
        backgrounds = screen.row_backgrounds(row_index)
        for background in ("ansi-235", "ansi-236"):
            if background not in backgrounds:
                continue
            first_index = backgrounds.index(background)
            assert first_index == 0, (
                f"row {row_index} has orphan {background} stripe starting at "
                f"column {first_index}: {screen.row_text(row_index)!r}"
            )


def test_pty_repeated_resize_keeps_black_out_of_input_box(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _slow_wattle_child_code(first_delay=1.4, later_delay=0.1),
        cwd=tmp_path,
        cols=90,
        rows=28,
    ) as session:
        session.read_until("press esc to interrupt", timeout=3)
        session.write("queued input that stays visible during resize")
        session.read_until("queued input", timeout=3)

        for cols in (38, 120, 24, 96):
            session.resize(cols=cols, rows=28)
            session.read_for(0.18)

        row = session.screen.find_row_containing(" > ")
        text = session.screen.row_text(row)
        assert "queued input" in text

        _assert_single_three_row_input_box(session.screen)
        _assert_no_right_side_prompt_or_status_stripes(session.screen)


def test_pty_resize_does_not_leave_reflowed_prompt_box_rows(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _slow_wattle_child_code(first_delay=2.0, later_delay=0.1),
        cwd=tmp_path,
        cols=120,
        rows=30,
    ) as session:
        session.read_until("press esc to interrupt", timeout=3)
        session.read_until(" > ", timeout=3)
        session.resize(cols=40, rows=30)
        session.read_for(0.35)

        _assert_single_three_row_input_box(session.screen)
        _assert_no_right_side_prompt_or_status_stripes(session.screen)


def test_pty_exit_command_works_while_assistant_is_working(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _slow_wattle_child_code(first_delay=3.0, later_delay=0.1),
        cwd=tmp_path,
        cols=100,
        rows=30,
    ) as session:
        session.read_until("press esc to interrupt", timeout=3)
        session.write("/exit\n")
        session.read_until("Goodbye.", timeout=3)

        assert "Messages to be submitted after next tool call" not in session.plain_output
        assert "done 1" not in session.plain_output
        assert session.process.wait(timeout=3) == 0


def test_pty_idle_resize_keeps_statusline_on_terminal_background(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _slow_wattle_child_code(first_delay=0.1, later_delay=0.1),
        cwd=tmp_path,
        cols=44,
        rows=24,
    ) as session:
        session.read_until("Worked for", timeout=3)
        session.resize(cols=96, rows=24)
        session.read_for(0.25)

        row = session.screen.find_row_containing("gpt-5.5 |")
        text = session.screen.row_text(row)
        assert text.startswith(" gpt-5.5")
        backgrounds = session.screen.row_backgrounds(row)
        assert all(background is None for background in backgrounds)


def test_pty_resize_redraw_keeps_single_welcome_and_transcript(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _slow_wattle_child_code(first_delay=0.1, later_delay=0.1, prompt=None),
        cwd=tmp_path,
        cols=120,
        rows=40,
    ) as session:
        session.read_until(">", timeout=3)
        session.write("hello\n")
        session.read_until("done 1", timeout=3)
        session.read_until("Worked for", timeout=3)
        session.read_until("gpt-5.5 |", timeout=3)
        session.resize(cols=80, rows=40)
        session.read_for(0.35)

        assert session.screen.text().count("Wattle Agent") == 1
        assert session.screen.text().count(" hello") == 1
        assert session.screen.text().count(" done 1") == 1


def test_pty_resize_does_not_insert_rules_around_user_message(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _slow_wattle_child_code(first_delay=0.1, later_delay=0.1, prompt=None),
        cwd=tmp_path,
        cols=120,
        rows=42,
    ) as session:
        session.read_until(">", timeout=3)
        session.write("resize rule regression\n")
        session.read_until("done 1", timeout=3)
        session.read_until("Worked for", timeout=3)

        session.resize(cols=80, rows=42)
        session.read_for(0.35)

        user_row = session.screen.find_row_containing(" resize rule regression")
        for row in (user_row - 1, user_row, user_row + 1):
            assert all(
                background == "ansi-235"
                for background in session.screen.row_backgrounds(row)
            )

        above_text = session.screen.row_text(user_row - 2).strip()
        below_text = session.screen.row_text(user_row + 2).strip()
        assert not above_text or set(above_text) != {"─"}
        assert not below_text or set(below_text) != {"─"}


def test_pty_clear_redraws_clean_session_screen(tmp_path: Path) -> None:
    with PtySession.spawn_python(
        _clear_wattle_child_code(),
        cwd=tmp_path,
        cols=96,
        rows=36,
    ) as session:
        session.read_until(">", timeout=3)
        session.write("first\n")
        session.read_until("done 1", timeout=3)
        session.read_until("Worked for", timeout=3)

        session.write("/clear\n")
        session.read_until("Last session usage", timeout=3)
        assert "\x1b[3J" in session.raw_output

        screen_text = session.screen.text()
        assert "Wattle Agent" in screen_text
        assert "Last session usage: last context: 10 tok" in screen_text
        assert "first" not in screen_text
        assert "done 1" not in screen_text
        assert "Conversation cleared." not in screen_text

        session.write("draft")
        session.read_until("draft", timeout=3)
        session.resize(cols=120, rows=42)
        session.read_for(0.35)

        resized_screen_text = session.screen.text()
        assert "Wattle Agent" in resized_screen_text
        assert "Last session usage: last context: 10 tok" in resized_screen_text
        assert "draft" in resized_screen_text
        assert "first" not in resized_screen_text
        assert "done 1" not in resized_screen_text
        assert session.raw_output.count("\x1b[H\x1b[2J\x1b[H") >= 2

        session.write("\x15")
        session.write("second\n")
        session.read_until("done 2", timeout=3)
        assert "done 2" in session.screen.text()


def test_pty_resize_does_not_leave_large_gap_between_user_and_assistant(
    tmp_path: Path,
) -> None:
    with PtySession.spawn_python(
        _slow_wattle_child_code(first_delay=1.0, later_delay=0.1, prompt=None),
        cwd=tmp_path,
        cols=48,
        rows=60,
    ) as session:
        session.read_until(">", timeout=3)
        session.write("hello\n")
        session.read_until("press esc to interrupt", timeout=3)
        for cols in (28, 96, 36):
            session.resize(cols=cols, rows=60)
            session.read_for(0.18)
        session.read_until("done 1", timeout=4)

        user_row = session.screen.find_row_containing(" hello")
        assistant_row = session.screen.find_row_containing(" done 1")
        assert assistant_row - user_row == 3


def test_pty_transcript_message_reflows_when_terminal_width_grows(
    tmp_path: Path,
) -> None:
    message = "this submitted message should reflow after the terminal gets wider"
    with PtySession.spawn_python(
        _slow_wattle_child_code(first_delay=0.1, later_delay=0.1, prompt=None),
        cwd=tmp_path,
        cols=24,
        rows=40,
    ) as session:
        session.read_until(">", timeout=3)
        session.write(f"{message}\n")
        session.read_until("done 1", timeout=3)

        session.resize(cols=96, rows=40)
        session.read_for(0.25)

        user_row = session.screen.find_row_containing(f" {message}")
        assert session.screen.row_text(user_row).rstrip() == f" {message}"


def test_pty_transcript_rows_do_not_store_zoom_reflow_fill_spaces(
    tmp_path: Path,
) -> None:
    with PtySession.spawn_python(
        _slow_wattle_child_code(first_delay=0.1, later_delay=0.1, prompt=None),
        cwd=tmp_path,
        cols=120,
        rows=32,
    ) as session:
        session.read_until(">", timeout=3)
        session.write("zoom regression\n")
        session.read_until("done 1", timeout=3)
        session.read_until("Worked for", timeout=3)

        user_row = session.screen.find_row_containing(" zoom regression")
        assistant_row = session.screen.find_row_containing(" done 1")
        assert session.screen.row_text(user_row).rstrip() == " zoom regression"
        assert session.screen.row_text(assistant_row).rstrip() == " done 1"

        for row in (user_row, assistant_row):
            backgrounds = session.screen.row_backgrounds(row)
            text = session.screen.row_text(row)
            visible_columns = [index for index, char in enumerate(text) if char != " "]
            assert visible_columns
            expected_background = "ansi-235" if row == user_row else None
            assert all(backgrounds[index] == expected_background for index in visible_columns)
            assert all(background == expected_background for background in backgrounds)
