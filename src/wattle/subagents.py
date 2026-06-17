from __future__ import annotations

import asyncio
import concurrent.futures
import queue
import threading
import time
import uuid
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal, cast

from wattle.loop import dispatch_tool_blocks_async
from wattle.message_history import monitor_event_text_blocks
from wattle.models import context_window_for_model, effort_levels_for_model, model_supports_modality
from wattle.permissions import PermissionGate, PermissionMode
from wattle.providers import (
    CompletionResponse,
    ContentBlock,
    Message,
    Provider,
    TextBlock,
    ToolUseBlock,
)
from wattle.request_preparation import RequestPreparer, acomplete_with_recovery
from wattle.skills import load_available_skills
from wattle.system_prompt import DEFAULT_SYSTEM_PROMPT, build_system_prompt
from wattle.tools.base import Tool, ToolSpec

SubagentStatus = Literal[
    "pending",
    "running",
    "completed",
    "failed",
    "closing",
    "closed",
]

COLLABORATION_TOOL_NAMES = frozenset(
    {"spawn_agent", "send_input", "wait_agent", "close_agent"}
)

DEFAULT_AGENT_TYPE = "default"


def _tools_for_model(tools_by_name: Mapping[str, Tool], model: str) -> dict[str, Tool]:
    if model_supports_modality(model, "image"):
        return dict(tools_by_name)
    return {name: tool for name, tool in tools_by_name.items() if name != "view_image"}


SUBAGENT_DISPLAY_NAMES = (
    "Hopper",
    "Grace",
    "Ada",
    "Lovelace",
    "Ampere",
    "Turing",
    "Volta",
    "Pascal",
    "Maxwell",
    "Kepler",
    "Fermi",
    "Blackwell",
    "Rubin",
    "Vera",
    "Thor",
    "Orin",
    "Xavier",
)


@dataclass
class SubagentRecord:
    subagent_id: str
    display_name: str
    role: str
    task: str
    instructions: str | None
    context: str | None
    model: str
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None
    workspace: str | None
    tool_names: list[str]
    status: SubagentStatus
    started_at: float
    updated_at: float
    ended_at: float | None = None
    result: str | None = None
    error: str | None = None
    turns: int = 0


@dataclass
class _SubagentConfig:
    provider: Provider
    tools_by_name: Mapping[str, Tool]
    full_tools_by_name: Mapping[str, Tool]
    system: str | None
    model: str
    max_tokens: int | None
    permission_gate: PermissionGate | None
    context_window: int | None
    thinking: bool
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None


@dataclass
class _SubagentSession:
    record: SubagentRecord
    system: str | None
    tools_by_name: dict[str, Tool]
    tool_specs: list[ToolSpec]
    provider: Provider
    max_tokens: int | None
    permission_gate: PermissionGate | None
    context_window: int | None
    thinking: bool
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None
    messages: list[Message] = field(default_factory=list)
    future: concurrent.futures.Future[None] | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)
    close_requested: bool = False
    update_queue: queue.Queue[str] = field(default_factory=queue.Queue)


class SubagentManager:
    """Runtime-owned managed subagent sessions.

    Subagents run in Wattle-managed threads and use the same provider/tool
    abstraction as the parent. They are not implemented by shelling out to a
    second Wattle CLI process.
    """

    def __init__(self) -> None:
        self._config: _SubagentConfig | None = None
        self._sessions: dict[str, _SubagentSession] = {}
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None

    def configure(
        self,
        *,
        provider: Provider,
        tools_by_name: Mapping[str, Tool],
        system: str | None,
        model: str,
        full_tools_by_name: Mapping[str, Tool] | None = None,
        max_tokens: int | None,
        permission_gate: PermissionGate | None = None,
        context_window: int | None = None,
        thinking: bool = False,
        effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
    ) -> None:
        with self._lock:
            self._config = _SubagentConfig(
                provider=provider,
                tools_by_name=tools_by_name,
                full_tools_by_name=full_tools_by_name or tools_by_name,
                system=system,
                model=model,
                max_tokens=max_tokens,
                permission_gate=permission_gate,
                context_window=context_window,
                thinking=thinking,
                effort=effort,
            )

    def spawn(
        self,
        *,
        task: str,
        agent_type: str | None = None,
        instructions: str | None = None,
        context: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        tool_names: list[str] | None = None,
    ) -> SubagentRecord:
        config = self._require_config()
        clean_task = task.strip()
        if not clean_task:
            raise ValueError("task must not be empty")

        resolved_model = model or config.model
        available_tools = _tools_for_model(config.full_tools_by_name, resolved_model)
        selected_tools = self._select_tools(available_tools, tool_names)
        resolved_effort = _effort_for_model(config.effort, resolved_model)
        resolved_context_window = context_window_for_model(resolved_model)
        if resolved_context_window is None:
            resolved_context_window = config.context_window
        resolved_agent_type = _normalize_agent_type(agent_type)
        subagent_id = f"subagent-{uuid.uuid4().hex[:12]}"
        now = time.time()
        with self._lock:
            display_name = SUBAGENT_DISPLAY_NAMES[
                len(self._sessions) % len(SUBAGENT_DISPLAY_NAMES)
            ]
        record = SubagentRecord(
            subagent_id=subagent_id,
            display_name=display_name,
            role=resolved_agent_type,
            task=clean_task,
            instructions=instructions.strip() if instructions else None,
            context=context.strip() if context else None,
            model=resolved_model,
            effort=resolved_effort,
            workspace=_workspace_from_tools(selected_tools),
            tool_names=sorted(selected_tools),
            status="pending",
            started_at=now,
            updated_at=now,
        )
        session = _SubagentSession(
            record=record,
            system=self._subagent_system(
                config.system,
                selected_tools,
                record.instructions,
                permission_mode=config.permission_gate.mode
                if config.permission_gate is not None
                else PermissionMode.YOLO,
            ),
            tools_by_name=selected_tools,
            tool_specs=[tool.spec() for tool in selected_tools.values()],
            provider=config.provider.fork(),
            max_tokens=max_tokens or config.max_tokens,
            permission_gate=config.permission_gate,
            context_window=resolved_context_window,
            thinking=config.thinking and resolved_effort is not None,
            effort=resolved_effort,
        )
        with self._lock:
            self._sessions[subagent_id] = session
        self._start_session_turn(session, self._initial_user_text(record))
        return self.snapshot_record(subagent_id)

    def send_input(self, subagent_id: str, message: str) -> SubagentRecord:
        clean_message = message.strip()
        if not clean_message:
            raise ValueError("message must not be empty")
        session = self._require_session(subagent_id)
        with session.lock:
            if session.record.status in {"running", "pending", "closing"}:
                raise RuntimeError(f"{subagent_id} is already {session.record.status}")
            if session.record.status == "closed" or session.close_requested:
                raise RuntimeError(f"{subagent_id} is closed")
            while not session.update_queue.empty():
                with suppress(queue.Empty):
                    session.update_queue.get_nowait()
            session.record.status = "pending"
            session.record.error = None
            session.record.result = None
            session.record.ended_at = None
            session.record.updated_at = time.time()
        self._start_session_turn(session, clean_message)
        return self.snapshot_record(subagent_id)

    def wait(self, subagent_id: str, timeout_seconds: float | None = None) -> SubagentRecord:
        session = self._require_session(subagent_id)
        deadline = (
            None
            if timeout_seconds is None
            else time.monotonic() + max(0.0, timeout_seconds)
        )
        while self._is_running(session):
            timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
            if timeout == 0.0:
                break
            with suppress(queue.Empty):
                session.update_queue.get(
                    timeout=timeout,
                )
                continue
            break
        return self.snapshot_record(subagent_id)

    async def await_wait(
        self,
        subagent_id: str,
        timeout_seconds: float | None = None,
    ) -> SubagentRecord:
        session = self._require_session(subagent_id)
        deadline = (
            None
            if timeout_seconds is None
            else time.monotonic() + max(0.0, timeout_seconds)
        )
        while self._is_running(session):
            timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
            if timeout == 0.0:
                break
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    asyncio.to_thread(session.update_queue.get),
                    timeout=timeout,
                )
                continue
            break
        return self.snapshot_record(subagent_id)

    def close(self, subagent_id: str) -> SubagentRecord:
        session = self._require_session(subagent_id)
        with session.lock:
            session.close_requested = True
            if session.record.status in {"pending", "running"}:
                session.record.status = "closing"
            else:
                session.record.status = "closed"
                session.record.ended_at = time.time()
            session.record.updated_at = time.time()
        return self.snapshot_record(subagent_id)

    async def await_close(self, subagent_id: str) -> SubagentRecord:
        return self.close(subagent_id)

    def snapshot_record(self, subagent_id: str) -> SubagentRecord:
        session = self._require_session(subagent_id)
        with session.lock:
            return SubagentRecord(**asdict(session.record))

    def snapshot(self, subagent_id: str) -> dict[str, object]:
        session = self._require_session(subagent_id)
        with session.lock:
            snapshot = asdict(session.record)
            snapshot["messages"] = tuple(_copy_message(message) for message in session.messages)
        return snapshot

    def snapshots(self) -> list[dict[str, object]]:
        with self._lock:
            ids = list(self._sessions)
        snapshots: list[dict[str, object]] = []
        for launch_index, subagent_id in enumerate(ids):
            snapshot = self.snapshot(subagent_id)
            snapshot["launch_index"] = launch_index
            snapshots.append(snapshot)
        return snapshots

    def cleanup(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            with session.lock:
                session.close_requested = True
                if session.record.status in {"pending", "running"}:
                    session.record.status = "closing"
                    session.record.updated_at = time.time()

    def _start_session_turn(self, session: _SubagentSession, user_text: str) -> None:
        with session.lock:
            if not session.close_requested:
                session.record.status = "running"
                session.record.updated_at = time.time()
        future = asyncio.run_coroutine_threadsafe(
            self._run_session_turn(session, user_text),
            self._ensure_loop(),
        )
        with session.lock:
            session.future = future

    async def _run_session_turn(self, session: _SubagentSession, user_text: str) -> None:
        with session.lock:
            if session.close_requested:
                session.record.status = "closed"
                session.record.ended_at = time.time()
                session.record.updated_at = session.record.ended_at
                return
            session.record.status = "running"
            session.record.updated_at = time.time()
            session.messages.append(Message(role="user", content=[TextBlock(text=user_text)]))

        preparer = RequestPreparer(
            provider=session.provider,
            model=session.record.model,
            system=session.system,
            tools=session.tool_specs,
            max_tokens=session.max_tokens,
            thinking=session.thinking,
            effort=session.effort,
            context_window=session.context_window,
        )

        try:
            while True:
                with session.lock:
                    if session.close_requested:
                        self._mark_closed_locked(session)
                        return
                    messages = list(session.messages)

                response = await acomplete_with_recovery(preparer, messages)
                with session.lock:
                    session.messages.append(
                        Message(role="assistant", content=list(response.content))
                    )
                    session.record.turns += 1
                    session.record.updated_at = time.time()

                self._notify_update(session, "turn")

                if response.stop_reason != "tool_use":
                    monitor_blocks = self._drain_runtime_event_blocks(session.tools_by_name)
                    if not monitor_blocks:
                        self._mark_completed(session, response)
                        return
                    with session.lock:
                        session.messages.append(Message(role="user", content=monitor_blocks))
                    self._notify_update(session, "monitor")
                    continue

                followup_blocks = await self._dispatch_tools(response, session)
                followup_blocks.extend(
                    self._drain_runtime_event_blocks(session.tools_by_name)
                )
                if not followup_blocks:
                    self._mark_completed(session, response)
                    return
                with session.lock:
                    session.messages.append(
                        Message(role="user", content=followup_blocks)
                    )
                self._notify_update(session, "tool_result")
        except Exception as exc:  # noqa: BLE001
            with session.lock:
                session.record.status = "failed"
                session.record.error = f"{type(exc).__name__}: {exc}"
                session.record.ended_at = time.time()
                session.record.updated_at = session.record.ended_at
            self._notify_update(session, "failed")
            self._publish_parent_event(session, "failed")

    async def _dispatch_tools(
        self,
        response: CompletionResponse,
        session: _SubagentSession,
    ) -> list[ContentBlock]:
        results: list[ContentBlock] = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            results.extend(await self._dispatch_tool(_as_tool_use_block(block), session))
        return results

    async def _dispatch_tool(
        self,
        block: ToolUseBlock,
        session: _SubagentSession,
    ) -> list[ContentBlock]:
        return await dispatch_tool_blocks_async(
            block,
            session.tools_by_name,
            session.permission_gate,
        )

    def _mark_completed(
        self,
        session: _SubagentSession,
        response: CompletionResponse,
        *,
        suffix: str = "",
    ) -> None:
        result = _response_text(response) or f"[stop_reason={response.stop_reason}]"
        with session.lock:
            if session.close_requested:
                self._mark_closed_locked(session)
                return
            session.record.status = "completed"
            session.record.result = f"{result}{suffix}"
            session.record.ended_at = time.time()
            session.record.updated_at = session.record.ended_at
        self._notify_update(session, "completed")
        self._publish_parent_event(session, "completed")

    @staticmethod
    def _mark_closed_locked(session: _SubagentSession) -> None:
        session.record.status = "closed"
        session.record.ended_at = time.time()
        session.record.updated_at = session.record.ended_at

    @staticmethod
    def _is_running(session: _SubagentSession) -> bool:
        with session.lock:
            return session.record.status in {"pending", "running", "closing"}

    @staticmethod
    def _notify_update(session: _SubagentSession, event: str) -> None:
        session.update_queue.put_nowait(event)
        if event in {"turn", "tool_result", "monitor"}:
            SubagentManager._publish_parent_event(session, event)

    @staticmethod
    def _publish_parent_event(session: _SubagentSession, event: str) -> None:
        runtime = _runtime_from_tools(session.tools_by_name)
        events = getattr(runtime, "events", None) if runtime is not None else None
        if events is None or not hasattr(events, "publish"):
            return
        events.publish(
            {
                "event_type": "subagent",
                "event": event,
                "subagent_id": session.record.subagent_id,
                "name": session.record.display_name,
                "role": session.record.role,
                "status": session.record.status,
                "model": session.record.model,
                "effort": session.record.effort or "default",
                "workspace": session.record.workspace or "",
                "task": session.record.task,
                "summary": (
                    f"{session.record.display_name} [{session.record.role}] {event}: "
                    f"{session.record.result or session.record.error or session.record.status}"
                ),
            }
        )

    @staticmethod
    def _drain_runtime_event_blocks(
        tools_by_name: Mapping[str, Tool],
    ) -> list[ContentBlock]:
        runtime = _runtime_from_tools(tools_by_name)
        if runtime is None:
            return []
        events = getattr(runtime, "events", None)
        if events is None or not hasattr(events, "drain"):
            return []
        drained = events.drain()
        if not drained:
            return []
        monitor_events = [
            event for event in drained if event.get("event_type") != "subagent"
        ]
        return list(monitor_event_text_blocks(monitor_events))

    def _select_tools(
        self,
        tools_by_name: Mapping[str, Tool],
        tool_names: list[str] | None,
    ) -> dict[str, Tool]:
        if tool_names is None:
            return {
                name: tool
                for name, tool in tools_by_name.items()
                if name not in COLLABORATION_TOOL_NAMES
            }
        selected: dict[str, Tool] = {}
        missing: list[str] = []
        for raw_name in tool_names:
            name = raw_name.strip()
            if not name:
                continue
            tool = tools_by_name.get(name)
            if tool is None:
                missing.append(name)
            else:
                selected[name] = tool
        if missing:
            raise ValueError(f"unknown subagent tool(s): {', '.join(sorted(missing))}")
        return selected

    @staticmethod
    def _subagent_system(
        base_system: str | None,
        tools_by_name: Mapping[str, Tool],
        instructions: str | None,
        *,
        permission_mode: PermissionMode,
    ) -> str:
        system = build_system_prompt(
            tools_by_name=tools_by_name,
            skills=load_available_skills(Path.cwd()),
            permission_mode=permission_mode,
        )
        parent_context = _parent_system_context(base_system, regenerated_system=system)
        if parent_context:
            system = f"{system}\n\nParent system context:\n{parent_context}"
        if not instructions:
            return system
        addition = (
            "You are running as a Wattle managed subagent. Focus only on the "
            "delegated task and return concise findings or results to the parent.\n\n"
            f"Subagent-specific instructions:\n{instructions}"
        )
        return f"{system}\n\n{addition}"

    @staticmethod
    def _initial_user_text(record: SubagentRecord) -> str:
        sections = ["Delegated task:", record.task]
        if record.context:
            sections.extend(["", "Additional context:", record.context])
        return "\n".join(sections)

    def _require_config(self) -> _SubagentConfig:
        with self._lock:
            config = self._config
        if config is None:
            raise RuntimeError("subagent runtime is not configured for this agent loop")
        return config

    def _require_session(self, subagent_id: str) -> _SubagentSession:
        with self._lock:
            session = self._sessions.get(subagent_id)
        if session is None:
            raise KeyError(f"unknown subagent: {subagent_id}")
        return session

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is not None and self._loop.is_running():
                return self._loop
            loop = asyncio.new_event_loop()
            ready = threading.Event()

            def run_loop() -> None:
                asyncio.set_event_loop(loop)
                ready.set()
                loop.run_forever()

            thread = threading.Thread(
                target=run_loop,
                name="wattle-subagents-asyncio",
                daemon=True,
            )
            thread.start()
            ready.wait()
            self._loop = loop
            self._loop_thread = thread
            return loop


def subagent_summary(record: SubagentRecord) -> str:
    lines = [
        f"subagent_id: {record.subagent_id}",
        f"name: {record.display_name}",
        f"role: {record.role}",
        f"status: {record.status}",
        f"model: {record.model}",
        f"effort: {record.effort or 'default'}",
        f"workspace: {record.workspace or ''}",
        f"task: {record.task}",
        f"turns: {record.turns}",
    ]
    if record.tool_names:
        lines.append(f"tools: {', '.join(record.tool_names)}")
    if record.result:
        lines.append(f"result:\n{record.result}")
    if record.error:
        lines.append(f"error: {record.error}")
    return "\n".join(lines)


def subagent_snapshot_summary(snapshot: Mapping[str, object]) -> str:
    return subagent_summary(
        SubagentRecord(
            subagent_id=cast(str, snapshot["subagent_id"]),
            display_name=cast(str, snapshot["display_name"]),
            role=cast(str, snapshot["role"]),
            task=cast(str, snapshot["task"]),
            instructions=cast(str | None, snapshot["instructions"]),
            context=cast(str | None, snapshot["context"]),
            model=cast(str, snapshot["model"]),
            effort=cast(
                Literal["low", "medium", "high", "xhigh", "max"] | None,
                snapshot["effort"],
            ),
            workspace=cast(str | None, snapshot["workspace"]),
            tool_names=list(cast(list[str], snapshot["tool_names"])),
            status=cast(SubagentStatus, snapshot["status"]),
            started_at=cast(float, snapshot["started_at"]),
            updated_at=cast(float, snapshot["updated_at"]),
            ended_at=cast(float | None, snapshot["ended_at"]),
            result=cast(str | None, snapshot["result"]),
            error=cast(str | None, snapshot["error"]),
            turns=cast(int, snapshot["turns"]),
        )
    )


def _copy_message(message: Message) -> Message:
    return replace(message, content=list(message.content))


def _response_text(response: CompletionResponse) -> str:
    parts: list[str] = []
    for block in response.content:
        if getattr(block, "type", None) != "text" or not hasattr(block, "text"):
            continue
        text = str(block.text).strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def _effort_for_model(
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None,
    model: str,
) -> Literal["low", "medium", "high", "xhigh", "max"] | None:
    if effort is None:
        return None
    levels = effort_levels_for_model(model)
    if effort in levels:
        return effort
    return levels[-1] if levels else None


def _parent_system_context(
    system: str | None,
    *,
    regenerated_system: str,
) -> str:
    if not system or system == regenerated_system:
        return ""
    if not _looks_like_wattle_system_prompt(system):
        return system
    return _custom_parent_system_suffix(system)


def _looks_like_wattle_system_prompt(system: str) -> bool:
    return (
        DEFAULT_SYSTEM_PROMPT in system
        and "Available tools:\n" in system
        and "Guidelines:\n" in system
    )


def _custom_parent_system_suffix(system: str) -> str:
    index = system.find(DEFAULT_SYSTEM_PROMPT)
    if index == -1:
        return ""
    parts = [system[:index].strip()]
    current_working_directory = "\n\nCurrent working directory: "
    cwd_index = system.rfind(current_working_directory)
    if cwd_index != -1:
        after_cwd = system.find("\n\n", cwd_index + len(current_working_directory))
        if after_cwd != -1:
            parts.append(system[after_cwd:].strip())
    return "\n\n".join(part for part in parts if part)


def _normalize_agent_type(agent_type: str | None) -> str:
    clean_agent_type = (agent_type or "").strip()
    return clean_agent_type or DEFAULT_AGENT_TYPE


def _workspace_from_tools(tools_by_name: Mapping[str, Tool]) -> str | None:
    runtime = _runtime_from_tools(tools_by_name)
    tasks = getattr(runtime, "tasks", None) if runtime is not None else None
    root = getattr(tasks, "root", None)
    return str(root) if root is not None else None


def _runtime_from_tools(tools_by_name: Mapping[str, Tool]) -> object | None:
    for tool in tools_by_name.values():
        runtime = getattr(tool, "runtime", None)
        if runtime is not None and hasattr(runtime, "events"):
            return runtime
    return None


def _as_tool_use_block(block: ContentBlock) -> ToolUseBlock:
    if isinstance(block, ToolUseBlock):
        return block
    return ToolUseBlock(
        id=str(getattr(block, "id", "")),
        name=str(getattr(block, "name", "")),
        input=dict(getattr(block, "input", {})),
    )
