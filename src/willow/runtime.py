from __future__ import annotations

import json
import os
import queue
import signal
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from willow.subagents import SubagentManager


class TaskStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    KILLED = "killed"
    LOST = "lost"


@dataclass
class AsyncTask:
    task_id: str
    kind: Literal["shell"]
    command: str
    pid: int
    pgid: int
    status: TaskStatus
    started_at: float
    ended_at: float | None
    exit_code: int | None
    log_path: Path
    status_path: Path


@dataclass
class MonitorRecord:
    monitor_id: str
    task_ids: list[str]
    patterns: list[dict[str, str]]
    interval_seconds: float
    min_push_interval_seconds: float
    tail_lines: int
    max_event_chars: int
    max_events: int
    status: TaskStatus
    started_at: float
    ended_at: float | None


class MonitorEventQueue:
    def __init__(self) -> None:
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._history: list[dict[str, Any]] = []
        self._subscribers: dict[str, Callable[[dict[str, Any]], None]] = {}
        self._lock = threading.RLock()

    def publish(self, event: dict[str, Any]) -> None:
        event_copy = dict(event)
        with self._lock:
            self._history.append(event_copy)
            subscribers = list(self._subscribers.values())
        self._queue.put(dict(event_copy))
        for subscriber in subscribers:
            try:
                subscriber(dict(event_copy))
            except Exception:
                continue

    def drain(self, max_events: int | None = None) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while max_events is None or len(events) < max_events:
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return events

    def history(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(event) for event in self._history]

    def subscribe(self, callback: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        subscriber_id = uuid.uuid4().hex
        with self._lock:
            self._subscribers[subscriber_id] = callback

        def unsubscribe() -> None:
            with self._lock:
                self._subscribers.pop(subscriber_id, None)

        return unsubscribe


class TaskRegistry:
    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path.cwd() if root is None else Path(root)
        self.jobs_dir = self.root / ".willow" / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._tasks: dict[str, AsyncTask] = {}
        self._lock = threading.RLock()

    def register_shell_task(
        self,
        *,
        command: str,
        pid: int,
        pgid: int,
        log_path: Path,
    ) -> AsyncTask:
        task_id = f"shell-{uuid.uuid4().hex[:12]}"
        task = AsyncTask(
            task_id=task_id,
            kind="shell",
            command=command,
            pid=pid,
            pgid=pgid,
            status=TaskStatus.RUNNING,
            started_at=time.time(),
            ended_at=None,
            exit_code=None,
            log_path=log_path,
            status_path=self.jobs_dir / f"{task_id}.json",
        )
        with self._lock:
            self._tasks[task_id] = task
            self._write_status_locked(task)
        return task

    def get(self, task_id: str) -> AsyncTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def snapshot(self, task_id: str) -> dict[str, object] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            return self._task_to_json(task)

    def snapshots(self) -> list[dict[str, object]]:
        with self._lock:
            return [self._task_to_json(task) for task in self._tasks.values()]

    def mark_terminal(
        self,
        task_id: str,
        *,
        status: TaskStatus,
        exit_code: int | None,
        ended_at: float | None = None,
    ) -> AsyncTask | None:
        if status == TaskStatus.RUNNING:
            raise ValueError("mark_terminal requires a terminal status")
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if task.status != TaskStatus.RUNNING:
                return task
            task.status = status
            task.exit_code = exit_code
            task.ended_at = time.time() if ended_at is None else ended_at
            self._write_status_locked(task)
            return task

    def cleanup(self) -> None:
        with self._lock:
            tasks = [
                task for task in self._tasks.values() if task.status == TaskStatus.RUNNING
            ]
        for task in tasks:
            try:
                os.killpg(task.pgid, signal.SIGTERM)
                status = TaskStatus.KILLED
            except ProcessLookupError:
                status = TaskStatus.LOST
            except PermissionError:
                status = TaskStatus.LOST
            self.mark_terminal(task.task_id, status=status, exit_code=None)

    def _write_status_locked(self, task: AsyncTask) -> None:
        task.status_path.write_text(
            json.dumps(self._task_to_json(task), indent=2, sort_keys=True) + "\n"
        )

    @staticmethod
    def _task_to_json(task: AsyncTask) -> dict[str, object]:
        data = asdict(task)
        data["status"] = task.status.value
        data["log_path"] = str(task.log_path)
        data["status_path"] = str(task.status_path)
        return data


class MonitorRegistry:
    def __init__(self) -> None:
        self._monitors: dict[str, MonitorRecord] = {}
        self._lock = threading.RLock()

    def register(
        self,
        *,
        task_ids: list[str],
        patterns: list[dict[str, str]],
        interval_seconds: float,
        min_push_interval_seconds: float,
        tail_lines: int,
        max_event_chars: int,
        max_events: int,
    ) -> MonitorRecord:
        monitor_id = f"monitor-{uuid.uuid4().hex[:12]}"
        monitor = MonitorRecord(
            monitor_id=monitor_id,
            task_ids=list(task_ids),
            patterns=[dict(pattern) for pattern in patterns],
            interval_seconds=interval_seconds,
            min_push_interval_seconds=min_push_interval_seconds,
            tail_lines=tail_lines,
            max_event_chars=max_event_chars,
            max_events=max_events,
            status=TaskStatus.RUNNING,
            started_at=time.time(),
            ended_at=None,
        )
        with self._lock:
            self._monitors[monitor_id] = monitor
        return monitor

    def get(self, monitor_id: str) -> MonitorRecord | None:
        with self._lock:
            return self._monitors.get(monitor_id)

    def snapshot(self, monitor_id: str) -> dict[str, object] | None:
        with self._lock:
            monitor = self._monitors.get(monitor_id)
            if monitor is None:
                return None
            return self._monitor_to_json(monitor)

    def snapshots(self) -> list[dict[str, object]]:
        with self._lock:
            return [self._monitor_to_json(monitor) for monitor in self._monitors.values()]

    def mark_terminal(
        self,
        monitor_id: str,
        *,
        status: TaskStatus = TaskStatus.COMPLETED,
        ended_at: float | None = None,
    ) -> MonitorRecord | None:
        if status == TaskStatus.RUNNING:
            raise ValueError("mark_terminal requires a terminal status")
        with self._lock:
            monitor = self._monitors.get(monitor_id)
            if monitor is None:
                return None
            if monitor.status != TaskStatus.RUNNING:
                return monitor
            monitor.status = status
            monitor.ended_at = time.time() if ended_at is None else ended_at
            return monitor

    @staticmethod
    def _monitor_to_json(monitor: MonitorRecord) -> dict[str, object]:
        data = asdict(monitor)
        data["status"] = monitor.status.value
        return data


class WillowRuntime:
    def __init__(self, root: Path | str | None = None) -> None:
        self.tasks = TaskRegistry(root=root)
        self.monitors = MonitorRegistry()
        self.events = MonitorEventQueue()
        self._subagents: SubagentManager | None = None

    @property
    def subagents(self) -> SubagentManager:
        if self._subagents is None:
            from willow.subagents import SubagentManager

            self._subagents = SubagentManager()
        return self._subagents

    def cleanup(self) -> None:
        if self._subagents is not None:
            self._subagents.cleanup()
        self.tasks.cleanup()
