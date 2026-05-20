from __future__ import annotations

import atexit
import os
import signal
import subprocess
import threading
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

from wattle.runtime import MonitorRecord, TaskStatus, WattleRuntime

from .base import Tool

_ACTIVE_COMMAND_PGIDS: set[int] = set()
_ACTIVE_COMMAND_PGIDS_LOCK = threading.RLock()


def _cleanup_command_monitor_process_groups() -> None:
    with _ACTIVE_COMMAND_PGIDS_LOCK:
        pgids = list(_ACTIVE_COMMAND_PGIDS)
    for pgid in pgids:
        with suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGTERM)


atexit.register(_cleanup_command_monitor_process_groups)


class MonitorTool(Tool):
    name = "monitor"
    description = (
        "Run a shell monitor command and push compact runtime events when stdout "
        "lines arrive or the monitor command finishes. Use this for log tails, "
        "readiness checks, or watch commands such as `tail -F app.log | grep ERROR`."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": (
                    "Shell command to run as the monitor. It should print relevant "
                    "notifications on stdout, for example with `tail -F app.log | "
                    "grep --line-buffered ERROR`."
                ),
            },
            "description": {
                "type": "string",
                "description": "Short human-readable description of the monitor command.",
            },
            "timeout_ms": {
                "type": "integer",
                "description": (
                    "Maximum runtime for command monitors in milliseconds. "
                    "Ignored when persistent is true. Defaults to 300000."
                ),
            },
            "persistent": {
                "type": "boolean",
                "description": (
                    "For command monitors, keep running until stopped or session teardown. "
                    "When true, timeout_ms is ignored. Defaults to false."
                ),
            },
            "max_event_chars": {
                "type": "integer",
                "description": "Maximum characters for summary/tail fields. Defaults to 2000.",
            },
            "max_events": {
                "type": "integer",
                "description": (
                    "Maximum non-terminal events emitted by this monitor. Defaults to 20."
                ),
            },
        },
        "required": ["command"],
    }

    def __init__(self, runtime: WattleRuntime | None = None) -> None:
        self.runtime = runtime if runtime is not None else WattleRuntime()

    def run(
        self,
        command: str,
        description: str | None = None,
        timeout_ms: int = 300_000,
        persistent: bool = False,
        max_event_chars: int = 2000,
        max_events: int = 20,
    ) -> str:
        return self._run_command_monitor(
            command=command,
            description=description,
            timeout_ms=timeout_ms,
            persistent=persistent,
            max_event_chars=max_event_chars,
            max_events=max_events,
        )

    def _run_command_monitor(
        self,
        *,
        command: str,
        description: str | None,
        timeout_ms: int,
        persistent: bool,
        max_event_chars: int,
        max_events: int,
    ) -> str:
        if not command.strip():
            return "[error] command must not be empty"

        monitor = self.runtime.monitors.register(
            task_ids=[],
            patterns=[],
            interval_seconds=0.0,
            min_push_interval_seconds=0.0,
            tail_lines=20,
            max_event_chars=max(200, int(max_event_chars)),
            max_events=max(1, int(max_events)),
        )
        log_path = (
            self.runtime.tasks.jobs_dir
            / f"{monitor.monitor_id}-{uuid.uuid4().hex[:8]}.stderr.log"
        )
        thread = threading.Thread(
            target=self._watch_command,
            kwargs={
                "monitor": monitor,
                "command": command,
                "description": description,
                "timeout_ms": timeout_ms,
                "persistent": persistent,
                "log_path": log_path,
            },
            name=f"wattle-monitor-command-{monitor.monitor_id}",
            daemon=True,
        )
        thread.start()

        lines = [
            f"monitor_id: {monitor.monitor_id}",
            f"command: {command}",
            f"stderr_log_path: {log_path}",
        ]
        if persistent:
            lines.append("persistent: true")
        else:
            lines.append(f"timeout_ms: {int(timeout_ms)}")
        return "\n".join(lines)

    def _watch_command(
        self,
        *,
        monitor: MonitorRecord,
        command: str,
        description: str | None,
        timeout_ms: int,
        persistent: bool,
        log_path: Path,
    ) -> None:
        process: subprocess.Popen[str] | None = None
        done = threading.Event()
        timed_out = threading.Event()
        emitted = 0
        stderr_file = log_path.open("wb")

        def kill_process_group(sig: int) -> None:
            if process is None:
                return
            with suppress(ProcessLookupError):
                os.killpg(process.pid, sig)

        def timeout_watch() -> None:
            if persistent:
                return
            timeout_seconds = max(0.001, int(timeout_ms) / 1000)
            if done.wait(timeout_seconds):
                return
            timed_out.set()
            kill_process_group(signal.SIGTERM)
            if done.wait(2):
                return
            kill_process_group(signal.SIGKILL)

        try:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=self.runtime.tasks.root,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                text=True,
                start_new_session=True,
            )
            with _ACTIVE_COMMAND_PGIDS_LOCK:
                _ACTIVE_COMMAND_PGIDS.add(process.pid)
            threading.Thread(
                target=timeout_watch,
                name=f"wattle-monitor-timeout-{monitor.monitor_id}",
                daemon=True,
            ).start()

            assert process.stdout is not None
            for line in process.stdout:
                if emitted < monitor.max_events:
                    text = line.rstrip("\n")
                    self.runtime.events.publish(
                        self._bounded_event(
                            monitor=monitor,
                            task_id=None,
                            event_type="command_output",
                            severity="info",
                            summary=self._command_summary(description, text),
                            status=TaskStatus.RUNNING.value,
                            exit_code=None,
                            log_path=log_path,
                            tail=text,
                        )
                    )
                    emitted += 1

            exit_code = process.wait()
            done.set()
            status = TaskStatus.TIMED_OUT.value if timed_out.is_set() else (
                TaskStatus.COMPLETED.value if exit_code == 0 else TaskStatus.FAILED.value
            )
            severity = "error" if status != TaskStatus.COMPLETED.value else "info"
            self.runtime.events.publish(
                self._bounded_event(
                    monitor=monitor,
                    task_id=None,
                    event_type="terminal_summary",
                    severity=severity,
                    summary=f"command monitor exited status={status} exit={exit_code}",
                    status=status,
                    exit_code=exit_code,
                    log_path=log_path,
                    tail=self._tail(log_path, monitor.tail_lines, monitor.max_event_chars),
                )
            )
            self.runtime.monitors.mark_terminal(
                monitor.monitor_id,
                status=TaskStatus(status),
            )
        except Exception as exc:
            done.set()
            self.runtime.events.publish(
                self._bounded_event(
                    monitor=monitor,
                    task_id=None,
                    event_type="monitor_error",
                    severity="error",
                    summary=f"command monitor failed: {exc}",
                    status=TaskStatus.FAILED.value,
                    exit_code=None,
                    log_path=log_path,
                    tail=self._tail(log_path, monitor.tail_lines, monitor.max_event_chars),
                )
            )
            self.runtime.monitors.mark_terminal(monitor.monitor_id, status=TaskStatus.FAILED)
        finally:
            done.set()
            if process is not None:
                with _ACTIVE_COMMAND_PGIDS_LOCK:
                    _ACTIVE_COMMAND_PGIDS.discard(process.pid)
            stderr_file.close()

    @staticmethod
    def _command_summary(description: str | None, text: str) -> str:
        prefix = f"{description}: " if description else "command output: "
        return f"{prefix}{text}"

    def _bounded_event(
        self,
        *,
        monitor: MonitorRecord,
        task_id: str | None,
        event_type: str,
        severity: str,
        summary: str,
        status: str,
        exit_code: object,
        log_path: Path | None,
        tail: str,
    ) -> dict[str, Any]:
        return {
            "monitor_id": monitor.monitor_id,
            "task_id": task_id,
            "event_type": event_type,
            "severity": severity,
            "summary": self._cap(summary, monitor.max_event_chars),
            "status": status,
            "exit_code": exit_code,
            "log_path": str(log_path) if log_path is not None else None,
            "tail": self._cap(tail, monitor.max_event_chars),
            "emitted_at": time.time(),
        }

    @classmethod
    def _tail(cls, path: Path, lines: int, max_chars: int) -> str:
        if not path.exists():
            return ""
        text = path.read_text(errors="replace")
        tail = "\n".join(text.splitlines()[-lines:])
        return cls._cap(tail, max_chars)

    @staticmethod
    def _cap(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return "...\n" + text[-max(0, max_chars - 4) :]
