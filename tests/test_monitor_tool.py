from __future__ import annotations

import shlex
import sys
import time
from pathlib import Path

from wattle.runtime import WattleRuntime
from wattle.tools.monitor import MonitorTool


def _wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def _python_command(code: str) -> str:
    return f"{shlex.quote(sys.executable)} -u -c {shlex.quote(code)}"


def test_monitor_command_emits_stdout_and_terminal_summary(tmp_path: Path) -> None:
    runtime = WattleRuntime(root=tmp_path)

    output = MonitorTool(runtime=runtime).run(
        command=_python_command("print('READY', flush=True)"),
        description="service readiness",
        timeout_ms=2000,
    )
    monitor_id = dict(line.split(": ", 1) for line in output.splitlines())["monitor_id"]

    def has_terminal_summary() -> bool:
        return any(event["event_type"] == "terminal_summary" for event in runtime.events.history())

    _wait_for(has_terminal_summary)
    events = runtime.events.drain()
    output_events = [event for event in events if event["event_type"] == "command_output"]
    terminal_events = [event for event in events if event["event_type"] == "terminal_summary"]

    assert output_events[0]["monitor_id"] == monitor_id
    assert output_events[0]["summary"] == "service readiness: READY"
    assert terminal_events[0]["status"] == "completed"
    assert terminal_events[0]["exit_code"] == 0


def test_monitor_command_timeout_marks_timed_out(tmp_path: Path) -> None:
    runtime = WattleRuntime(root=tmp_path)

    output = MonitorTool(runtime=runtime).run(
        command=_python_command("import time; print('started', flush=True); time.sleep(60)"),
        timeout_ms=50,
    )
    monitor_id = dict(line.split(": ", 1) for line in output.splitlines())["monitor_id"]

    def has_timeout() -> bool:
        return any(
            event["event_type"] == "terminal_summary" and event["status"] == "timed_out"
            for event in runtime.events.history()
        )

    _wait_for(has_timeout)

    snapshot = runtime.monitors.snapshot(monitor_id)
    assert snapshot is not None
    assert snapshot["status"] == "timed_out"


def test_monitor_command_timeout_kills_process_group(tmp_path: Path) -> None:
    runtime = WattleRuntime(root=tmp_path)
    leaked_path = tmp_path / "leaked"
    child = (
        "import pathlib, time; "
        "time.sleep(1); "
        f"pathlib.Path({str(leaked_path)!r}).write_text('leaked')"
    )
    parent = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
        "time.sleep(60)"
    )

    MonitorTool(runtime=runtime).run(command=_python_command(parent), timeout_ms=50)

    def has_timeout() -> bool:
        return any(
            event["event_type"] == "terminal_summary" and event["status"] == "timed_out"
            for event in runtime.events.history()
        )

    _wait_for(has_timeout)
    time.sleep(1.2)

    assert not leaked_path.exists()


def test_monitor_command_persistent_ignores_timeout_ms(tmp_path: Path) -> None:
    runtime = WattleRuntime(root=tmp_path)

    MonitorTool(runtime=runtime).run(
        command=_python_command("import time; time.sleep(0.2); print('done', flush=True)"),
        timeout_ms=1,
        persistent=True,
    )

    def has_completed() -> bool:
        return any(
            event["event_type"] == "terminal_summary" and event["status"] == "completed"
            for event in runtime.events.history()
        )

    _wait_for(has_completed)


def test_monitor_caps_command_output_events(tmp_path: Path) -> None:
    runtime = WattleRuntime(root=tmp_path)

    MonitorTool(runtime=runtime).run(
        command=_python_command("print('one', flush=True); print('two', flush=True)"),
        max_events=1,
    )

    def has_terminal_summary() -> bool:
        return any(event["event_type"] == "terminal_summary" for event in runtime.events.history())

    _wait_for(has_terminal_summary)

    output_events = [
        event for event in runtime.events.history() if event["event_type"] == "command_output"
    ]
    assert len(output_events) == 1
    assert output_events[0]["tail"] == "one"


def test_monitor_rejects_empty_command(tmp_path: Path) -> None:
    runtime = WattleRuntime(root=tmp_path)

    output = MonitorTool(runtime=runtime).run(command=" ")

    assert output == "[error] command must not be empty"
    assert runtime.monitors.snapshots() == []
