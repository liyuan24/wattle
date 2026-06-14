from __future__ import annotations

import shlex
import sys
import time
from pathlib import Path

from wattle.runtime import TaskRegistry, TaskStatus, WattleRuntime
from wattle.tools.bash import BashTool


def _python_command(code: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def _wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def test_registry_registers_snapshots_and_marks_terminal(tmp_path: Path) -> None:
    registry = TaskRegistry(root=tmp_path)
    log_path = registry.jobs_dir / "manual.log"
    log_path.write_text("")

    task = registry.register_shell_task(
        command="echo hi",
        pid=111,
        pgid=111,
        log_path=log_path,
    )

    assert task.status_path.is_file()
    snapshot = registry.snapshot(task.task_id)
    assert snapshot is not None
    assert snapshot["status"] == "running"
    assert snapshot["log_path"] == str(log_path)
    assert registry.snapshots()[0]["task_id"] == task.task_id

    marked = registry.mark_terminal(
        task.task_id,
        status=TaskStatus.COMPLETED,
        exit_code=0,
    )

    assert marked is not None
    assert marked.status == TaskStatus.COMPLETED
    assert registry.snapshot(task.task_id)["exit_code"] == 0  # type: ignore[index]


def test_runtime_event_queue_drains_in_order(tmp_path: Path) -> None:
    runtime = WattleRuntime(root=tmp_path)

    runtime.events.publish({"event_type": "first"})
    runtime.events.publish({"event_type": "second"})

    assert [event["event_type"] for event in runtime.events.drain(max_events=1)] == [
        "first"
    ]
    assert [event["event_type"] for event in runtime.events.drain()] == ["second"]
    assert [event["event_type"] for event in runtime.events.history()] == [
        "first",
        "second",
    ]


def test_runtime_event_queue_notifies_subscribers(tmp_path: Path) -> None:
    runtime = WattleRuntime(root=tmp_path)
    received: list[dict[str, object]] = []
    unsubscribe = runtime.events.subscribe(received.append)

    runtime.events.publish({"event_type": "pattern_match"})
    unsubscribe()
    runtime.events.publish({"event_type": "terminal_summary"})

    assert [event["event_type"] for event in received] == ["pattern_match"]


def test_registry_stop_all_running_terminates_background_process(tmp_path: Path) -> None:
    runtime = WattleRuntime(root=tmp_path)
    tool = BashTool(runtime=runtime, cwd=tmp_path)
    output = tool.run(_python_command("import time; time.sleep(30)"), background=True)
    task_id = dict(line.split(": ", 1) for line in output.splitlines())["task_id"]

    stopped, lost = runtime.tasks.stop_all_running()

    assert (stopped, lost) == (1, 0)

    def is_terminal() -> bool:
        snapshot = runtime.tasks.snapshot(task_id)
        return snapshot is not None and snapshot["status"] in {"killed", "lost"}

    _wait_for(is_terminal)
    assert runtime.tasks.snapshot(task_id)["exit_code"] is None  # type: ignore[index]


def test_registry_cleanup_terminates_running_background_process(tmp_path: Path) -> None:
    runtime = WattleRuntime(root=tmp_path)
    tool = BashTool(runtime=runtime, cwd=tmp_path)
    output = tool.run(_python_command("import time; time.sleep(30)"), background=True)
    task_id = dict(line.split(": ", 1) for line in output.splitlines())["task_id"]

    runtime.tasks.cleanup()

    def is_terminal() -> bool:
        snapshot = runtime.tasks.snapshot(task_id)
        return snapshot is not None and snapshot["status"] in {"killed", "lost"}

    _wait_for(is_terminal)
    assert runtime.tasks.snapshot(task_id)["exit_code"] is None  # type: ignore[index]
