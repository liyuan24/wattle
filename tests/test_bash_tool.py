from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path

import pytest

from wattle.runtime import WattleRuntime
from wattle.tools.bash import BashTool, _kill_new_processes_under_cwd


def _python_command(code: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def _wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _cleanup_pid(pid: int) -> None:
    with suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


def _spawn_sleeping_child_command(pid_file: Path, *, start_new_session: bool = False) -> str:
    child = "import time; time.sleep(30)"
    code = (
        "import subprocess, sys; "
        "p = subprocess.Popen("
        f"[sys.executable, '-c', {child!r}], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, "
        f"start_new_session={start_new_session!r}"
        "); "
        f"open({str(pid_file)!r}, 'w').write(str(p.pid)); "
        "print(p.pid, flush=True)"
    )
    return _python_command(code)


def test_foreground_output_elapsed_and_exit_code(tmp_path: Path) -> None:
    tool = BashTool(cwd=tmp_path)
    command = _python_command(
        "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"
    )

    output = tool.run(command)

    assert "out" in output
    assert "[stderr]\nerr" in output
    assert "[exit 3]" in output
    assert "[elapsed " in output


def test_foreground_uses_workdir_without_cd(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "marker.txt").write_text("ok")
    tool = BashTool(cwd=tmp_path)

    result = tool.run_execution_result("pwd && ls marker.txt", workdir="project")
    output = str(result.content)

    assert str(project) in output
    assert "marker.txt" in output
    assert result.metadata["workdir"] == str(project)


def test_foreground_rejects_invalid_workdir(tmp_path: Path) -> None:
    tool = BashTool(cwd=tmp_path)

    with pytest.raises(ValueError, match="workdir does not exist"):
        tool.run("pwd", workdir="missing")


def test_bash_schema_exposes_workdir() -> None:
    schema = BashTool.input_schema

    assert "workdir" in schema["properties"]
    assert "cd" in schema["properties"]["workdir"]["description"]


def test_foreground_replaces_invalid_utf8_output(tmp_path: Path) -> None:
    tool = BashTool(cwd=tmp_path)
    command = _python_command(
        "import sys; "
        "sys.stdout.buffer.write(bytes([0x61, 0xcb, 0x62])); "
        "sys.stderr.buffer.write(bytes([0x65, 0xcb, 0x66]))"
    )

    output = tool.run(command)

    assert "a\ufffdb" in output
    assert "[stderr]\ne\ufffdf" in output
    assert "UnicodeDecodeError" not in output


def test_foreground_uses_closed_stdin(tmp_path: Path) -> None:
    tool = BashTool(cwd=tmp_path)
    command = _python_command(
        "import sys; data = sys.stdin.read(); print('stdin-eof' if data == '' else 'stdin-data')"
    )

    output = tool.run(command, timeout=2)

    assert "stdin-eof" in output


def test_foreground_large_output_is_externalized(tmp_path: Path) -> None:
    tool = BashTool(cwd=tmp_path)
    command = _python_command("import sys; sys.stdout.write('x' * 30000)")

    output = tool.run(command)
    fields = dict(line.split(": ", 1) for line in output.splitlines() if ": " in line)
    full_output_path = Path(fields["full_output_path"])

    assert output.startswith("[output truncated: 30000 chars]")
    assert str(full_output_path).startswith(str(tmp_path / ".wattle" / "artifacts"))
    assert full_output_path.read_text() == "x" * 30000
    assert len(output) < 14000


def test_foreground_accepts_model_output_budget_alias(tmp_path: Path) -> None:
    tool = BashTool(cwd=tmp_path)
    command = _python_command("import sys; sys.stdout.write('x' * 1000)")

    output = tool.run(command, max_event_chars=100)

    assert output.startswith("[output truncated: 1000 chars]")
    assert "full_output_path:" in output


def test_foreground_timeout_clamps(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(BashTool, "MAX_TIMEOUT_SECONDS", 0.05)
    command = _python_command("import time; time.sleep(30)")

    started_at = time.monotonic()
    output = BashTool(cwd=tmp_path).run(command, timeout=999)
    elapsed = time.monotonic() - started_at

    assert elapsed < 2.0
    assert "Status: timed out" in output
    assert "Requested timeout: 0.05s" in output


def test_foreground_timeout_returns_when_descendant_keeps_pipe_open(tmp_path: Path) -> None:
    tool = BashTool(cwd=tmp_path)
    command = _python_command(
        "import subprocess, sys, time; "
        "subprocess.Popen("
        "[sys.executable, '-c', 'import time; time.sleep(3)'], "
        "stdout=sys.stdout, stderr=sys.stderr, start_new_session=True"
        "); "
        "time.sleep(30)"
    )

    started_at = time.monotonic()
    output = tool.run(command, timeout=0.1)
    elapsed = time.monotonic() - started_at

    assert elapsed < 2.0
    assert "Status: timed out" in output
    assert "Requested timeout: 0.1s" in output
    assert "descendant process kept stdout/stderr open" not in output


def test_foreground_piped_kills_descendants_after_command_exits(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"

    output = BashTool(cwd=tmp_path).run(_spawn_sleeping_child_command(pid_file))

    child_pid = int(pid_file.read_text())
    try:
        assert re.search(rf"\b{child_pid}\b", output)
        _wait_for(lambda: not _pid_is_alive(child_pid))
    finally:
        _cleanup_pid(child_pid)


def test_foreground_piped_kills_new_session_descendants_under_cwd(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"

    output = BashTool(cwd=tmp_path).run(
        _spawn_sleeping_child_command(pid_file, start_new_session=True)
    )

    child_pid = int(pid_file.read_text())
    try:
        assert re.search(rf"\b{child_pid}\b", output)
        _wait_for(lambda: not _pid_is_alive(child_pid))
    finally:
        _cleanup_pid(child_pid)


def test_foreground_parallel_commands_same_cwd_do_not_kill_each_other(tmp_path: Path) -> None:
    tool = BashTool(cwd=tmp_path)
    short = _python_command("print('short')")
    slow = _python_command("import time; time.sleep(0.2); print('slow')")

    async def run() -> tuple[str, str]:
        short_task = asyncio.create_task(
            tool.arun_with_events(
                emit=lambda _event: None,
                tool_use_id="short",
                command=short,
                timeout=2,
            )
        )
        slow_task = asyncio.create_task(
            tool.arun_with_events(
                emit=lambda _event: None,
                tool_use_id="slow",
                command=slow,
                timeout=2,
            )
        )
        return await asyncio.gather(short_task, slow_task)

    short_output, slow_output = asyncio.run(run())

    assert "short" in short_output
    assert "slow" in slow_output
    assert "Status: timed out" not in short_output
    assert "Status: timed out" not in slow_output


def test_foreground_cleanup_does_not_kill_unrelated_process_under_cwd(tmp_path: Path) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _kill_new_processes_under_cwd(
            BashTool.RUN_ID_ENV,
            "different-run",
            tmp_path,
        )

        assert process.poll() is None
    finally:
        _cleanup_pid(process.pid)
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=1)


def test_foreground_tty_allocates_terminal(tmp_path: Path) -> None:
    tool = BashTool(cwd=tmp_path)
    command = _python_command("import os; print(f'tty={os.isatty(1)}')")

    output = tool.run(command, tty=True)

    assert "tty=True" in output
    assert "[elapsed " in output


def test_foreground_tty_timeout_uses_process_lifecycle(tmp_path: Path) -> None:
    tool = BashTool(cwd=tmp_path)
    command = _python_command("import time; print('tty-start', flush=True); time.sleep(30)")

    started_at = time.monotonic()
    output = tool.run(command, tty=True, timeout=0.05)
    elapsed = time.monotonic() - started_at

    assert elapsed < 2.0
    assert "Status: timed out" in output
    assert "Requested timeout: 0.05s" in output
    assert "tty-start" in output


def test_foreground_tty_kills_descendants_after_command_exits(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"

    output = BashTool(cwd=tmp_path).run(
        _spawn_sleeping_child_command(pid_file),
        tty=True,
    )

    child_pid = int(pid_file.read_text())
    try:
        assert re.search(rf"\b{child_pid}\b", output)
        _wait_for(lambda: not _pid_is_alive(child_pid))
    finally:
        _cleanup_pid(child_pid)


def test_foreground_tty_kills_new_session_descendants_under_cwd(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"

    output = BashTool(cwd=tmp_path).run(
        _spawn_sleeping_child_command(pid_file, start_new_session=True),
        tty=True,
    )

    child_pid = int(pid_file.read_text())
    try:
        assert re.search(rf"\b{child_pid}\b", output)
        _wait_for(lambda: not _pid_is_alive(child_pid))
    finally:
        _cleanup_pid(child_pid)


def test_foreground_piped_emits_output_events(tmp_path: Path) -> None:
    tool = BashTool(cwd=tmp_path)
    events = []
    command = _python_command(
        "import sys; print('out', flush=True); print('err', file=sys.stderr, flush=True)"
    )

    output = asyncio.run(
        tool.arun_with_events(emit=events.append, tool_use_id="call_1", command=command)
    )

    assert "out" in output
    assert "[stderr]\nerr" in output
    assert events[0].kind == "started"
    assert events[0].text == command
    assert events[-1].kind == "completed"
    assert any(
        event.kind == "output" and event.stream == "stdout" and "out" in event.text
        for event in events
    )
    assert any(
        event.kind == "output" and event.stream == "stderr" and "err" in event.text
        for event in events
    )


def test_foreground_piped_events_waits_for_process_after_pipe_eof(tmp_path: Path) -> None:
    tool = BashTool(cwd=tmp_path)
    events = []
    command = _python_command(
        "import os, sys, time; os.close(1); os.close(2); time.sleep(0.05); sys.exit(7)"
    )

    output = asyncio.run(
        tool.arun_with_events(
            emit=events.append,
            tool_use_id="call_1",
            command=command,
            timeout=2,
        )
    )

    assert "Status: timed out" not in output
    assert "[exit 7]" in output
    assert events[0].kind == "started"
    assert events[-1].kind == "completed"


def test_foreground_piped_events_timeout_after_pipe_eof(tmp_path: Path) -> None:
    tool = BashTool(cwd=tmp_path)
    command = _python_command(
        "import os, time; os.close(1); os.close(2); time.sleep(30)"
    )

    started_at = time.monotonic()
    output = asyncio.run(
        tool.arun_with_events(
            emit=lambda _event: None,
            tool_use_id="call_1",
            command=command,
            timeout=0.05,
        )
    )
    elapsed = time.monotonic() - started_at

    assert elapsed < 1.0
    assert "Status: timed out" in output
    assert "Requested timeout: 0.05s" in output


def test_foreground_tty_emits_combined_output_events(tmp_path: Path) -> None:
    tool = BashTool(cwd=tmp_path)
    events = []
    command = _python_command("print('tty-event', flush=True)")

    output = asyncio.run(
        tool.arun_with_events(
            emit=events.append,
            tool_use_id="call_1",
            command=command,
            tty=True,
        )
    )

    assert "tty-event" in output
    assert any(
        event.kind == "output" and event.stream == "combined" and "tty-event" in event.text
        for event in events
    )


def test_foreground_piped_can_be_cancelled(tmp_path: Path) -> None:
    tool = BashTool(cwd=tmp_path)
    cancel_event = threading.Event()
    command = _python_command("import time; print('started', flush=True); time.sleep(30)")

    async def run() -> str:
        task = asyncio.create_task(
            tool.arun_with_events(
                emit=lambda _event: None,
                tool_use_id="call_1",
                command=command,
                cancel_event=cancel_event,
            )
        )
        await asyncio.sleep(0.2)
        cancel_event.set()
        return await asyncio.wait_for(task, timeout=3.0)

    output = asyncio.run(run())

    assert "[stopped by user after" in output
    assert "started" in output


def test_foreground_tty_can_be_cancelled(tmp_path: Path) -> None:
    tool = BashTool(cwd=tmp_path)
    cancel_event = threading.Event()
    command = _python_command("import time; print('tty-started', flush=True); time.sleep(30)")

    async def run() -> str:
        task = asyncio.create_task(
            tool.arun_with_events(
                emit=lambda _event: None,
                tool_use_id="call_1",
                command=command,
                tty=True,
                cancel_event=cancel_event,
            )
        )
        await asyncio.sleep(0.2)
        cancel_event.set()
        return await asyncio.wait_for(task, timeout=3.0)

    output = asyncio.run(run())

    assert "[stopped by user after" in output
    assert "tty-started" in output


def test_background_task_metadata_log_and_status_update(tmp_path: Path) -> None:
    runtime = WattleRuntime(root=tmp_path)
    tool = BashTool(runtime=runtime, cwd=tmp_path)
    command = _python_command("print('background-output', flush=True)")

    output = tool.run(command, background=True)
    fields = dict(line.split(": ", 1) for line in output.splitlines())

    assert fields["task_id"].startswith("shell-")
    assert int(fields["pid"]) > 0
    assert fields["pgid"] == fields["pid"]
    assert Path(fields["log_path"]).is_file()
    assert Path(fields["status_path"]).is_file()

    task_id = fields["task_id"]

    def is_complete() -> bool:
        snapshot = runtime.tasks.snapshot(task_id)
        return snapshot is not None and snapshot["status"] == "completed"

    _wait_for(is_complete)

    status_data = json.loads(Path(fields["status_path"]).read_text())
    assert status_data["status"] == "completed"
    assert status_data["exit_code"] == 0
    assert status_data["ended_at"] is not None
    assert "background-output" in Path(fields["log_path"]).read_text()


def test_background_tty_logs_output_and_status_update(tmp_path: Path) -> None:
    runtime = WattleRuntime(root=tmp_path)
    tool = BashTool(runtime=runtime, cwd=tmp_path)
    command = _python_command("import os; print(f'tty={os.isatty(1)}', flush=True)")

    output = tool.run(command, background=True, tty=True)
    fields = dict(line.split(": ", 1) for line in output.splitlines())

    assert fields["tty"] == "true"
    task_id = fields["task_id"]

    def is_complete() -> bool:
        snapshot = runtime.tasks.snapshot(task_id)
        return snapshot is not None and snapshot["status"] == "completed"

    _wait_for(is_complete)

    status_data = json.loads(Path(fields["status_path"]).read_text())
    assert status_data["status"] == "completed"
    assert status_data["exit_code"] == 0
    assert "tty=True" in Path(fields["log_path"]).read_text()
