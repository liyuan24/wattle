from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from wattle.runtime import WattleRuntime
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


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _cleanup_pid(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _spawn_sleeping_child_command(pid_file: Path) -> str:
    child = "import time; time.sleep(30)"
    code = (
        "import subprocess, sys; "
        "p = subprocess.Popen("
        f"[sys.executable, '-c', {child!r}], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL"
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


def test_foreground_timeout_clamps_and_kills_process_group(monkeypatch, tmp_path: Path) -> None:
    communicate_timeouts: list[float | None] = []
    killed_pgids: list[int] = []

    class FakeProcess:
        pid = 12345
        returncode = None

        def communicate(self, timeout=None):
            communicate_timeouts.append(timeout)
            if timeout == 600.0:
                raise subprocess.TimeoutExpired(cmd="sleep forever", timeout=timeout)
            self.returncode = -9
            return "", ""

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr("wattle.tools.bash.os.killpg", lambda pgid, sig: killed_pgids.append(pgid))

    output = BashTool(cwd=tmp_path).run("sleep forever", timeout=999)

    assert communicate_timeouts[0] == 600.0
    assert communicate_timeouts[1] == BashTool.PIPE_DRAIN_TIMEOUT_SECONDS
    assert killed_pgids == [12345]
    assert "[timeout after 600s;" in output


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
    assert "[timeout after 0.1s;" in output
    assert "descendant process kept stdout/stderr open" in output


def test_foreground_piped_kills_descendants_after_command_exits(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"

    output = BashTool(cwd=tmp_path).run(_spawn_sleeping_child_command(pid_file))

    child_pid = int(pid_file.read_text())
    try:
        assert re.search(rf"\b{child_pid}\b", output)
        _wait_for(lambda: not _pid_is_alive(child_pid))
    finally:
        _cleanup_pid(child_pid)


def test_foreground_tty_allocates_terminal(tmp_path: Path) -> None:
    tool = BashTool(cwd=tmp_path)
    command = _python_command("import os; print(f'tty={os.isatty(1)}')")

    output = tool.run(command, tty=True)

    assert "tty=True" in output
    assert "[elapsed " in output


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
