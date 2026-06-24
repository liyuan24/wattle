from __future__ import annotations

import asyncio
import errno
import os
import pty
import select
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

from wattle.command_summary import analyze_shell_chain
from wattle.git_attribution import apply_git_attribution_env
from wattle.runtime import TaskStatus, WattleRuntime
from wattle.settings import load_settings
from wattle.tool_events import ToolRunEvent

from .base import Tool, ToolExecutionResult
from .utils.output import externalize_large_output


class _ProcessCancelled(Exception):
    def __init__(self, stdout: bytes = b"", stderr: bytes = b"") -> None:
        super().__init__("process cancelled")
        self.stdout = stdout
        self.stderr = stderr


class BashTool(Tool):
    MAX_TIMEOUT_SECONDS = 600.0
    PIPE_DRAIN_TIMEOUT_SECONDS = 1.0

    name = "bash"
    description = (
        "Execute a shell command in the user's default shell and return its combined "
        "stdout/stderr. Non-zero exit codes are reported in the output, not raised. "
        "Set workdir instead of prefixing commands with cd. "
        "Do not use bash to create or edit files; use write for new files and edit "
        "for existing file changes."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "workdir": {
                "type": "string",
                "description": (
                    "Directory to run the command in. Prefer this over `cd ... &&`; "
                    "relative paths resolve against Wattle's current working directory."
                ),
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds. Defaults to 120.",
            },
            "background": {
                "type": "boolean",
                "description": "Run the command in the background. Defaults to false.",
                "default": False,
            },
            "tty": {
                "type": "boolean",
                "description": (
                    "Allocate a pseudo-terminal for commands that require TTY "
                    "semantics. Defaults to false."
                ),
                "default": False,
            },
            "max_output_chars": {
                "type": "integer",
                "description": (
                    "Optional inline output character budget before Wattle writes "
                    "the full output to an artifact."
                ),
            },
            "max_event_chars": {
                "type": "integer",
                "description": (
                    "Alias for max_output_chars, accepted for compatibility with "
                    "models that use event-budget terminology."
                ),
            },
        },
        "required": ["command"],
    }

    def __init__(
        self,
        runtime: WattleRuntime | None = None,
        cwd: Path | None = None,
    ) -> None:
        self.cwd = Path.cwd() if cwd is None else Path(cwd)
        self.runtime = runtime if runtime is not None else WattleRuntime(root=self.cwd)

    def run(
        self,
        command: str,
        workdir: str | None = None,
        timeout: float = 120.0,
        background: bool = False,
        tty: bool = False,
        max_output_chars: int | None = None,
        max_event_chars: int | None = None,
    ) -> str:
        return str(
            self.run_execution_result(
                command=command,
                workdir=workdir,
                timeout=timeout,
                background=background,
                tty=tty,
                max_output_chars=max_output_chars,
                max_event_chars=max_event_chars,
            ).content
        )

    def run_execution_result(
        self,
        command: str,
        workdir: str | None = None,
        timeout: float = 120.0,
        background: bool = False,
        tty: bool = False,
        max_output_chars: int | None = None,
        max_event_chars: int | None = None,
    ) -> ToolExecutionResult:
        resolved_workdir = self._resolve_workdir(workdir)
        if background:
            return self._run_background(command, cwd=resolved_workdir, tty=tty)
        output_limit = _output_limit(max_output_chars, max_event_chars)
        if tty:
            return self._run_foreground_tty(
                command,
                timeout,
                cwd=resolved_workdir,
                max_output_chars=output_limit,
            )
        return self._run_foreground_piped(
            command,
            timeout,
            cwd=resolved_workdir,
            max_output_chars=output_limit,
        )

    async def arun_with_events(
        self,
        *,
        emit: Callable[[ToolRunEvent], None],
        tool_use_id: str,
        cancel_event: threading.Event | None = None,
        command: str,
        workdir: str | None = None,
        timeout: float = 120.0,
        background: bool = False,
        tty: bool = False,
        max_output_chars: int | None = None,
        max_event_chars: int | None = None,
    ) -> str:
        result = await self.arun_execution_result_with_events(
            emit=emit,
            tool_use_id=tool_use_id,
            cancel_event=cancel_event,
            command=command,
            workdir=workdir,
            timeout=timeout,
            background=background,
            tty=tty,
            max_output_chars=max_output_chars,
            max_event_chars=max_event_chars,
        )
        return str(result.content)

    async def arun_execution_result_with_events(
        self,
        *,
        emit: Callable[[ToolRunEvent], None],
        tool_use_id: str,
        cancel_event: threading.Event | None = None,
        command: str,
        workdir: str | None = None,
        timeout: float = 120.0,
        background: bool = False,
        tty: bool = False,
        max_output_chars: int | None = None,
        max_event_chars: int | None = None,
    ) -> ToolExecutionResult:
        return await asyncio.to_thread(
            self._run_with_events,
            emit=emit,
            tool_use_id=tool_use_id,
            command=command,
            workdir=workdir,
            timeout=timeout,
            background=background,
            tty=tty,
            max_output_chars=max_output_chars,
            max_event_chars=max_event_chars,
            cancel_event=cancel_event,
        )

    def _run_with_events(
        self,
        *,
        emit: Callable[[ToolRunEvent], None],
        tool_use_id: str,
        command: str,
        workdir: str | None,
        timeout: float,
        background: bool,
        tty: bool,
        max_output_chars: int | None,
        max_event_chars: int | None,
        cancel_event: threading.Event | None,
    ) -> ToolExecutionResult:
        resolved_workdir = self._resolve_workdir(workdir)
        if background:
            return self._run_background(command, cwd=resolved_workdir, tty=tty)
        output_limit = _output_limit(max_output_chars, max_event_chars)
        emit(ToolRunEvent(tool_use_id, self.name, "started", command))
        try:
            if tty:
                return self._run_foreground_tty(
                    command,
                    timeout,
                    cwd=resolved_workdir,
                    max_output_chars=output_limit,
                    emit=lambda text: emit(
                        ToolRunEvent(tool_use_id, self.name, "output", text, "combined")
                    ),
                    cancel_event=cancel_event,
                )
            return self._run_foreground_piped(
                command,
                timeout,
                cwd=resolved_workdir,
                max_output_chars=output_limit,
                emit=lambda text, stream: emit(
                    ToolRunEvent(tool_use_id, self.name, "output", text, stream)
                ),
                cancel_event=cancel_event,
            )
        finally:
            emit(ToolRunEvent(tool_use_id, self.name, "completed"))

    def _run_foreground_piped(
        self,
        command: str,
        timeout: float,
        *,
        cwd: Path,
        max_output_chars: int | None = None,
        emit: Callable[[str, str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> ToolExecutionResult:
        clamped_timeout = min(float(timeout), self.MAX_TIMEOUT_SECONDS)
        started_at = time.monotonic()
        attribution = self._git_attribution_env(command, cwd=cwd)
        existing_pids = _process_snapshot()
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            env=attribution.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            if emit is None and cancel_event is None:
                stdout, stderr = process.communicate(timeout=clamped_timeout)
            else:
                stdout, stderr = _communicate_with_output_events(
                    process,
                    timeout=clamped_timeout,
                    emit=emit,
                    cancel_event=cancel_event,
                )
        except _ProcessCancelled as cancelled:
            _terminate_process_group(process.pid)
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    with suppress(OSError):
                        pipe.close()
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=0.2)
            stdout = cancelled.stdout
            stderr = cancelled.stderr
            elapsed = time.monotonic() - started_at
            parts = [f"[stopped by user after {elapsed:.2f}s]"]
            stdout_text = _decode_output(stdout)
            stderr_text = _decode_output(stderr)
            if stdout_text:
                parts.append(stdout_text.rstrip("\n"))
            if stderr_text:
                parts.append(f"[stderr]\n{stderr_text.rstrip(chr(10))}")
            _kill_new_processes_under_cwd(existing_pids, cwd)
            output = externalize_large_output(
                _prepend_warnings("\n".join(parts), attribution.warnings),
                root=self.cwd,
                tool_name=self.name,
                **_externalize_limits(max_output_chars),
            )
            return _bash_result(
                output,
                command=command,
                cwd=cwd,
                status="cancelled",
                exit_code=None,
                elapsed_seconds=elapsed,
                timeout_seconds=clamped_timeout,
                stdout_tail=stdout_text,
                stderr_tail=stderr_text,
            )
        except subprocess.TimeoutExpired as timeout_error:
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, signal.SIGKILL)
            capture_incomplete = False
            try:
                if emit is None:
                    stdout, stderr = process.communicate(timeout=self.PIPE_DRAIN_TIMEOUT_SECONDS)
                else:
                    stdout, stderr = _communicate_with_output_events(
                        process,
                        timeout=self.PIPE_DRAIN_TIMEOUT_SECONDS,
                        emit=emit,
                    )
            except subprocess.TimeoutExpired as drain_error:
                capture_incomplete = True
                stdout = _timeout_output(drain_error.output) or _timeout_output(
                    timeout_error.output
                )
                stderr = _timeout_output(drain_error.stderr) or _timeout_output(
                    timeout_error.stderr
                )
                for pipe in (process.stdout, process.stderr):
                    if pipe is not None:
                        with suppress(OSError):
                            pipe.close()
                with suppress(ProcessLookupError, PermissionError):
                    os.killpg(process.pid, signal.SIGKILL)
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=0.1)
            elapsed = time.monotonic() - started_at
            stdout_text = _decode_output(stdout)
            stderr_text = _decode_output(stderr)
            parts = _structured_command_output(
                status="timed_out",
                elapsed_seconds=elapsed,
                timeout_seconds=clamped_timeout,
                exit_code=None,
                stdout_text=stdout_text,
                stderr_text=stderr_text,
                output_capture_stopped=capture_incomplete,
            )
            output = externalize_large_output(
                _prepend_warnings("\n".join(parts), attribution.warnings),
                root=self.cwd,
                tool_name=self.name,
                **_externalize_limits(max_output_chars),
            )
            return _bash_result(
                output,
                command=command,
                cwd=cwd,
                status="timed_out",
                exit_code=None,
                elapsed_seconds=elapsed,
                timeout_seconds=clamped_timeout,
                stdout_tail=stdout_text,
                stderr_tail=stderr_text,
                output_capture_stopped=capture_incomplete,
            )

        _kill_process_group(process.pid)
        _kill_new_processes_under_cwd(existing_pids, cwd)
        parts = list(attribution.warnings)
        stdout_text = _decode_output(stdout)
        stderr_text = _decode_output(stderr)
        if stdout_text:
            parts.append(stdout_text.rstrip("\n"))
        if stderr_text:
            parts.append(f"[stderr]\n{stderr_text.rstrip(chr(10))}")
        if process.returncode != 0:
            parts.append(f"[exit {process.returncode}]")
        if not parts:
            parts.append("[no output]")
        output = "\n".join(parts)
        elapsed = time.monotonic() - started_at
        externalized = externalize_large_output(
            output,
            root=self.cwd,
            tool_name=self.name,
            **_externalize_limits(max_output_chars),
        )
        if externalized != output:
            output = externalized
        else:
            output = "\n".join([output, f"[elapsed {elapsed:.2f}s]"])
        return _bash_result(
            output,
            command=command,
            cwd=cwd,
            status="completed" if process.returncode == 0 else "failed",
            exit_code=process.returncode,
            elapsed_seconds=elapsed,
            timeout_seconds=clamped_timeout,
            stdout_tail=stdout_text,
            stderr_tail=stderr_text,
        )

    def _run_foreground_tty(
        self,
        command: str,
        timeout: float,
        *,
        cwd: Path,
        max_output_chars: int | None = None,
        emit: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> ToolExecutionResult:
        clamped_timeout = min(float(timeout), self.MAX_TIMEOUT_SECONDS)
        started_at = time.monotonic()
        master_fd, slave_fd = pty.openpty()
        attribution = self._git_attribution_env(command, cwd=cwd)
        existing_pids = _process_snapshot()
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            env=attribution.env,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
            start_new_session=True,
        )
        os.close(slave_fd)

        chunks: list[bytes] = []
        timed_out = False
        cancelled = False
        try:
            deadline = started_at + clamped_timeout
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    _terminate_process_group(process.pid)
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    break

                readable, _, _ = select.select([master_fd], [], [], min(0.1, remaining))
                if readable:
                    data = _read_pty(master_fd)
                    if data:
                        chunks.append(data)
                        if emit is not None:
                            emit(_decode_output(data))
                    elif process.poll() is not None:
                        break

                if process.poll() is not None:
                    while select.select([master_fd], [], [], 0)[0]:
                        data = _read_pty(master_fd)
                        if not data:
                            break
                        chunks.append(data)
                        if emit is not None:
                            emit(_decode_output(data))
                    break
        finally:
            _kill_process_group(process.pid)
            _kill_new_processes_under_cwd(existing_pids, cwd)
            with suppress(OSError):
                os.close(master_fd)

        output_text = b"".join(chunks).decode(errors="replace").rstrip("\n")
        parts: list[str] = list(attribution.warnings)
        if cancelled:
            elapsed = time.monotonic() - started_at
            parts.append(f"[stopped by user after {elapsed:.2f}s]")
        elif timed_out:
            elapsed = time.monotonic() - started_at
            parts.extend(
                _structured_command_output(
                    status="timed_out",
                    elapsed_seconds=elapsed,
                    timeout_seconds=clamped_timeout,
                    exit_code=None,
                    stdout_text=output_text,
                    stderr_text="",
                    output_capture_stopped=False,
                )
            )
        if output_text and not timed_out:
            parts.append(output_text)
        if not timed_out and not cancelled and process.returncode not in (0, None):
            parts.append(f"[exit {process.returncode}]")
        if not parts:
            parts.append("[no output]")
        output = "\n".join(parts)
        elapsed = time.monotonic() - started_at
        externalized = externalize_large_output(
            output,
            root=self.cwd,
            tool_name=self.name,
            **_externalize_limits(max_output_chars),
        )
        if externalized != output:
            output = externalized
        else:
            output = "\n".join([output, f"[elapsed {elapsed:.2f}s]"])
        if cancelled:
            status = "cancelled"
        elif timed_out:
            status = "timed_out"
        elif process.returncode == 0:
            status = "completed"
        else:
            status = "failed"
        return _bash_result(
            output,
            command=command,
            cwd=cwd,
            status=status,
            exit_code=process.returncode,
            elapsed_seconds=elapsed,
            timeout_seconds=clamped_timeout,
            stdout_tail=output_text,
            stderr_tail="",
            output_capture_stopped=False,
        )

    def _run_background(self, command: str, *, cwd: Path, tty: bool) -> ToolExecutionResult:
        started_at = time.time()
        log_path = self.runtime.tasks.jobs_dir / f"shell-{int(started_at * 1000)}-{os.getpid()}.log"
        attribution = self._git_attribution_env(command, cwd=cwd)
        log_file = log_path.open("ab")
        try:
            if tty:
                master_fd, slave_fd = pty.openpty()
                process = subprocess.Popen(
                    command,
                    shell=True,
                    cwd=cwd,
                    env=attribution.env,
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    close_fds=True,
                    start_new_session=True,
                )
                os.close(slave_fd)
            else:
                master_fd = None
                if attribution.warnings:
                    log_file.write(("\n".join(attribution.warnings) + "\n").encode())
                    log_file.flush()
                process = subprocess.Popen(
                    command,
                    shell=True,
                    cwd=cwd,
                    env=attribution.env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except Exception:
            log_file.close()
            raise

        task = self.runtime.tasks.register_shell_task(
            command=command,
            pid=process.pid,
            pgid=process.pid,
            log_path=log_path,
        )

        def watch() -> None:
            try:
                if master_fd is not None:
                    _copy_pty_to_log(master_fd, process, log_file)
                exit_code = process.wait()
                status = TaskStatus.COMPLETED if exit_code == 0 else TaskStatus.FAILED
                self.runtime.tasks.mark_terminal(
                    task.task_id,
                    status=status,
                    exit_code=exit_code,
                )
            finally:
                if master_fd is not None:
                    with suppress(OSError):
                        os.close(master_fd)
                log_file.close()

        threading.Thread(target=watch, name=f"wattle-watch-{task.task_id}", daemon=True).start()

        output = "\n".join(
            [
                f"task_id: {task.task_id}",
                f"pid: {task.pid}",
                f"pgid: {task.pgid}",
                f"log_path: {task.log_path}",
                f"status_path: {task.status_path}",
                *([f"tty: {str(tty).lower()}"] if tty else []),
            ]
        )
        return _bash_result(
            output,
            command=command,
            cwd=cwd,
            status="background",
            exit_code=None,
            elapsed_seconds=0.0,
            timeout_seconds=None,
            stdout_tail="",
            stderr_tail="",
            background_task_id=task.task_id,
            log_path=str(task.log_path),
            status_path=str(task.status_path),
        )

    def _resolve_workdir(self, workdir: str | None) -> Path:
        if workdir is None or str(workdir).strip() == "":
            candidate = self.cwd
        else:
            raw = Path(str(workdir)).expanduser()
            candidate = raw if raw.is_absolute() else self.cwd / raw
        try:
            resolved = candidate.resolve()
        except OSError as exc:
            raise ValueError(f"workdir does not exist or cannot be resolved: {workdir}") from exc
        if not resolved.exists():
            raise ValueError(f"workdir does not exist: {resolved}")
        if not resolved.is_dir():
            raise ValueError(f"workdir is not a directory: {resolved}")
        return resolved

    def _git_attribution_env(self, command: str, *, cwd: Path):
        return apply_git_attribution_env(
            os.environ,
            cwd=cwd,
            state_root=self.runtime.tasks.root,
            command=command,
            enabled=load_settings().git_commit_attribution,
        )


def _communicate_with_output_events(
    process: subprocess.Popen[bytes],
    *,
    timeout: float,
    emit: Callable[[str, str], None] | None,
    cancel_event: threading.Event | None = None,
) -> tuple[bytes, bytes]:
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    streams: dict[int, tuple[str, list[bytes]]] = {}
    if process.stdout is not None:
        os.set_blocking(process.stdout.fileno(), False)
        streams[process.stdout.fileno()] = ("stdout", stdout_chunks)
    if process.stderr is not None:
        os.set_blocking(process.stderr.fileno(), False)
        streams[process.stderr.fileno()] = ("stderr", stderr_chunks)
    deadline = time.monotonic() + timeout
    while streams:
        if cancel_event is not None and cancel_event.is_set():
            raise _ProcessCancelled(
                b"".join(stdout_chunks),
                b"".join(stderr_chunks),
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        readable, _, _ = select.select(list(streams), [], [], min(0.1, remaining))
        if not readable:
            if process.poll() is not None:
                continue
            continue
        for fd in readable:
            stream, chunks = streams[fd]
            try:
                data = os.read(fd, 4096)
            except BlockingIOError:
                continue
            if data:
                chunks.append(data)
                if emit is not None:
                    emit(_decode_output(data), stream)
            else:
                streams.pop(fd, None)
    process.wait(timeout=0)
    return b"".join(stdout_chunks), b"".join(stderr_chunks)


def _read_pty(fd: int) -> bytes:
    try:
        return os.read(fd, 4096)
    except OSError as exc:
        if exc.errno == errno.EIO:
            return b""
        raise


def _terminate_process_group(pgid: int, *, grace_seconds: float = 0.5) -> None:
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            break
        time.sleep(0.02)
    _kill_process_group(pgid)


def _kill_process_group(pgid: int) -> None:
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)


def _process_snapshot() -> set[int]:
    proc = Path("/proc")
    if not proc.exists():
        return set()
    return {int(path.name) for path in proc.iterdir() if path.name.isdigit()}


def _kill_new_processes_under_cwd(existing_pids: set[int], cwd: Path) -> None:
    try:
        root = cwd.resolve()
    except OSError:
        return
    if root == Path("/"):
        return

    proc = Path("/proc")
    if not proc.exists():
        return
    current_pid = os.getpid()
    for path in proc.iterdir():
        if not path.name.isdigit():
            continue
        pid = int(path.name)
        if pid in existing_pids or pid == current_pid:
            continue
        try:
            process_cwd = (path / "cwd").resolve()
        except (FileNotFoundError, PermissionError, OSError):
            continue
        if not _is_relative_to(process_cwd, root):
            continue
        with suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _decode_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _prepend_warnings(output: str, warnings: tuple[str, ...]) -> str:
    if not warnings:
        return output
    return "\n".join([*warnings, output])


def _structured_command_output(
    *,
    status: str,
    elapsed_seconds: float,
    timeout_seconds: float | None,
    exit_code: int | None,
    stdout_text: str,
    stderr_text: str,
    output_capture_stopped: bool,
) -> list[str]:
    parts = [
        f"Status: {_display_status(status)}",
        f"Wall time: {elapsed_seconds:.2f}s",
    ]
    if timeout_seconds is not None:
        parts.append(f"Requested timeout: {timeout_seconds:g}s")
    parts.append(f"Exit code: {exit_code if exit_code is not None else 'unknown'}")
    if output_capture_stopped:
        parts.append("Output capture stopped: descendant process kept stdout/stderr open")
    if stdout_text or stderr_text:
        parts.append("Output:")
        if stdout_text:
            parts.append(stdout_text.rstrip("\n"))
        if stderr_text:
            parts.append(f"[stderr]\n{stderr_text.rstrip(chr(10))}")
    return parts


def _display_status(status: str) -> str:
    if status == "timed_out":
        return "timed out"
    return status.replace("_", " ")


def _output_limit(
    max_output_chars: int | None,
    max_event_chars: int | None,
) -> int | None:
    raw_limit = max_output_chars if max_output_chars is not None else max_event_chars
    if raw_limit is None:
        return None
    return max(1, int(raw_limit))


def _externalize_limits(max_output_chars: int | None) -> dict[str, int]:
    if max_output_chars is None:
        return {}
    excerpt_chars = min(max_output_chars, max(1, max_output_chars // 2))
    return {
        "max_inline_chars": max_output_chars,
        "max_excerpt_chars": excerpt_chars,
    }


def _bash_result(
    content: str,
    *,
    command: str,
    cwd: Path,
    status: str,
    exit_code: int | None,
    elapsed_seconds: float,
    timeout_seconds: float | None,
    stdout_tail: str,
    stderr_tail: str,
    output_capture_stopped: bool = False,
    background_task_id: str | None = None,
    log_path: str | None = None,
    status_path: str | None = None,
) -> ToolExecutionResult:
    metadata: dict[str, object] = {
        "kind": "command",
        "command": command,
        "cwd": str(cwd),
        "workdir": str(cwd),
        "original_command": command,
        "status": status,
        "exit_code": exit_code,
        "elapsed_seconds": elapsed_seconds,
        "timeout_seconds": timeout_seconds,
        "output_capture_stopped": output_capture_stopped,
        "background_task_id": background_task_id,
        "log_path": log_path,
        "status_path": status_path,
        "stdout_tail": _tail_text(stdout_tail),
        "stderr_tail": _tail_text(stderr_tail),
    }
    chain = analyze_shell_chain(command)
    if chain is not None and chain.command_count > 1:
        metadata["is_shell_chain"] = True
        metadata["shell_chain_segments"] = [" ".join(segment) for segment in chain.segments]
        metadata["shell_chain_operators"] = list(chain.operators)
    output_artifact = _full_output_path_from_text(content)
    if output_artifact is not None:
        metadata["output_artifact"] = output_artifact
    return ToolExecutionResult(content=content, metadata=metadata)


def _tail_text(text: str, *, max_chars: int = 2000) -> str:
    return text if len(text) <= max_chars else text[-max_chars:]


def _full_output_path_from_text(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("full_output_path: "):
            return line.split(": ", 1)[1]
    return None


def _copy_pty_to_log(master_fd: int, process: subprocess.Popen[bytes], log_file) -> None:
    while True:
        readable, _, _ = select.select([master_fd], [], [], 0.1)
        if readable:
            data = _read_pty(master_fd)
            if data:
                log_file.write(data)
                log_file.flush()
            elif process.poll() is not None:
                break
        if process.poll() is not None:
            while select.select([master_fd], [], [], 0)[0]:
                data = _read_pty(master_fd)
                if not data:
                    break
                log_file.write(data)
            log_file.flush()
            break
