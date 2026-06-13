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

from wattle.git_attribution import apply_git_attribution_env
from wattle.runtime import TaskStatus, WattleRuntime
from wattle.settings import load_settings
from wattle.tool_events import ToolRunEvent

from .base import Tool
from .utils.output import externalize_large_output


class BashTool(Tool):
    MAX_TIMEOUT_SECONDS = 600.0
    PIPE_DRAIN_TIMEOUT_SECONDS = 1.0

    name = "bash"
    description = (
        "Execute a shell command in the user's default shell and return its combined "
        "stdout/stderr. Non-zero exit codes are reported in the output, not raised. "
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
        timeout: float = 120.0,
        background: bool = False,
        tty: bool = False,
        max_output_chars: int | None = None,
        max_event_chars: int | None = None,
    ) -> str:
        if background:
            return self._run_background(command, tty=tty)
        output_limit = _output_limit(max_output_chars, max_event_chars)
        if tty:
            return self._run_foreground_tty(command, timeout, max_output_chars=output_limit)
        return self._run_foreground_piped(command, timeout, max_output_chars=output_limit)

    async def arun_with_events(
        self,
        *,
        emit: Callable[[ToolRunEvent], None],
        tool_use_id: str,
        command: str,
        timeout: float = 120.0,
        background: bool = False,
        tty: bool = False,
        max_output_chars: int | None = None,
        max_event_chars: int | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self._run_with_events,
            emit=emit,
            tool_use_id=tool_use_id,
            command=command,
            timeout=timeout,
            background=background,
            tty=tty,
            max_output_chars=max_output_chars,
            max_event_chars=max_event_chars,
        )

    def _run_with_events(
        self,
        *,
        emit: Callable[[ToolRunEvent], None],
        tool_use_id: str,
        command: str,
        timeout: float,
        background: bool,
        tty: bool,
        max_output_chars: int | None,
        max_event_chars: int | None,
    ) -> str:
        if background:
            return self._run_background(command, tty=tty)
        output_limit = _output_limit(max_output_chars, max_event_chars)
        emit(ToolRunEvent(tool_use_id, self.name, "started", command))
        try:
            if tty:
                return self._run_foreground_tty(
                    command,
                    timeout,
                    max_output_chars=output_limit,
                    emit=lambda text: emit(
                        ToolRunEvent(tool_use_id, self.name, "output", text, "combined")
                    ),
                )
            return self._run_foreground_piped(
                command,
                timeout,
                max_output_chars=output_limit,
                emit=lambda text, stream: emit(
                    ToolRunEvent(tool_use_id, self.name, "output", text, stream)
                ),
            )
        finally:
            emit(ToolRunEvent(tool_use_id, self.name, "completed"))

    def _run_foreground_piped(
        self,
        command: str,
        timeout: float,
        *,
        max_output_chars: int | None = None,
        emit: Callable[[str, str], None] | None = None,
    ) -> str:
        clamped_timeout = min(float(timeout), self.MAX_TIMEOUT_SECONDS)
        started_at = time.monotonic()
        attribution = self._git_attribution_env(command)
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=self.cwd,
            env=attribution.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            if emit is None:
                stdout, stderr = process.communicate(timeout=clamped_timeout)
            else:
                stdout, stderr = _communicate_with_output_events(
                    process,
                    timeout=clamped_timeout,
                    emit=emit,
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
            parts = [f"[timeout after {clamped_timeout:g}s; elapsed {elapsed:.2f}s]"]
            if capture_incomplete:
                parts.append("[output capture stopped: descendant process kept stdout/stderr open]")
            stdout_text = _decode_output(stdout)
            stderr_text = _decode_output(stderr)
            if stdout_text:
                parts.append(stdout_text.rstrip("\n"))
            if stderr_text:
                parts.append(f"[stderr]\n{stderr_text.rstrip(chr(10))}")
            return externalize_large_output(
                _prepend_warnings("\n".join(parts), attribution.warnings),
                root=self.cwd,
                tool_name=self.name,
                **_externalize_limits(max_output_chars),
            )

        _kill_process_group(process.pid)
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
        externalized = externalize_large_output(
            output,
            root=self.cwd,
            tool_name=self.name,
            **_externalize_limits(max_output_chars),
        )
        if externalized != output:
            return externalized
        return "\n".join([output, f"[elapsed {time.monotonic() - started_at:.2f}s]"])

    def _run_foreground_tty(
        self,
        command: str,
        timeout: float,
        *,
        max_output_chars: int | None = None,
        emit: Callable[[str], None] | None = None,
    ) -> str:
        clamped_timeout = min(float(timeout), self.MAX_TIMEOUT_SECONDS)
        started_at = time.monotonic()
        master_fd, slave_fd = pty.openpty()
        attribution = self._git_attribution_env(command)
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=self.cwd,
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
        try:
            deadline = started_at + clamped_timeout
            while True:
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
            with suppress(OSError):
                os.close(master_fd)

        output_text = b"".join(chunks).decode(errors="replace").rstrip("\n")
        parts: list[str] = list(attribution.warnings)
        if timed_out:
            elapsed = time.monotonic() - started_at
            parts.append(f"[timeout after {clamped_timeout:g}s; elapsed {elapsed:.2f}s]")
        if output_text:
            parts.append(output_text)
        if process.returncode not in (0, None):
            parts.append(f"[exit {process.returncode}]")
        if not parts:
            parts.append("[no output]")
        output = "\n".join(parts)
        externalized = externalize_large_output(
            output,
            root=self.cwd,
            tool_name=self.name,
            **_externalize_limits(max_output_chars),
        )
        if externalized != output:
            return externalized
        return "\n".join([output, f"[elapsed {time.monotonic() - started_at:.2f}s]"])

    def _run_background(self, command: str, *, tty: bool) -> str:
        started_at = time.time()
        log_path = self.runtime.tasks.jobs_dir / f"shell-{int(started_at * 1000)}-{os.getpid()}.log"
        attribution = self._git_attribution_env(command)
        log_file = log_path.open("ab")
        try:
            if tty:
                master_fd, slave_fd = pty.openpty()
                process = subprocess.Popen(
                    command,
                    shell=True,
                    cwd=self.cwd,
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
                    cwd=self.cwd,
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

        return "\n".join(
            [
                f"task_id: {task.task_id}",
                f"pid: {task.pid}",
                f"pgid: {task.pgid}",
                f"log_path: {task.log_path}",
                f"status_path: {task.status_path}",
                *([f"tty: {str(tty).lower()}"] if tty else []),
            ]
        )

    def _git_attribution_env(self, command: str):
        return apply_git_attribution_env(
            os.environ,
            cwd=self.cwd,
            state_root=self.runtime.tasks.root,
            command=command,
            enabled=load_settings().git_commit_attribution,
        )


def _communicate_with_output_events(
    process: subprocess.Popen[bytes],
    *,
    timeout: float,
    emit: Callable[[str, str], None],
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


def _kill_process_group(pgid: int) -> None:
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)


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
