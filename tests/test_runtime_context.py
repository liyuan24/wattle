from __future__ import annotations

from pathlib import Path

from wattle.runtime_context import (
    RuntimeContextStore,
    command_family,
    render_runtime_context_projection,
)


def test_command_family_uses_executable_and_meaningful_subcommand() -> None:
    assert command_family("pytest tests") == "pytest"
    assert command_family("npm test -- --watch=false") == "npm test"
    assert command_family("cargo test -q") == "cargo test"
    assert command_family("python -m pytest tests") == "python -m pytest"
    assert command_family("fasttext supervised -input train.txt") == "fasttext supervised"


def test_projection_surfaces_repeated_failures_artifacts_and_metric_sources(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"x" * 128)
    store = RuntimeContextStore(root=tmp_path)

    for index in range(2):
        store.record_tool_metadata(
            tool_use_id=f"toolu_timeout_{index}",
            tool_name="bash",
            metadata={
                "command": "python train.py --epochs 100",
                "cwd": str(tmp_path),
                "status": "timed_out",
                "exit_code": None,
                "elapsed_seconds": 600.0,
                "timeout_seconds": 600.0,
                "stdout_tail": "",
                "stderr_tail": "timeout",
            },
        )
    store.record_tool_metadata(
        tool_use_id="toolu_metric_1",
        tool_name="bash",
        metadata={
            "command": f"python evaluate.py {tmp_path / 'proxy.txt'}",
            "cwd": str(tmp_path),
            "status": "success",
            "exit_code": 0,
            "elapsed_seconds": 2.0,
            "timeout_seconds": 120.0,
            "stdout_tail": "score: 0.91\n",
            "stderr_tail": "",
            "artifacts": [str(artifact)],
        },
    )
    store.record_tool_metadata(
        tool_use_id="toolu_metric_2",
        tool_name="bash",
        metadata={
            "command": f"python evaluate.py {tmp_path / 'raw.txt'}",
            "cwd": str(tmp_path),
            "status": "success",
            "exit_code": 0,
            "elapsed_seconds": 2.0,
            "timeout_seconds": 120.0,
            "stdout_tail": "score: 0.58\n",
            "stderr_tail": "",
        },
    )
    store.record_tool_metadata(
        tool_use_id="toolu_read",
        tool_name="bash",
        metadata={
            "command": "ls -la",
            "cwd": str(tmp_path),
            "status": "success",
            "exit_code": 0,
            "elapsed_seconds": 0.1,
            "timeout_seconds": 120.0,
            "stdout_tail": "total 0",
            "stderr_tail": "",
        },
    )

    projection = store.project()

    assert projection is not None
    rendered = render_runtime_context_projection(projection)
    assert "2 similar `python` commands timed out" in rendered
    assert "Metric `score` was observed with different values" in rendered
    assert str(artifact) in rendered
    assert "score=0.58" in rendered
    assert "score=0.91" in rendered
    assert "ls -la" not in rendered


def test_projection_includes_active_task_snapshot(tmp_path: Path) -> None:
    store = RuntimeContextStore(
        root=tmp_path,
        tasks_snapshot=lambda: [
            {
                "task_id": "shell-abc",
                "command": "python serve.py",
                "status": "running",
                "started_at": 1.0,
            }
        ],
    )

    projection = store.project()

    assert projection is not None
    assert projection.active_tasks[0].key == "active_task:shell-abc"
    assert "`shell-abc`" in projection.active_tasks[0].text
