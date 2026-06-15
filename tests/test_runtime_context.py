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


def test_path_discovery_ignores_heredoc_code_tokens(tmp_path: Path) -> None:
    store = RuntimeContextStore(root=tmp_path)

    store.record_tool_metadata(
        tool_use_id="toolu_code",
        tool_name="bash",
        metadata={
            "command": (
                "python - <<'PY'\n"
                "import importlib.util\n"
                "p='data/train-00000-of-00001.parquet'\n"
                "pf=pq.ParquetFile(path)\n"
                "tbl=pf.read_row_group(rg, columns=[label,text])\n"
                "print('ok')\n"
                "PY"
            ),
            "cwd": str(tmp_path),
            "status": "success",
            "exit_code": 0,
            "elapsed_seconds": 1.0,
            "timeout_seconds": 120.0,
            "stdout_tail": "ok\n",
            "stderr_tail": "",
        },
    )

    projection = store.project()

    assert projection is not None
    artifact_text = "\n".join(fact.text for fact in projection.artifacts)
    assert "importlib.util" not in artifact_text
    assert "train-00000-of-00001.parquet" not in artifact_text
    assert "read_row_group" not in artifact_text
    assert "ParquetFile" not in artifact_text


def test_path_discovery_keeps_common_bare_artifact_names(tmp_path: Path) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    store = RuntimeContextStore(root=tmp_path)

    store.record_tool_metadata(
        tool_use_id="toolu_model",
        tool_name="bash",
        metadata={
            "command": "cp /tmp/source model.bin",
            "cwd": str(tmp_path),
            "status": "success",
            "exit_code": 0,
            "elapsed_seconds": 1.0,
            "timeout_seconds": 120.0,
            "stdout_tail": "",
            "stderr_tail": "",
        },
    )

    projection = store.project()

    assert projection is not None
    assert str(model) in render_runtime_context_projection(projection)


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
