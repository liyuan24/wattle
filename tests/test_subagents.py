from __future__ import annotations

from pathlib import Path

from willow.providers import CompletionResponse, StubProvider, TextBlock
from willow.runtime import WillowRuntime
from willow.tools import build_tools


def _field(output: str, name: str) -> str:
    prefix = f"{name}: "
    for line in output.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix)
    raise AssertionError(f"missing field {name!r} in output:\n{output}")


def test_spawn_agent_runs_managed_child_session(tmp_path: Path) -> None:
    runtime = WillowRuntime(root=tmp_path)
    tools = build_tools(runtime)
    provider = StubProvider(
        [
            CompletionResponse(
                content=[TextBlock(text="child result")],
                stop_reason="end_turn",
            )
        ]
    )
    runtime.subagents.configure(
        provider=provider,
        tools_by_name=tools,
        system="base system",
        model="stub-model",
        max_tokens=512,

    )

    spawn_output = tools["spawn_agent"].run(
        task="inspect the parser",
        agent_type="explorer",
        instructions="Report only facts.",
    )
    subagent_id = _field(spawn_output, "subagent_id")
    assert _field(spawn_output, "name") == "Euclid"
    assert _field(spawn_output, "role") == "explorer"
    assert _field(spawn_output, "workspace") == str(tmp_path)
    assert _field(spawn_output, "task") == "inspect the parser"
    wait_output = tools["wait_agent"].run(subagent_id, timeout_seconds=2)

    assert "status: completed" in wait_output
    assert "result:\nchild result" in wait_output

    request = provider.requests[0]
    assert request.model == "stub-model"
    assert request.system is not None
    assert "base system" in request.system
    assert "Report only facts." in request.system
    assert request.messages[0].content[0].type == "text"
    assert request.messages[0].content[0].text == "Delegated task:\ninspect the parser"
    tool_names = {spec["name"] for spec in request.tools}
    assert "bash" in tool_names
    assert "spawn_agent" not in tool_names


def test_spawn_agent_defaults_to_default_role_without_prompt_inference(
    tmp_path: Path,
) -> None:
    runtime = WillowRuntime(root=tmp_path)
    tools = build_tools(runtime)
    provider = StubProvider(
        [
            CompletionResponse(
                content=[TextBlock(text="child result")],
                stop_reason="end_turn",
            )
        ]
    )
    runtime.subagents.configure(
        provider=provider,
        tools_by_name=tools,
        system=None,
        model="stub-model",
        max_tokens=512,
    )

    spawn_output = tools["spawn_agent"].run(task="inspect the parser")

    assert _field(spawn_output, "name") == "Euclid"
    assert _field(spawn_output, "role") == "default"


def test_send_input_continues_existing_subagent_history(tmp_path: Path) -> None:
    runtime = WillowRuntime(root=tmp_path)
    tools = build_tools(runtime)
    provider = StubProvider(
        [
            CompletionResponse(content=[TextBlock(text="first")], stop_reason="end_turn"),
            CompletionResponse(content=[TextBlock(text="second")], stop_reason="end_turn"),
        ]
    )
    runtime.subagents.configure(
        provider=provider,
        tools_by_name=tools,
        system=None,
        model="stub-model",
        max_tokens=512,

    )

    subagent_id = _field(tools["spawn_agent"].run(task="first task"), "subagent_id")
    tools["wait_agent"].run(subagent_id, timeout_seconds=2)

    send_output = tools["send_input"].run(
        subagent_id,
        message="now answer the follow-up",
    )
    assert "status: running" in send_output or "status: completed" in send_output
    wait_output = tools["wait_agent"].run(subagent_id, timeout_seconds=2)

    assert "status: completed" in wait_output
    assert "result:\nsecond" in wait_output
    second_request = provider.requests[1]
    assert [message.role for message in second_request.messages] == [
        "user",
        "assistant",
        "user",
    ]
    assert second_request.messages[2].content[0].type == "text"
    assert second_request.messages[2].content[0].text == "now answer the follow-up"
