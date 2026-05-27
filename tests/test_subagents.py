from __future__ import annotations

from pathlib import Path

from wattle.providers import CompletionResponse, StubProvider, TextBlock
from wattle.runtime import WattleRuntime
from wattle.system_prompt import build_system_prompt
from wattle.tools import build_tools
from wattle.tools.base import Tool


def _field(output: str, name: str) -> str:
    prefix = f"{name}: "
    for line in output.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix)
    raise AssertionError(f"missing field {name!r} in output:\n{output}")


def test_spawn_agent_runs_managed_child_session(tmp_path: Path) -> None:
    runtime = WattleRuntime(root=tmp_path)
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
    assert _field(spawn_output, "name") == "Hopper"
    assert _field(spawn_output, "role") == "explorer"
    assert _field(spawn_output, "workspace") == str(tmp_path)
    assert _field(spawn_output, "task") == "inspect the parser"
    wait_output = tools["wait_agent"].run(subagent_id, timeout_seconds=2)

    assert "status: completed" in wait_output
    assert "result:\nchild result" in wait_output

    request = provider.requests[0]
    assert request.model == "stub-model"
    assert request.system is not None
    assert "Available tools:" in request.system
    assert "Parent system context:\nbase system" in request.system
    assert "Report only facts." in request.system
    assert request.messages[0].content[0].type == "text"
    assert request.messages[0].content[0].text == "Delegated task:\ninspect the parser"
    tool_names = {spec["name"] for spec in request.tools}
    assert "bash" in tool_names
    assert "spawn_agent" not in tool_names


def test_spawn_agent_defaults_to_default_role_without_prompt_inference(
    tmp_path: Path,
) -> None:
    runtime = WattleRuntime(root=tmp_path)
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

    assert _field(spawn_output, "name") == "Hopper"
    assert _field(spawn_output, "role") == "default"


class _DummyViewImageTool(Tool):
    name = "view_image"
    description = "Attach an image."
    input_schema = {"type": "object", "properties": {}}

    def run(self) -> str:
        return "image"


class _DummyTextTool(Tool):
    name = "text_tool"
    description = "Text tool."
    input_schema = {"type": "object", "properties": {}}

    def run(self) -> str:
        return "text"


def test_subagent_model_override_filters_from_full_tool_set(tmp_path: Path) -> None:
    runtime = WattleRuntime(root=tmp_path)
    parent_tools = {"text_tool": _DummyTextTool()}
    full_tools = {**parent_tools, "view_image": _DummyViewImageTool()}
    provider = StubProvider(
        [CompletionResponse(content=[TextBlock(text="child result")], stop_reason="end_turn")]
    )
    runtime.subagents.configure(
        provider=provider,
        tools_by_name=parent_tools,
        full_tools_by_name=full_tools,
        system=None,
        model="deepseek-v4-flash",
        max_tokens=512,
    )

    spawn_output = runtime.subagents.spawn(
        task="inspect image",
        model="gpt-5.5",
        tool_names=["view_image"],
    )
    runtime.subagents.wait(spawn_output.subagent_id, timeout_seconds=2)

    assert spawn_output.tool_names == ["view_image"]
    tool_names = {spec["name"] for spec in provider.requests[0].tools}
    assert "view_image" in tool_names


def test_text_only_subagent_does_not_append_parent_view_image_prompt(
    tmp_path: Path,
) -> None:
    runtime = WattleRuntime(root=tmp_path)
    full_tools = build_tools(runtime)
    parent_system = build_system_prompt(
        tools_by_name=full_tools,
        skills=[],
    )
    provider = StubProvider(
        [CompletionResponse(content=[TextBlock(text="child result")], stop_reason="end_turn")]
    )
    runtime.subagents.configure(
        provider=provider,
        tools_by_name=full_tools,
        full_tools_by_name=full_tools,
        system=parent_system,
        model="gpt-5.5",
        max_tokens=512,
    )

    record = runtime.subagents.spawn(
        task="inspect text only",
        model="deepseek-v4-flash",
    )
    runtime.subagents.wait(record.subagent_id, timeout_seconds=2)

    request = provider.requests[0]
    assert "view_image" not in {spec["name"] for spec in request.tools}
    assert request.system is not None
    assert "view_image" not in request.system
    assert "Parent system context:" not in request.system


def test_text_only_subagent_preserves_custom_prefix_without_view_image_prompt(
    tmp_path: Path,
) -> None:
    runtime = WattleRuntime(root=tmp_path)
    full_tools = build_tools(runtime)
    parent_system = (
        "Custom parent instruction.\n\n"
        + build_system_prompt(
            tools_by_name=full_tools,
            skills=[],
        )
    )
    provider = StubProvider(
        [CompletionResponse(content=[TextBlock(text="child result")], stop_reason="end_turn")]
    )
    runtime.subagents.configure(
        provider=provider,
        tools_by_name=full_tools,
        full_tools_by_name=full_tools,
        system=parent_system,
        model="gpt-5.5",
        max_tokens=512,
    )

    record = runtime.subagents.spawn(
        task="inspect text only",
        model="deepseek-v4-flash",
    )
    runtime.subagents.wait(record.subagent_id, timeout_seconds=2)

    request = provider.requests[0]
    assert request.system is not None
    assert "Custom parent instruction." in request.system
    assert "view_image" not in request.system


def test_subagent_model_override_coerces_effort_and_context_window(tmp_path: Path) -> None:
    runtime = WattleRuntime(root=tmp_path)
    tools = build_tools(runtime)
    provider = StubProvider(
        [CompletionResponse(content=[TextBlock(text="child result")], stop_reason="end_turn")]
    )
    runtime.subagents.configure(
        provider=provider,
        tools_by_name=tools,
        full_tools_by_name=tools,
        system=None,
        model="gpt-5.5",
        max_tokens=512,
        context_window=123,
        thinking=True,
        effort="max",
    )

    record = runtime.subagents.spawn(
        task="inspect with override",
        model="gpt-5.2",
    )
    runtime.subagents.wait(record.subagent_id, timeout_seconds=2)

    assert record.effort == "xhigh"
    request = provider.requests[0]
    assert request.model == "gpt-5.2"
    assert request.effort == "xhigh"
    assert request.thinking is True


def test_subagent_model_override_preserves_configured_window_for_unknown_model(
    tmp_path: Path,
) -> None:
    runtime = WattleRuntime(root=tmp_path)
    tools = build_tools(runtime)
    provider = StubProvider(
        [CompletionResponse(content=[TextBlock(text="child result")], stop_reason="end_turn")]
    )
    runtime.subagents.configure(
        provider=provider,
        tools_by_name=tools,
        full_tools_by_name=tools,
        system=None,
        model="gpt-5.5",
        max_tokens=512,
        context_window=123,
    )

    record = runtime.subagents.spawn(
        task="inspect with override",
        model="custom-model",
    )
    runtime.subagents.wait(record.subagent_id, timeout_seconds=2)

    assert provider.requests[0].model == "custom-model"
    assert runtime.subagents._sessions[record.subagent_id].context_window == 123


def test_custom_parent_system_with_tool_markers_is_preserved(tmp_path: Path) -> None:
    runtime = WattleRuntime(root=tmp_path)
    tools = build_tools(runtime)
    custom_system = "Custom parent.\n\nAvailable tools:\nnone\n\nGuidelines:\nbe brief"
    provider = StubProvider(
        [CompletionResponse(content=[TextBlock(text="child result")], stop_reason="end_turn")]
    )
    runtime.subagents.configure(
        provider=provider,
        tools_by_name=tools,
        full_tools_by_name=tools,
        system=custom_system,
        model="gpt-5.5",
        max_tokens=512,
    )

    record = runtime.subagents.spawn(task="inspect text only", model="deepseek-v4-flash")
    runtime.subagents.wait(record.subagent_id, timeout_seconds=2)

    request = provider.requests[0]
    assert request.system is not None
    assert custom_system in request.system


def test_text_only_subagent_preserves_custom_suffix_without_view_image_prompt(
    tmp_path: Path,
) -> None:
    runtime = WattleRuntime(root=tmp_path)
    full_tools = build_tools(runtime)
    parent_system = (
        build_system_prompt(
            tools_by_name=full_tools,
            skills=[],
        )
        + "\n\nCustom suffix constraint."
    )
    provider = StubProvider(
        [CompletionResponse(content=[TextBlock(text="child result")], stop_reason="end_turn")]
    )
    runtime.subagents.configure(
        provider=provider,
        tools_by_name=full_tools,
        full_tools_by_name=full_tools,
        system=parent_system,
        model="gpt-5.5",
        max_tokens=512,
    )

    record = runtime.subagents.spawn(
        task="inspect text only",
        model="deepseek-v4-flash",
    )
    runtime.subagents.wait(record.subagent_id, timeout_seconds=2)

    request = provider.requests[0]
    assert request.system is not None
    assert "Custom suffix constraint." in request.system
    assert "view_image" not in request.system


def test_send_input_continues_existing_subagent_history(tmp_path: Path) -> None:
    runtime = WattleRuntime(root=tmp_path)
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
