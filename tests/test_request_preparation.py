from __future__ import annotations

import asyncio
from pathlib import Path

from wattle.compaction import RuntimeCompaction, estimate_request_context_tokens
from wattle.deadline import RunDeadline, append_runtime_deadline_notice, run_deadline_from_env
from wattle.providers import (
    CompletionResponse,
    ImageBlock,
    Message,
    StubProvider,
    TextBlock,
    ToolResultBlock,
)
from wattle.request_preparation import (
    POST_TOOL_OBSERVATION_CHECKPOINT,
    RequestPreparer,
    append_post_tool_observation_checkpoint,
    append_runtime_deadline_status,
    project_messages_for_model_modalities,
)


def test_project_messages_replaces_images_for_text_only_model(tmp_path: Path) -> None:
    image = tmp_path / "shot.png"
    image.write_bytes(b"fake-png")
    messages = [
        Message(
            role="user",
            content=[
                TextBlock(text="look [image#1]"),
                ImageBlock(
                    path=str(image),
                    media_type="image/png",
                    filename=image.name,
                    size_bytes=image.stat().st_size,
                ),
            ],
        )
    ]

    projected = project_messages_for_model_modalities(messages, model="deepseek-v4-flash")

    assert projected[0].content == [
        TextBlock(text="look [image#1]"),
        TextBlock(text="[image omitted: model deepseek-v4-flash does not support image inputs]"),
    ]
    assert messages[0].content[1].type == "image"


def test_project_messages_preserves_images_for_image_capable_model(tmp_path: Path) -> None:
    image = tmp_path / "shot.png"
    image.write_bytes(b"fake-png")
    messages = [
        Message(
            role="user",
            content=[
                ImageBlock(
                    path=str(image),
                    media_type="image/png",
                    filename=image.name,
                    size_bytes=image.stat().st_size,
                )
            ],
        )
    ]

    assert project_messages_for_model_modalities(messages, model="kimi-k2.6") is messages


def test_project_messages_replaces_unavailable_images_for_image_capable_model(
    tmp_path: Path,
) -> None:
    image = tmp_path / "moved.png"
    messages = [
        Message(
            role="user",
            content=[
                TextBlock(text="look at this"),
                ImageBlock(
                    path=str(image),
                    media_type="image/png",
                    filename=image.name,
                    size_bytes=12345,
                ),
            ],
        )
    ]

    projected = project_messages_for_model_modalities(messages, model="kimi-k2.6")

    assert projected is not messages
    assert projected[0].content[0] == TextBlock(text="look at this")
    replacement = projected[0].content[1]
    assert isinstance(replacement, TextBlock)
    assert "image omitted" in replacement.text
    assert str(image) in replacement.text
    assert "filename=moved.png" in replacement.text
    assert "media_type=image/png" in replacement.text
    assert "size_bytes=12345" in replacement.text
    assert messages[0].content[1].type == "image"


def test_prepare_omits_unavailable_image_without_mutating_history(tmp_path: Path) -> None:
    image = tmp_path / "gone.png"
    messages = [
        Message(
            role="user",
            content=[
                ImageBlock(
                    path=str(image),
                    media_type="image/png",
                    filename=image.name,
                    size_bytes=99,
                )
            ],
        )
    ]
    preparer = RequestPreparer(
        provider=StubProvider([CompletionResponse(content=[], stop_reason="end_turn")]),
        model="kimi-k2.6",
        system=None,
        tools=[],
        max_tokens=1024,
    )

    prepared = asyncio.run(preparer.aprepare(messages))

    projected_block = prepared.request.messages[0].content[0]
    assert isinstance(projected_block, TextBlock)
    assert "attached file is no longer available" in projected_block.text
    assert messages[0].content[0].type == "image"


def test_prepare_context_estimate_uses_projected_text_only_messages(
    tmp_path: Path,
) -> None:
    image = tmp_path / "shot.png"
    image.write_bytes(b"fake-png")
    messages = [
        Message(
            role="user",
            content=[
                TextBlock(text="look [image#1]"),
                ImageBlock(
                    path=str(image),
                    media_type="image/png",
                    filename=image.name,
                    size_bytes=image.stat().st_size,
                ),
            ],
        )
    ]
    preparer = RequestPreparer(
        provider=StubProvider([CompletionResponse(content=[], stop_reason="end_turn")]),
        model="deepseek-v4-flash",
        system=None,
        tools=[],
        max_tokens=1024,
    )

    prepared = asyncio.run(preparer.aprepare(messages))

    expected_messages = project_messages_for_model_modalities(
        messages,
        model="deepseek-v4-flash",
    )
    assert prepared.request.messages == expected_messages
    assert prepared.context_tokens == estimate_request_context_tokens(
        system=None,
        messages=expected_messages,
        tools=[],
    )
    assert prepared.context_tokens < 100


def test_prepare_reads_provider_context_tokens_lazily() -> None:
    values = iter([None, 800])
    provider = StubProvider(
        [
            CompletionResponse(content=[TextBlock(text="updated summary")], stop_reason="end_turn"),
        ]
    )
    messages = [
        Message(
            role="user" if index % 2 else "assistant",
            content=[TextBlock(text="x" * 80)],
        )
        for index in range(50)
    ]
    messages.append(Message(role="user", content=[TextBlock(text="a")]))
    state = RuntimeCompaction(
        summary="initial summary",
        summarized_until=41,
        first_kept_index=41,
    )
    preparer = RequestPreparer(
        provider=provider,
        model="test-model",
        system=None,
        tools=[],
        max_tokens=10,
        context_window=1_000,
        state=state,
        provider_context_tokens=lambda: next(values),
    )

    prepared_before_pressure = asyncio.run(preparer.aprepare(messages))
    assert preparer.state == state

    prepared_after_pressure = asyncio.run(preparer.aprepare(messages))

    assert preparer.state is not None
    assert preparer.state != state
    assert preparer.state.summary == "updated summary"
    assert preparer.state.summarized_until == len(messages) - 1
    assert "initial summary" in prepared_before_pressure.request.messages[0].content[0].text
    assert "updated summary" in prepared_after_pressure.request.messages[0].content[0].text
    assert prepared_after_pressure.request.messages[-1] == messages[-1]


def test_prepare_prefers_provider_context_tokens_from_message_history() -> None:
    provider = StubProvider(
        [
            CompletionResponse(content=[TextBlock(text="history summary")], stop_reason="end_turn"),
        ]
    )
    messages = [
        Message(
            role="assistant",
            content=[TextBlock(text="previous answer")],
            input_tokens=800,
        ),
        Message(role="user", content=[TextBlock(text="next request")]),
    ]
    preparer = RequestPreparer(
        provider=provider,
        model="test-model",
        system=None,
        tools=[],
        max_tokens=10,
        context_window=1_000,
        provider_context_tokens=lambda: None,
    )

    prepared = asyncio.run(preparer.aprepare(messages))

    assert preparer.state is not None
    assert preparer.state.summary == "history summary"
    assert "history summary" in prepared.request.messages[0].content[0].text
    assert prepared.request.messages[-1] == messages[-1]


def test_prepare_keeps_deadline_out_of_system_and_appends_tail_status() -> None:
    now = [1000.0]
    deadline = RunDeadline(epoch_ms=1_900_000, clock=lambda: now[0])
    events: list[dict[str, object]] = []
    preparer = RequestPreparer(
        provider=StubProvider([]),
        model="test-model",
        system="Base system.",
        tools=[],
        max_tokens=1024,
        run_deadline=deadline,
        on_runtime_event=events.append,
    )
    messages = [Message(role="user", content=[TextBlock(text="do it")])]

    first = asyncio.run(preparer.aprepare(messages))
    now[0] = 1850.0
    second = asyncio.run(preparer.aprepare(messages))

    assert first.request.system is not None
    assert first.request.system == "Base system."
    assert second.request.system is not None
    assert first.request.system == second.request.system
    assert "about 15 minutes" in first.request.messages[-1].content[0].text
    assert "about 50 seconds" in second.request.messages[-1].content[0].text
    system_hashes = [
        event["data"]["system_sha256"]
        for event in events
        if event.get("type") == "provider_request_prepared"
    ]
    assert len(set(system_hashes)) == 1


def test_runtime_deadline_status_is_appended_at_request_tail() -> None:
    deadline = RunDeadline(epoch_ms=160_000, clock=lambda: 100.0)
    messages = [Message(role="user", content=[TextBlock(text="do it")])]

    with_status = append_runtime_deadline_status(messages, deadline)

    assert with_status[:-1] == messages
    assert with_status[-1].role == "user"
    assert "about 60 seconds" in with_status[-1].content[0].text
    assert append_runtime_deadline_status(messages, None) is messages


def test_prepare_adds_provider_only_observation_checkpoint_after_tool_result() -> None:
    preparer = RequestPreparer(
        provider=StubProvider([]),
        model="test-model",
        system="Base system.",
        tools=[],
        max_tokens=1024,
    )
    messages = [
        Message(role="user", content=[TextBlock(text="run it")]),
        Message(role="assistant", content=[TextBlock(text="tool call placeholder")]),
        Message(
            role="user",
            content=[
                ToolResultBlock(
                    tool_use_id="call_1",
                    content="observed output",
                    is_error=False,
                )
            ],
        ),
    ]

    prepared = asyncio.run(preparer.aprepare(messages))

    assert prepared.request.system == "Base system."
    assert prepared.request.messages[:-1] == messages[:-1]
    assert prepared.request.messages[-1].content[:-1] == messages[-1].content
    checkpoint = prepared.request.messages[-1].content[-1]
    assert isinstance(checkpoint, TextBlock)
    assert checkpoint.text == POST_TOOL_OBSERVATION_CHECKPOINT
    assert checkpoint.text.startswith("[system reminder]")
    assert "derived file, command, or artifact" in checkpoint.text
    assert "preserves the meaning of the observed inputs" in checkpoint.text
    assert "check whether the user's request is actually complete" in checkpoint.text
    assert "what was and was not verified" in checkpoint.text
    assert messages[-1].content == [
        ToolResultBlock(tool_use_id="call_1", content="observed output", is_error=False)
    ]


def test_post_tool_observation_checkpoint_is_not_added_without_latest_tool_result() -> None:
    messages = [
        Message(
            role="user",
            content=[
                ToolResultBlock(
                    tool_use_id="call_1",
                    content="observed output",
                    is_error=False,
                )
            ],
        ),
        Message(role="assistant", content=[TextBlock(text="next")]),
    ]

    assert append_post_tool_observation_checkpoint(messages) is messages
    with_checkpoint = append_post_tool_observation_checkpoint(messages[:1])
    assert (
        append_post_tool_observation_checkpoint(with_checkpoint)
        == with_checkpoint
    )


def test_runtime_deadline_notice_can_be_loaded_from_env() -> None:
    deadline = run_deadline_from_env(
        {"WATTLE_RUN_DEADLINE_EPOCH_MS": "160000"},
        clock=lambda: 100.0,
    )

    system = append_runtime_deadline_notice(None, deadline)

    assert system is None
    assert "about 60 seconds" in deadline.request_status()
