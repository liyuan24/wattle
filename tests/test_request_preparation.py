from __future__ import annotations

import asyncio
from pathlib import Path

from wattle.compaction import RuntimeCompaction, estimate_request_context_tokens
from wattle.deadline import RunDeadline, append_runtime_deadline_notice, run_deadline_from_env
from wattle.providers import CompletionResponse, ImageBlock, Message, StubProvider, TextBlock
from wattle.request_preparation import RequestPreparer, project_messages_for_model_modalities
from wattle.runtime_context import RuntimeContextProjection, RuntimeFact


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


def test_prepare_adds_fresh_runtime_deadline_notice_to_request_system() -> None:
    now = [1000.0]
    deadline = RunDeadline(epoch_ms=1_900_000, clock=lambda: now[0])
    preparer = RequestPreparer(
        provider=StubProvider([]),
        model="test-model",
        system="Base system.",
        tools=[],
        max_tokens=1024,
        run_deadline=deadline,
    )
    messages = [Message(role="user", content=[TextBlock(text="do it")])]

    first = asyncio.run(preparer.aprepare(messages))
    now[0] = 1850.0
    second = asyncio.run(preparer.aprepare(messages))

    assert first.request.system is not None
    assert first.request.system.startswith("Base system.")
    assert "Wall-clock budget remaining for this run: about 15 minutes." in (
        first.request.system
    )
    assert second.request.system is not None
    assert "Wall-clock budget remaining for this run: about 50 seconds." in (
        second.request.system
    )
    assert "only start commands that fit in the remaining time" in second.request.system


def test_runtime_deadline_notice_can_be_loaded_from_env() -> None:
    deadline = run_deadline_from_env(
        {"WATTLE_RUN_DEADLINE_EPOCH_MS": "160000"},
        clock=lambda: 100.0,
    )

    system = append_runtime_deadline_notice(None, deadline)

    assert system is not None
    assert "Wall-clock budget remaining for this run: about 60 seconds." in system


def test_prepare_injects_runtime_context_projection_without_messages() -> None:
    events: list[dict[str, object]] = []
    projection = RuntimeContextProjection(
        warnings=[
            RuntimeFact(
                section="warnings",
                key="command_family:pytest:failed",
                score=95,
                text="2 similar `pytest` commands failed.",
            )
        ],
        signals=[
            RuntimeFact(
                section="signals",
                key="metric:score:validation.txt",
                score=70,
                text="metric: `score=0.62` from `python eval.py validation.txt`",
            )
        ],
    )
    preparer = RequestPreparer(
        provider=StubProvider([]),
        model="test-model",
        system="Base system.",
        tools=[{"name": "bash", "description": "run", "input_schema": {}}],
        max_tokens=1024,
        runtime_context_provider=lambda: projection,
        on_runtime_event=events.append,
        provider_name="stub",
    )
    messages = [Message(role="user", content=[TextBlock(text="continue")])]

    prepared = asyncio.run(preparer.aprepare(messages))

    assert prepared.request.messages == messages
    assert prepared.request.system is not None
    assert "Base system." in prepared.request.system
    assert "Runtime context:" in prepared.request.system
    assert "2 similar `pytest` commands failed." in prepared.request.system
    assert [event["type"] for event in events] == [
        "runtime_context_projection",
        "provider_request_prepared",
    ]
    assert events[0]["data"]["fact_keys"] == [
        "command_family:pytest:failed",
        "metric:score:validation.txt",
    ]
    assert events[1]["data"]["runtime_projection_sha256"] == events[0]["data"]["sha256"]
