from __future__ import annotations

import asyncio
from pathlib import Path

from wattle.compaction import estimate_request_context_tokens
from wattle.providers import CompletionResponse, ImageBlock, Message, StubProvider, TextBlock
from wattle.request_preparation import RequestPreparer, project_messages_for_model_modalities


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
