"""Anthropic Messages API provider.

Translates Wattle's normalized `CompletionRequest` / `CompletionResponse`
shape to and from the `anthropic` SDK's `messages.create` API.

The Wattle content-block dataclasses already mirror Anthropic's wire shape
one-for-one, so the translation here is mechanical: dataclass -> dict on
the way out, response object -> dataclass on the way back.

Extended thinking
-----------------
Three modes are supported, selected by the `thinking`, `effort`, and
`budget` fields on `CompletionRequest`:

  * Disabled (`thinking=False`, the default): no `thinking` or
    `output_config` kwarg is sent.
  * Manual (`thinking=True` with `budget` set): emits
    `thinking={"type": "enabled", "budget_tokens": budget}`. `effort` is
    ignored — the API does not accept an effort hint in manual mode.
  * Adaptive (`thinking=True` without `budget`): emits
    `thinking={"type": "adaptive"}`, plus a top-level
    `output_config={"effort": effort}` when `effort` is supplied.

Two reasoning block shapes round-trip:

  * `ThinkingBlock` <-> `{"type": "thinking", "thinking": ..., "signature": ...}`
    — the `signature` is required on replay when extended thinking is
    combined with tool use, so this serializer always emits it when present.
  * `RedactedThinkingBlock` <-> `{"type": "redacted_thinking", "data": ...}`
    — opaque encrypted segments. Modeled as a separate dataclass (rather
    than overloading `ThinkingBlock` with a sentinel signature prefix) so
    the type system reflects the wire shape rather than encoding metadata
    inside string fields.

Prompt caching
--------------
Anthropic prompt caching is enabled for every request:

  * A block-level cache breakpoint is placed on the system prompt when one is
    present. Because Anthropic's cache prefix order is tools -> system ->
    messages, this also lets stable tool definitions participate in that
    prefix.
  * A top-level ``cache_control`` value enables Anthropic's automatic moving
    breakpoint at the end of the current conversation history, which is the
    recommended shape for multi-turn conversations.
"""

from __future__ import annotations

import base64
import inspect
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import anthropic

from .base import (
    CompletionRequest,
    CompletionResponse,
    ContentBlock,
    ImageBlock,
    Message,
    Provider,
    RedactedThinkingBlock,
    StopReason,
    StreamComplete,
    StreamEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseDelta,
)

CACHE_CONTROL_EPHEMERAL: dict[str, str] = {"type": "ephemeral"}


class AnthropicProvider(Provider):
    """Provider plugin backed by `anthropic.AsyncAnthropic`.

    The async client is constructed lazily so tests can inject a mock without
    touching the network or requiring an API key in the environment.
    """

    def __init__(
        self,
        async_client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self._async_client = async_client

    @property
    def async_client(self) -> anthropic.AsyncAnthropic:
        if self._async_client is None:
            self._async_client = anthropic.AsyncAnthropic()
        return self._async_client

    def fork(self) -> AnthropicProvider:
        return AnthropicProvider(async_client=self._async_client)

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        response = await _maybe_await(
            self.async_client.messages.create(**self._build_kwargs(request))
        )
        return _response_from_api(response)

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        """Async-native streaming path backed by ``anthropic.AsyncAnthropic``."""
        kwargs = self._build_kwargs(request)

        async with self.async_client.messages.stream(**kwargs) as stream:
            current_tool_use_id: str | None = None
            async for event in stream:
                etype = event.type
                if etype == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        current_tool_use_id = block.id
                        yield ToolUseDelta(
                            id=block.id,
                            name=block.name,
                            partial_json=None,
                        )
                elif etype == "content_block_delta":
                    delta = event.delta
                    dtype = delta.type
                    if dtype == "text_delta":
                        yield TextDelta(text=delta.text)
                    elif dtype == "thinking_delta":
                        yield ThinkingDelta(thinking=delta.thinking)
                    elif dtype == "input_json_delta":
                        assert current_tool_use_id is not None
                        yield ToolUseDelta(
                            id=current_tool_use_id,
                            name=None,
                            partial_json=delta.partial_json,
                        )

            final_message = await _maybe_await(stream.get_final_message())

        yield StreamComplete(response=_response_from_api(final_message))

    def _build_kwargs(self, request: CompletionRequest) -> dict[str, Any]:
        # Any: forwarded as `**kwargs` to `messages.create` / `messages.stream`,
        # both of which accept a heterogeneous keyword set (model, max_tokens,
        # message dicts, tool specs, optional system/thinking/etc.).
        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "cache_control": CACHE_CONTROL_EPHEMERAL,
            "messages": [_message_to_api(m) for m in request.messages],
            "tools": request.tools,
        }
        if request.system is not None:
            kwargs["system"] = [_system_text_block(request.system)]
        if request.thinking:
            if request.budget is not None:
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": request.budget,
                }
            else:
                kwargs["thinking"] = {"type": "adaptive"}
                if request.effort is not None:
                    kwargs["output_config"] = {"effort": request.effort}
        return kwargs


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value



# Any: SDK Message/RawMessage object. We read `.content`, `.stop_reason`,
# `.usage`. The Anthropic SDK exposes these as parameterized typed objects
# with stub gaps; the boundary stays Any and is narrowed via attribute access.
def _response_from_api(response: Any) -> CompletionResponse:
    content: list[ContentBlock] = [_block_from_api(b) for b in response.content]
    stop_reason: StopReason = response.stop_reason
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    cache_creation = getattr(response.usage, "cache_creation_input_tokens", None)
    if isinstance(cache_creation, int):
        usage["cache_creation_input_tokens"] = cache_creation
    cache_read = getattr(response.usage, "cache_read_input_tokens", None)
    if isinstance(cache_read, int):
        usage["cache_read_input_tokens"] = cache_read
        if cache_read > 0:
            usage["cached_tokens"] = cache_read
    return CompletionResponse(
        content=content,
        stop_reason=stop_reason,
        usage=usage,
    )


# Any: an Anthropic Messages API wire message — JSON object whose `content`
# carries a list of role-tagged block dicts. The wire spec is the source of
# truth; the outer envelope is fixed but block shapes vary by `type`.
def _message_to_api(message: Message) -> dict[str, Any]:
    return {
        "role": message.role,
        "content": [_block_to_api(b) for b in message.content],
    }


def _system_text_block(system: str) -> dict[str, Any]:
    return {
        "type": "text",
        "text": system,
        "cache_control": CACHE_CONTROL_EPHEMERAL,
    }


# Any: an Anthropic content block on the wire — fields differ by `type`
# (`tool_use` carries `id`+`input`; `thinking` carries `signature`; etc.).
# The Messages API wire spec is the source of truth.
def _block_to_api(block: ContentBlock) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ImageBlock):
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": block.media_type,
                "data": base64.b64encode(Path(block.path).read_bytes()).decode("ascii"),
            },
        }
    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if isinstance(block, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
            "is_error": block.is_error,
        }
    if isinstance(block, ThinkingBlock):
        wire: dict[str, Any] = {"type": "thinking", "thinking": block.thinking}
        if block.signature is not None:
            wire["signature"] = block.signature
        return wire
    if isinstance(block, RedactedThinkingBlock):
        return {"type": "redacted_thinking", "data": block.data}
    raise TypeError(f"Unknown content block type: {type(block).__name__}")


# Any: SDK content-block object — a parameterized union (TextBlock,
# ToolUseBlock, ThinkingBlock, RedactedThinkingBlock, ...). We discriminate
# on `.type` and read attributes that exist on the matching variant.
def _block_from_api(block: Any) -> ContentBlock:
    if block.type == "text":
        return TextBlock(text=block.text)
    if block.type == "tool_use":
        return ToolUseBlock(id=block.id, name=block.name, input=block.input)
    if block.type == "thinking":
        return ThinkingBlock(
            thinking=block.thinking,
            signature=getattr(block, "signature", None),
        )
    if block.type == "redacted_thinking":
        return RedactedThinkingBlock(data=block.data)
    raise ValueError(f"Unexpected content block type from API: {block.type!r}")
