"""Anthropic Messages API provider.

Translates Willow's normalized `CompletionRequest` / `CompletionResponse`
shape to and from the `anthropic` SDK's `messages.create` API.

The Willow content-block dataclasses already mirror Anthropic's wire shape
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
from collections.abc import Iterator
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
    """Provider plugin backed by `anthropic.Anthropic`.

    The client is constructed lazily so tests can inject a mock without
    touching the network or requiring an API key in the environment.
    """

    def __init__(self, client: anthropic.Anthropic | None = None) -> None:
        self.client = client if client is not None else anthropic.Anthropic()

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        response = self.client.messages.create(**self._build_kwargs(request))
        return _response_from_api(response)

    def stream(self, request: CompletionRequest) -> Iterator[StreamEvent]:
        """Stream a turn against the Anthropic Messages API.

        Uses ``client.messages.stream(...)`` (a context manager exposing an
        iterator of typed SDK events) and translates each SDK event into the
        Willow ``StreamEvent`` vocabulary:

          * ``content_block_start`` for a ``tool_use`` block emits a
            ``ToolUseDelta`` carrying the block id and tool name (start).
          * ``content_block_delta`` of type ``text_delta`` emits a
            ``TextDelta``; ``thinking_delta`` emits a ``ThinkingDelta``;
            ``input_json_delta`` emits a continuation ``ToolUseDelta`` for
            the most recently started tool_use block.
          * ``redacted_thinking`` blocks emit no incremental event — they
            still surface in the assembled response.

        After the SDK stream is fully drained, a single ``StreamComplete``
        is yielded carrying the same ``CompletionResponse`` shape that
        ``complete()`` would have returned (assembled via
        ``stream.get_final_message()``).
        """
        kwargs = self._build_kwargs(request)

        with self.client.messages.stream(**kwargs) as stream:
            current_tool_use_id: str | None = None
            for event in stream:
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
                    # text / thinking / redacted_thinking: no event yet.
                elif etype == "content_block_delta":
                    delta = event.delta
                    dtype = delta.type
                    if dtype == "text_delta":
                        yield TextDelta(text=delta.text)
                    elif dtype == "thinking_delta":
                        yield ThinkingDelta(thinking=delta.thinking)
                    elif dtype == "input_json_delta":
                        # input_json fragments only ever follow a tool_use start.
                        assert current_tool_use_id is not None
                        yield ToolUseDelta(
                            id=current_tool_use_id,
                            name=None,
                            partial_json=delta.partial_json,
                        )
                    # signature_delta, citations_delta: no Willow event.
                # message_start / message_delta / message_stop /
                # content_block_stop: assembly is handled by the SDK
                # snapshot; we only need them to keep iteration alive.

            final_message = stream.get_final_message()

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
