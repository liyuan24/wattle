"""OpenAI Chat Completions API provider.

Translates Willow's normalized `CompletionRequest` / `CompletionResponse`
shape to and from the `openai` SDK's `chat.completions.create` API.

The Chat Completions wire format is structurally different from Willow's
internal block-based shape in three ways, all handled here:

  * `system` is prepended as a `{"role": "system"}` wire message rather than
    a top-level parameter.
  * Assistant turns split text and tool calls across the parallel
    `content` / `tool_calls` fields; we collapse our ordered block list
    onto that pair.
  * Tool results live on a dedicated `"tool"` wire role; each
    `ToolResultBlock` in a Willow user message is lifted to its own wire
    message (one-to-many).

OpenAI's tool role has no `is_error` flag, so an error result is conveyed
by prepending a sentinel `"[error] "` prefix to the wire `content`. This
is documented and tested; the convention is deterministic and round-trip
visible to the model.

Stop-reason vocabulary is Anthropic's; OpenAI's `finish_reason` is mapped
into it (`stop` -> `end_turn`, `tool_calls` -> `tool_use`,
`length` -> `max_tokens`, anything else -> `stop_sequence`).

Reasoning is not preservable on this provider
---------------------------------------------
Chat Completions reasoning models do reason internally, but the API surface
exposes no content-block representation of that reasoning and no input
field that would let a client replay reasoning between turns. The only
reasoning-related signal returned is a token count
(`usage.completion_tokens_details.reasoning_tokens`).

Concretely: any `ThinkingBlock`s in `request.messages` are silently dropped
on the way out (they have no wire-format home), and the deserialized
response never contains a `ThinkingBlock`.

The provider does, however, support a `reasoning_effort` knob that
influences how much the model thinks internally. `request.thinking` plus
`request.effort` / `request.budget` now flow through to the wire as a
top-level `reasoning_effort` kwarg (with `effort="max"` clamped to
`"xhigh"` and a bare `budget` bucketed into an effort level). ThinkingBlocks
in inputs are still dropped (no wire representation); responses still
never contain ThinkingBlocks (no reasoning content channel).

This is a real limitation: a multi-turn agent run that depends on the
model carrying earlier reasoning forward through tool calls cannot use
this provider for reasoning continuity. Pick the Anthropic or OpenAI
Responses provider for reasoning-heavy tasks.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal, TypedDict

import openai

from willow.tools.base import ToolSpec

from .base import (
    CompletionRequest,
    CompletionResponse,
    ContentBlock,
    ImageBlock,
    Message,
    Provider,
    StopReason,
    StreamComplete,
    StreamEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseDelta,
)

_ERROR_PREFIX = "[error] "

_FINISH_REASON_MAP: dict[str, StopReason] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
}


def _map_effort(
    effort: Literal["low", "medium", "high", "xhigh", "max"],
) -> Literal["low", "medium", "high", "xhigh"]:
    return "xhigh" if effort == "max" else effort


def _budget_to_effort(budget: int) -> Literal["low", "medium", "high", "xhigh"]:
    if budget < 2048:
        return "low"
    if budget < 8192:
        return "medium"
    if budget < 32768:
        return "high"
    return "xhigh"


class _ToolCallAcc(TypedDict):
    """Per-tool-call accumulator for streaming reassembly.

    Chat Completions splits a tool call across many delta chunks indexed by
    `delta.tool_calls[i].index`. We accumulate the id, name, and a running
    `arguments` JSON string until the stream completes; then we json.loads
    the assembled arguments into a `ToolUseBlock.input`.
    """

    id: str
    name: str
    arguments: str


class OpenAICompletionsProvider(Provider):
    """Provider plugin backed by `openai.OpenAI` Chat Completions.

    The client is constructed lazily so tests can inject a mock without
    touching the network or requiring an API key in the environment.
    """

    def __init__(self, client: openai.OpenAI | None = None) -> None:
        self.client = client if client is not None else openai.OpenAI()

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        kwargs = self._build_kwargs(request)
        response = self.client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        content: list[ContentBlock] = []
        if choice.message.content:
            content.append(TextBlock(text=choice.message.content))
        for tool_call in choice.message.tool_calls or []:
            content.append(
                ToolUseBlock(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    input=json.loads(tool_call.function.arguments),
                )
            )

        stop_reason: StopReason = _FINISH_REASON_MAP.get(
            choice.finish_reason, "stop_sequence"
        )
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
        cached_tokens = _cached_tokens_from_usage(response.usage)
        if cached_tokens > 0:
            usage["cached_tokens"] = cached_tokens
        return CompletionResponse(
            content=content,
            stop_reason=stop_reason,
            usage=usage,
        )

    def stream(self, request: CompletionRequest) -> Iterator[StreamEvent]:
        """Stream a turn against Chat Completions.

        Calls ``client.chat.completions.create(stream=True, ...)`` with
        ``stream_options={"include_usage": True}`` so the final chunk
        carries usage tokens (Chat Completions only emits usage on the
        terminal chunk when this option is set).

        Iterating the response yields per-token chunks; each chunk's
        ``choices[0].delta`` is one of:

          * ``delta.content`` text fragment -> ``TextDelta``
          * ``delta.tool_calls[i]`` arrival -> per-tool-call-index tracking:
            id+name arrive once at the start, ``arguments`` arrive in
            fragments. The first sighting of an index emits a start
            ``ToolUseDelta`` (``name`` set, ``partial_json=None``);
            subsequent fragments emit continuation deltas
            (``name=None``, ``partial_json=fragment``).
          * ``finish_reason`` non-None marks the terminal chunk. We then
            assemble the accumulated text + tool calls into a
            ``CompletionResponse`` and yield ``StreamComplete``.
        """
        kwargs = self._build_kwargs(request)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        text_parts: list[str] = []
        # Per-tool-call-index accumulators. Indices in `delta.tool_calls`
        # are stable across chunks for the same tool call; we keep a dict
        # so iteration order is preserved (insertion order) when assembling.
        tool_calls: dict[int, _ToolCallAcc] = {}
        finish_reason: str | None = None
        prompt_tokens: int = 0
        completion_tokens: int = 0
        cached_tokens: int = 0

        for chunk in self.client.chat.completions.create(**kwargs):
            # Usage-only chunks have an empty `choices` list when
            # `include_usage=True`; harvest usage and continue.
            if chunk.usage is not None:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens
                cached_tokens = _cached_tokens_from_usage(chunk.usage)
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if delta.content:
                text_parts.append(delta.content)
                yield TextDelta(text=delta.content)

            for tool_call_delta in delta.tool_calls or []:
                index = tool_call_delta.index
                entry = tool_calls.get(index)
                if entry is None:
                    # First sighting of this tool call: id + name arrive now.
                    name = (
                        tool_call_delta.function.name
                        if tool_call_delta.function is not None
                        else None
                    )
                    entry = _ToolCallAcc(
                        id=tool_call_delta.id or "",
                        name=name or "",
                        arguments="",
                    )
                    tool_calls[index] = entry
                    yield ToolUseDelta(
                        id=entry["id"],
                        name=entry["name"],
                        partial_json=None,
                    )
                else:
                    # Some providers re-send id/name on later chunks; treat
                    # them as authoritative if previously empty, but never
                    # emit another start delta.
                    if not entry["id"] and tool_call_delta.id:
                        entry["id"] = tool_call_delta.id
                    if (
                        not entry["name"]
                        and tool_call_delta.function is not None
                        and tool_call_delta.function.name
                    ):
                        entry["name"] = tool_call_delta.function.name

                fragment = (
                    tool_call_delta.function.arguments
                    if tool_call_delta.function is not None
                    else None
                )
                if fragment:
                    entry["arguments"] += fragment
                    yield ToolUseDelta(
                        id=entry["id"],
                        name=None,
                        partial_json=fragment,
                    )

            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason

        # Assemble the final response.
        content: list[ContentBlock] = []
        text = "".join(text_parts)
        if text:
            content.append(TextBlock(text=text))
        for entry in tool_calls.values():
            content.append(
                ToolUseBlock(
                    id=entry["id"],
                    name=entry["name"],
                    input=json.loads(entry["arguments"]) if entry["arguments"] else {},
                )
            )

        stop_reason: StopReason = _FINISH_REASON_MAP.get(
            finish_reason or "", "stop_sequence"
        )
        usage = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
        }
        if cached_tokens > 0:
            usage["cached_tokens"] = cached_tokens
        yield StreamComplete(
            response=CompletionResponse(
                content=content,
                stop_reason=stop_reason,
                usage=usage,
            )
        )

    def _build_kwargs(self, request: CompletionRequest) -> dict[str, Any]:
        # Any: forwarded as `**kwargs` to `chat.completions.create`, which
        # accepts a heterogeneous keyword set (str model, int max_tokens,
        # list of message dicts, optional tool list, optional stream flag,
        # etc.). The OpenAI SDK's overloaded signature is the source of
        # truth; mirroring it as a TypedDict here would be churn.
        # Chat Completions has no on-wire representation of reasoning, so
        # any ThinkingBlocks the loop has accumulated are dropped before
        # serialization. See module docstring for the rationale.
        # Any: a Chat Completions wire message — a JSON object whose
        # required fields depend on `role` (assistant has optional
        # `tool_calls`, tool results have `tool_call_id`, etc.). The wire
        # spec is the source of truth; encoding the unions in TypedDicts
        # would obscure the translation logic.
        wire_messages: list[dict[str, Any]] = []
        if request.system is not None:
            wire_messages.append({"role": "system", "content": request.system})
        for message in request.messages:
            wire_messages.extend(_message_to_wire(message))

        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": wire_messages,
        }
        if request.tools:
            kwargs["tools"] = [_tool_spec_to_wire(spec) for spec in request.tools]
        if request.thinking:
            if request.effort is not None:
                kwargs["reasoning_effort"] = _map_effort(request.effort)
            elif request.budget is not None:
                kwargs["reasoning_effort"] = _budget_to_effort(request.budget)
        return kwargs


def _cached_tokens_from_usage(usage: Any) -> int:
    details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = getattr(details, "cached_tokens", None)
    return cached_tokens if isinstance(cached_tokens, int) else 0


def _tool_spec_to_wire(spec: ToolSpec) -> dict[str, Any]:
    # Any: the OpenAI Chat Completions tool wire shape is a JSON object whose
    # `parameters` field carries an arbitrary JSON Schema document. The outer
    # envelope is fixed; the schema's value types are not.
    return {
        "type": "function",
        "function": {
            "name": spec["name"],
            "description": spec["description"],
            "parameters": spec["input_schema"],
        },
    }


def _message_to_wire(message: Message) -> list[dict[str, Any]]:
    """Translate one Willow message into zero or more OpenAI wire messages.

    User messages carrying `ToolResultBlock`s expand into one wire message
    per block on the dedicated `"tool"` role. Everything else collapses
    into a single wire message.

    `ThinkingBlock`s are dropped: Chat Completions has no wire shape for
    reasoning content. An assistant turn whose only blocks were thinking
    blocks therefore yields no wire message at all.

    Any: each wire message is a JSON object whose required fields depend on
    role (`role="tool"` carries `tool_call_id`; `role="assistant"` may carry
    `tool_calls` whose entries are themselves nested role-dependent objects).
    The OpenAI wire schema is the source of truth.
    """
    # Strip ThinkingBlocks up front: they have no Chat Completions wire shape.
    blocks = [b for b in message.content if not isinstance(b, ThinkingBlock)]

    if message.role == "user":
        tool_results = [b for b in blocks if isinstance(b, ToolResultBlock)]
        image_blocks = [b for b in blocks if isinstance(b, ImageBlock)]
        text = _concat_text(blocks, separator="\n\n")
        user_content: str | list[dict[str, Any]]
        if image_blocks:
            user_content = []
            if text:
                user_content.append({"type": "text", "text": text})
            user_content.extend(_image_to_wire_part(block) for block in image_blocks)
        else:
            user_content = text
        if tool_results:
            user_wire: list[dict[str, Any]] = [
                _tool_result_to_wire(b) for b in tool_results
            ]
            if text or image_blocks:
                user_wire.append({"role": "user", "content": user_content})
            return user_wire
        return [{"role": "user", "content": user_content}]

    # assistant
    text = _concat_text(blocks)
    tool_uses = [b for b in blocks if isinstance(b, ToolUseBlock)]
    if not text and not tool_uses:
        # An assistant message of only ThinkingBlocks (now stripped) has
        # nothing to put on the wire. Skipping it is the only principled
        # outcome — emitting `{"role": "assistant", "content": None}` would
        # be a malformed wire message.
        return []
    assistant_wire: dict[str, Any] = {
        "role": "assistant",
        "content": text if text else None,
    }
    if tool_uses:
        assistant_wire["tool_calls"] = [
            {
                "id": b.id,
                "type": "function",
                "function": {
                    "name": b.name,
                    "arguments": json.dumps(b.input),
                },
            }
            for b in tool_uses
        ]
    return [assistant_wire]


def _tool_result_to_wire(block: ToolResultBlock) -> dict[str, Any]:
    # Any: the Chat Completions tool-role message is a JSON object; the
    # outer fields are fixed but `content` is a model-visible string.
    content = f"{_ERROR_PREFIX}{block.content}" if block.is_error else block.content
    return {
        "role": "tool",
        "tool_call_id": block.tool_use_id,
        "content": content,
    }


def _image_to_wire_part(block: ImageBlock) -> dict[str, Any]:
    data = base64.b64encode(Path(block.path).read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{block.media_type};base64,{data}"},
    }


def _concat_text(content: Iterable[ContentBlock], *, separator: str = "") -> str:
    return separator.join(b.text for b in content if isinstance(b, TextBlock))
