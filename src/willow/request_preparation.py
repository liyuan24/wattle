"""Shared provider-request preparation.

This module owns the runtime projection Willow sends to providers. Durable
session history can stay complete while request preparation estimates size,
compacts when needed, and resets stateful providers before sending a compacted
projection.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from typing import Literal

from willow.compaction import (
    RuntimeCompaction,
    amaybe_compact_messages,
    estimate_request_context_tokens,
    maybe_compact_messages,
)
from willow.provider_errors import is_context_length_error
from willow.providers import CompletionRequest, CompletionResponse, Message, Provider, StreamEvent
from willow.session import message_to_dict

MODEL_CONTEXT_TOKENS: dict[str, int] = {
    "gpt-5.5": 1_050_000,
    "gpt-5.4": 1_050_000,
    "gpt-5.4-mini": 400_000,
    "gpt-5.3-codex": 400_000,
    "gpt-5.3-codex-spark": 400_000,
    "gpt-5.2": 400_000,
    "claude-sonnet-4-6": 200_000,
    "claude-opus-4-6": 200_000,
    "claude-haiku-4-6": 200_000,
    "deepseek-v4-flash": 1_000_000,
    "deepseek-v4-pro": 1_000_000,
    "kimi-k2.6": 256_000,
    "kimi-k2.5": 256_000,
    "MiniMax-M2.7": 204_800,
    "MiniMax-M2.7-highspeed": 204_800,
}


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    request: CompletionRequest
    context_tokens: int
    request_bytes: int
    compacted: bool


class RequestPreparer:
    """Build provider requests from complete session history."""

    def __init__(
        self,
        *,
        provider: Provider,
        model: str,
        system: str | None,
        tools: list[dict[str, object]],
        max_tokens: int,
        thinking: bool = False,
        effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
        context_window: int | None = None,
        state: RuntimeCompaction | None = None,
        on_compaction_start: Callable[[], None] | None = None,
        on_compaction_end: Callable[[], None] | None = None,
        on_compaction_record: (
            Callable[[RuntimeCompaction, str, int, int], None] | None
        ) = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.system = system
        self.tools = tools
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.effort = effort
        self.context_window = (
            context_window_for_model(model) if context_window is None else context_window
        )
        self.state = state
        self.on_compaction_start = on_compaction_start
        self.on_compaction_end = on_compaction_end
        self.on_compaction_record = on_compaction_record

    def prepare(
        self,
        messages: list[Message],
        *,
        force_compaction: bool = False,
        reset_provider: bool = True,
    ) -> PreparedRequest:
        previous_state = self.state
        raw_context_tokens = estimate_request_context_tokens(
            system=self.system,
            messages=messages,
            tools=self.tools,
        )
        request_messages, state = maybe_compact_messages(
            provider=self.provider,
            model=self.model,
            system=self.system,
            messages=messages,
            tools=self.tools,
            max_tokens=self.max_tokens,
            context_window=self.context_window,
            state=self.state,
            on_start=self.on_compaction_start,
            on_end=self.on_compaction_end,
            force=force_compaction,
        )
        self.state = state
        compacted = state is not None
        if compacted and reset_provider:
            self.provider.reset_conversation()
        context_tokens = estimate_request_context_tokens(
            system=self.system,
            messages=request_messages,
            tools=self.tools,
        )
        if (
            self.on_compaction_record is not None
            and state is not None
            and state != previous_state
        ):
            reason = "overflow" if force_compaction else "threshold"
            self.on_compaction_record(state, reason, raw_context_tokens, context_tokens)
        return PreparedRequest(
            request=CompletionRequest(
                model=self.model,
                messages=request_messages,
                max_tokens=self.max_tokens,
                system=self.system,
                tools=self.tools,
                thinking=self.thinking,
                effort=self.effort,
            ),
            context_tokens=context_tokens,
            request_bytes=estimate_serialized_request_bytes(
                model=self.model,
                messages=request_messages,
                max_tokens=self.max_tokens,
                system=self.system,
                tools=self.tools,
                thinking=self.thinking,
                effort=self.effort,
            ),
            compacted=compacted,
        )

    async def aprepare(
        self,
        messages: list[Message],
        *,
        force_compaction: bool = False,
    ) -> PreparedRequest:
        previous_state = self.state
        raw_context_tokens = estimate_request_context_tokens(
            system=self.system,
            messages=messages,
            tools=self.tools,
        )
        request_messages, state = await amaybe_compact_messages(
            provider=self.provider,
            model=self.model,
            system=self.system,
            messages=messages,
            tools=self.tools,
            max_tokens=self.max_tokens,
            context_window=self.context_window,
            state=self.state,
            on_start=self.on_compaction_start,
            on_end=self.on_compaction_end,
            force=force_compaction,
        )
        self.state = state
        compacted = state is not None
        if compacted:
            await self.provider.areset_conversation()
        context_tokens = estimate_request_context_tokens(
            system=self.system,
            messages=request_messages,
            tools=self.tools,
        )
        if (
            self.on_compaction_record is not None
            and state is not None
            and state != previous_state
        ):
            reason = "overflow" if force_compaction else "threshold"
            self.on_compaction_record(state, reason, raw_context_tokens, context_tokens)
        return PreparedRequest(
            request=CompletionRequest(
                model=self.model,
                messages=request_messages,
                max_tokens=self.max_tokens,
                system=self.system,
                tools=self.tools,
                thinking=self.thinking,
                effort=self.effort,
            ),
            context_tokens=context_tokens,
            request_bytes=estimate_serialized_request_bytes(
                model=self.model,
                messages=request_messages,
                max_tokens=self.max_tokens,
                system=self.system,
                tools=self.tools,
                thinking=self.thinking,
                effort=self.effort,
            ),
            compacted=compacted,
        )


def complete_with_recovery(
    preparer: RequestPreparer,
    messages: list[Message],
) -> CompletionResponse:
    prepared = preparer.prepare(messages)
    try:
        return preparer.provider.complete(prepared.request)
    except Exception as exc:
        if not is_context_length_error(exc):
            raise
    prepared = preparer.prepare(messages, force_compaction=True)
    return preparer.provider.complete(prepared.request)


async def acomplete_with_recovery(
    preparer: RequestPreparer,
    messages: list[Message],
) -> CompletionResponse:
    prepared = await preparer.aprepare(messages)
    try:
        return await preparer.provider.acomplete(prepared.request)
    except Exception as exc:
        if not is_context_length_error(exc):
            raise
    prepared = await preparer.aprepare(messages, force_compaction=True)
    return await preparer.provider.acomplete(prepared.request)


def stream_with_recovery(
    preparer: RequestPreparer,
    messages: list[Message],
) -> Iterator[StreamEvent]:
    prepared = preparer.prepare(messages)
    try:
        yield from preparer.provider.stream(prepared.request)
        return
    except Exception as exc:
        if not is_context_length_error(exc):
            raise
    prepared = preparer.prepare(messages, force_compaction=True)
    yield from preparer.provider.stream(prepared.request)


async def astream_with_recovery(
    preparer: RequestPreparer,
    messages: list[Message],
) -> AsyncIterator[StreamEvent]:
    prepared = await preparer.aprepare(messages)
    try:
        async for event in preparer.provider.astream(prepared.request):
            yield event
        return
    except Exception as exc:
        if not is_context_length_error(exc):
            raise
    prepared = await preparer.aprepare(messages, force_compaction=True)
    async for event in preparer.provider.astream(prepared.request):
        yield event


def context_window_for_model(model: str) -> int | None:
    if model in MODEL_CONTEXT_TOKENS:
        return MODEL_CONTEXT_TOKENS[model]
    if model.startswith("gpt-"):
        return 400_000
    if model.startswith("claude-"):
        return 200_000
    if model.startswith("deepseek-"):
        return 1_000_000
    if model.startswith("kimi-"):
        return 256_000
    if model.startswith("MiniMax-"):
        return 204_800
    return None


def estimate_serialized_request_bytes(
    *,
    model: str,
    messages: list[Message],
    max_tokens: int,
    system: str | None,
    tools: list[dict[str, object]],
    thinking: bool = False,
    effort: str | None = None,
) -> int:
    payload = {
        "model": model,
        "messages": [message_to_dict(message) for message in messages],
        "max_tokens": max_tokens,
        "system": system,
        "tools": tools,
        "thinking": thinking,
        "effort": effort,
    }
    return len(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
