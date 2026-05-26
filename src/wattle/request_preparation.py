"""Shared provider-request preparation.

This module owns the runtime projection Wattle sends to providers. Durable
session history can stay complete while request preparation estimates size,
compacts when needed, and resets stateful providers before sending a compacted
projection.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Literal

from wattle.compaction import (
    RuntimeCompaction,
    amaybe_compact_messages,
    estimate_request_context_tokens,
)
from wattle.models import context_window_for_model
from wattle.provider_errors import is_context_length_error
from wattle.providers import (
    CompletionRequest,
    CompletionResponse,
    IncompleteStreamError,
    Message,
    Provider,
    StreamEvent,
    TransientProviderError,
)
from wattle.providers.base import stream_idle_timeout_seconds_from_env
from wattle.session import message_to_dict

DEFAULT_STREAM_MAX_RETRIES = 3

type RetryableProviderError = IncompleteStreamError | TransientProviderError
type StreamRetryCallback = Callable[[int, int, BaseException, float], None]


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
        compaction_keep_recent_tokens: int = 20_000,
        stream_max_retries: int = DEFAULT_STREAM_MAX_RETRIES,
        stream_idle_timeout_seconds: float | None = None,
        on_stream_retry: StreamRetryCallback | None = None,
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
        self.compaction_keep_recent_tokens = compaction_keep_recent_tokens
        self.stream_max_retries = max(0, stream_max_retries)
        self.stream_idle_timeout_seconds = (
            stream_idle_timeout_seconds_from_env()
            if stream_idle_timeout_seconds is None
            else max(0.001, stream_idle_timeout_seconds)
        )
        self.on_stream_retry = on_stream_retry

    async def aprepare(
        self,
        messages: list[Message],
        *,
        force_compaction: bool = False,
        reset_provider: bool = True,
        compaction_instructions: str | None = None,
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
            keep_recent_tokens=self.compaction_keep_recent_tokens,
            compaction_instructions=compaction_instructions,
        )
        self.state = state
        compacted = state is not None
        if compacted and reset_provider:
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


async def acomplete_with_recovery(
    preparer: RequestPreparer,
    messages: list[Message],
) -> CompletionResponse:
    prepared = await preparer.aprepare(messages)
    try:
        return await _acomplete_prepared_with_retries(preparer, prepared.request)
    except Exception as exc:
        if not is_context_length_error(exc):
            raise
    prepared = await preparer.aprepare(messages, force_compaction=True)
    return await _acomplete_prepared_with_retries(preparer, prepared.request)


async def astream_with_recovery(
    preparer: RequestPreparer,
    messages: list[Message],
) -> AsyncIterator[StreamEvent]:
    prepared = await preparer.aprepare(messages)
    try:
        async for event in _astream_prepared_with_retries(preparer, prepared.request):
            yield event
        return
    except Exception as exc:
        if not is_context_length_error(exc):
            raise
    prepared = await preparer.aprepare(messages, force_compaction=True)
    async for event in _astream_prepared_with_retries(preparer, prepared.request):
        yield event


async def _acomplete_prepared_with_retries(
    preparer: RequestPreparer,
    request: CompletionRequest,
) -> CompletionResponse:
    for failures in range(preparer.stream_max_retries + 1):
        try:
            return await preparer.provider.acomplete(request)
        except (IncompleteStreamError, TransientProviderError) as exc:
            if failures >= preparer.stream_max_retries:
                raise
            delay = _retry_delay_for_error(failures + 1, exc)
            _notify_stream_retry(preparer, failures + 1, exc, delay)
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


async def _astream_prepared_with_retries(
    preparer: RequestPreparer,
    request: CompletionRequest,
) -> AsyncIterator[StreamEvent]:
    for failures in range(preparer.stream_max_retries + 1):
        try:
            async for event in _iter_stream_with_idle_timeout(
                preparer.provider.astream(request),
                timeout_seconds=preparer.stream_idle_timeout_seconds,
            ):
                yield event
            return
        except (IncompleteStreamError, TransientProviderError) as exc:
            if failures >= preparer.stream_max_retries:
                raise
            delay = _retry_delay_for_error(failures + 1, exc)
            _notify_stream_retry(preparer, failures + 1, exc, delay)
            await asyncio.sleep(delay)


async def _iter_stream_with_idle_timeout(
    events: AsyncIterator[StreamEvent],
    *,
    timeout_seconds: float,
) -> AsyncIterator[StreamEvent]:
    try:
        while True:
            try:
                event = await asyncio.wait_for(
                    anext(events),
                    timeout=timeout_seconds,
                )
            except StopAsyncIteration:
                return
            except TimeoutError as exc:
                raise TransientProviderError(
                    f"Provider stream was idle for {timeout_seconds:g}s.",
                ) from exc
            yield event
    finally:
        aclose = getattr(events, "aclose", None)
        if callable(aclose):
            await aclose()


def _retry_delay_for_error(attempt: int, error: RetryableProviderError) -> float:
    if isinstance(error, TransientProviderError) and error.retry_after is not None:
        return min(30.0, error.retry_after)
    return _stream_retry_delay(attempt)


def _stream_retry_delay(attempt: int) -> float:
    return min(8.0, 0.5 * (2 ** max(0, attempt - 1)))


def _notify_stream_retry(
    preparer: RequestPreparer,
    attempt: int,
    exc: RetryableProviderError,
    delay: float,
) -> None:
    if preparer.on_stream_retry is not None:
        preparer.on_stream_retry(attempt, preparer.stream_max_retries, exc, delay)




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
