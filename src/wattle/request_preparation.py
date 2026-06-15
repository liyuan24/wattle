"""Shared provider-request preparation.

This module owns the runtime projection Wattle sends to providers. Durable
session history can stay complete while request preparation estimates size,
compacts when needed, and resets stateful providers before sending a compacted
projection.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Literal

from wattle.compaction import (
    RuntimeCompaction,
    amaybe_compact_messages,
    estimate_request_context_tokens,
)
from wattle.deadline import RunDeadline, append_runtime_deadline_notice, run_deadline_from_env
from wattle.models import (
    context_window_for_model,
    model_supports_modality,
    resolve_max_tokens,
)
from wattle.provider_errors import is_context_length_error
from wattle.providers import (
    CompletionRequest,
    CompletionResponse,
    ImageBlock,
    IncompleteStreamError,
    Message,
    Provider,
    StreamEvent,
    TextBlock,
    TransientProviderError,
)
from wattle.providers.base import stream_idle_timeout_seconds_from_env
from wattle.runtime_context import (
    RuntimeContextProjection,
    append_runtime_context_projection,
    projection_hash,
    render_runtime_context_projection,
)
from wattle.session import message_to_dict

DEFAULT_STREAM_MAX_RETRIES = 3

type RetryableProviderError = IncompleteStreamError | TransientProviderError
type StreamRetryCallback = Callable[[int, int, BaseException, float], None]
type ProviderContextTokens = int | Callable[[], int | None] | None
type RuntimeContextProvider = Callable[[], RuntimeContextProjection | None]
type RuntimeEventCallback = Callable[[dict[str, object]], None]


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
        max_tokens: int | None = None,
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
        provider_context_tokens: ProviderContextTokens = None,
        stream_max_retries: int = DEFAULT_STREAM_MAX_RETRIES,
        stream_idle_timeout_seconds: float | None = None,
        on_stream_retry: StreamRetryCallback | None = None,
        run_deadline: RunDeadline | None = None,
        runtime_context_provider: RuntimeContextProvider | None = None,
        on_runtime_event: RuntimeEventCallback | None = None,
        provider_name: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.system = system
        self.tools = tools
        self.max_tokens = resolve_max_tokens(model, max_tokens)
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
        self.provider_context_tokens = provider_context_tokens
        self.stream_max_retries = max(0, stream_max_retries)
        self.stream_idle_timeout_seconds = (
            stream_idle_timeout_seconds_from_env()
            if stream_idle_timeout_seconds is None
            else max(0.001, stream_idle_timeout_seconds)
        )
        self.on_stream_retry = on_stream_retry
        self.run_deadline = run_deadline if run_deadline is not None else run_deadline_from_env()
        self.runtime_context_provider = runtime_context_provider
        self.on_runtime_event = on_runtime_event
        self.provider_name = provider_name

    def _provider_context_tokens(self, messages: list[Message]) -> int | None:
        history_tokens = _latest_provider_context_tokens(messages)
        if history_tokens is not None:
            return history_tokens
        if callable(self.provider_context_tokens):
            return self.provider_context_tokens()
        return self.provider_context_tokens

    async def aprepare(
        self,
        messages: list[Message],
        *,
        force_compaction: bool = False,
        reset_provider: bool = True,
        compaction_instructions: str | None = None,
    ) -> PreparedRequest:
        previous_state = self.state
        projected_input_messages = project_messages_for_model_modalities(
            messages,
            model=self.model,
        )
        request_system = append_runtime_deadline_notice(self.system, self.run_deadline)
        projection_text: str | None = None
        if self.runtime_context_provider is not None and _runtime_context_enabled():
            projection = self.runtime_context_provider()
            if projection is not None and not projection.is_empty():
                projection_text = render_runtime_context_projection(projection)
                request_system = append_runtime_context_projection(
                    request_system,
                    projection,
                )
                self._emit_runtime_event(
                    {
                        "type": "runtime_context_projection",
                        "source": {},
                        "data": {
                            "rendered": projection_text,
                            "sha256": projection_hash(projection_text),
                            "fact_keys": projection.fact_keys,
                        },
                    }
                )
        raw_context_tokens = estimate_request_context_tokens(
            system=request_system,
            messages=projected_input_messages,
            tools=self.tools,
        )
        request_messages, state = await amaybe_compact_messages(
            provider=self.provider,
            model=self.model,
            system=request_system,
            messages=projected_input_messages,
            tools=self.tools,
            max_tokens=self.max_tokens,
            context_window=self.context_window,
            state=self.state,
            on_start=self.on_compaction_start,
            on_end=self.on_compaction_end,
            force=force_compaction,
            keep_recent_tokens=self.compaction_keep_recent_tokens,
            compaction_instructions=compaction_instructions,
            provider_context_tokens=self._provider_context_tokens(projected_input_messages),
        )
        self.state = state
        compacted = state is not None
        if compacted and reset_provider:
            await self.provider.areset_conversation()
        projected_messages = project_messages_for_model_modalities(
            request_messages,
            model=self.model,
        )
        context_tokens = estimate_request_context_tokens(
            system=request_system,
            messages=projected_messages,
            tools=self.tools,
        )
        if (
            self.on_compaction_record is not None
            and state is not None
            and state != previous_state
        ):
            reason = "overflow" if force_compaction else "threshold"
            self.on_compaction_record(state, reason, raw_context_tokens, context_tokens)
        request_event_data = {
            "provider": self.provider_name or type(self.provider).__name__,
            "model": self.model,
            "message_count": len(projected_messages),
            "system_sha256": _optional_sha256(request_system),
            "runtime_projection_sha256": projection_hash(projection_text),
            "tool_names": [
                str(tool.get("name"))
                for tool in self.tools
                if isinstance(tool, dict) and tool.get("name") is not None
            ],
            "context_tokens": context_tokens,
            "request_bytes": estimate_serialized_request_bytes(
                model=self.model,
                messages=projected_messages,
                max_tokens=self.max_tokens,
                system=request_system,
                tools=self.tools,
                thinking=self.thinking,
                effort=self.effort,
            ),
            "compacted": compacted,
        }
        self._emit_runtime_event(
            {
                "type": "provider_request_prepared",
                "source": {},
                "data": request_event_data,
            }
        )
        return PreparedRequest(
            request=CompletionRequest(
                model=self.model,
                messages=projected_messages,
                max_tokens=self.max_tokens,
                system=request_system,
                tools=self.tools,
                thinking=self.thinking,
                effort=self.effort,
            ),
            context_tokens=context_tokens,
            request_bytes=int(request_event_data["request_bytes"]),
            compacted=compacted,
        )

    def _emit_runtime_event(self, event: dict[str, object]) -> None:
        if self.on_runtime_event is None:
            return
        self.on_runtime_event(event)


def project_messages_for_model_modalities(
    messages: list[Message],
    *,
    model: str,
) -> list[Message]:
    """Return the provider request projection allowed by model input modalities."""

    if model_supports_modality(model, "image"):
        return messages
    return [_replace_images_with_text(message, model=model) for message in messages]


def _latest_provider_context_tokens(messages: list[Message]) -> int | None:
    return next(
        (
            message.input_tokens
            for message in reversed(messages)
            if message.role == "assistant" and message.input_tokens > 0
        ),
        None,
    )


def _replace_images_with_text(message: Message, *, model: str) -> Message:
    content = [
        (
            TextBlock(text=_unsupported_image_text(block, model=model))
            if isinstance(block, ImageBlock)
            else block
        )
        for block in message.content
    ]
    if content == message.content:
        return message
    return Message(
        role=message.role,
        content=content,
        input_tokens=message.input_tokens,
        output_tokens=message.output_tokens,
        cached_tokens=message.cached_tokens,
    )


def _unsupported_image_text(block: ImageBlock, *, model: str) -> str:
    return f"[image omitted: model {model} does not support image inputs]"


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
            return await asyncio.wait_for(
                preparer.provider.acomplete(request),
                timeout=preparer.stream_idle_timeout_seconds,
            )
        except TimeoutError as exc:
            timeout_error = TransientProviderError(
                f"Provider completion was idle for "
                f"{preparer.stream_idle_timeout_seconds:g}s.",
            )
            if failures >= preparer.stream_max_retries:
                raise timeout_error from exc
            delay = _retry_delay_for_error(failures + 1, timeout_error)
            _notify_stream_retry(preparer, failures + 1, timeout_error, delay)
            await asyncio.sleep(delay)
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


def _runtime_context_enabled() -> bool:
    raw = os.environ.get("WATTLE_RUNTIME_CONTEXT_INJECTION")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _optional_sha256(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
