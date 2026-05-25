"""OpenAI Responses API provider.

Translates Wattle's normalized `CompletionRequest` / `CompletionResponse`
shape to and from the `openai` SDK's `responses.create` API.

Stateful chaining
-----------------
The Responses API is designed around server-side conversation state. This
provider uses it natively:

  * The first call sends the conversation as `input` items and `store=True`,
    causing the server to persist the response.
  * Subsequent calls pass `previous_response_id=<id>` and send only the
    *delta* — the messages newly appended by the loop since the last call.
    The server has all prior items in its chain, so prior turns (including
    reasoning items) are never resent.

State is per-instance and per-agent-run: ``wattle.agent.run_agent``
constructs a fresh ``OpenAIResponsesProvider`` for each run, so the chain
naturally scopes to one conversation. There is no global state.

Cursor tracking
~~~~~~~~~~~~~~~
``_seen_messages_count`` is the index in ``request.messages`` past which
items still need to be delivered to the server. Building input from
``request.messages[_seen_messages_count:]`` therefore yields exactly the
delta. After a successful call we set
``_seen_messages_count = len(request.messages) + 1`` — the ``+1``
anticipates the loop appending one assistant message (the response we just
returned) before it builds the next ``CompletionRequest``. Whatever the loop
appends after that (typically a single user-tool-result message) is then
the only delta on the next call.

Reasoning items
---------------
Under chaining the server already has all prior reasoning items in its
chain, so the *outgoing* delta never needs to carry reasoning blocks. In
practice the delta only ever consists of `function_call_output` items
(plus, rarely, additional user input), neither of which carries reasoning
— so this falls out of the cursor logic for free. Reasoning items still
appear in *incoming* responses and are translated to ``ThinkingBlock``
entries so the loop's chat history is faithful; they end up at indices
strictly less than ``_seen_messages_count`` after the next call updates
the cursor, so they are naturally excluded from future deltas.

Translation choices that are not forced by the wire format:
  * `ToolResultBlock.is_error=True` becomes a prefix `"[error] "` on the
    `function_call_output.output` string. The Responses API has no
    `is_error` field on `function_call_output`, and the model only ever
    sees `output` as text — a deterministic textual marker is the only
    principled way to surface the failure flag to the next turn.

ThinkingBlock fields used in serialization (when the rare case arises that
a delta does include one — kept correct for completeness):
  * `ThinkingBlock.signature` carries the reasoning item's `id`. A block
    without an `id` cannot be replayed (the API requires one), so it is
    silently dropped on the way out.
  * `ThinkingBlock.encrypted_content`, when present, is replayed as the
    reasoning item's `encrypted_content` field.
  * On the way back, each `reasoning` output item becomes one `ThinkingBlock`
    with `thinking` set to the concatenated `summary_text` parts.

Reasoning knobs (`thinking` / `effort` / `budget`) -> `reasoning={"effort": ...}`
---------------------------------------------------------------------------------
The Responses API's reasoning knob is categorical (`reasoning={"effort":
...}`). When `request.thinking` is `False` no `reasoning` kwarg is emitted.
When `True`, the effort is resolved in this order:

  * `request.effort` set: passed through, with `"max"` clamped to `"xhigh"`.
  * else `request.budget` set: bucketed via:

        budget < 2048   -> "low"
        budget < 8192   -> "medium"
        budget < 32768  -> "high"
        otherwise       -> "xhigh"

  * else: no `reasoning` kwarg is emitted at all (API default applies).
"""

from __future__ import annotations

import base64
import inspect
import json
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

import openai

from wattle.tools.base import ToolSpec

from .base import (
    CompletionRequest,
    CompletionResponse,
    ContentBlock,
    ImageBlock,
    IncompleteStreamError,
    Message,
    Provider,
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
    TransientProviderError,
)

_ERROR_PREFIX = "[error] "

_LOW_THRESHOLD = 2048
_MEDIUM_THRESHOLD = 8192
_HIGH_THRESHOLD = 32768
_RETRYABLE_STATUS_CODES = {408, 500, 502, 503, 504}


def _budget_to_effort(budget: int) -> Literal["low", "medium", "high", "xhigh"]:
    if budget < _LOW_THRESHOLD:
        return "low"
    if budget < _MEDIUM_THRESHOLD:
        return "medium"
    if budget < _HIGH_THRESHOLD:
        return "high"
    return "xhigh"


def _map_effort(
    effort: Literal["low", "medium", "high", "xhigh", "max"],
) -> Literal["low", "medium", "high", "xhigh"]:
    if effort == "max":
        return "xhigh"
    return effort


class OpenAIResponsesProvider(Provider):
    """Provider plugin backed by `openai.AsyncOpenAI` using the Responses API.

    The async client is constructed lazily so tests can inject a mock without
    touching the network or requiring an API key in the environment.

    Stateful chaining is per-instance: ``_previous_response_id`` and
    ``_seen_messages_count`` accumulate over successive ``complete()``
    calls on the same provider object. Construct a new provider to start
    a new chain.
    """

    def __init__(self, async_client: openai.AsyncOpenAI | None = None) -> None:
        self._async_client = async_client
        # The id of the most recent successful response. While `None`, the
        # next call is the chain's first turn and must send the full
        # conversation; once set, subsequent calls send only the delta and
        # pass this id as `previous_response_id`.
        self._previous_response_id: str | None = None
        # Cursor: messages at indices `[0:_seen_messages_count]` have already
        # been delivered to the server (or, for the assistant response from
        # the most recent call, are present in the server-side chain even
        # though the loop has not yet appended them locally — see
        # `complete()` for the `+1` reasoning).
        self._seen_messages_count: int = 0

    @property
    def async_client(self) -> openai.AsyncOpenAI:
        if self._async_client is None:
            self._async_client = openai.AsyncOpenAI()
        return self._async_client

    def fork(self) -> OpenAIResponsesProvider:
        return OpenAIResponsesProvider(async_client=self._async_client)

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        kwargs = self._build_kwargs(request)
        try:
            response = await _maybe_await(self.async_client.responses.create(**kwargs))
        except Exception as exc:
            _raise_transient_openai_error(exc)
            raise
        completion = _completion_from_response(response)
        self._advance_state(response, request)
        return completion

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        """Async-native streaming path backed by ``openai.AsyncOpenAI``."""
        kwargs = self._build_kwargs(request)
        kwargs["stream"] = True

        # `function_call` items each have their own `call_id`; the SDK
        # interleaves arguments-delta events for any active function_call
        # by `output_index` (or `item_id`). The contract says deltas after
        # the start carry the *current* tool's id, so we track the most
        # recently started one. In practice the server emits all argument
        # deltas for a function_call contiguously after its
        # `output_item.added`, so a single cursor is enough.
        current_call_id: str | None = None
        # Any: the SDK exposes a union of Response objects; we capture the
        # one carried on `response.completed` and let the downstream helpers
        # narrow via attribute access. Stubs for these types are gappy.
        final_response: Any = None

        try:
            stream = await _maybe_await(self.async_client.responses.create(**kwargs))
            async for event in _aiter(stream):
                etype = event.type
                if etype == "response.output_text.delta":
                    yield TextDelta(text=event.delta)
                elif etype in (
                    "response.reasoning_summary_text.delta",
                    "response.reasoning_text.delta",
                ):
                    yield ThinkingDelta(thinking=event.delta)
                elif etype == "response.output_item.added":
                    item = event.item
                    if item.type == "function_call":
                        current_call_id = item.call_id
                        yield ToolUseDelta(
                            id=item.call_id,
                            name=item.name,
                            partial_json=None,
                        )
                elif etype == "response.function_call_arguments.delta":
                    # The SDK doesn't put `call_id` directly on this event; we
                    # track it via the most recent function_call start.
                    assert current_call_id is not None
                    yield ToolUseDelta(
                        id=current_call_id,
                        name=None,
                        partial_json=event.delta,
                    )
                elif etype == "response.completed":
                    final_response = event.response
                # Other events (in_progress, content_part_added, item_done,
                # etc.) carry no Wattle-visible signal during streaming.
        except Exception as exc:
            _raise_transient_openai_error(exc)
            raise

        if final_response is None:
            raise IncompleteStreamError(
                "Responses stream ended without a `response.completed` event."
            )

        completion = _completion_from_response(final_response)
        self._advance_state(final_response, request)
        yield StreamComplete(response=completion)

    def _build_kwargs(self, request: CompletionRequest) -> dict[str, Any]:
        # Any: forwarded as `**kwargs` to `responses.create`, which accepts
        # a heterogeneous keyword set (model, max_output_tokens, input items
        # of varying shapes, optional reasoning/tools/instructions/etc.).
        # The OpenAI SDK's overloaded signature is the source of truth.
        # Build only the delta: messages the server hasn't seen yet.
        delta_messages = request.messages[self._seen_messages_count :]

        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_output_tokens": request.max_tokens,
            "input": _messages_to_input(delta_messages),
            # `store=True` is required for `previous_response_id` chaining
            # (and is also the SDK default). Made explicit so the intent is
            # visible in one place.
            "store": True,
        }
        if self._previous_response_id is not None:
            kwargs["previous_response_id"] = self._previous_response_id
        if request.system is not None:
            kwargs["instructions"] = request.system
        if request.tools:
            kwargs["tools"] = [_tool_spec_to_api(spec) for spec in request.tools]
        if request.thinking:
            effort: Literal["low", "medium", "high", "xhigh"] | None
            if request.effort is not None:
                effort = _map_effort(request.effort)
            elif request.budget is not None:
                effort = _budget_to_effort(request.budget)
            else:
                effort = None
            if effort is not None:
                kwargs["reasoning"] = {"effort": effort}
        return kwargs

    # Any: SDK Response object. Its full type is a vendor-private union
    # whose stubs are gappy; we only read `.id`.
    def _advance_state(self, response: Any, request: CompletionRequest) -> None:
        # Advance state only after a successful call, so a thrown SDK error
        # leaves the provider in a usable, replayable state.
        self._previous_response_id = response.id
        # `+1` accounts for the assistant message the loop will append next
        # — carrying THIS response's content. The server already has it in
        # its chain (because we just stored this response), so we must not
        # resend it as part of the next delta. By advancing past it here,
        # `request.messages[self._seen_messages_count:]` on the next call
        # naturally yields just whatever the loop appends after the
        # assistant message (typically the user tool_result message).
        self._seen_messages_count = len(request.messages) + 1

    async def areset_conversation(self) -> None:
        self._previous_response_id = None
        self._seen_messages_count = 0


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _aiter(iterable: Any) -> AsyncIterator[Any]:
    if hasattr(iterable, "__aiter__"):
        async for item in iterable:
            yield item
        return
    for item in iterable:
        yield item


def _raise_transient_openai_error(exc: BaseException) -> None:
    status_code = _status_code_from_exception(exc)
    error_payload = _openai_error_payload(exc)
    if error_payload is not None:
        _raise_for_openai_error_payload(
            error_payload,
            status_code=status_code,
            retry_after=_retry_after_from_exception(exc),
            fallback_message=str(exc) or type(exc).__name__,
        )

    if status_code in _RETRYABLE_STATUS_CODES:
        raise TransientProviderError(
            str(exc) or type(exc).__name__,
            status_code=status_code,
            retry_after=_retry_after_from_exception(exc),
        ) from exc

    retryable_types = tuple(
        typ
        for typ in (
            getattr(openai, "APIConnectionError", None),
            getattr(openai, "APITimeoutError", None),
            getattr(openai, "InternalServerError", None),
        )
        if isinstance(typ, type)
    )
    if retryable_types and isinstance(exc, retryable_types):
        raise TransientProviderError(
            str(exc) or type(exc).__name__,
            status_code=status_code,
            retry_after=_retry_after_from_exception(exc),
        ) from exc


def _openai_error_payload(exc: BaseException) -> dict[str, Any] | None:
    for value in (
        getattr(exc, "body", None),
        getattr(exc, "response", None),
    ):
        payload = _error_payload_from_value(value)
        if payload is not None:
            return payload
    return None


def _error_payload_from_value(value: object) -> dict[str, Any] | None:
    if isinstance(value, dict):
        error = value.get("error")
        if isinstance(error, dict):
            return error
        if any(isinstance(value.get(key), str) for key in ("code", "type", "message")):
            return value
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return _error_payload_from_text(text)
    content = getattr(value, "content", None)
    if isinstance(content, (str, bytes)):
        return _error_payload_from_text(
            content.decode("utf-8", errors="replace") if isinstance(content, bytes) else content
        )
    return None


def _error_payload_from_text(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return _error_payload_from_value(parsed)


def _raise_for_openai_error_payload(
    error: dict[str, Any],
    *,
    status_code: int | None,
    retry_after: float | None,
    fallback_message: str,
) -> None:
    code = _openai_error_code(error)
    message = _openai_error_message(error) or fallback_message
    retry_after = retry_after if retry_after is not None else _retry_after_from_message(message)
    if code == "context_length_exceeded":
        raise RuntimeError(f"context_length_exceeded: {message}")
    if code in {"server_is_overloaded", "slow_down", "rate_limit_exceeded"}:
        raise TransientProviderError(
            message,
            status_code=status_code,
            retry_after=retry_after,
        )
    if code in {"insufficient_quota", "usage_limit_reached"}:
        raise RuntimeError(message or "Quota exceeded. Check your plan and billing details.")
    if code == "usage_not_included":
        raise RuntimeError(message or "Usage is not included for this plan.")
    if code == "cyber_policy":
        raise RuntimeError(
            message or "This request has been flagged for possible cybersecurity risk."
        )
    if code == "invalid_prompt":
        raise RuntimeError(message or "Invalid request.")


def _openai_error_code(error: dict[str, Any]) -> str | None:
    for key in ("code", "type"):
        value = error.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _openai_error_message(error: dict[str, Any]) -> str | None:
    value = error.get("message")
    return value if isinstance(value, str) and value else None


def _retry_after_from_message(message: str | None) -> float | None:
    if not message:
        return None
    match = re.search(
        r"(?i)try again in\s*(\d+(?:\.\d+)?)\s*(ms|s|seconds?)",
        message,
    )
    if match is None:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "ms":
        return value / 1000
    return value


def _status_code_from_exception(exc: BaseException) -> int | None:
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _retry_after_from_exception(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    get = getattr(headers, "get", None)
    if not callable(get):
        return None
    value = get("retry-after") or get("Retry-After")
    if value is None:
        return None
    try:
        retry_after = float(value)
    except (TypeError, ValueError):
        return None
    return retry_after if retry_after >= 0 else None


# Any: SDK Response object — we read `.usage.input_tokens`, `.output`,
# `.status`. The Responses SDK exposes a union of vendor-private types here
# and the stubs do not narrow cleanly, so the boundary stays Any.
def _completion_from_response(response: Any) -> CompletionResponse:
    content = _content_from_response(response)
    stop_reason = _stop_reason_from_response(response, content)
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    cached_tokens = _cached_tokens_from_usage(response.usage)
    if cached_tokens > 0:
        usage["cached_tokens"] = cached_tokens
    return CompletionResponse(
        content=content,
        stop_reason=stop_reason,
        usage=usage,
    )


def _cached_tokens_from_usage(usage: Any) -> int:
    details = getattr(usage, "input_tokens_details", None)
    cached_tokens = getattr(details, "cached_tokens", None)
    return cached_tokens if isinstance(cached_tokens, int) else 0


# ---------------------------------------------------------------------------
# Request translation
# ---------------------------------------------------------------------------


def _tool_spec_to_api(spec: ToolSpec) -> dict[str, Any]:
    # Any: the Responses API tool wire shape carries an arbitrary JSON
    # Schema document under `parameters`. The outer envelope is fixed.
    return {
        "type": "function",
        "name": spec["name"],
        "description": spec["description"],
        "parameters": spec["input_schema"],
    }


def _messages_to_input(messages: list[Message]) -> list[dict[str, Any]]:
    """Lift Wattle messages into a flat list of Responses API input items.

    Each ToolUseBlock, ToolResultBlock, and round-trippable ThinkingBlock
    becomes its own top-level item, preserving emission order with
    surrounding text blocks. ThinkingBlocks with no `signature` (no `id`)
    cannot be replayed — the API requires the id — and are dropped.

    Any: each input item is a JSON object whose required keys depend on
    `type` (function_call carries `call_id`+`arguments`; reasoning carries
    `id`+`summary`; message carries role-tagged `content` parts). The
    Responses API wire spec is the source of truth.
    """
    items: list[dict[str, Any]] = []
    for message in messages:
        for block in message.content:
            item = _block_to_input_item(message.role, block)
            if item is not None:
                items.append(item)
    return items


# Any: same Responses API wire-shape rationale as `_messages_to_input`.
def _block_to_input_item(role: str, block: ContentBlock) -> dict[str, Any] | None:
    if isinstance(block, TextBlock):
        if role == "assistant":
            return {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": block.text}],
            }
        return {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": block.text}],
        }
    if isinstance(block, ImageBlock):
        if role != "user":
            return None
        return {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": _image_data_url(block),
                }
            ],
        }
    if isinstance(block, ToolUseBlock):
        return {
            "type": "function_call",
            "call_id": block.id,
            "name": block.name,
            "arguments": json.dumps(block.input),
        }
    if isinstance(block, ToolResultBlock):
        output = _ERROR_PREFIX + block.content if block.is_error else block.content
        return {
            "type": "function_call_output",
            "call_id": block.tool_use_id,
            "output": output,
        }
    if isinstance(block, ThinkingBlock):
        if block.signature is None:
            return None
        item: dict[str, Any] = {
            "type": "reasoning",
            "id": block.signature,
            "summary": (
                [{"type": "summary_text", "text": block.thinking}] if block.thinking else []
            ),
        }
        if block.encrypted_content is not None:
            item["encrypted_content"] = block.encrypted_content
        return item
    raise TypeError(f"Unknown content block type: {type(block).__name__}")


def _image_data_url(block: ImageBlock) -> str:
    data = base64.b64encode(Path(block.path).read_bytes()).decode("ascii")
    return f"data:{block.media_type};base64,{data}"


# ---------------------------------------------------------------------------
# Response translation
# ---------------------------------------------------------------------------


# Any: SDK Response object — see `_completion_from_response`.
def _content_from_response(response: Any) -> list[ContentBlock]:
    content: list[ContentBlock] = []
    for item in response.output:
        if item.type == "message":
            for part in item.content:
                if part.type == "output_text":
                    content.append(TextBlock(text=part.text))
        elif item.type == "function_call":
            content.append(
                ToolUseBlock(
                    id=item.call_id,
                    name=item.name,
                    input=json.loads(item.arguments),
                )
            )
        elif item.type == "reasoning":
            summary_parts = getattr(item, "summary", None) or []
            thinking = "".join(part.text for part in summary_parts if part.type == "summary_text")
            content.append(
                ThinkingBlock(
                    thinking=thinking,
                    signature=item.id,
                    encrypted_content=getattr(item, "encrypted_content", None),
                )
            )
    return content


# Any: SDK Response object — we only read `.status` and `.incomplete_details`.
def _stop_reason_from_response(response: Any, content: list[ContentBlock]) -> StopReason:
    if any(isinstance(b, ToolUseBlock) for b in content):
        return "tool_use"
    if response.status == "incomplete":
        details = response.incomplete_details
        if details is not None and details.reason == "max_output_tokens":
            return "max_tokens"
        reason = getattr(details, "reason", None) or "unknown"
        raise IncompleteStreamError(f"Incomplete Responses response returned, reason: {reason}")
    return "end_turn"
