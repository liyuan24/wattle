"""OpenAI Codex Responses provider.

This provider targets ChatGPT/Codex OAuth tokens, not OpenAI Platform API
keys. It mirrors the transport shape used by Codex-style clients:

    https://chatgpt.com/backend-api/codex/responses

with the account id extracted from the OAuth JWT and sent in
``chatgpt-account-id``.
"""

from __future__ import annotations

import json
import platform
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any, Literal
from uuid import uuid4

from wattle.auth import _jwt_payload
from wattle.providers.openai_responses import _messages_to_input, _tool_spec_to_api

from .base import (
    CompletionRequest,
    CompletionResponse,
    ContentBlock,
    Provider,
    StopReason,
    StreamComplete,
    StreamEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolUseBlock,
    ToolUseDelta,
)

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api"
JWT_CLAIM_PATH = "https://api.openai.com/auth"
OPENAI_BETA_RESPONSES = "responses=experimental"
DEFAULT_CODEX_INSTRUCTIONS = (
    "You are Wattle, a coding agent working in the user's current repository. "
    "Be concise, inspect files before editing, preserve user changes, and run "
    "relevant checks when feasible."
)

UrlOpen = Callable[[urllib.request.Request], Any]


class OpenAICodexResponsesProvider(Provider):
    """Provider backed by ChatGPT's Codex Responses endpoint."""

    def __init__(
        self,
        bearer_token: str,
        *,
        base_url: str = DEFAULT_CODEX_BASE_URL,
        urlopen: UrlOpen | None = None,
        session_id: str | None = None,
        thread_id: str | None = None,
    ) -> None:
        self.bearer_token = bearer_token
        self.base_url = base_url
        self.urlopen = urlopen or urllib.request.urlopen
        self.session_id = session_id or str(uuid4())
        self.thread_id = thread_id or str(uuid4())

    def fork(self) -> OpenAICodexResponsesProvider:
        return OpenAICodexResponsesProvider(
            bearer_token=self.bearer_token,
            base_url=self.base_url,
            urlopen=self.urlopen,
        )

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        response: CompletionResponse | None = None
        for event in self.stream(request):
            if isinstance(event, StreamComplete):
                response = event.response
        if response is None:
            raise RuntimeError("Codex stream ended without a StreamComplete event.")
        return response

    def stream(self, request: CompletionRequest) -> Iterator[StreamEvent]:
        body = self._build_body(request)
        http_request = urllib.request.Request(
            _codex_responses_url(self.base_url),
            data=json.dumps(body).encode("utf-8"),
            headers=_build_headers(
                self.bearer_token,
                session_id=self.session_id,
                thread_id=self.thread_id,
            ),
            method="POST",
        )

        try:
            with self.urlopen(http_request) as response:
                yield from _events_from_sse(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(_codex_error_message(exc.code, detail)) from exc

    def _build_body(self, request: CompletionRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": request.model,
            "store": False,
            "stream": True,
            "instructions": request.system or DEFAULT_CODEX_INSTRUCTIONS,
            "input": _messages_to_input(request.messages),
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "prompt_cache_key": self.thread_id,
        }
        if request.tools:
            body["tools"] = [_codex_tool_spec_to_api(spec) for spec in request.tools]
        if request.thinking:
            effort: Literal["low", "medium", "high", "xhigh"] | None
            if request.effort is not None:
                effort = "xhigh" if request.effort == "max" else request.effort
            elif request.budget is not None:
                if request.budget < 2048:
                    effort = "low"
                elif request.budget < 8192:
                    effort = "medium"
                elif request.budget < 32768:
                    effort = "high"
                else:
                    effort = "xhigh"
            else:
                effort = None
            if effort is not None:
                body["reasoning"] = {"effort": effort, "summary": "auto"}
        return body


def _codex_tool_spec_to_api(spec: dict[str, Any]) -> dict[str, Any]:
    tool = _tool_spec_to_api(spec)
    # The Codex endpoint accepts the same function tool shape but Pi sends
    # `strict: null`; keep that exact compatibility marker.
    tool["strict"] = None
    return tool


def _codex_responses_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/codex/responses"):
        return normalized
    if normalized.endswith("/codex"):
        return f"{normalized}/responses"
    return f"{normalized}/codex/responses"


def _build_headers(token: str, *, session_id: str, thread_id: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "chatgpt-account-id": _extract_account_id(token),
        "originator": "wattle",
        "User-Agent": _user_agent(),
        "OpenAI-Beta": OPENAI_BETA_RESPONSES,
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "session-id": session_id,
        "thread-id": thread_id,
        "x-client-request-id": thread_id,
    }


def _extract_account_id(token: str) -> str:
    payload = _jwt_payload(token) or {}
    auth_claim = payload.get(JWT_CLAIM_PATH)
    if isinstance(auth_claim, dict):
        account_id = auth_claim.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id:
            return account_id
    raise ValueError("OpenAI Codex OAuth token is missing chatgpt_account_id.")


def _user_agent() -> str:
    return f"wattle ({platform.system().lower()} {platform.release()}; {platform.machine()})"


def _events_from_sse(
    response: Any,
    *,
    on_final_response: Callable[[dict[str, Any]], None] | None = None,
) -> Iterator[StreamEvent]:
    current_call_id: str | None = None
    final_response: dict[str, Any] | None = None
    fallback_blocks: list[ContentBlock] = []
    text_buffer: list[str] = []
    tool_arg_buffers: dict[str, list[str]] = {}
    tool_block_indices: dict[str, int] = {}

    for raw_event in _iter_sse_events(response):
        event = _parse_sse_event(raw_event)
        if event is not None:
            mapped, current_call_id, final_response = _map_event(
                event,
                current_call_id,
                final_response,
            )
            if mapped is not None:
                _capture_streamed_block(
                    mapped,
                    fallback_blocks,
                    text_buffer,
                    tool_arg_buffers,
                    tool_block_indices,
                )
                yield mapped
            if final_response is not None:
                if on_final_response is not None:
                    on_final_response(final_response)
                yield StreamComplete(
                    response=_completion_from_response(
                        final_response,
                        fallback_blocks=_finalize_fallback_blocks(
                            fallback_blocks,
                            text_buffer,
                            tool_arg_buffers,
                            tool_block_indices,
                        ),
                    )
                )
                return

    raise RuntimeError("Codex stream ended without a completion event.")


def _iter_sse_events(response: Any) -> Iterator[str]:
    raw_lines: list[str] = []
    for line in _iter_response_lines(response):
        if not line.strip():
            if raw_lines:
                yield "".join(raw_lines)
                raw_lines = []
            continue
        raw_lines.append(line)
    if raw_lines:
        yield "".join(raw_lines)


def _iter_response_lines(response: Any) -> Iterator[str]:
    readline = getattr(response, "readline", None)
    if callable(readline):
        while True:
            line = readline()
            if not line:
                return
            yield line.decode("utf-8") if isinstance(line, bytes) else str(line)
    else:
        buffer = ""
        while True:
            chunk = response.read(8192)
            if not chunk:
                break
            buffer += chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                yield f"{line}\n"
        if buffer:
            yield buffer


def _parse_sse_event(raw_event: str) -> dict[str, Any] | None:
    data_lines = [
        line.removeprefix("data:").strip()
        for line in raw_event.splitlines()
        if line.startswith("data:")
    ]
    if not data_lines:
        return None
    data = "\n".join(data_lines).strip()
    if not data or data == "[DONE]":
        return None
    parsed = json.loads(data)
    return parsed if isinstance(parsed, dict) else None


def _map_event(
    event: dict[str, Any],
    current_call_id: str | None,
    final_response: dict[str, Any] | None,
) -> tuple[StreamEvent | None, str | None, dict[str, Any] | None]:
    event_type = event.get("type")
    if event_type == "error":
        raise RuntimeError(f"Codex error: {event.get('message') or event}")
    if event_type == "response.failed":
        response = event.get("response")
        message = None
        if isinstance(response, dict):
            error = response.get("error")
            if isinstance(error, dict):
                message = error.get("message")
        raise RuntimeError(str(message or "Codex response failed"))
    if event_type == "response.output_text.delta":
        return TextDelta(text=str(event.get("delta", ""))), current_call_id, final_response
    if event_type in {
        "response.reasoning_summary_text.delta",
        "response.reasoning_text.delta",
    }:
        return ThinkingDelta(thinking=str(event.get("delta", ""))), current_call_id, final_response
    if event_type == "response.output_item.added":
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") == "function_call":
            call_id = str(item.get("call_id", ""))
            return (
                ToolUseDelta(
                    id=call_id,
                    name=str(item.get("name", "")),
                    partial_json=None,
                ),
                call_id,
                final_response,
            )
    if event_type == "response.function_call_arguments.delta":
        if current_call_id is None:
            raise RuntimeError("Codex function-call arguments arrived before a call id.")
        return (
            ToolUseDelta(
                id=current_call_id,
                name=None,
                partial_json=str(event.get("delta", "")),
            ),
            current_call_id,
            final_response,
        )
    if event_type in {"response.done", "response.completed", "response.incomplete"}:
        response = event.get("response")
        if not isinstance(response, dict):
            raise RuntimeError("Codex completion event did not include a response object.")
        return None, current_call_id, response
    return None, current_call_id, final_response


def _capture_streamed_block(
    event: StreamEvent,
    fallback_blocks: list[ContentBlock],
    text_buffer: list[str],
    tool_arg_buffers: dict[str, list[str]],
    tool_block_indices: dict[str, int],
) -> None:
    if isinstance(event, TextDelta):
        text_buffer.append(event.text)
        return
    if not isinstance(event, ToolUseDelta):
        return
    if event.name is not None:
        _flush_text_buffer(fallback_blocks, text_buffer)
        tool_block_indices[event.id] = len(fallback_blocks)
        tool_arg_buffers[event.id] = []
        fallback_blocks.append(ToolUseBlock(id=event.id, name=event.name, input={}))
    elif event.partial_json is not None:
        tool_arg_buffers.setdefault(event.id, []).append(event.partial_json)


def _finalize_fallback_blocks(
    fallback_blocks: list[ContentBlock],
    text_buffer: list[str],
    tool_arg_buffers: dict[str, list[str]],
    tool_block_indices: dict[str, int],
) -> list[ContentBlock]:
    _flush_text_buffer(fallback_blocks, text_buffer)
    finalized = list(fallback_blocks)
    for call_id, index in tool_block_indices.items():
        block = finalized[index]
        if not isinstance(block, ToolUseBlock):
            continue
        raw_args = "".join(tool_arg_buffers.get(call_id, []))
        finalized[index] = ToolUseBlock(
            id=block.id,
            name=block.name,
            input=json.loads(raw_args or "{}"),
        )
    return finalized


def _flush_text_buffer(
    fallback_blocks: list[ContentBlock],
    text_buffer: list[str],
) -> None:
    if not text_buffer:
        return
    fallback_blocks.append(TextBlock(text="".join(text_buffer)))
    text_buffer.clear()


def _completion_from_response(
    response: dict[str, Any],
    *,
    fallback_blocks: list[ContentBlock] | None = None,
) -> CompletionResponse:
    content = _content_from_response(response)
    if fallback_blocks and not any(
        isinstance(block, TextBlock | ToolUseBlock) for block in content
    ):
        content = fallback_blocks
    usage = response.get("usage")
    return CompletionResponse(
        content=content,
        stop_reason=_stop_reason_from_response(response, content),
        usage=_usage_from_response(usage if isinstance(usage, dict) else {}),
    )


def _content_from_response(response: dict[str, Any]) -> list[ContentBlock]:
    content: list[ContentBlock] = []
    output = response.get("output")
    if not isinstance(output, list):
        return content
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message":
            parts = item.get("content")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text = part.get("text")
                    if isinstance(text, str):
                        content.append(TextBlock(text=text))
        elif item_type == "function_call":
            call_id = item.get("call_id")
            name = item.get("name")
            arguments = item.get("arguments", "{}")
            if isinstance(call_id, str) and isinstance(name, str):
                content.append(
                    ToolUseBlock(
                        id=call_id,
                        name=name,
                        input=json.loads(arguments if isinstance(arguments, str) else "{}"),
                    )
                )
        elif item_type == "reasoning":
            summary = item.get("summary") or []
            thinking = ""
            if isinstance(summary, list):
                thinking = "".join(
                    str(part.get("text", ""))
                    for part in summary
                    if isinstance(part, dict) and part.get("type") == "summary_text"
                )
            signature = item.get("id")
            encrypted_content = item.get("encrypted_content")
            content.append(
                ThinkingBlock(
                    thinking=thinking,
                    signature=signature if isinstance(signature, str) else None,
                    encrypted_content=(
                        encrypted_content if isinstance(encrypted_content, str) else None
                    ),
                )
            )
    return content


def _stop_reason_from_response(
    response: dict[str, Any], content: list[ContentBlock]
) -> StopReason:
    if any(isinstance(block, ToolUseBlock) for block in content):
        return "tool_use"
    if response.get("status") == "incomplete":
        details = response.get("incomplete_details")
        if isinstance(details, dict) and details.get("reason") == "max_output_tokens":
            return "max_tokens"
    return "end_turn"


def _usage_from_response(usage: dict[str, Any]) -> dict[str, int]:
    normalized = {
        "input_tokens": _int_usage(usage.get("input_tokens")),
        "output_tokens": _int_usage(usage.get("output_tokens")),
    }
    cached_tokens = _cached_tokens_from_usage(usage)
    if cached_tokens > 0:
        normalized["cached_tokens"] = cached_tokens
    return normalized


def _cached_tokens_from_usage(usage: dict[str, Any]) -> int:
    for details_key in ("input_tokens_details", "prompt_tokens_details"):
        details = usage.get(details_key)
        if isinstance(details, dict):
            cached_tokens = details.get("cached_tokens")
            if isinstance(cached_tokens, int):
                return cached_tokens
    return 0


def _int_usage(value: Any) -> int:
    return value if isinstance(value, int) else 0


def _codex_error_message(status: int, detail: str) -> str:
    try:
        parsed = json.loads(detail)
    except json.JSONDecodeError:
        return f"Codex request failed with HTTP {status}: {detail}"
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str):
                return message
    return f"Codex request failed with HTTP {status}: {detail}"
