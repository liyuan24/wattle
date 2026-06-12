"""The provider-agnostic agent loop.

Drives the standard tool-using conversation:

    user input
       -> assistant turn (text and/or tool_use blocks)
       -> if tool_use: dispatch all tools, append a single user message
          carrying the ToolResultBlocks, repeat
       -> else: stop

The loop never reaches into a provider SDK. It only ever speaks the types
defined in `wattle.providers.base`.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Callable, Mapping, MutableSequence
from typing import Any, Literal, cast

from .message_history import monitor_event_text_blocks
from .models import model_supports_modality
from .permissions import PermissionGate
from .providers.base import (
    CompletionResponse,
    ContentBlock,
    ImageBlock,
    MalformedToolCallError,
    Message,
    Provider,
    StreamComplete,
    StreamEvent,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from .request_preparation import (
    RequestPreparer,
    acomplete_with_recovery,
    astream_with_recovery,
)
from .tool_events import ToolRunEvent
from .tools.base import Tool

MAX_MALFORMED_TOOL_CALL_REPAIRS = 1


def run(
    provider: Provider,
    tools_by_name: Mapping[str, Tool],
    system: str | None,
    user_input: str,
    model: str,
    max_tokens: int | None = None,
    permission_gate: PermissionGate | None = None,
    context_window: int | None = None,
    thinking: bool = False,
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
    messages_out: MutableSequence[Message] | None = None,
) -> CompletionResponse:
    """Run the agent loop until the model stops.

    Use tool-specific timeouts/close operations for runtime control.

    Multiple `ToolUseBlock`s in a single assistant turn are dispatched in
    emission order, and their results are bundled into one user message of
    `ToolResultBlock`s — this matches Anthropic's expected shape and is the
    natural interleaving for the OpenAI plugins to lift to their wire format.

    Tool exceptions never propagate. They are converted into
    `ToolResultBlock(is_error=True)` carrying the exception type and message,
    so the model can react to the failure on the next turn.
    """
    return asyncio.run(
        arun(
            provider=provider,
            tools_by_name=tools_by_name,
            system=system,
            user_input=user_input,
            model=model,
            max_tokens=max_tokens,
            permission_gate=permission_gate,
            context_window=context_window,
            thinking=thinking,
            effort=effort,
            messages_out=messages_out,
        )
    )


def _tools_for_model(tools_by_name: Mapping[str, Tool], model: str) -> dict[str, Tool]:
    if model_supports_modality(model, "image"):
        return dict(tools_by_name)
    return {name: tool for name, tool in tools_by_name.items() if name != "view_image"}


async def arun(
    provider: Provider,
    tools_by_name: Mapping[str, Tool],
    system: str | None,
    user_input: str,
    model: str,
    max_tokens: int | None = None,
    permission_gate: PermissionGate | None = None,
    context_window: int | None = None,
    thinking: bool = False,
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
    messages_out: MutableSequence[Message] | None = None,
) -> CompletionResponse:
    """Async version of :func:`run`."""

    model_tools_by_name = _tools_for_model(tools_by_name, model)
    tool_specs = [t.spec() for t in model_tools_by_name.values()]
    runtime = _runtime_from_tools(model_tools_by_name)
    _configure_subagent_runtime(
        runtime,
        provider=provider,
        tools_by_name=model_tools_by_name,
        full_tools_by_name=tools_by_name,
        system=system,
        model=model,
        max_tokens=max_tokens,
        permission_gate=permission_gate,
        context_window=context_window,
        thinking=thinking,
        effort=effort,
    )
    messages: list[Message] = [
        Message(role="user", content=[TextBlock(text=user_input)])
    ]
    preparer = RequestPreparer(
        provider=provider,
        model=model,
        system=system,
        tools=tool_specs,
        max_tokens=max_tokens,
        thinking=thinking,
        effort=effort,
        context_window=context_window,
    )

    malformed_tool_call_repairs = 0
    for _ in itertools.count():
        try:
            response = await acomplete_with_recovery(preparer, messages)
        except MalformedToolCallError as exc:
            if malformed_tool_call_repairs >= MAX_MALFORMED_TOOL_CALL_REPAIRS:
                raise
            malformed_tool_call_repairs += 1
            messages.append(_malformed_tool_call_repair_message(exc))
            continue
        messages.append(_assistant_message(response))
        malformed_tool_call_repairs = 0

        if response.stop_reason != "tool_use":
            monitor_blocks = _drain_monitor_event_blocks(runtime)
            if not monitor_blocks:
                if messages_out is not None:
                    messages_out[:] = messages
                return response
            messages.append(Message(role="user", content=monitor_blocks))
            continue

        tool_results: list[ContentBlock] = []
        for block in response.content:
            if not isinstance(block, ToolUseBlock):
                continue
            tool_results.extend(
                await dispatch_tool_blocks_async(block, model_tools_by_name, permission_gate)
            )
        monitor_blocks = _drain_monitor_event_blocks(runtime)

        # An assistant turn with stop_reason="tool_use" but no ToolUseBlocks
        # would be a provider bug; bail rather than spin.
        followup_blocks = [*tool_results, *monitor_blocks]
        if not followup_blocks:
            if messages_out is not None:
                messages_out[:] = messages
            return response

        messages.append(Message(role="user", content=followup_blocks))

    raise RuntimeError("unreachable agent loop exit")


def run_streaming(
    provider: Provider,
    tools_by_name: Mapping[str, Tool],
    system: str | None,
    user_input: str,
    model: str,
    max_tokens: int | None = None,
    on_event: Callable[[StreamEvent], None] = lambda _e: None,
    permission_gate: PermissionGate | None = None,
    context_window: int | None = None,
    thinking: bool = False,
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
) -> CompletionResponse:
    """Run the agent loop using streaming for each model turn.

    Same semantics as `run()`, but each model turn is driven through
    `provider.astream()`. Every `StreamEvent` emitted by the provider is
    forwarded to `on_event` for incremental rendering. The terminal
    `StreamComplete.response` is used for tool dispatch and termination
    decisions, and is what this function ultimately returns.

    Returns the final `CompletionResponse` from the last streamed turn.

    `on_event` is called synchronously, in event-emission order, before the
    loop touches the assembled response. Exceptions from `on_event`
    propagate (the loop does not catch them).
    """
    return asyncio.run(
        arun_streaming(
            provider=provider,
            tools_by_name=tools_by_name,
            system=system,
            user_input=user_input,
            model=model,
            max_tokens=max_tokens,
            on_event=on_event,
            permission_gate=permission_gate,
            context_window=context_window,
            thinking=thinking,
            effort=effort,
        )
    )


async def arun_streaming(
    provider: Provider,
    tools_by_name: Mapping[str, Tool],
    system: str | None,
    user_input: str,
    model: str,
    max_tokens: int | None = None,
    on_event: Callable[[StreamEvent], None] = lambda _e: None,
    permission_gate: PermissionGate | None = None,
    context_window: int | None = None,
    thinking: bool = False,
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
) -> CompletionResponse:
    """Async version of :func:`run_streaming`."""

    model_tools_by_name = _tools_for_model(tools_by_name, model)
    tool_specs = [t.spec() for t in model_tools_by_name.values()]
    runtime = _runtime_from_tools(model_tools_by_name)
    _configure_subagent_runtime(
        runtime,
        provider=provider,
        tools_by_name=model_tools_by_name,
        full_tools_by_name=tools_by_name,
        system=system,
        model=model,
        max_tokens=max_tokens,
        permission_gate=permission_gate,
        context_window=context_window,
        thinking=thinking,
        effort=effort,
    )
    messages: list[Message] = [
        Message(role="user", content=[TextBlock(text=user_input)])
    ]
    preparer = RequestPreparer(
        provider=provider,
        model=model,
        system=system,
        tools=tool_specs,
        max_tokens=max_tokens,
        thinking=thinking,
        effort=effort,
        context_window=context_window,
    )

    malformed_tool_call_repairs = 0
    for _ in itertools.count():
        response = None
        try:
            async for event in astream_with_recovery(preparer, messages):
                on_event(event)
                if isinstance(event, StreamComplete):
                    response = event.response
        except MalformedToolCallError as exc:
            if malformed_tool_call_repairs >= MAX_MALFORMED_TOOL_CALL_REPAIRS:
                raise
            malformed_tool_call_repairs += 1
            messages.append(_malformed_tool_call_repair_message(exc))
            continue
        if response is None:
            raise RuntimeError(
                "Provider stream ended without a StreamComplete event."
            )

        messages.append(_assistant_message(response))
        malformed_tool_call_repairs = 0

        if response.stop_reason != "tool_use":
            monitor_blocks = _drain_monitor_event_blocks(runtime)
            if not monitor_blocks:
                return response
            messages.append(Message(role="user", content=monitor_blocks))
            continue

        tool_results: list[ContentBlock] = []
        for block in response.content:
            if not isinstance(block, ToolUseBlock):
                continue
            tool_results.extend(
                await dispatch_tool_blocks_async(block, model_tools_by_name, permission_gate)
            )
        monitor_blocks = _drain_monitor_event_blocks(runtime)

        followup_blocks = [*tool_results, *monitor_blocks]
        if not followup_blocks:
            return response

        messages.append(Message(role="user", content=followup_blocks))

    raise RuntimeError("unreachable streaming agent loop exit")


async def dispatch_tool_async(
    block: ToolUseBlock,
    tools_by_name: Mapping[str, Tool],
    permission_gate: PermissionGate | None = None,
) -> ToolResultBlock:
    """Async tool dispatch, packaging failures as ToolResultBlocks."""
    blocks = await dispatch_tool_blocks_async(block, tools_by_name, permission_gate)
    return _first_tool_result(block, blocks)


async def dispatch_tool_blocks_async(
    block: ToolUseBlock,
    tools_by_name: Mapping[str, Tool],
    permission_gate: PermissionGate | None = None,
    tool_event_callback: Callable[[ToolRunEvent], None] | None = None,
) -> list[ContentBlock]:
    """Async tool dispatch, allowing tools to return extra content blocks."""

    tool = tools_by_name.get(block.name)
    if tool is None:
        return [_tool_error(block, f"Unknown tool: {block.name!r}")]
    if permission_gate is not None:
        permission = permission_gate.check(block)
        if not permission.allowed:
            return [_tool_error(block, permission.denial or "Tool execution denied.")]
    try:
        if tool_event_callback is not None:
            output = await tool.arun_with_events(
                emit=tool_event_callback,
                tool_use_id=block.id,
                **block.input,
            )
        else:
            arun = getattr(tool, "arun", None)
            if arun is None:
                output = await asyncio.to_thread(tool.run, **block.input)
            else:
                output = await arun(**block.input)
    except Exception as exc:  # noqa: BLE001 — surface anything as a tool error
        return [_tool_error(block, f"{type(exc).__name__}: {exc}")]
    return _normalize_tool_output(block, output)


def dispatch_tool(
    block: ToolUseBlock,
    tools_by_name: Mapping[str, Tool],
    permission_gate: PermissionGate | None = None,
) -> ToolResultBlock:
    """Run one tool call, packaging any failure as a ToolResultBlock."""
    blocks = dispatch_tool_blocks(block, tools_by_name, permission_gate)
    return _first_tool_result(block, blocks)


def dispatch_tool_blocks(
    block: ToolUseBlock,
    tools_by_name: Mapping[str, Tool],
    permission_gate: PermissionGate | None = None,
) -> list[ContentBlock]:
    """Run one tool call, allowing tools to return extra content blocks."""
    tool = tools_by_name.get(block.name)
    if tool is None:
        return [_tool_error(block, f"Unknown tool: {block.name!r}")]
    if permission_gate is not None:
        permission = permission_gate.check(block)
        if not permission.allowed:
            return [_tool_error(block, permission.denial or "Tool execution denied.")]
    try:
        output = tool.run(**block.input)
    except Exception as exc:  # noqa: BLE001 — surface anything as a tool error
        return [_tool_error(block, f"{type(exc).__name__}: {exc}")]
    return _normalize_tool_output(block, output)


def _tool_error(block: ToolUseBlock, content: str) -> ToolResultBlock:
    return ToolResultBlock(tool_use_id=block.id, content=content, is_error=True)


def _first_tool_result(
    block: ToolUseBlock,
    blocks: list[ContentBlock],
) -> ToolResultBlock:
    for result in blocks:
        if isinstance(result, ToolResultBlock):
            return result
    return ToolResultBlock(
        tool_use_id=block.id,
        content=f"Tool returned no textual result: {block.name!r}",
        is_error=True,
    )


def _normalize_tool_output(block: ToolUseBlock, output: object) -> list[ContentBlock]:
    if isinstance(output, ToolResultBlock):
        return [output]
    if isinstance(output, str):
        return [ToolResultBlock(tool_use_id=block.id, content=output)]
    if isinstance(output, ImageBlock) or _is_image_like(output):
        image = _as_image_block(output)
        return [
            ToolResultBlock(tool_use_id=block.id, content=_attached_image_text(image)),
            image,
        ]
    if isinstance(output, list):
        content = [_canonical_content_block(item) for item in output]
        content = [item for item in content if item is not None]
        if not content:
            return [ToolResultBlock(tool_use_id=block.id, content=str(output))]
        if any(isinstance(item, ToolResultBlock) for item in content):
            return content
        images = [item for item in content if isinstance(item, ImageBlock)]
        if images:
            summary = "\n".join(_attached_image_text(image) for image in images)
        else:
            summary = f"Attached {len(content)} content block(s)."
        return [ToolResultBlock(tool_use_id=block.id, content=summary), *content]
    return [ToolResultBlock(tool_use_id=block.id, content=str(output))]


def _attached_image_text(block: ImageBlock) -> str:
    return (
        f"Attached image: {block.filename} "
        f"({block.media_type}, {block.size_bytes} bytes)."
    )


def _canonical_content_block(item: object) -> ContentBlock | None:
    if isinstance(item, (TextBlock, ImageBlock, ToolResultBlock)):
        return item
    if _is_image_like(item):
        return _as_image_block(item)
    return None


def _is_image_like(item: object) -> bool:
    return (
        getattr(item, "type", None) == "image"
        and isinstance(getattr(item, "path", None), str)
        and isinstance(getattr(item, "media_type", None), str)
        and isinstance(getattr(item, "filename", None), str)
        and isinstance(getattr(item, "size_bytes", None), int)
    )


def _as_image_block(item: object) -> ImageBlock:
    if isinstance(item, ImageBlock):
        return item
    image = cast(Any, item)
    return ImageBlock(
        path=str(image.path),
        media_type=str(image.media_type),
        filename=str(image.filename),
        size_bytes=int(image.size_bytes),
    )


def _assistant_message(response: CompletionResponse) -> Message:
    return Message(
        role="assistant",
        content=list(response.content),
        input_tokens=response.usage.get("input_tokens", 0),
        output_tokens=response.usage.get("output_tokens", 0),
        cached_tokens=_cached_tokens_from_usage(response.usage),
    )


def _malformed_tool_call_repair_message(error: MalformedToolCallError) -> Message:
    raw_arguments = _clip_text(error.raw_arguments, max_chars=1200)
    truncated_text = (
        " The provider reported that the response stopped because it hit the output limit."
        if error.was_truncated
        else ""
    )
    return Message(
        role="user",
        content=[
            TextBlock(
                text=(
                    "Your previous tool call could not be executed because its JSON "
                    f"arguments for `{error.tool_name}` were malformed or truncated."
                    f"{truncated_text}\n\n"
                    "Malformed arguments snippet:\n"
                    f"{raw_arguments}\n\n"
                    "Re-issue exactly one tool call now. Keep the JSON arguments "
                    "minimal and valid. Do not include explanatory text."
                )
            )
        ],
    )


def _clip_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n[... {omitted} more characters omitted]"


def _cached_tokens_from_usage(usage: Mapping[str, int]) -> int:
    for key in ("cached_tokens", "cache_read_input_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return 0


def _runtime_from_tools(tools_by_name: Mapping[str, Tool]) -> object | None:
    for tool in tools_by_name.values():
        runtime = getattr(tool, "runtime", None)
        if runtime is not None and hasattr(runtime, "events"):
            return runtime
    return None


def _drain_monitor_event_blocks(runtime: object | None) -> list[ContentBlock]:
    if runtime is None:
        return []
    events = getattr(runtime, "events", None)
    if events is None or not hasattr(events, "drain"):
        return []
    drained = events.drain()
    if not drained:
        return []
    return list(monitor_event_text_blocks(drained))


def _configure_subagent_runtime(
    runtime: object | None,
    *,
    provider: Provider,
    tools_by_name: Mapping[str, Tool],
    full_tools_by_name: Mapping[str, Tool],
    system: str | None,
    model: str,
    max_tokens: int | None,
    permission_gate: PermissionGate | None,
    context_window: int | None,
    thinking: bool,
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None,
) -> None:
    if runtime is None:
        return
    subagents = getattr(runtime, "subagents", None)
    if subagents is None or not hasattr(subagents, "configure"):
        return
    subagents.configure(
        provider=provider,
        tools_by_name=tools_by_name,
        system=system,
        model=model,
        full_tools_by_name=full_tools_by_name,
        max_tokens=max_tokens,
        permission_gate=permission_gate,
        context_window=context_window,
        thinking=thinking,
        effort=effort,
    )
