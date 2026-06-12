"""Runtime context compaction for long Wattle sessions.

The persisted session history remains the complete transcript. This module
only builds a compacted projection for provider requests.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from wattle.providers import (
    CompletionRequest,
    ImageBlock,
    Message,
    Provider,
    StreamComplete,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)

COMPACTION_TRIGGER_RATIO = 0.80
DEFAULT_KEEP_RECENT_TOKENS = 20_000
SUMMARY_MAX_TOKENS = 2048

SUMMARY_SYSTEM_PROMPT = (
    "You are a context summarization assistant for a coding agent. "
    "Do not continue the conversation. Produce only a concise structured "
    "summary that will let another model continue the work."
)


@dataclass(slots=True)
class RuntimeCompaction:
    """In-memory compaction state for one live session."""

    summary: str
    summarized_until: int
    first_kept_index: int
    read_files: tuple[str, ...] = ()
    modified_files: tuple[str, ...] = ()


async def amaybe_compact_messages(
    *,
    provider: Provider,
    model: str,
    system: str | None,
    messages: list[Message],
    tools: list[dict[str, object]],
    max_tokens: int,
    context_window: int | None,
    state: RuntimeCompaction | None,
    on_start: Callable[[], None] | None = None,
    on_end: Callable[[], None] | None = None,
    force: bool = False,
    keep_recent_tokens: int = DEFAULT_KEEP_RECENT_TOKENS,
    compaction_instructions: str | None = None,
    provider_context_tokens: int | None = None,
) -> tuple[list[Message], RuntimeCompaction | None]:
    """Return request messages, compacting in memory if needed."""

    provider_pressure = _provider_context_pressure_exceeds_compaction_threshold(
        provider_context_tokens=provider_context_tokens,
        context_window=context_window,
    )
    should_start = state is None and (
        _should_start_compaction(
            system=system,
            messages=messages,
            tools=tools,
            context_window=context_window,
        )
        or provider_pressure
    )
    should_start = should_start or (force and state is None)
    if not force and not should_start and state is None:
        return list(messages), None

    if state is not None and not force:
        projected_messages = _build_compacted_messages(
            messages,
            state.first_kept_index,
            state.summary,
        )
        projected_tokens = estimate_request_context_tokens(
            system=system,
            messages=projected_messages,
            tools=tools,
        )
        estimated_pressure = _context_pressure_exceeds_compaction_threshold(
            estimated_tokens=projected_tokens,
            provider_context_tokens=None,
            context_window=context_window,
        )
        if not estimated_pressure and not provider_pressure:
            return projected_messages, state
        if provider_pressure:
            first_end, last_start = _provider_pressure_compaction_bounds(messages)
        else:
            first_end, last_start = _compaction_bounds(
                messages,
                force=force,
                context_window=context_window,
                keep_recent_tokens=keep_recent_tokens,
            )
            if state.summarized_until >= last_start:
                return projected_messages, state

    if state is None or force:
        first_end, last_start = _compaction_bounds(
            messages,
            force=force,
            context_window=context_window,
            keep_recent_tokens=keep_recent_tokens,
        )
    if last_start <= first_end and should_start:
        if provider_pressure and not force:
            first_end, last_start = _provider_pressure_compaction_bounds(messages)
        else:
            first_end, last_start = _compaction_bounds(
                messages,
                force=True,
                context_window=context_window,
                keep_recent_tokens=keep_recent_tokens,
            )
    if last_start <= first_end:
        if state is None:
            return list(messages), None
        return _build_compacted_messages(
            messages,
            state.first_kept_index,
            state.summary,
        ), state

    if on_start is not None:
        on_start()
    try:
        await provider.areset_conversation()
        if state is None or force:
            summary = await _asummarize_messages(
                provider=provider,
                model=model,
                messages=messages[first_end:last_start],
                previous_summary=None,
                compaction_instructions=compaction_instructions,
            )
        else:
            summary = await _asummarize_messages(
                provider=provider,
                model=model,
                messages=messages[state.summarized_until:last_start],
                previous_summary=state.summary,
                compaction_instructions=compaction_instructions,
            )
        await provider.areset_conversation()
    finally:
        if on_end is not None:
            on_end()

    read_files, modified_files = _merge_file_metadata(
        state,
        messages[first_end:last_start],
    )
    next_state = RuntimeCompaction(
        summary=summary,
        summarized_until=last_start,
        first_kept_index=last_start,
        read_files=read_files,
        modified_files=modified_files,
    )
    return _build_compacted_messages(messages, last_start, summary), next_state


def compacted_message_count() -> int:
    return 1


def _provider_pressure_compaction_bounds(messages: list[Message]) -> tuple[int, int]:
    if len(messages) <= 1:
        return len(messages), len(messages)
    return 0, _live_tail_start(messages)


def _live_tail_start(messages: list[Message]) -> int:
    if len(messages) < 2:
        return len(messages)
    previous, latest = messages[-2], messages[-1]
    has_tool_use = any(isinstance(block, ToolUseBlock) for block in previous.content)
    has_tool_result = any(isinstance(block, ToolResultBlock) for block in latest.content)
    if previous.role == "assistant" and latest.role == "user" and has_tool_use and has_tool_result:
        return len(messages) - 2
    return len(messages) - 1


def _compaction_bounds(
    messages: list[Message],
    *,
    force: bool,
    context_window: int | None,
    keep_recent_tokens: int,
) -> tuple[int, int]:
    if not force:
        first_end = 0
        last_start = _last_keep_start_by_tokens(
            messages,
            first_end,
            _effective_keep_recent_tokens(
                context_window=context_window,
                keep_recent_tokens=keep_recent_tokens,
            ),
        )
        return first_end, last_start

    if len(messages) <= 1:
        return len(messages), len(messages)

    first_end = 0
    # Recovery compaction must be able to remove a huge most-recent tool result
    # after a provider context-length error.
    return first_end, len(messages)


def _should_start_compaction(
    *,
    system: str | None,
    messages: list[Message],
    tools: list[dict[str, object]],
    context_window: int | None,
) -> bool:
    estimate = estimate_request_context_tokens(system=system, messages=messages, tools=tools)
    return _context_pressure_exceeds_compaction_threshold(
        estimated_tokens=estimate,
        provider_context_tokens=None,
        context_window=context_window,
    )


def _context_pressure_exceeds_compaction_threshold(
    *,
    estimated_tokens: int,
    provider_context_tokens: int | None,
    context_window: int | None,
) -> bool:
    if context_window is None or context_window <= 0:
        return False
    threshold = int(context_window * COMPACTION_TRIGGER_RATIO)
    pressures = [estimated_tokens]
    if provider_context_tokens is not None and provider_context_tokens > 0:
        pressures.append(provider_context_tokens)
    return max(pressures) >= threshold


def _provider_context_pressure_exceeds_compaction_threshold(
    *,
    provider_context_tokens: int | None,
    context_window: int | None,
) -> bool:
    if provider_context_tokens is None or provider_context_tokens <= 0:
        return False
    return _context_pressure_exceeds_compaction_threshold(
        estimated_tokens=0,
        provider_context_tokens=provider_context_tokens,
        context_window=context_window,
    )


def _build_compacted_messages(
    messages: list[Message],
    first_kept_index: int,
    summary: str,
) -> list[Message]:
    summary_message = Message(
        role="user",
        content=[
            TextBlock(
                text=(
                    "The conversation history before this point was compacted into "
                    "the following summary. "
                    "Use this as prior conversation context, not as a new user request.\n\n"
                    f"{summary}"
                )
            )
        ],
    )
    return [summary_message, *messages[first_kept_index:]]


async def _asummarize_messages(
    *,
    provider: Provider,
    model: str,
    messages: list[Message],
    previous_summary: str | None,
    compaction_instructions: str | None,
) -> str:
    if not messages and previous_summary:
        return previous_summary

    prompt = _summary_prompt(
        messages,
        previous_summary=previous_summary,
        compaction_instructions=compaction_instructions,
    )
    request = CompletionRequest(
        model=model,
        messages=[Message(role="user", content=[TextBlock(text=prompt)])],
        max_tokens=SUMMARY_MAX_TOKENS,
        system=SUMMARY_SYSTEM_PROMPT,
        tools=[],
    )

    final_text: list[str] = []
    async for event in provider.astream(request):
        if isinstance(event, StreamComplete):
            final_text = [
                block.text for block in event.response.content if isinstance(block, TextBlock)
            ]
    summary = "\n".join(final_text).strip()
    if not summary:
        raise RuntimeError("compaction summarization returned no text")
    return summary


def _summary_prompt(
    messages: list[Message],
    *,
    previous_summary: str | None,
    compaction_instructions: str | None,
) -> str:
    sections = [
        "Summarize the following middle section of a Wattle coding-agent session.",
        "Preserve exact file paths, commands, errors, decisions, user constraints, "
        "completed work, current work, and next steps.",
    ]
    if previous_summary:
        sections.append(
            "<previous-summary>\n"
            f"{previous_summary}\n"
            "</previous-summary>\n\n"
            "Update the previous summary with the new middle messages below."
        )
    if compaction_instructions:
        sections.append(
            "<user-compaction-instructions>\n"
            f"{compaction_instructions}\n"
            "</user-compaction-instructions>"
        )
    sections.append(f"<messages>\n{serialize_messages(messages)}\n</messages>")
    sections.append(
        "Use this format:\n"
        "## Goal\n"
        "## Constraints & Preferences\n"
        "## Progress\n"
        "## Key Decisions\n"
        "## Next Steps\n"
        "## Critical Context"
    )
    return "\n\n".join(sections)


def serialize_messages(messages: list[Message]) -> str:
    parts: list[str] = []
    for index, message in enumerate(messages, start=1):
        blocks = "\n".join(_serialize_block(block) for block in message.content)
        parts.append(f"[{index}] {message.role}:\n{blocks}")
    return "\n\n".join(parts)


def _serialize_block(block: object) -> str:
    if isinstance(block, TextBlock):
        return block.text
    if isinstance(block, ImageBlock):
        return (
            f"[image] {block.filename} media_type={block.media_type} "
            f"size_bytes={block.size_bytes}"
        )
    if isinstance(block, ToolUseBlock):
        args = json.dumps(block.input, sort_keys=True, default=str)
        return f"[tool use] id={block.id} name={block.name} input={args}"
    if isinstance(block, ToolResultBlock):
        return (
            f"[tool result] tool_use_id={block.tool_use_id} "
            f"is_error={block.is_error}\n{_truncate_tool_result(block.content)}"
        )
    if isinstance(block, ThinkingBlock):
        return f"[thinking]\n{block.thinking}"
    data = getattr(block, "data", None)
    if isinstance(data, str):
        return f"[redacted thinking]\n{data}"
    return str(block)


def _truncate_tool_result(text: str, max_chars: int = 2000) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n\n[... {len(text) - max_chars} more characters truncated]"


def _effective_keep_recent_tokens(
    *,
    context_window: int | None,
    keep_recent_tokens: int,
) -> int:
    if context_window is None or context_window <= 0:
        return max(1, keep_recent_tokens)
    available = max(1, int(context_window * COMPACTION_TRIGGER_RATIO))
    return max(1, min(keep_recent_tokens, available // 4))


def _last_keep_start_by_tokens(
    messages: list[Message],
    first_end: int,
    keep_recent_tokens: int,
) -> int:
    total = 0
    start = len(messages)
    for index in range(len(messages) - 1, first_end - 1, -1):
        message_tokens = _estimate_message_tokens(messages[index])
        if start < len(messages) and total + message_tokens > keep_recent_tokens:
            break
        total += message_tokens
        start = index
    return _expand_start_to_tool_pair(messages, max(first_end, start), first_end)


def _expand_start_to_tool_pair(
    messages: list[Message],
    start: int,
    first_end: int,
) -> int:
    while start < len(messages) and _is_tool_result_message(messages[start]):
        tool_ids = _tool_result_ids(messages[start])
        previous_start = start
        for index in range(start - 1, first_end - 1, -1):
            if _assistant_has_tool_use(messages[index], tool_ids):
                start = index
                break
        if start == previous_start:
            break
    return start


def _estimate_message_tokens(message: Message) -> int:
    return (
        4
        + _estimate_text_tokens(message.role)
        + sum(_estimate_content_block_tokens(block) for block in message.content)
    )


def _is_tool_result_message(message: Message) -> bool:
    return any(isinstance(block, ToolResultBlock) for block in message.content)


def _tool_result_ids(message: Message) -> set[str]:
    return {
        block.tool_use_id
        for block in message.content
        if isinstance(block, ToolResultBlock)
    }


def _assistant_has_tool_use(message: Message, tool_ids: set[str]) -> bool:
    return message.role == "assistant" and any(
        isinstance(block, ToolUseBlock) and block.id in tool_ids
        for block in message.content
    )


def _merge_file_metadata(
    state: RuntimeCompaction | None,
    messages: list[Message],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    read_files = set(state.read_files if state is not None else ())
    modified_files = set(state.modified_files if state is not None else ())
    for message in messages:
        for block in message.content:
            if not isinstance(block, ToolUseBlock):
                continue
            path = block.input.get("path")
            if not isinstance(path, str) or not path:
                continue
            if block.name in {"read", "view_image"}:
                read_files.add(path)
            elif block.name in {"write", "edit"}:
                modified_files.add(path)
    return tuple(sorted(read_files)), tuple(sorted(modified_files))


def estimate_request_context_tokens(
    *,
    system: str | None,
    messages: list[Message],
    tools: list[dict[str, object]],
) -> int:
    total = _estimate_text_tokens(system or "")
    for message in messages:
        total += 4 + _estimate_text_tokens(message.role)
        total += sum(_estimate_content_block_tokens(block) for block in message.content)
    for tool in tools:
        total += _estimate_text_tokens(json.dumps(tool, sort_keys=True, default=str))
    return total


def _estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _estimate_content_block_tokens(block: object) -> int:
    if isinstance(block, TextBlock):
        return _estimate_text_tokens(block.text)
    if isinstance(block, ImageBlock):
        return 1024 + _estimate_text_tokens(block.filename)
    if isinstance(block, ToolResultBlock):
        return _estimate_text_tokens(block.tool_use_id) + _estimate_text_tokens(block.content)
    if isinstance(block, ToolUseBlock):
        return (
            _estimate_text_tokens(block.id)
            + _estimate_text_tokens(block.name)
            + _estimate_text_tokens(json.dumps(block.input, sort_keys=True, default=str))
        )
    if isinstance(block, ThinkingBlock):
        return (
            _estimate_text_tokens(block.thinking)
            + _estimate_text_tokens(block.signature or "")
            + _estimate_text_tokens(block.encrypted_content or "")
        )
    data = getattr(block, "data", None)
    if isinstance(data, str):
        return _estimate_text_tokens(data)
    return _estimate_text_tokens(str(block))
