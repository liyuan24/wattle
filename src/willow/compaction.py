"""Runtime context compaction for long Willow sessions.

The persisted session history remains the complete transcript. This module
only builds a compacted projection for provider requests.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from willow.providers import (
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
FIRST_MESSAGES_TO_KEEP = 10
LAST_MESSAGES_TO_KEEP = 10
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


def maybe_compact_messages(
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
) -> tuple[list[Message], RuntimeCompaction | None]:
    """Return request messages, compacting in memory if needed."""

    should_start = state is None and _should_start_compaction(
        system=system,
        messages=messages,
        tools=tools,
        max_tokens=max_tokens,
        context_window=context_window,
    )
    should_start = should_start or (force and state is None)
    if (
        not force
        and not should_start
        and state is None
        and len(messages) <= FIRST_MESSAGES_TO_KEEP + LAST_MESSAGES_TO_KEEP
    ):
        return list(messages), None

    first_end, last_start = _compaction_bounds(messages, force=force)
    if last_start <= first_end and should_start:
        first_end, last_start = _compaction_bounds(messages, force=True)
    if last_start <= first_end:
        if state is None:
            return list(messages), None
        return _build_compacted_messages(messages, first_end, last_start, state.summary), state
    should_update = state is not None and (force or state.summarized_until < last_start)

    if not should_start and not should_update:
        if state is None:
            return list(messages), None
        return _build_compacted_messages(messages, first_end, last_start, state.summary), state

    if on_start is not None:
        on_start()
    try:
        provider.reset_conversation()
        if state is None or force:
            summary = _summarize_messages(
                provider=provider,
                model=model,
                messages=messages[first_end:last_start],
                previous_summary=None,
            )
        else:
            summary = _summarize_messages(
                provider=provider,
                model=model,
                messages=messages[state.summarized_until:last_start],
                previous_summary=state.summary,
            )
        provider.reset_conversation()
    finally:
        if on_end is not None:
            on_end()

    next_state = RuntimeCompaction(summary=summary, summarized_until=last_start)
    return _build_compacted_messages(messages, first_end, last_start, summary), next_state


def compacted_message_count() -> int:
    return FIRST_MESSAGES_TO_KEEP + 1 + LAST_MESSAGES_TO_KEEP


def _compaction_bounds(messages: list[Message], *, force: bool) -> tuple[int, int]:
    if not force:
        first_end = _first_keep_end(messages, FIRST_MESSAGES_TO_KEEP)
        last_start = _last_keep_start(messages, first_end, LAST_MESSAGES_TO_KEEP)
        return first_end, last_start

    if len(messages) <= 1:
        return len(messages), len(messages)
    if len(messages) <= 3:
        return 1, len(messages)

    first_end = _first_keep_end(messages, min(2, len(messages)))
    # Recovery compaction should shrink much more aggressively than the
    # proactive path while still preserving the freshest tool/result pair.
    last_keep = min(8, max(1, len(messages) - first_end - 1))
    last_start = _last_keep_start(messages, first_end, last_keep)
    if last_start <= first_end and first_end < len(messages) - 1:
        last_start = first_end + 1
    return first_end, last_start


def _should_start_compaction(
    *,
    system: str | None,
    messages: list[Message],
    tools: list[dict[str, object]],
    max_tokens: int,
    context_window: int | None,
) -> bool:
    if context_window is None or context_window <= 0:
        return False
    estimate = estimate_request_context_tokens(system=system, messages=messages, tools=tools)
    return estimate + max_tokens >= int(context_window * COMPACTION_TRIGGER_RATIO)


def _build_compacted_messages(
    messages: list[Message],
    first_end: int,
    last_start: int,
    summary: str,
) -> list[Message]:
    summary_message = Message(
        role="user",
        content=[
            TextBlock(
                text=(
                    "Context summary of omitted middle messages. "
                    "Use this as prior conversation context, not as a new user request.\n\n"
                    f"{summary}"
                )
            )
        ],
    )
    return [*messages[:first_end], summary_message, *messages[last_start:]]


def _summarize_messages(
    *,
    provider: Provider,
    model: str,
    messages: list[Message],
    previous_summary: str | None,
) -> str:
    if not messages and previous_summary:
        return previous_summary

    prompt = _summary_prompt(messages, previous_summary=previous_summary)
    request = CompletionRequest(
        model=model,
        messages=[Message(role="user", content=[TextBlock(text=prompt)])],
        max_tokens=SUMMARY_MAX_TOKENS,
        system=SUMMARY_SYSTEM_PROMPT,
        tools=[],
    )

    final_text: list[str] = []
    for event in provider.stream(request):
        if isinstance(event, StreamComplete):
            final_text = [
                block.text for block in event.response.content if isinstance(block, TextBlock)
            ]
    summary = "\n".join(final_text).strip()
    if not summary:
        raise RuntimeError("compaction summarization returned no text")
    return summary


def _summary_prompt(messages: list[Message], *, previous_summary: str | None) -> str:
    sections = [
        "Summarize the following middle section of a Willow coding-agent session.",
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


def _first_keep_end(messages: list[Message], desired: int) -> int:
    end = min(desired, len(messages))
    while end < len(messages) and _is_tool_result_message(messages[end]):
        end += 1
    return end


def _last_keep_start(messages: list[Message], first_end: int, desired: int) -> int:
    start = max(first_end, len(messages) - desired)
    if start < len(messages) and _is_tool_result_message(messages[start]):
        tool_ids = _tool_result_ids(messages[start])
        for index in range(start - 1, first_end - 1, -1):
            if _assistant_has_tool_use(messages[index], tool_ids):
                start = index
                break
    return start


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
