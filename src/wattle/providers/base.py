"""Provider abstraction for Wattle.

Three target APIs (Anthropic Messages, OpenAI Chat Completions, OpenAI Responses)
are normalized to one shape. The loop only ever speaks this shape; provider
plugins translate at the boundary in their `complete()` implementation.

Design notes:
  * Content blocks are explicit dataclasses, never untyped dicts. The
    Chat Completions style of "assistant.content + assistant.tool_calls as
    parallel fields" is flattened into a single ordered list of blocks per
    message — that is the lowest-common-denominator shape and matches both
    Anthropic and the Responses API natively.
  * Tool results live in a `Message(role="user", content=[ToolResultBlock, ...])`.
    This matches Anthropic exactly and is lifted to OpenAI's `tool` role
    messages by the OpenAI plugin (one `tool` message per ToolResultBlock).
  * `system` is a single string. Anthropic puts it in the top-level `system`
    parameter; OpenAI plugins prepend a `{"role": "system"}` message.
  * `tools` is a list of `Tool.spec()` dicts (`{name, description, input_schema}`),
    matching the existing `wattle.tools.Tool` ABC. Provider plugins reshape into
    their own tool-declaration format.
  * `StopReason` is the Anthropic vocabulary; OpenAI plugins normalize their
    `finish_reason` into it (`stop` -> `end_turn`, `tool_calls` -> `tool_use`,
    `length` -> `max_tokens`).
  * `usage` is a free-form `dict[str, int]` of token counts. Recommended keys
    are `input_tokens`, `output_tokens`, and normalized `cached_tokens`, but
    providers may add their own cache-specific fields too.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Self

from wattle.tools.base import ToolSpec

# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextBlock:
    """A run of plain text emitted by the assistant or supplied by the user."""

    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True, slots=True)
class ImageBlock:
    """A local image attachment supplied by the user.

    ``path`` points at a readable local file, usually copied into Wattle's
    session assets directory. Provider adapters read and encode the file at
    request time so session records stay inspectable instead of embedding
    base64 payloads inline.
    """

    path: str
    media_type: str
    filename: str
    size_bytes: int
    type: Literal["image"] = "image"


@dataclass(frozen=True, slots=True)
class ToolUseBlock:
    """An assistant request to invoke a tool.

    `id` is the provider-issued correlation id; the matching ToolResultBlock
    must carry the same value in `tool_use_id`.
    """

    id: str
    name: str
    # Any: model-emitted JSON object (the tool's arguments). The shape is
    # whatever the tool's `input_schema` declares — only known at runtime.
    input: dict[str, Any]
    type: Literal["tool_use"] = "tool_use"


@dataclass(frozen=True, slots=True)
class ToolResultBlock:
    """The outcome of a tool invocation, returned to the assistant on the next turn.

    `content` is the tool's stringified result. The Wattle `Tool.run()` ABC
    returns `str`, so this stays `str` end-to-end. Errors during dispatch are
    reported by setting `is_error=True` and putting the error text in `content`.
    """

    tool_use_id: str
    content: str
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


@dataclass(frozen=True, slots=True)
class ThinkingBlock:
    """A block of model-internal reasoning that callers may need to round-trip.

    Different providers expose reasoning very differently and the abstraction
    only normalizes the parts that the loop and serializers actually need:

      * `thinking`  - the human-visible reasoning text. Anthropic emits this
        directly on its `thinking` blocks; the OpenAI Responses provider fills
        it from the concatenated `summary_text` parts of a reasoning item.
      * `signature` - the opaque round-trip key. For Anthropic this is the
        block's `signature` field (required to replay the block on the next
        request when extended thinking + tool use are combined). For OpenAI
        Responses this is the reasoning item's `id` (required to replay the
        item in `input` when running stateless with tool use).
      * `encrypted_content` - opaque ciphertext that some providers attach to
        a reasoning item (currently OpenAI Responses' `encrypted_content`).
        Modeled as a first-class optional field rather than smuggled inside
        `signature` so the round-trip is honest about what is being carried.

    A `signature` of `None` means the block is not round-trippable; provider
    plugins drop such blocks on the way out instead of producing an invalid
    request.
    """

    thinking: str
    signature: str | None = None
    encrypted_content: str | None = None
    type: Literal["thinking"] = "thinking"


@dataclass(frozen=True, slots=True)
class RedactedThinkingBlock:
    """A reasoning segment that the provider has redacted/encrypted in full.

    Anthropic's `redacted_thinking` blocks expose only an opaque `data`
    payload (no `thinking` text, no `signature`). They must still be replayed
    verbatim on the next turn or the API rejects the call.

    A separate type (rather than overloading `ThinkingBlock`) keeps the type
    system honest: redacted blocks have a different field set and a different
    wire shape, and treating them as a distinct dataclass avoids encoding
    metadata into a string field.
    """

    data: str
    type: Literal["redacted_thinking"] = "redacted_thinking"


ContentBlock = (
    TextBlock
    | ImageBlock
    | ToolUseBlock
    | ToolResultBlock
    | ThinkingBlock
    | RedactedThinkingBlock
)


# ---------------------------------------------------------------------------
# Messages and request/response envelopes
# ---------------------------------------------------------------------------


Role = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    """One turn in the conversation.

    Assistant messages may contain TextBlock and ToolUseBlock entries.
    User messages may contain TextBlock, ImageBlock, and ToolResultBlock entries.
    The loop never produces a raw string — content is always a list of blocks.

    Content is a list because a single turn can carry multiple blocks: an
    assistant turn may interleave prose with several parallel tool calls, and
    a user turn must return one ToolResultBlock per preceding tool call (plus
    any new text input).
    """

    role: Role
    content: list[ContentBlock]
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0


StopReason = Literal["end_turn", "tool_use", "max_tokens", "stop_sequence"]


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    """Inputs to a single provider call.

    `tools` is a list of `Tool.spec()` dicts; the provider plugin reshapes
    them into its own tool-declaration format.

    Reasoning is configured by three independent knobs:

      * `thinking: bool` — master switch. `False` (default) requests no
        extended reasoning; `True` enables it.
      * `effort: Literal["low","medium","high","xhigh","max"] | None` —
        soft guidance on how much reasoning to do. `None` lets the provider
        use its default. `"max"` is Anthropic-only; OpenAI providers clamp
        it to `"xhigh"`.
      * `budget: int | None` — explicit reasoning-token budget. When set
        with `thinking=True`, runs in manual mode on Anthropic
        (`thinking={"type": "enabled", "budget_tokens": N}`) and shadows
        `effort`. On OpenAI providers a budget without an explicit `effort`
        is bucketed into an effort level.

    Per-provider mapping when `thinking=True`:

      * Anthropic Messages:
          - `budget` set → `thinking={"type":"enabled","budget_tokens":N}`
            (manual mode; `effort` ignored).
          - `effort` set, no `budget` → `thinking={"type":"adaptive"}` plus
            top-level `output_config={"effort": effort}`.
          - neither → `thinking={"type":"adaptive"}` (model picks default).
      * OpenAI Responses: emits `reasoning={...}`. `effort` is mapped
        (`"max"`→`"xhigh"`); a `budget` without `effort` is bucketed; with
        neither, the `effort` key is omitted (API default).
      * OpenAI Chat Completions: emits top-level `reasoning_effort=...`
        (mapped/bucketed the same way). The wire format has no
        reasoning-content channel, so ThinkingBlocks still cannot
        round-trip through this provider; only the effort knob takes effect.

    When `thinking=False`, `effort` and `budget` are ignored and providers
    omit all reasoning-related kwargs.
    """

    model: str
    messages: list[Message]
    max_tokens: int
    system: str | None = None
    tools: list[ToolSpec] = field(default_factory=list)
    thinking: bool = False
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    budget: int | None = None


@dataclass(frozen=True, slots=True)
class CompletionResponse:
    """Output of a single provider call.

    `content` is the assistant's full output for this turn — interleaved
    TextBlock and ToolUseBlock entries in emission order. `stop_reason` has
    already been normalized to the Anthropic vocabulary by the plugin.
    `usage` is provider-defined; the loop does not read it.
    """

    content: list[ContentBlock]
    stop_reason: StopReason
    usage: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Streaming events
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextDelta:
    """An incremental fragment of assistant text emitted during streaming.

    Concatenating every `TextDelta.text` for a given streaming turn yields
    the full text of the assembled assistant `TextBlock`s in emission order.
    """

    text: str
    type: Literal["text_delta"] = "text_delta"


@dataclass(frozen=True, slots=True)
class ThinkingDelta:
    """An incremental fragment of model-internal reasoning text.

    Emitted by providers that surface reasoning during streaming (Anthropic
    extended thinking, OpenAI Responses reasoning summaries). Providers that
    have no reasoning channel (OpenAI Chat Completions) never emit this.
    """

    thinking: str
    type: Literal["thinking_delta"] = "thinking_delta"


@dataclass(frozen=True, slots=True)
class ToolUseDelta:
    """An incremental fragment of an in-flight tool_use block.

    The block id is the same across every delta belonging to the same
    tool_use. `name` is set exactly once on the first delta for a block (the
    "start" delta) and is `None` thereafter; `partial_json` is `None` on the
    start delta and carries a JSON-string fragment on every subsequent
    delta. Concatenating all `partial_json` fragments for a given id yields
    the JSON-encoded `input` of the assembled `ToolUseBlock`.
    """

    id: str
    name: str | None
    partial_json: str | None
    type: Literal["tool_use_delta"] = "tool_use_delta"


@dataclass(frozen=True, slots=True)
class StreamComplete:
    """Terminal event of a stream, carrying the fully assembled response.

    The `response` is the same `CompletionResponse` value that `complete()`
    would have returned for this request. State updates that mirror
    `complete()` (e.g. previous_response_id chaining for the Responses
    provider) happen on emission of this event, never mid-stream.
    """

    response: CompletionResponse
    type: Literal["stream_complete"] = "stream_complete"


StreamEvent = TextDelta | ThinkingDelta | ToolUseDelta | StreamComplete


class IncompleteStreamError(RuntimeError):
    """Raised when a provider stream closes before its terminal completion event."""


# ---------------------------------------------------------------------------
# Provider ABC
# ---------------------------------------------------------------------------


class Provider(ABC):
    """The contract every LLM-provider plugin satisfies.

    A plugin's job is exactly two translations:
      1. CompletionRequest -> the wire format its SDK expects.
      2. The SDK's response -> CompletionResponse, with stop_reason normalized.

    Everything else (the agent loop, tool dispatch, message accumulation)
    is provider-agnostic and lives in `wattle.loop`.
    """

    def fork(self) -> Self:
        """Return an independent wrapper for a separate conversation.

        Stateless providers may return ``self``. Stateful providers should
        override this to share underlying API clients while keeping conversation
        cursors isolated.
        """

        return self

    @abstractmethod
    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        """Run one round-trip against the underlying LLM API."""
        ...

    @abstractmethod
    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        """Stream the response, ending with one StreamComplete event."""
        ...

    async def areset_conversation(self) -> None:
        """Forget provider-side continuation state, if any."""

        return
