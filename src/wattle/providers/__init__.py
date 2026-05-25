"""LLM-provider abstraction.

The full surface that provider plugins (Anthropic / OpenAI Completion / OpenAI
Responses) target lives here. Plugins implement async `Provider.acomplete()` and
`Provider.astream()`; everything else is provider-agnostic.
"""

from .anthropic import AnthropicProvider
from .base import (
    CompletionRequest,
    CompletionResponse,
    ContentBlock,
    ImageBlock,
    IncompleteStreamError,
    Message,
    Provider,
    RedactedThinkingBlock,
    Role,
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
from .openai_codex import OpenAICodexResponsesProvider
from .openai_completions import OpenAICompletionsProvider
from .openai_responses import OpenAIResponsesProvider
from .stub import StubProvider

__all__ = [
    "AnthropicProvider",
    "CompletionRequest",
    "CompletionResponse",
    "ContentBlock",
    "ImageBlock",
    "IncompleteStreamError",
    "Message",
    "OpenAICompletionsProvider",
    "OpenAICodexResponsesProvider",
    "OpenAIResponsesProvider",
    "Provider",
    "RedactedThinkingBlock",
    "Role",
    "StopReason",
    "StreamComplete",
    "StreamEvent",
    "StubProvider",
    "TextBlock",
    "TextDelta",
    "ThinkingBlock",
    "ThinkingDelta",
    "ToolResultBlock",
    "ToolUseBlock",
    "ToolUseDelta",
    "TransientProviderError",
]
