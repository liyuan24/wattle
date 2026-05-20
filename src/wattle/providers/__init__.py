"""LLM-provider abstraction.

The full surface that provider plugins (Anthropic / OpenAI Completion / OpenAI
Responses) target lives here. Plugins implement `Provider.complete()`;
everything else is provider-agnostic.
"""

from .anthropic import AnthropicProvider
from .base import (
    CompletionRequest,
    CompletionResponse,
    ContentBlock,
    ImageBlock,
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
]
