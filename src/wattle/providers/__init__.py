"""LLM-provider abstraction.

The full surface that provider plugins (Anthropic / OpenAI Completion / OpenAI
Responses) target lives here. Plugins implement async `Provider.acomplete()` and
`Provider.astream()`; everything else is provider-agnostic.
"""

from .base import (
    CompletionRequest,
    CompletionResponse,
    ContentBlock,
    ImageBlock,
    IncompleteStreamError,
    Message,
    Provider,
    ProviderAuthError,
    ProviderBillingError,
    ProviderError,
    ProviderInvalidRequestError,
    ProviderPermissionError,
    ProviderPolicyError,
    ProviderQuotaError,
    ProviderRequestTooLargeError,
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
from .stub import StubProvider


def __getattr__(name: str):
    if name == "AnthropicProvider":
        from .anthropic import AnthropicProvider

        return AnthropicProvider
    if name == "OpenAICodexResponsesProvider":
        from .openai_codex import OpenAICodexResponsesProvider

        return OpenAICodexResponsesProvider
    if name == "OpenAICompletionsProvider":
        from .openai_completions import OpenAICompletionsProvider

        return OpenAICompletionsProvider
    if name == "OpenAIResponsesProvider":
        from .openai_responses import OpenAIResponsesProvider

        return OpenAIResponsesProvider
    raise AttributeError(name)


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
    "ProviderAuthError",
    "ProviderBillingError",
    "ProviderError",
    "ProviderInvalidRequestError",
    "ProviderPermissionError",
    "ProviderPolicyError",
    "ProviderQuotaError",
    "ProviderRequestTooLargeError",
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
