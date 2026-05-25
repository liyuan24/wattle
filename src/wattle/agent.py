"""High-level agent runner.

`run_agent` is Wattle's single entry point for "pick a provider by name,
wire up its credential, and run the loop with all registered tools." Provider
selection, vendor-to-credential resolution, SDK client construction, and provider
wrapping are unified into one principled dispatch table — adding a new
provider is a one-line change to ``_PROVIDER_DISPATCH``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import anthropic
import openai

from wattle import loop
from wattle.auth import get_api_key_credential, get_credential, get_openai_codex_credential
from wattle.permissions import PermissionGate, PermissionMode
from wattle.providers import (
    AnthropicProvider,
    CompletionResponse,
    Message,
    OpenAICodexResponsesProvider,
    OpenAICompletionsProvider,
    OpenAIResponsesProvider,
    Provider,
)
from wattle.skills import load_available_skills
from wattle.system_prompt import build_system_prompt
from wattle.tools import TOOLS_BY_NAME

# Public mapping: provider name -> vendor name in wattle.auth. Multiple
# OpenAI-backed provider adapters can still use distinct vendor credentials.
PROVIDER_TO_VENDOR: dict[str, str] = {
    "anthropic": "anthropic",
    "deepseek": "deepseek",
    "kimi": "kimi",
    "minimax": "minimax",
    "openai_codex": "openai",
    "openai_completions": "openai",
    "openai_responses": "openai",
}


# Generic over the SDK client type. Each `_ProviderSpec` instance is
# internally consistent: the `provider_factory` accepts exactly the kind of
# client the `client_factory` produces. The `build()` method ties the two
# halves together and erases `ClientT` at the boundary, so the dispatch
# dict below can hold heterogeneous specs (different `T` per entry) while
# still being callable through a single typed surface.
@dataclass(frozen=True)
class _ProviderSpec[ClientT]:
    """How to build one provider end-to-end from a bearer token."""

    vendor: str
    client_factory: Callable[[str], ClientT]
    provider_factory: Callable[[ClientT], Provider]

    def build(self, bearer_token: str) -> Provider:
        """Construct the SDK client and wrap it in a Provider."""
        return self.provider_factory(self.client_factory(bearer_token))


type _DispatchSpec = (
    _ProviderSpec[anthropic.AsyncAnthropic]
    | _ProviderSpec[openai.AsyncOpenAI]
    | _ProviderSpec[str]
)


# Single principled dispatch. Adding a new provider = one entry here. The
# value type erases `ClientT` (each entry's SDK client may differ); the
# `build()` method preserves the per-entry consistency internally.
_PROVIDER_DISPATCH: dict[str, _DispatchSpec] = {
    "anthropic": _ProviderSpec[anthropic.AsyncAnthropic](
        vendor="anthropic",
        client_factory=lambda key: anthropic.AsyncAnthropic(api_key=key),
        provider_factory=lambda client: AnthropicProvider(async_client=client),
    ),
    "deepseek": _ProviderSpec[openai.AsyncOpenAI](
        vendor="deepseek",
        client_factory=lambda key: openai.AsyncOpenAI(
            api_key=key,
            base_url="https://api.deepseek.com",
        ),
        provider_factory=lambda client: OpenAICompletionsProvider(async_client=client),
    ),
    "kimi": _ProviderSpec[openai.AsyncOpenAI](
        vendor="kimi",
        client_factory=lambda key: openai.AsyncOpenAI(
            api_key=key,
            base_url="https://api.moonshot.ai/v1",
        ),
        provider_factory=lambda client: OpenAICompletionsProvider(async_client=client),
    ),
    "minimax": _ProviderSpec[openai.AsyncOpenAI](
        vendor="minimax",
        client_factory=lambda key: openai.AsyncOpenAI(
            api_key=key,
            base_url="https://api.minimax.io/v1",
        ),
        provider_factory=lambda client: OpenAICompletionsProvider(async_client=client),
    ),
    "openai_codex": _ProviderSpec[str](
        vendor="openai",
        client_factory=lambda key: key,
        provider_factory=lambda token: OpenAICodexResponsesProvider(bearer_token=token),
    ),
    "openai_completions": _ProviderSpec[openai.AsyncOpenAI](
        vendor="openai",
        client_factory=lambda key: openai.AsyncOpenAI(api_key=key),
        provider_factory=lambda client: OpenAICompletionsProvider(async_client=client),
    ),
    "openai_responses": _ProviderSpec[openai.AsyncOpenAI](
        vendor="openai",
        client_factory=lambda key: openai.AsyncOpenAI(api_key=key),
        provider_factory=lambda client: OpenAIResponsesProvider(async_client=client),
    ),
}

# Sanity: keep the public mapping in lockstep with the internal dispatch.
assert {name: spec.vendor for name, spec in _PROVIDER_DISPATCH.items()} == PROVIDER_TO_VENDOR


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """Result of a headless agent run with its persisted transcript."""

    response: CompletionResponse
    messages: list[Message]
    system: str | None


def _build_provider_and_system(
    provider_name: str,
    permission_mode: PermissionMode,
) -> tuple[Provider, str]:
    spec = _PROVIDER_DISPATCH.get(provider_name)
    if spec is None:
        raise ValueError(
            f"Unknown provider: {provider_name!r}. "
            f"Choices: {sorted(PROVIDER_TO_VENDOR)}"
        )

    if provider_name == "openai_codex":
        credential = get_openai_codex_credential()
    elif provider_name in {"openai_completions", "openai_responses"}:
        credential = get_api_key_credential(spec.vendor)
    else:
        credential = get_credential(spec.vendor)
    provider = spec.build(credential.bearer_token)
    built_system = build_system_prompt(
        tools_by_name=TOOLS_BY_NAME,
        skills=load_available_skills(Path.cwd()),
        permission_mode=permission_mode,
    )
    return provider, built_system


def run_agent(
    provider_name: str,
    model: str,
    user_input: str,
    *,
    max_tokens: int = 4096,
    permission_mode: PermissionMode = PermissionMode.YOLO,
    thinking: bool = False,
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
) -> CompletionResponse:
    """Construct the right provider, wire its credential from ``wattle.auth``,
    and run the agent loop with all registered tools.

    Args:
        provider_name: any key in :data:`PROVIDER_TO_VENDOR`.
        model: Model id passed through to the provider unchanged.
        user_input: First user turn.
        max_tokens: Forwarded to the provider per turn.
        thinking: Enable provider reasoning controls.
        effort: Requested reasoning effort when ``thinking`` is enabled.

    Raises:
        ValueError: If ``provider_name`` is not a recognized provider.
        FileNotFoundError: If the auth file is missing (from :mod:`wattle.auth`).
        KeyError: If the vendor's entry is missing or malformed (from
            :mod:`wattle.auth`).
    """
    return asyncio.run(
        arun_agent(
            provider_name,
            model,
            user_input,
            max_tokens=max_tokens,
            permission_mode=permission_mode,
            thinking=thinking,
            effort=effort,
        )
    )


async def arun_agent(
    provider_name: str,
    model: str,
    user_input: str,
    *,
    max_tokens: int = 4096,
    permission_mode: PermissionMode = PermissionMode.YOLO,
    thinking: bool = False,
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
) -> CompletionResponse:
    """Async implementation of :func:`run_agent`."""
    provider, built_system = _build_provider_and_system(provider_name, permission_mode)
    return await loop.arun(
        provider,
        TOOLS_BY_NAME,
        built_system,
        user_input,
        model,
        max_tokens,
        permission_gate=(
            None if permission_mode == PermissionMode.YOLO else PermissionGate(permission_mode)
        ),
        thinking=thinking,
        effort=effort,
    )


def run_agent_with_history(
    provider_name: str,
    model: str,
    user_input: str,
    *,
    max_tokens: int = 4096,
    permission_mode: PermissionMode = PermissionMode.YOLO,
    thinking: bool = False,
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
) -> AgentRunResult:
    """Run the headless agent and return the full transcript for persistence."""
    return asyncio.run(
        arun_agent_with_history(
            provider_name,
            model,
            user_input,
            max_tokens=max_tokens,
            permission_mode=permission_mode,
            thinking=thinking,
            effort=effort,
        )
    )


async def arun_agent_with_history(
    provider_name: str,
    model: str,
    user_input: str,
    *,
    max_tokens: int = 4096,
    permission_mode: PermissionMode = PermissionMode.YOLO,
    thinking: bool = False,
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
) -> AgentRunResult:
    """Async headless agent runner returning the full transcript."""
    provider, built_system = _build_provider_and_system(provider_name, permission_mode)
    messages: list[Message] = []
    response = await loop.arun(
        provider,
        TOOLS_BY_NAME,
        built_system,
        user_input,
        model,
        max_tokens,
        permission_gate=(
            None if permission_mode == PermissionMode.YOLO else PermissionGate(permission_mode)
        ),
        thinking=thinking,
        effort=effort,
        messages_out=messages,
    )
    return AgentRunResult(response=response, messages=messages, system=built_system)
