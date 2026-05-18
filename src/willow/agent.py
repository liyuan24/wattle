"""High-level agent runner.

`run_agent` is Willow's single entry point for "pick a provider by name,
wire up its credential, and run the loop with all registered tools." Provider
selection, vendor-to-credential resolution, SDK client construction, and provider
wrapping are unified into one principled dispatch table — adding a new
provider is a one-line change to ``_PROVIDER_DISPATCH``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import anthropic
import openai

from willow import loop
from willow.auth import get_credential
from willow.permissions import PermissionGate, PermissionMode
from willow.providers import (
    AnthropicProvider,
    CompletionResponse,
    OpenAICodexResponsesProvider,
    OpenAICompletionsProvider,
    OpenAIResponsesProvider,
    Provider,
)
from willow.skills import load_available_skills
from willow.system_prompt import build_system_prompt
from willow.tools import TOOLS_BY_NAME

# Public mapping: provider name -> vendor name in willow.auth. Multiple
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
    _ProviderSpec[anthropic.Anthropic] | _ProviderSpec[openai.OpenAI] | _ProviderSpec[str]
)


# Single principled dispatch. Adding a new provider = one entry here. The
# value type erases `ClientT` (each entry's SDK client may differ); the
# `build()` method preserves the per-entry consistency internally.
_PROVIDER_DISPATCH: dict[str, _DispatchSpec] = {
    "anthropic": _ProviderSpec[anthropic.Anthropic](
        vendor="anthropic",
        client_factory=lambda key: anthropic.Anthropic(api_key=key),
        provider_factory=lambda client: AnthropicProvider(client=client),
    ),
    "deepseek": _ProviderSpec[openai.OpenAI](
        vendor="deepseek",
        client_factory=lambda key: openai.OpenAI(
            api_key=key,
            base_url="https://api.deepseek.com",
        ),
        provider_factory=lambda client: OpenAICompletionsProvider(client=client),
    ),
    "kimi": _ProviderSpec[openai.OpenAI](
        vendor="kimi",
        client_factory=lambda key: openai.OpenAI(
            api_key=key,
            base_url="https://api.moonshot.ai/v1",
        ),
        provider_factory=lambda client: OpenAICompletionsProvider(client=client),
    ),
    "minimax": _ProviderSpec[openai.OpenAI](
        vendor="minimax",
        client_factory=lambda key: openai.OpenAI(
            api_key=key,
            base_url="https://api.minimax.io/v1",
        ),
        provider_factory=lambda client: OpenAICompletionsProvider(client=client),
    ),
    "openai_codex": _ProviderSpec[str](
        vendor="openai",
        client_factory=lambda key: key,
        provider_factory=lambda token: OpenAICodexResponsesProvider(bearer_token=token),
    ),
    "openai_completions": _ProviderSpec[openai.OpenAI](
        vendor="openai",
        client_factory=lambda key: openai.OpenAI(api_key=key),
        provider_factory=lambda client: OpenAICompletionsProvider(client=client),
    ),
    "openai_responses": _ProviderSpec[openai.OpenAI](
        vendor="openai",
        client_factory=lambda key: openai.OpenAI(api_key=key),
        provider_factory=lambda client: OpenAIResponsesProvider(client=client),
    ),
}

# Sanity: keep the public mapping in lockstep with the internal dispatch.
assert {name: spec.vendor for name, spec in _PROVIDER_DISPATCH.items()} == PROVIDER_TO_VENDOR


def run_agent(
    provider_name: str,
    model: str,
    user_input: str,
    *,
    max_tokens: int = 4096,
    max_iterations: int = 20,
    permission_mode: PermissionMode = PermissionMode.YOLO,
    thinking: bool = False,
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
) -> CompletionResponse:
    """Construct the right provider, wire its credential from ``willow.auth``,
    and run the agent loop with all registered tools.

    Args:
        provider_name: any key in :data:`PROVIDER_TO_VENDOR`.
        model: Model id passed through to the provider unchanged.
        user_input: First user turn.
        max_tokens: Forwarded to the provider per turn.
        max_iterations: Hard cap on assistant turns before the loop exits.
        thinking: Enable provider reasoning controls.
        effort: Requested reasoning effort when ``thinking`` is enabled.

    Raises:
        ValueError: If ``provider_name`` is not a recognized provider.
        FileNotFoundError: If the auth file is missing (from :mod:`willow.auth`).
        KeyError: If the vendor's entry is missing or malformed (from
            :mod:`willow.auth`).
    """
    spec = _PROVIDER_DISPATCH.get(provider_name)
    if spec is None:
        raise ValueError(
            f"Unknown provider: {provider_name!r}. "
            f"Choices: {sorted(PROVIDER_TO_VENDOR)}"
        )

    credential = get_credential(spec.vendor)
    provider = spec.build(credential.bearer_token)
    built_system = build_system_prompt(
        tools_by_name=TOOLS_BY_NAME,
        skills=load_available_skills(Path.cwd()),
        permission_mode=permission_mode,
    )

    if permission_mode == PermissionMode.YOLO:
        return loop.run(
            provider,
            TOOLS_BY_NAME,
            built_system,
            user_input,
            model,
            max_tokens,
            max_iterations,
            thinking=thinking,
            effort=effort,
        )

    return loop.run(
        provider,
        TOOLS_BY_NAME,
        built_system,
        user_input,
        model,
        max_tokens,
        max_iterations,
        permission_gate=PermissionGate(permission_mode),
        thinking=thinking,
        effort=effort,
    )
