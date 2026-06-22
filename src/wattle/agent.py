"""High-level agent runner.

`run_agent` is Wattle's single entry point for "pick a provider by name,
wire up its credential, and run the loop with all registered tools." Provider
selection, vendor-to-credential resolution, SDK client construction, and provider
wrapping are unified into one principled dispatch table — adding a new
provider is a one-line change to ``_PROVIDER_DISPATCH``.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import anthropic
import openai

from wattle import loop
from wattle.auth import (
    ANTHROPIC_OAUTH_BETA_HEADER,
    AuthCredential,
    get_api_key_credential,
    get_credential,
    get_openai_codex_credential,
)
from wattle.goal import GoalState, GoalTurnStopHook
from wattle.permissions import PermissionMode
from wattle.providers import (
    AnthropicProvider,
    CompletionResponse,
    Message,
    OpenAICodexResponsesProvider,
    OpenAICompletionsProvider,
    OpenAIResponsesProvider,
    Provider,
)
from wattle.session import SessionEvent
from wattle.skills import load_available_skills
from wattle.system_prompt import build_system_prompt
from wattle.tools import TOOLS_BY_NAME
from wattle.tools.base import Tool
from wattle.tools.goal import UpdateGoalTool

# Public mapping: provider name -> vendor name in wattle.auth. Multiple
# OpenAI-backed provider adapters can still use distinct vendor credentials.
PROVIDER_TO_VENDOR: dict[str, str] = {
    "anthropic": "anthropic",
    "deepseek": "deepseek",
    "kimi": "kimi",
    "minimax": "minimax",
    "xiaomi-token-plan-sgp": "xiaomi-token-plan-sgp",
    "openai_codex": "openai",
    "openai_completions": "openai",
    "openai_responses": "openai",
}

DEFAULT_PROVIDER_REQUEST_TIMEOUT_SECONDS = 300.0
PROVIDER_REQUEST_TIMEOUT_ENV = "WATTLE_PROVIDER_REQUEST_TIMEOUT_SECONDS"
MAX_HEADLESS_GOAL_CONTINUATIONS = 20


def _provider_request_timeout_seconds() -> float:
    raw_value = os.environ.get(PROVIDER_REQUEST_TIMEOUT_ENV)
    if raw_value is None:
        return DEFAULT_PROVIDER_REQUEST_TIMEOUT_SECONDS
    try:
        value = float(raw_value)
    except ValueError:
        return DEFAULT_PROVIDER_REQUEST_TIMEOUT_SECONDS
    if value <= 0:
        return DEFAULT_PROVIDER_REQUEST_TIMEOUT_SECONDS
    return value


def _sdk_timeout_kwargs() -> dict[str, float]:
    return {"timeout": _provider_request_timeout_seconds()}


# Generic over the SDK client type. Each `_ProviderSpec` instance is
# internally consistent: the `provider_factory` accepts exactly the kind of
# client the `client_factory` produces. The `build()` method ties the two
# halves together and erases `ClientT` at the boundary, so the dispatch
# dict below can hold heterogeneous specs (different `T` per entry) while
# still being callable through a single typed surface.
@dataclass(frozen=True)
class _ProviderSpec[ClientT]:
    """How to build one provider end-to-end from a credential."""

    vendor: str
    client_factory: Callable[[AuthCredential], ClientT]
    provider_factory: Callable[[ClientT], Provider]

    def build(self, credential: AuthCredential) -> Provider:
        """Construct the SDK client and wrap it in a Provider."""
        return self.provider_factory(self.client_factory(credential))


type _DispatchSpec = (
    _ProviderSpec[anthropic.AsyncAnthropic]
    | _ProviderSpec[openai.AsyncOpenAI]
    | _ProviderSpec[str]
)


def _anthropic_auth_kwargs(credential: AuthCredential) -> dict[str, object]:
    """Return Anthropic SDK auth kwargs for an API-key or OAuth credential."""
    if credential.kind == "oauth":
        return {
            "auth_token": credential.bearer_token,
            "default_headers": {"anthropic-beta": ANTHROPIC_OAUTH_BETA_HEADER},
        }
    return {"api_key": credential.bearer_token}


# Single principled dispatch. Adding a new provider = one entry here. The
# value type erases `ClientT` (each entry's SDK client may differ); the
# `build()` method preserves the per-entry consistency internally.
_PROVIDER_DISPATCH: dict[str, _DispatchSpec] = {
    "anthropic": _ProviderSpec[anthropic.AsyncAnthropic](
        vendor="anthropic",
        client_factory=lambda credential: anthropic.AsyncAnthropic(
            **_anthropic_auth_kwargs(credential),
            **_sdk_timeout_kwargs(),
        ),
        provider_factory=lambda client: AnthropicProvider(async_client=client),
    ),
    "deepseek": _ProviderSpec[openai.AsyncOpenAI](
        vendor="deepseek",
        client_factory=lambda credential: openai.AsyncOpenAI(
            api_key=credential.bearer_token,
            base_url="https://api.deepseek.com",
            **_sdk_timeout_kwargs(),
        ),
        provider_factory=lambda client: OpenAICompletionsProvider(async_client=client),
    ),
    "kimi": _ProviderSpec[openai.AsyncOpenAI](
        vendor="kimi",
        client_factory=lambda credential: openai.AsyncOpenAI(
            api_key=credential.bearer_token,
            base_url="https://api.moonshot.ai/v1",
            **_sdk_timeout_kwargs(),
        ),
        provider_factory=lambda client: OpenAICompletionsProvider(async_client=client),
    ),
    "minimax": _ProviderSpec[openai.AsyncOpenAI](
        vendor="minimax",
        client_factory=lambda credential: openai.AsyncOpenAI(
            api_key=credential.bearer_token,
            base_url="https://api.minimax.io/v1",
            **_sdk_timeout_kwargs(),
        ),
        provider_factory=lambda client: OpenAICompletionsProvider(async_client=client),
    ),
    "xiaomi-token-plan-sgp": _ProviderSpec[openai.AsyncOpenAI](
        vendor="xiaomi-token-plan-sgp",
        client_factory=lambda credential: openai.AsyncOpenAI(
            api_key=credential.bearer_token,
            base_url="https://token-plan-sgp.xiaomimimo.com/v1",
            **_sdk_timeout_kwargs(),
        ),
        provider_factory=lambda client: OpenAICompletionsProvider(async_client=client),
    ),
    "openai_codex": _ProviderSpec[str](
        vendor="openai",
        client_factory=lambda credential: credential.bearer_token,
        provider_factory=lambda token: OpenAICodexResponsesProvider(bearer_token=token),
    ),
    "openai_completions": _ProviderSpec[openai.AsyncOpenAI](
        vendor="openai",
        client_factory=lambda credential: openai.AsyncOpenAI(
            api_key=credential.bearer_token,
            **_sdk_timeout_kwargs(),
        ),
        provider_factory=lambda client: OpenAICompletionsProvider(async_client=client),
    ),
    "openai_responses": _ProviderSpec[openai.AsyncOpenAI](
        vendor="openai",
        client_factory=lambda credential: openai.AsyncOpenAI(
            api_key=credential.bearer_token,
            **_sdk_timeout_kwargs(),
        ),
        provider_factory=lambda client: OpenAIResponsesProvider(async_client=client),
    ),
}

_API_KEY_ONLY_PROVIDERS = frozenset(
    {
        "openai_completions",
        "openai_responses",
        "xiaomi-token-plan-sgp",
    }
)

# Sanity: keep the public mapping in lockstep with the internal dispatch.
assert {name: spec.vendor for name, spec in _PROVIDER_DISPATCH.items()} == PROVIDER_TO_VENDOR


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """Result of a headless agent run with its persisted transcript."""

    response: CompletionResponse
    messages: list[Message]
    system: str | None
    events: list[SessionEvent] = field(default_factory=list)
    goal: GoalState | None = None


@dataclass(frozen=True, slots=True)
class AgentRunSnapshot:
    """Intermediate headless run state suitable for live persistence."""

    system: str | None
    messages: list[Message] = field(default_factory=list)
    events: list[SessionEvent] = field(default_factory=list)
    goal: GoalState | None = None


def _build_provider_and_system(
    provider_name: str,
    permission_mode: PermissionMode,
    *,
    model: str,
    tools_by_name: Mapping[str, Tool] | None = None,
) -> tuple[Provider, str, Mapping[str, Tool]]:
    spec = _PROVIDER_DISPATCH.get(provider_name)
    if spec is None:
        raise ValueError(
            f"Unknown provider: {provider_name!r}. "
            f"Choices: {sorted(PROVIDER_TO_VENDOR)}"
        )

    if provider_name == "openai_codex":
        credential = get_openai_codex_credential()
    elif provider_name in _API_KEY_ONLY_PROVIDERS:
        credential = get_api_key_credential(spec.vendor)
    else:
        credential = get_credential(spec.vendor)
    provider = spec.build(credential)
    available_tools = tools_by_name or TOOLS_BY_NAME
    model_tools_by_name = loop._tools_for_model(available_tools, model)
    built_system = build_system_prompt(
        tools_by_name=model_tools_by_name,
        skills=load_available_skills(Path.cwd()),
        permission_mode=permission_mode,
    )
    return provider, built_system, available_tools


def _tools_for_goal(goal_box: dict[str, GoalState | None]) -> dict[str, Tool]:
    tools = dict(TOOLS_BY_NAME)
    tools[UpdateGoalTool.name] = UpdateGoalTool(
        get_goal=lambda: goal_box["goal"],
        set_goal=lambda goal: goal_box.__setitem__("goal", goal),
    )
    return tools


def run_agent(
    provider_name: str,
    model: str,
    user_input: str,
    *,
    max_tokens: int | None = None,
    permission_mode: PermissionMode = PermissionMode.YOLO,
    thinking: bool = False,
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
    goal: GoalState | None = None,
) -> CompletionResponse:
    """Construct the right provider, wire its credential from ``wattle.auth``,
    and run the agent loop with all registered tools.

    Args:
        provider_name: any key in :data:`PROVIDER_TO_VENDOR`.
        model: Model id passed through to the provider unchanged.
        user_input: First user turn.
        max_tokens: Per-turn output cap; None resolves to the model's
            documented max output limit (see ``wattle.models``).
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
            goal=goal,
        )
    )


async def arun_agent(
    provider_name: str,
    model: str,
    user_input: str,
    *,
    max_tokens: int | None = None,
    permission_mode: PermissionMode = PermissionMode.YOLO,
    thinking: bool = False,
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
    goal: GoalState | None = None,
) -> CompletionResponse:
    """Async implementation of :func:`run_agent`."""
    goal_box: dict[str, GoalState | None] = {"goal": goal}
    goal_tools = _tools_for_goal(goal_box) if goal is not None else None
    provider, built_system, tools_by_name = _build_provider_and_system(
        provider_name,
        permission_mode,
        model=model,
        tools_by_name=goal_tools,
    )
    run_kwargs: dict[str, object] = {
        "permission_gate": None,
        "thinking": thinking,
        "effort": effort,
    }
    if goal is not None:
        run_kwargs["turn_stop_hooks"] = [GoalTurnStopHook(lambda: goal_box["goal"])]
        run_kwargs["max_turn_stop_continuations"] = MAX_HEADLESS_GOAL_CONTINUATIONS
    return await loop.arun(
        provider,
        tools_by_name,
        built_system,
        user_input,
        model,
        max_tokens,
        **run_kwargs,
    )


def run_agent_with_history(
    provider_name: str,
    model: str,
    user_input: str,
    *,
    max_tokens: int | None = None,
    permission_mode: PermissionMode = PermissionMode.YOLO,
    thinking: bool = False,
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
    on_snapshot: Callable[[AgentRunSnapshot], None] | None = None,
    goal: GoalState | None = None,
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
            on_snapshot=on_snapshot,
            goal=goal,
        )
    )


async def arun_agent_with_history(
    provider_name: str,
    model: str,
    user_input: str,
    *,
    max_tokens: int | None = None,
    permission_mode: PermissionMode = PermissionMode.YOLO,
    thinking: bool = False,
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
    on_snapshot: Callable[[AgentRunSnapshot], None] | None = None,
    goal: GoalState | None = None,
) -> AgentRunResult:
    """Async headless agent runner returning the full transcript."""
    goal_box: dict[str, GoalState | None] = {"goal": goal}
    goal_tools = _tools_for_goal(goal_box) if goal is not None else None
    provider, built_system, tools_by_name = _build_provider_and_system(
        provider_name,
        permission_mode,
        model=model,
        tools_by_name=goal_tools,
    )
    messages: list[Message] = []
    raw_events: list[dict[str, object]] = []
    events: list[SessionEvent] = []

    def publish_snapshot() -> None:
        if on_snapshot is not None:
            on_snapshot(
                AgentRunSnapshot(
                    system=built_system,
                    messages=list(messages),
                    events=list(events),
                    goal=goal_box["goal"],
                )
            )

    def on_messages(snapshot: list[Message]) -> None:
        messages[:] = snapshot
        publish_snapshot()

    def on_runtime_event(event: dict[str, object]) -> None:
        session_event = _session_event_from_runtime(event)
        events.append(session_event)
        publish_snapshot()

    publish_snapshot()
    run_kwargs = {
        "permission_gate": None,
        "thinking": thinking,
        "effort": effort,
        "messages_out": messages,
        "runtime_events_out": raw_events,
        "messages_callback": on_messages,
        "runtime_event_callback": on_runtime_event,
    }
    if goal is not None:
        run_kwargs["turn_stop_hooks"] = [GoalTurnStopHook(lambda: goal_box["goal"])]
        run_kwargs["max_turn_stop_continuations"] = MAX_HEADLESS_GOAL_CONTINUATIONS
    response = await loop.arun(
        provider,
        tools_by_name,
        built_system,
        user_input,
        model,
        max_tokens,
        **run_kwargs,
    )
    publish_snapshot()
    return AgentRunResult(
        response=response,
        messages=messages,
        system=built_system,
        events=events,
        goal=goal_box["goal"],
    )


def _session_event_from_runtime(raw_event: dict[str, object]) -> SessionEvent:
    source = raw_event.get("source")
    data = raw_event.get("data")
    created_at = raw_event.get("created_at")
    fallback_created_at = SessionEvent("runtime_event").created_at
    return SessionEvent(
        type=str(raw_event.get("type") or "runtime_event"),
        created_at=created_at if isinstance(created_at, str) else fallback_created_at,
        source=source if isinstance(source, dict) else {},
        data=data if isinstance(data, dict) else {},
    )
