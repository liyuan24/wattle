"""Wattle's command-line entry point.

Default usage mirrors coding-agent CLIs:

    wattle
    wattle "follow this prompt"
    wattle -p "follow this prompt"

Running without ``-p/--print`` opens the native terminal UI. A positional
prompt is submitted as the first interactive user message. Running with
``-p/--print`` executes one prompt headlessly and prints only the final
assistant text.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import anthropic
import openai

from wattle.agent import AgentRunResult, _ProviderSpec, run_agent, run_agent_with_history
from wattle.auth import get_api_key_credential, get_credential, get_openai_codex_credential
from wattle.models import (
    MODEL_CHOICES_BY_MODEL,
    first_available_model_choice,
    first_catalog_model_choice,
    first_catalog_model_choice_for_provider,
)
from wattle.permissions import PermissionMode
from wattle.providers import (
    AnthropicProvider,
    OpenAICodexResponsesProvider,
    OpenAICompletionsProvider,
    OpenAIResponsesProvider,
    Provider,
    TextBlock,
)
from wattle.session import new_session, save_session
from wattle.settings import WattleSettings, load_settings

# ---------------------------------------------------------------------------
# Provider construction
# ---------------------------------------------------------------------------


# `_ProviderSpec` is reused from `wattle.agent` so the dispatch tables share
# the same typed surface while letting the TUI own a longer-lived provider.
type _DispatchSpec = (
    _ProviderSpec[anthropic.AsyncAnthropic]
    | _ProviderSpec[openai.AsyncOpenAI]
    | _ProviderSpec[str]
)

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
    "xiaomi-token-plan-sgp": _ProviderSpec[openai.AsyncOpenAI](
        vendor="xiaomi-token-plan-sgp",
        client_factory=lambda key: openai.AsyncOpenAI(
            api_key=key,
            base_url="https://token-plan-sgp.xiaomimimo.com/v1",
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

_API_KEY_ONLY_PROVIDERS = frozenset(
    {
        "openai_completions",
        "openai_responses",
        "xiaomi-token-plan-sgp",
    }
)


def _build_provider(provider_name: str) -> Provider:
    """Resolve a provider name to a fully-wired Provider instance."""
    spec = _PROVIDER_DISPATCH.get(provider_name)
    if spec is None:
        raise ValueError(
            f"Unknown provider: {provider_name!r}. "
            f"Choices: {sorted(_PROVIDER_DISPATCH)}"
        )
    if provider_name == "openai_codex":
        credential = get_openai_codex_credential()
    elif provider_name in _API_KEY_ONLY_PROVIDERS:
        credential = get_api_key_credential(spec.vendor)
    else:
        credential = get_credential(spec.vendor)
    return spec.build(credential.bearer_token)


def _provider_auth_available(provider_name: str) -> bool:
    spec = _PROVIDER_DISPATCH.get(provider_name)
    if spec is None:
        return False
    try:
        if provider_name == "openai_codex":
            get_openai_codex_credential()
        elif provider_name in _API_KEY_ONLY_PROVIDERS:
            get_api_key_credential(spec.vendor)
        else:
            get_credential(spec.vendor)
    except (FileNotFoundError, KeyError, ValueError):
        return False
    return True


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _run_headless(args: argparse.Namespace) -> int:
    """Run one prompt and print only the final assistant text."""
    permission_mode = PermissionMode.YOLO

    if bool(getattr(args, "persist", False)):
        result = run_agent_with_history(
            args.provider,
            args.model,
            args.print_prompt,
            max_tokens=args.max_tokens,
            permission_mode=permission_mode,
            thinking=bool(getattr(args, "thinking", False)),
            effort=getattr(args, "effort", None),
        )
        response = result.response
        session_path = _persist_headless_session(args, result)
    else:
        response = run_agent(
            args.provider,
            args.model,
            args.print_prompt,
            max_tokens=args.max_tokens,
            permission_mode=permission_mode,
            thinking=bool(getattr(args, "thinking", False)),
            effort=getattr(args, "effort", None),
        )
        session_path = None

    text = "".join(
        block.text for block in response.content if isinstance(block, TextBlock)
    )
    if text:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
    if session_path is not None:
        sys.stderr.write(f"Saved session: {session_path}\n")
        sys.stderr.flush()
    return 0


def _persist_headless_session(args: argparse.Namespace, result: AgentRunResult) -> Path:
    record = new_session(
        provider=args.provider,
        model=args.model,
        system=result.system,
        max_tokens=args.max_tokens,
        thinking=bool(getattr(args, "thinking", False)),
        effort=getattr(args, "effort", None),
    )
    return save_session(replace(record, messages=list(result.messages)))


def _run_tui(args: argparse.Namespace) -> int:
    """Lazy-import wrapper so headless mode doesn't pay the TUI import cost."""
    from wattle.tui import run_tui

    return run_tui(args)


# ---------------------------------------------------------------------------
# argparse plumbing
# ---------------------------------------------------------------------------


def _add_permission_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--yolo",
        dest="permission_mode",
        action="store_const",
        const=PermissionMode.YOLO,
        default=PermissionMode.YOLO,
        help="Run requested tools without asking for confirmation (default and only mode).",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wattle",
        description="Wattle — a pure-Python coding agent.",
    )
    parser.add_argument(
        "-p",
        "--print",
        dest="print_prompt",
        metavar="PROMPT",
        default=None,
        help="Run one prompt headlessly and print the final response.",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Save a headless -p session using the same session store as the TUI.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Prompt to submit as the first message in the interactive TUI.",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(_PROVIDER_DISPATCH),
        default=None,
        help=(
            "Provider to talk to (default: settings.json provider, otherwise "
            "the provider for the selected/default catalog model)."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model id forwarded to the provider (default: settings.json model, "
            "otherwise the first model of the first authenticated provider)."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Per-turn output token cap (default: 4096).",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable provider reasoning controls for supported models.",
    )
    parser.add_argument(
        "--effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        default=None,
        help="Reasoning effort to request when --thinking is enabled.",
    )
    parser.add_argument(
        "-r",
        "--resume",
        nargs="?",
        const="",
        default=None,
        metavar="SESSION",
        help=(
            "Resume a saved TUI session. With SESSION, use that session id or JSONL path; "
            "without SESSION, choose from recent sessions."
        ),
    )
    _add_permission_args(parser)
    return parser


def _apply_settings_defaults(
    args: argparse.Namespace,
    argv: list[str],
    settings: WattleSettings,
) -> None:
    model_explicit = _has_flag(argv, "--model")
    provider_explicit = _has_flag(argv, "--provider")
    if provider_explicit and not model_explicit and settings.model is None:
        provider_default = first_catalog_model_choice_for_provider(args.provider)
        if provider_default is not None:
            args.model = provider_default.model
    if not model_explicit and args.model is None:
        if settings.model is not None:
            args.model = settings.model
        else:
            default_choice = first_available_model_choice() or first_catalog_model_choice()
            args.model = default_choice.model
            if not provider_explicit:
                args.provider = default_choice.provider
    if not provider_explicit:
        provider = _default_provider_for_model(args.model)
        if provider is not None:
            args.provider = provider
        elif settings.provider in _PROVIDER_DISPATCH:
            args.provider = settings.provider
        else:
            default_choice = first_available_model_choice() or first_catalog_model_choice()
            args.provider = default_choice.provider
            if not model_explicit and settings.model is None:
                args.model = default_choice.model
    if not _has_flag(argv, "--max-tokens"):
        args.max_tokens = settings.max_tokens
    if not _has_flag(argv, "--thinking") and not _has_flag(argv, "--effort"):
        args.thinking = settings.thinking
        args.effort = settings.effort
    elif not _has_flag(argv, "--effort") and getattr(args, "effort", None) is None:
        args.effort = settings.effort if args.thinking else None
    args.permission_mode = PermissionMode.YOLO
    args.statusline_fields = settings.tui.statusline_fields
    args.statusline = bool(args.statusline_fields)
    args.enabled_models = settings.enabled_models
    args.compaction_keep_recent_tokens = settings.compaction_keep_recent_tokens


def _default_provider_for_model(model: str) -> str | None:
    choice = MODEL_CHOICES_BY_MODEL.get(model)
    return choice.provider if choice is not None else None


def _has_flag(argv: list[str], flag: str) -> bool:
    return any(item == flag or item.startswith(f"{flag}=") for item in argv)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    _apply_settings_defaults(args, raw_argv, load_settings())
    if args.print_prompt is not None and args.prompt is not None:
        parser.error("positional prompt cannot be used with -p/--print")
    if args.persist and args.print_prompt is None:
        parser.error("--persist can only be used with -p/--print")
    if args.effort is not None:
        args.thinking = True
    if args.print_prompt is not None:
        return _run_headless(args)
    return _run_tui(args)


if __name__ == "__main__":
    raise SystemExit(main())
