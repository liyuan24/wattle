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

import anthropic
import openai

from wattle.agent import _ProviderSpec, run_agent
from wattle.auth import get_credential
from wattle.permissions import PermissionMode
from wattle.providers import (
    AnthropicProvider,
    OpenAICodexResponsesProvider,
    OpenAICompletionsProvider,
    OpenAIResponsesProvider,
    Provider,
    TextBlock,
)

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


def _build_provider(provider_name: str) -> Provider:
    """Resolve a provider name to a fully-wired Provider instance."""
    spec = _PROVIDER_DISPATCH.get(provider_name)
    if spec is None:
        raise ValueError(
            f"Unknown provider: {provider_name!r}. "
            f"Choices: {sorted(_PROVIDER_DISPATCH)}"
        )
    credential = get_credential(spec.vendor)
    return spec.build(credential.bearer_token)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _run_headless(args: argparse.Namespace) -> int:
    """Run one prompt and print only the final assistant text."""
    permission_mode = getattr(args, "permission_mode", PermissionMode.YOLO)
    if permission_mode == PermissionMode.ASK:
        sys.stderr.write(
            "wattle -p does not support --ask-for-permission; "
            "use --yolo or --read-only.\n"
        )
        sys.stderr.flush()
        return 2

    response = run_agent(
        args.provider,
        args.model,
        args.print_prompt,
        max_tokens=args.max_tokens,
        permission_mode=permission_mode,
        thinking=bool(getattr(args, "thinking", False)),
        effort=getattr(args, "effort", None),
    )

    text = "".join(
        block.text for block in response.content if isinstance(block, TextBlock)
    )
    if text:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
    return 0


def _run_tui(args: argparse.Namespace) -> int:
    """Lazy-import wrapper so headless mode doesn't pay the TUI import cost."""
    from wattle.tui import run_tui

    return run_tui(args)


# ---------------------------------------------------------------------------
# argparse plumbing
# ---------------------------------------------------------------------------


def _add_permission_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--yolo",
        dest="permission_mode",
        action="store_const",
        const=PermissionMode.YOLO,
        default=PermissionMode.YOLO,
        help="Run requested tools without asking for confirmation (default).",
    )
    group.add_argument(
        "--read-only",
        dest="permission_mode",
        action="store_const",
        const=PermissionMode.READ_ONLY,
        help="Only allow read-only tools.",
    )
    group.add_argument(
        "--ask-for-permission",
        dest="permission_mode",
        action="store_const",
        const=PermissionMode.ASK,
        help="Ask before executing each tool. Not supported with -p.",
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
        "prompt",
        nargs="?",
        default=None,
        help="Prompt to submit as the first message in the interactive TUI.",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(_PROVIDER_DISPATCH),
        default="openai_codex",
        help="Provider to talk to (default: openai_codex).",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.5",
        help="Model id forwarded to the provider (default: gpt-5.5).",
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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.print_prompt is not None and args.prompt is not None:
        parser.error("positional prompt cannot be used with -p/--print")
    if args.effort is not None:
        args.thinking = True
    if args.print_prompt is not None:
        return _run_headless(args)
    return _run_tui(args)


if __name__ == "__main__":
    raise SystemExit(main())
