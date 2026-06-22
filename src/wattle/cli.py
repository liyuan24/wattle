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
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard

from wattle.permissions import PermissionMode
from wattle.update import maybe_latest_update, prompt_for_tui_update, run_manual_upgrade
from wattle.version import get_wattle_version

AUTH_REQUIRED_MESSAGE = (
    "No authenticated provider is configured. Run `wattle` and use /login, "
    "or set an API-key or supported OAuth environment variable."
)

if TYPE_CHECKING:
    from wattle.agent import AgentRunResult as AgentRunResultType
    from wattle.agent import _ProviderSpec
    from wattle.goal import GoalState
    from wattle.providers import CompletionResponse, Provider, TextBlock
    from wattle.settings import WattleSettings


@dataclass
class _LazyModule:
    module_name: str

    def __getattr__(self, name: str) -> Any:
        return getattr(import_module(self.module_name), name)


@dataclass
class _LazyAttribute:
    module_name: str
    attribute_name: str

    def _target(self) -> Any:
        return getattr(import_module(self.module_name), self.attribute_name)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target(), name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._target()(*args, **kwargs)

    def get(self, *args: Any, **kwargs: Any) -> Any:
        return self._target().get(*args, **kwargs)


anthropic = _LazyModule("anthropic")
openai = _LazyModule("openai")
AgentRunResult = _LazyAttribute("wattle.agent", "AgentRunResult")


# ---------------------------------------------------------------------------
# Provider construction
# ---------------------------------------------------------------------------


_PROVIDER_CHOICES = (
    "anthropic",
    "deepseek",
    "kimi",
    "minimax",
    "openai_codex",
    "openai_completions",
    "openai_responses",
    "xiaomi-token-plan-sgp",
)
_PROVIDER_TO_VENDOR: dict[str, str] = {
    "anthropic": "anthropic",
    "deepseek": "deepseek",
    "kimi": "kimi",
    "minimax": "minimax",
    "xiaomi-token-plan-sgp": "xiaomi-token-plan-sgp",
    "openai_codex": "openai",
    "openai_completions": "openai",
    "openai_responses": "openai",
}

_API_KEY_ONLY_PROVIDERS = frozenset(
    {
        "openai_completions",
        "openai_responses",
        "xiaomi-token-plan-sgp",
    }
)


def _provider_dispatch() -> dict[str, _ProviderSpec[Any]]:
    from wattle.agent import _anthropic_auth_kwargs, _ProviderSpec, _sdk_timeout_kwargs

    return {
        "anthropic": _ProviderSpec[Any](
            vendor="anthropic",
            client_factory=lambda credential: anthropic.AsyncAnthropic(
                **_anthropic_auth_kwargs(credential),
                **_sdk_timeout_kwargs(),
            ),
            provider_factory=lambda client: globals()["AnthropicProvider"](async_client=client),
        ),
        "deepseek": _ProviderSpec[Any](
            vendor="deepseek",
            client_factory=lambda credential: openai.AsyncOpenAI(
                api_key=credential.bearer_token,
                base_url="https://api.deepseek.com",
                **_sdk_timeout_kwargs(),
            ),
            provider_factory=lambda client: globals()["OpenAICompletionsProvider"](
                async_client=client
            ),
        ),
        "kimi": _ProviderSpec[Any](
            vendor="kimi",
            client_factory=lambda credential: openai.AsyncOpenAI(
                api_key=credential.bearer_token,
                base_url="https://api.moonshot.ai/v1",
                **_sdk_timeout_kwargs(),
            ),
            provider_factory=lambda client: globals()["OpenAICompletionsProvider"](
                async_client=client
            ),
        ),
        "minimax": _ProviderSpec[Any](
            vendor="minimax",
            client_factory=lambda credential: openai.AsyncOpenAI(
                api_key=credential.bearer_token,
                base_url="https://api.minimax.io/v1",
                **_sdk_timeout_kwargs(),
            ),
            provider_factory=lambda client: globals()["OpenAICompletionsProvider"](
                async_client=client
            ),
        ),
        "xiaomi-token-plan-sgp": _ProviderSpec[Any](
            vendor="xiaomi-token-plan-sgp",
            client_factory=lambda credential: openai.AsyncOpenAI(
                api_key=credential.bearer_token,
                base_url="https://token-plan-sgp.xiaomimimo.com/v1",
                **_sdk_timeout_kwargs(),
            ),
            provider_factory=lambda client: globals()["OpenAICompletionsProvider"](
                async_client=client
            ),
        ),
        "openai_codex": _ProviderSpec[str](
            vendor="openai",
            client_factory=lambda credential: credential.bearer_token,
            provider_factory=lambda token: globals()["OpenAICodexResponsesProvider"](
                bearer_token=token
            ),
        ),
        "openai_completions": _ProviderSpec[Any](
            vendor="openai",
            client_factory=lambda credential: openai.AsyncOpenAI(
                api_key=credential.bearer_token,
                **_sdk_timeout_kwargs(),
            ),
            provider_factory=lambda client: globals()["OpenAICompletionsProvider"](
                async_client=client
            ),
        ),
        "openai_responses": _ProviderSpec[Any](
            vendor="openai",
            client_factory=lambda credential: openai.AsyncOpenAI(
                api_key=credential.bearer_token,
                **_sdk_timeout_kwargs(),
            ),
            provider_factory=lambda client: globals()["OpenAIResponsesProvider"](
                async_client=client
            ),
        ),
    }


def get_credential(vendor: str):  # type: ignore[no-untyped-def]
    from wattle.auth import get_credential as real_get_credential

    return real_get_credential(vendor)


def get_api_key_credential(vendor: str):  # type: ignore[no-untyped-def]
    from wattle.auth import get_api_key_credential as real_get_api_key_credential

    return real_get_api_key_credential(vendor)


def get_openai_codex_credential():  # type: ignore[no-untyped-def]
    from wattle.auth import get_openai_codex_credential as real_get_openai_codex_credential

    return real_get_openai_codex_credential()


def AnthropicProvider(*args: Any, **kwargs: Any) -> Any:
    from wattle.providers import AnthropicProvider as real_provider

    return real_provider(*args, **kwargs)


def OpenAICodexResponsesProvider(*args: Any, **kwargs: Any) -> Any:
    from wattle.providers import OpenAICodexResponsesProvider as real_provider

    return real_provider(*args, **kwargs)


def OpenAICompletionsProvider(*args: Any, **kwargs: Any) -> Any:
    from wattle.providers import OpenAICompletionsProvider as real_provider

    return real_provider(*args, **kwargs)


def OpenAIResponsesProvider(*args: Any, **kwargs: Any) -> Any:
    from wattle.providers import OpenAIResponsesProvider as real_provider

    return real_provider(*args, **kwargs)


def _build_provider(provider_name: str) -> Provider:
    """Resolve a provider name to a fully-wired Provider instance."""
    dispatch = _provider_dispatch()
    spec = dispatch.get(provider_name)
    if spec is None:
        raise ValueError(
            f"Unknown provider: {provider_name!r}. "
            f"Choices: {sorted(_PROVIDER_CHOICES)}"
        )
    if provider_name == "openai_codex":
        credential = get_openai_codex_credential()
    elif provider_name in _API_KEY_ONLY_PROVIDERS:
        credential = get_api_key_credential(spec.vendor)
    else:
        credential = get_credential(spec.vendor)
    return spec.build(credential)


def _provider_auth_available(provider_name: str) -> bool:
    vendor = _PROVIDER_TO_VENDOR.get(provider_name)
    if vendor is None:
        return False
    try:
        if provider_name == "openai_codex":
            get_openai_codex_credential()
        elif provider_name in _API_KEY_ONLY_PROVIDERS:
            get_api_key_credential(vendor)
        else:
            get_credential(vendor)
    except (FileNotFoundError, KeyError, ValueError):
        return False
    return True


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def run_agent(*args: Any, **kwargs: Any) -> CompletionResponse:
    from wattle.agent import run_agent as real_run_agent

    return real_run_agent(*args, **kwargs)


def run_agent_with_history(*args: Any, **kwargs: Any) -> AgentRunResultType:
    from wattle.agent import run_agent_with_history as real_run_agent_with_history

    return real_run_agent_with_history(*args, **kwargs)


def _is_text_block(block: object) -> TypeGuard[TextBlock]:
    return getattr(block, "type", None) == "text" and isinstance(getattr(block, "text", None), str)


def _goal_objective_from_print_prompt(prompt: str) -> str | None:
    stripped = prompt.strip()
    if stripped == "/goal":
        return ""
    if stripped.startswith("/goal "):
        return stripped[len("/goal ") :].strip()
    return None


class _HeadlessSessionWriter:
    def __init__(
        self,
        args: argparse.Namespace,
        goal: GoalState | None = None,
    ) -> None:
        from wattle.session import new_session, save_session

        self._save_session = save_session
        self._record = new_session(
            provider=args.provider,
            model=args.model,
            system=None,
            max_tokens=args.max_tokens,
            thinking=bool(getattr(args, "thinking", False)),
            effort=getattr(args, "effort", None),
        )
        if goal is not None:
            self._record = replace(self._record, goal=goal)
        self.path = self._save_session(self._record)

    def update(self, snapshot: object) -> None:
        from wattle.session import SessionSettings

        self._record = replace(
            self._record,
            settings=SessionSettings(
                provider=self._record.settings.provider,
                model=self._record.settings.model,
                system=getattr(snapshot, "system", None),
                max_tokens=self._record.settings.max_tokens,
                thinking=self._record.settings.thinking,
                effort=self._record.settings.effort,
            ),
            messages=list(getattr(snapshot, "messages", [])),
            events=list(getattr(snapshot, "events", [])),
            goal=getattr(snapshot, "goal", self._record.goal),
        )
        self.path = self._save_session(self._record, self.path)

    def finish(self, result: AgentRunResultType) -> Path:
        from wattle.session import SessionSettings

        self._record = replace(
            self._record,
            settings=SessionSettings(
                provider=self._record.settings.provider,
                model=self._record.settings.model,
                system=result.system,
                max_tokens=self._record.settings.max_tokens,
                thinking=self._record.settings.thinking,
                effort=self._record.settings.effort,
            ),
            messages=list(result.messages),
            events=list(result.events),
            goal=result.goal,
        )
        self.path = self._save_session(self._record, self.path)
        return self.path


def _run_headless(args: argparse.Namespace) -> int:
    """Run one prompt and print only the final assistant text."""
    permission_mode = PermissionMode.YOLO
    goal = None
    print_prompt = args.print_prompt
    goal_objective = _goal_objective_from_print_prompt(args.print_prompt)
    if goal_objective is not None:
        if not goal_objective:
            sys.stderr.write("[error] Usage: /goal <objective>\n")
            sys.stderr.flush()
            return 1
        from wattle.goal import build_goal_continuation_prompt, create_goal

        goal = create_goal(goal_objective)
        print_prompt = build_goal_continuation_prompt(goal)

    try:
        if bool(getattr(args, "persist", False)):
            session_writer = _HeadlessSessionWriter(args, goal=goal)
            run_kwargs = {
                "max_tokens": args.max_tokens,
                "permission_mode": permission_mode,
                "thinking": bool(getattr(args, "thinking", False)),
                "effort": getattr(args, "effort", None),
                "on_snapshot": session_writer.update,
            }
            if goal is not None:
                run_kwargs["goal"] = goal
            result = run_agent_with_history(
                args.provider,
                args.model,
                print_prompt,
                **run_kwargs,
            )
            response = result.response
            session_path = session_writer.finish(result)
        else:
            run_kwargs = {
                "max_tokens": args.max_tokens,
                "permission_mode": permission_mode,
                "thinking": bool(getattr(args, "thinking", False)),
                "effort": getattr(args, "effort", None),
            }
            if goal is not None:
                run_kwargs["goal"] = goal
            response = run_agent(
                args.provider,
                args.model,
                print_prompt,
                **run_kwargs,
            )
            session_path = None
    except RuntimeError as exc:
        if _is_attachment_unavailable_error(exc):
            sys.stderr.write(f"[error] {exc}\n")
            sys.stderr.flush()
            return 1
        if not _is_normalized_provider_error(exc):
            raise
        sys.stderr.write(f"[error] {_headless_provider_error_text(exc)}\n")
        sys.stderr.flush()
        return 1

    text = "".join(block.text for block in response.content if _is_text_block(block))
    if text:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
    if session_path is not None:
        sys.stderr.write(f"Saved session: {session_path}\n")
        sys.stderr.flush()
    return 0


def _normalized_provider_error_types() -> tuple[type[BaseException], ...]:
    from wattle.providers import ProviderError, TransientProviderError

    return (ProviderError, TransientProviderError)


def _is_attachment_unavailable_error(error: BaseException) -> bool:
    from wattle.attachments import AttachmentUnavailableError

    return isinstance(error, AttachmentUnavailableError)


def _is_normalized_provider_error(error: BaseException) -> bool:
    if isinstance(error, _normalized_provider_error_types()):
        return True
    return _is_named_provider_error(error, "ProviderError") or _is_named_provider_error(
        error,
        "TransientProviderError",
    )


def _is_named_provider_error(error: BaseException, class_name: str) -> bool:
    return any(
        cls.__module__.startswith("wattle.providers") and cls.__name__ == class_name
        for cls in type(error).__mro__
    )


def _headless_provider_error_text(error: BaseException) -> str:
    from wattle.providers import TransientProviderError

    text = str(error).strip() or type(error).__name__
    if isinstance(error, TransientProviderError) or _is_named_provider_error(
        error,
        "TransientProviderError",
    ):
        return f"Temporary provider error after retries: {text}"
    return text


def _run_tui(args: argparse.Namespace) -> int:
    """Lazy-import wrapper so headless mode doesn't pay the TUI import cost."""
    latest = maybe_latest_update(get_wattle_version())
    if latest is not None and prompt_for_tui_update(get_wattle_version(), latest):
        return 0

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
        "-v",
        "--version",
        action="store_true",
        help="Show version number and exit.",
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Update Wattle to the latest published release and exit.",
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
        choices=sorted(_PROVIDER_CHOICES),
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
            "Catalog model id to use (default: settings.json model, "
            "otherwise the first model of the first authenticated provider)."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Per-turn output token cap (default: the model's max output limit).",
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
        provider_default = globals()["first_catalog_model_choice_for_provider"](args.provider)
        if provider_default is not None:
            args.model = provider_default.model
    if not model_explicit and args.model is None:
        if (
            settings.model is not None
            and globals()["MODEL_CHOICES_BY_MODEL"].get(settings.model) is not None
        ):
            args.model = settings.model
        else:
            default_choice = globals()["first_available_model_choice"]()
            if default_choice is not None:
                args.model = default_choice.model
            if default_choice is not None and not provider_explicit:
                args.provider = default_choice.provider
    if not provider_explicit:
        provider = _default_provider_for_model(args.model) if args.model is not None else None
        if provider is not None:
            args.provider = provider
        elif settings.provider in _PROVIDER_CHOICES:
            args.provider = settings.provider
            if not model_explicit and args.model is None:
                provider_default = globals()["first_catalog_model_choice_for_provider"](
                    settings.provider
                )
                if provider_default is not None:
                    args.model = provider_default.model
        else:
            default_choice = globals()["first_available_model_choice"]()
            if default_choice is not None:
                args.provider = default_choice.provider
            if default_choice is not None and not model_explicit and settings.model is None:
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
    args.compaction_keep_recent_tokens = settings.compaction_keep_recent_tokens


def _validate_catalog_model(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.model is None:
        return
    if globals()["MODEL_CHOICES_BY_MODEL"].get(args.model) is None:
        parser.error(f"unknown model: {args.model}")


def _default_provider_for_model(model: str) -> str | None:
    choice = globals()["MODEL_CHOICES_BY_MODEL"].get(model)
    return choice.provider if choice is not None else None


def load_settings() -> WattleSettings:
    from wattle.settings import load_settings as real_load_settings

    return real_load_settings()


def first_available_model_choice() -> Any:
    from wattle.models import first_available_model_choice as real_choice

    return real_choice()


def first_catalog_model_choice() -> Any:
    from wattle.models import first_catalog_model_choice as real_choice

    return real_choice()


def first_catalog_model_choice_for_provider(provider: str) -> Any:
    from wattle.models import first_catalog_model_choice_for_provider as real_choice

    return real_choice(provider)


MODEL_CHOICES_BY_MODEL = _LazyAttribute("wattle.models", "MODEL_CHOICES_BY_MODEL")


def _has_flag(argv: list[str], flag: str) -> bool:
    return any(item == flag or item.startswith(f"{flag}=") for item in argv)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    if args.version:
        print(get_wattle_version())
        return 0
    if args.upgrade:
        return run_manual_upgrade(get_wattle_version())
    _apply_settings_defaults(args, raw_argv, load_settings())
    if args.print_prompt is not None and args.prompt is not None:
        parser.error("positional prompt cannot be used with -p/--print")
    if args.persist and args.print_prompt is None:
        parser.error("--persist can only be used with -p/--print")
    _validate_catalog_model(args, parser)
    if args.print_prompt is not None and (args.provider is None or args.model is None):
        parser.error(AUTH_REQUIRED_MESSAGE)
    if args.effort is not None:
        args.thinking = True
    if args.print_prompt is not None:
        return _run_headless(args)
    return _run_tui(args)


if __name__ == "__main__":
    raise SystemExit(main())
