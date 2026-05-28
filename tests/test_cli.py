"""Tests for the top-level `wattle` command."""

from __future__ import annotations

import argparse
import io
import sys
from typing import Any
from unittest.mock import patch

import pytest

from wattle import cli, models
from wattle.auth import AuthCredential
from wattle.permissions import PermissionMode
from wattle.providers import CompletionResponse, Message, TextBlock
from wattle.session import load_session
from wattle.settings import TuiSettings, WattleSettings


def test_build_parser_defaults_to_tui_settings() -> None:
    parser = cli._build_parser()
    args = parser.parse_args([])

    assert args.prompt is None
    assert args.provider is None
    assert args.model is None
    assert args.max_tokens == 4096
    assert args.thinking is False
    assert args.effort is None
    assert args.print_prompt is None
    assert args.resume is None
    assert args.persist is False
    assert args.permission_mode == cli.PermissionMode.YOLO


def test_apply_settings_defaults_when_cli_flags_are_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = cli._build_parser()
    args = parser.parse_args([])
    monkeypatch.setattr(cli, "_provider_auth_available", lambda _provider: True)

    cli._apply_settings_defaults(
        args,
        [],
        WattleSettings(
            provider="openai_responses",
            model="custom-platform-model",
            max_tokens=1234,
            thinking=True,
            effort="high",
            permission_mode=PermissionMode.YOLO,
            statusline=False,
            tui=TuiSettings(statusline=()),
            enabled_models=("gpt-5.4",),
            compaction_keep_recent_tokens=5000,
        ),
    )

    assert args.provider == "openai_responses"
    assert args.model == "custom-platform-model"
    assert args.max_tokens == 1234
    assert args.thinking is True
    assert args.effort == "high"
    assert args.permission_mode == PermissionMode.YOLO
    assert args.statusline is False
    assert args.statusline_fields == ()
    assert args.enabled_models == ("gpt-5.4",)
    assert args.compaction_keep_recent_tokens == 5000


def test_apply_settings_defaults_prefers_catalog_provider_for_known_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = cli._build_parser()
    args = parser.parse_args([])
    monkeypatch.setattr(cli, "_provider_auth_available", lambda _provider: True)

    cli._apply_settings_defaults(
        args,
        [],
        WattleSettings(provider="openai_responses", model="gpt-5.5"),
    )

    assert args.provider == "openai_codex"
    assert args.model == "gpt-5.5"


def test_apply_settings_defaults_uses_saved_provider_for_unknown_saved_model() -> None:
    parser = cli._build_parser()
    args = parser.parse_args([])

    cli._apply_settings_defaults(
        args,
        [],
        WattleSettings(provider="openai_responses", model="custom-platform-model"),
    )

    assert args.provider == "openai_responses"
    assert args.model == "custom-platform-model"


def test_apply_settings_defaults_falls_back_to_first_available_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = cli._build_parser()
    args = parser.parse_args([])
    monkeypatch.setattr(
        cli,
        "first_available_model_choice",
        lambda: models.ModelChoice(
            model="mimo-v2.5-pro",
            provider="xiaomi-token-plan-sgp",
            vendor="xiaomi-token-plan-sgp",
            description="Xiaomi MiMo V2.5 Pro model.",
        ),
    )

    cli._apply_settings_defaults(
        args,
        [],
        WattleSettings(),
    )

    assert args.provider == "xiaomi-token-plan-sgp"
    assert args.model == "mimo-v2.5-pro"


def test_apply_settings_defaults_keeps_explicit_model_when_provider_has_no_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["--model", "gpt-5.5"])
    monkeypatch.setattr(
        cli,
        "first_available_model_choice",
        lambda: models.ModelChoice(
            model="mimo-v2.5-pro",
            provider="xiaomi-token-plan-sgp",
            vendor="xiaomi-token-plan-sgp",
            description="Xiaomi MiMo V2.5 Pro model.",
        ),
    )

    cli._apply_settings_defaults(
        args,
        ["--model", "gpt-5.5"],
        WattleSettings(provider="openai_codex", model="gpt-5.5"),
    )

    assert args.provider == "openai_codex"
    assert args.model == "gpt-5.5"


def test_apply_settings_defaults_uses_first_catalog_model_for_explicit_provider() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["--provider", "xiaomi-token-plan-sgp"])

    cli._apply_settings_defaults(
        args,
        ["--provider", "xiaomi-token-plan-sgp"],
        WattleSettings(),
    )

    assert args.provider == "xiaomi-token-plan-sgp"
    assert args.model == "mimo-v2.5-pro"


def test_apply_settings_defaults_falls_back_to_first_catalog_model_without_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = cli._build_parser()
    args = parser.parse_args([])
    monkeypatch.setattr(cli, "first_available_model_choice", lambda: None)

    cli._apply_settings_defaults(args, [], WattleSettings())

    assert args.provider == "openai_codex"
    assert args.model == "gpt-5.5"


def test_build_parser_print_mode_accepts_prompt_and_shared_flags() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "--provider",
            "anthropic",
            "--model",
            "claude-sonnet-4-6",
            "--max-tokens",
            "512",
            "--thinking",
            "--effort",
            "high",
            "--yolo",
            "--persist",
            "-p",
            "follow the prompt",
        ]
    )

    assert args.print_prompt == "follow the prompt"
    assert args.prompt is None
    assert args.provider == "anthropic"
    assert args.model == "claude-sonnet-4-6"
    assert args.max_tokens == 512
    assert args.thinking is True
    assert args.effort == "high"
    assert args.permission_mode == cli.PermissionMode.YOLO
    assert args.persist is True


def test_build_parser_resume_accepts_optional_session() -> None:
    parser = cli._build_parser()

    choose_args = parser.parse_args(["--resume"])
    direct_args = parser.parse_args(["--resume", "sess_123"])
    short_args = parser.parse_args(["-r", "sess_456"])

    assert choose_args.resume == ""
    assert direct_args.resume == "sess_123"
    assert short_args.resume == "sess_456"


def test_build_parser_accepts_positional_prompt_for_tui() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(["do the task"])

    assert args.prompt == "do the task"
    assert args.print_prompt is None


def test_build_parser_rejects_extra_positionals() -> None:
    parser = cli._build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["headless", "prompt"])


def test_main_rejects_print_prompt_with_positional_prompt() -> None:
    with pytest.raises(SystemExit):
        cli.main(["-p", "headless", "interactive"])


def test_build_parser_only_supports_yolo_permission_mode() -> None:
    parser = cli._build_parser()

    assert parser.parse_args(["--yolo", "-p", "prompt"]).permission_mode == cli.PermissionMode.YOLO
    with pytest.raises(SystemExit):
        parser.parse_args(["--read-only"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--ask-for-permission"])


def test_main_without_prompt_runs_tui(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[argparse.Namespace] = []

    def fake_run_tui(args: argparse.Namespace) -> int:
        calls.append(args)
        return 42

    monkeypatch.setattr(cli, "_run_tui", fake_run_tui)
    monkeypatch.setattr(cli, "_run_headless", lambda _args: pytest.fail("headless called"))

    assert cli.main([]) == 42
    assert len(calls) == 1
    assert calls[0].prompt is None


def test_main_with_print_prompt_runs_headless(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[argparse.Namespace] = []

    def fake_run_headless(args: argparse.Namespace) -> int:
        calls.append(args)
        return 7

    monkeypatch.setattr(cli, "_run_tui", lambda _args: pytest.fail("tui called"))
    monkeypatch.setattr(cli, "_run_headless", fake_run_headless)

    assert cli.main(["-p", "follow the prompt"]) == 7
    assert len(calls) == 1
    assert calls[0].print_prompt == "follow the prompt"


def test_main_with_positional_prompt_runs_tui(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[argparse.Namespace] = []

    def fake_run_tui(args: argparse.Namespace) -> int:
        calls.append(args)
        return 42

    monkeypatch.setattr(cli, "_run_tui", fake_run_tui)
    monkeypatch.setattr(cli, "_run_headless", lambda _args: pytest.fail("headless called"))

    assert cli.main(["follow the prompt"]) == 42
    assert len(calls) == 1
    assert calls[0].prompt == "follow the prompt"


@pytest.mark.parametrize(
    ("provider_name", "vendor", "base_url"),
    [
        ("deepseek", "deepseek", "https://api.deepseek.com"),
        ("kimi", "kimi", "https://api.moonshot.ai/v1"),
        ("minimax", "minimax", "https://api.minimax.io/v1"),
    ],
)
def test_build_provider_wires_openai_compatible_base_url(
    provider_name: str,
    vendor: str,
    base_url: str,
) -> None:
    fake_client = object()
    fake_provider = object()

    with (
        patch.object(
            cli,
            "get_credential",
            return_value=AuthCredential(
                kind="api_key",
                bearer_token=f"fake-{vendor}-key",
                source="test",
            ),
        ) as gc,
        patch.object(cli.openai, "AsyncOpenAI", return_value=fake_client) as openai_client,
        patch.object(
            cli, "OpenAICompletionsProvider", return_value=fake_provider
        ) as provider_factory,
    ):
        provider = cli._build_provider(provider_name)

    gc.assert_called_once_with(vendor)
    openai_client.assert_called_once_with(
        api_key=f"fake-{vendor}-key",
        base_url=base_url,
    )
    provider_factory.assert_called_once_with(async_client=fake_client)
    assert provider is fake_provider


def test_build_provider_wires_openai_codex_credential() -> None:
    fake_provider = object()

    with (
        patch.object(
            cli,
            "get_openai_codex_credential",
            return_value=AuthCredential(
                kind="oauth",
                bearer_token="fake-codex-token",
                source="test",
            ),
        ) as gc,
        patch.object(cli, "get_credential") as generic_gc,
        patch.object(
            cli, "OpenAICodexResponsesProvider", return_value=fake_provider
        ) as provider_factory,
    ):
        provider = cli._build_provider("openai_codex")

    gc.assert_called_once_with()
    generic_gc.assert_not_called()
    provider_factory.assert_called_once_with(bearer_token="fake-codex-token")
    assert provider is fake_provider


def test_build_provider_wires_openai_responses_api_key() -> None:
    fake_client = object()
    fake_provider = object()

    with (
        patch.object(
            cli,
            "get_api_key_credential",
            return_value=AuthCredential(
                kind="api_key",
                bearer_token="fake-openai-key",
                source="test",
            ),
        ) as gc,
        patch.object(cli, "get_credential") as generic_gc,
        patch.object(cli.openai, "AsyncOpenAI", return_value=fake_client) as oa,
        patch.object(
            cli, "OpenAIResponsesProvider", return_value=fake_provider
        ) as provider_factory,
    ):
        provider = cli._build_provider("openai_responses")

    gc.assert_called_once_with("openai")
    generic_gc.assert_not_called()
    oa.assert_called_once_with(api_key="fake-openai-key")
    provider_factory.assert_called_once_with(async_client=fake_client)
    assert provider is fake_provider


def test_build_provider_wires_xiaomi_token_plan_api_key() -> None:
    fake_client = object()
    fake_provider = object()

    with (
        patch.object(
            cli,
            "get_api_key_credential",
            return_value=AuthCredential(
                kind="api_key",
                bearer_token="fake-xiaomi-key",
                source="test",
            ),
        ) as gc,
        patch.object(cli, "get_credential") as generic_gc,
        patch.object(cli.openai, "AsyncOpenAI", return_value=fake_client) as openai_client,
        patch.object(
            cli, "OpenAICompletionsProvider", return_value=fake_provider
        ) as provider_factory,
    ):
        provider = cli._build_provider("xiaomi-token-plan-sgp")

    gc.assert_called_once_with("xiaomi-token-plan-sgp")
    generic_gc.assert_not_called()
    openai_client.assert_called_once_with(
        api_key="fake-xiaomi-key",
        base_url="https://token-plan-sgp.xiaomimimo.com/v1",
    )
    provider_factory.assert_called_once_with(async_client=fake_client)
    assert provider is fake_provider


def test_headless_calls_run_agent_and_prints_final_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = CompletionResponse(
        content=[TextBlock(text="hello"), TextBlock(text=" world")],
        stop_reason="end_turn",
    )
    calls: list[tuple[Any, ...]] = []

    def fake_run_agent(*args: Any, **kwargs: Any) -> CompletionResponse:
        calls.append((args, kwargs))
        return response

    stdout = io.StringIO()
    monkeypatch.setattr(cli, "run_agent", fake_run_agent)
    monkeypatch.setattr(sys, "stdout", stdout)

    rc = cli._run_headless(
        argparse.Namespace(
            provider="anthropic",
            model="claude-sonnet-4-6",
            max_tokens=512,
            permission_mode=cli.PermissionMode.YOLO,
            print_prompt="follow the prompt",
            prompt=None,
        )
    )

    assert rc == 0
    assert stdout.getvalue() == "hello world\n"
    assert calls == [
        (
            ("anthropic", "claude-sonnet-4-6", "follow the prompt"),
            {
                "max_tokens": 512,
                "permission_mode": cli.PermissionMode.YOLO,
                "thinking": False,
                "effort": None,
            },
        )
    ]


def test_main_rejects_persist_without_print_prompt() -> None:
    with pytest.raises(SystemExit):
        cli.main(["--persist"])


def test_headless_with_persist_saves_session_and_prints_final_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    response = CompletionResponse(
        content=[TextBlock(text="saved response")],
        stop_reason="end_turn",
    )
    messages = [
        Message(role="user", content=[TextBlock(text="persist this")]),
        Message(role="assistant", content=[TextBlock(text="saved response")]),
    ]
    calls: list[tuple[Any, ...]] = []

    def fake_run_agent_with_history(*args: Any, **kwargs: Any) -> cli.AgentRunResult:
        calls.append((args, kwargs))
        return cli.AgentRunResult(response=response, messages=messages, system="system prompt")

    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setenv("WATTLE_SESSION_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "run_agent", lambda *args, **kwargs: pytest.fail("run_agent called"))
    monkeypatch.setattr(cli, "run_agent_with_history", fake_run_agent_with_history)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    rc = cli._run_headless(
        argparse.Namespace(
            provider="anthropic",
            model="claude-sonnet-4-6",
            max_tokens=512,
            permission_mode=cli.PermissionMode.YOLO,
            print_prompt="persist this",
            prompt=None,
            persist=True,
            thinking=True,
            effort="high",
        )
    )

    assert rc == 0
    assert stdout.getvalue() == "saved response\n"
    assert "Saved session:" in stderr.getvalue()
    assert calls == [
        (
            ("anthropic", "claude-sonnet-4-6", "persist this"),
            {
                "max_tokens": 512,
                "permission_mode": cli.PermissionMode.YOLO,
                "thinking": True,
                "effort": "high",
            },
        )
    ]

    session_files = list(tmp_path.glob("*.jsonl"))
    assert len(session_files) == 1
    record = load_session(session_files[0])
    assert record.settings.provider == "anthropic"
    assert record.settings.model == "claude-sonnet-4-6"
    assert record.settings.system == "system prompt"
    assert record.settings.max_tokens == 512
    assert record.settings.thinking is True
    assert record.settings.effort == "high"
    assert [message.role for message in record.messages] == ["user", "assistant"]
