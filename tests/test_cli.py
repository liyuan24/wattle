"""Tests for the top-level `wattle` command."""

from __future__ import annotations

import argparse
import io
import sys
from typing import Any
from unittest.mock import patch

import pytest

from wattle import cli
from wattle.auth import AuthCredential
from wattle.providers import CompletionResponse, TextBlock


def test_build_parser_defaults_to_tui_settings() -> None:
    parser = cli._build_parser()
    args = parser.parse_args([])

    assert args.prompt is None
    assert args.provider == "openai_codex"
    assert args.model == "gpt-5.5"
    assert args.max_tokens == 4096
    assert args.thinking is False
    assert args.effort is None
    assert args.print_prompt is None
    assert args.resume is None
    assert args.permission_mode == cli.PermissionMode.YOLO


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
            "--read-only",
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
    assert args.permission_mode == cli.PermissionMode.READ_ONLY


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


def test_build_parser_permission_modes_are_mutually_exclusive() -> None:
    parser = cli._build_parser()

    assert parser.parse_args(["--read-only"]).permission_mode == cli.PermissionMode.READ_ONLY
    assert (
        parser.parse_args(["--ask-for-permission"]).permission_mode
        == cli.PermissionMode.ASK
    )
    assert parser.parse_args(["--yolo", "-p", "prompt"]).permission_mode == cli.PermissionMode.YOLO
    with pytest.raises(SystemExit):
        parser.parse_args(["--yolo", "--read-only"])


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
            permission_mode=cli.PermissionMode.READ_ONLY,
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
                "permission_mode": cli.PermissionMode.READ_ONLY,
                "thinking": False,
                "effort": None,
            },
        )
    ]


def test_headless_rejects_ask_for_permission(monkeypatch: pytest.MonkeyPatch) -> None:
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)

    rc = cli._run_headless(
        argparse.Namespace(
            provider="openai_responses",
            model="gpt-5.5",
            max_tokens=4096,
            permission_mode=cli.PermissionMode.ASK,
            print_prompt="use a tool",
            prompt=None,
        )
    )

    assert rc == 2
    assert "--ask-for-permission" in stderr.getvalue()
