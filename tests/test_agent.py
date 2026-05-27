"""Tests for the high-level `wattle.agent.run_agent` entry point.

All external surfaces are mocked: no real API calls, no SDK construction,
no auth-file reads. We're verifying the wiring between provider name ->
vendor -> credential -> SDK client -> Provider wrapper -> loop.run.
"""

from __future__ import annotations

import sys
from collections.abc import Generator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

# If the auth module hasn't landed yet (parallel branch), inject a stub so the
# `from wattle.auth import get_credential` at the top of `wattle.agent` succeeds
# at import time. Once real auth lands, the real module is used and this stub
# is never installed (so the auth tests' own contract — that `import wattle`
# does not eagerly bind `wattle.auth` — is unaffected by us).
if "wattle.auth" not in sys.modules:
    try:
        import wattle.auth  # noqa: F401  # real module preferred when available
    except ImportError:
        sys.modules["wattle.auth"] = SimpleNamespace(  # type: ignore[assignment]
            AUTH_PATH=None,
            AuthCredential=SimpleNamespace,
            load_auth=lambda: {},
            get_credential=lambda vendor: SimpleNamespace(
                kind="api_key",
                bearer_token=f"fake-{vendor}-key",
                source="stub",
                expires_at=None,
            ),
            get_api_key_credential=lambda vendor: SimpleNamespace(
                kind="api_key",
                bearer_token=f"fake-{vendor}-key",
                source="stub",
                expires_at=None,
            ),
            get_openai_codex_credential=lambda: SimpleNamespace(
                kind="oauth",
                bearer_token="fake-openai-codex-token",
                source="stub",
                expires_at=None,
            ),
        )

import pytest

from wattle import agent
from wattle.agent import PROVIDER_TO_VENDOR, run_agent
from wattle.auth import AuthCredential
from wattle.providers import Message, TextBlock


@pytest.fixture
def loop_run_sentinel() -> Generator[AsyncMock, None, None]:
    """A patched loop.arun that returns a sentinel CompletionResponse."""
    sentinel = object()
    with (
        patch.object(agent.loop, "arun", new_callable=AsyncMock, return_value=sentinel) as mock_run,
        patch.object(agent, "build_system_prompt", return_value="built system"),
    ):
        mock_run.sentinel = sentinel  # type: ignore[attr-defined]
        yield mock_run


def test_provider_to_vendor_mapping_is_exact() -> None:
    assert PROVIDER_TO_VENDOR == {
        "anthropic": "anthropic",
        "deepseek": "deepseek",
        "kimi": "kimi",
        "minimax": "minimax",
        "xiaomi-token-plan-sgp": "xiaomi-token-plan-sgp",
        "openai_codex": "openai",
        "openai_completions": "openai",
        "openai_responses": "openai",
    }


def test_anthropic_provider_wires_anthropic_vendor_key(
    loop_run_sentinel: MagicMock,
) -> None:
    fake_client = object()
    fake_provider = object()

    with (
        patch.object(
            agent,
            "get_credential",
            return_value=AuthCredential(
                kind="api_key",
                bearer_token="fake-anthropic-key",
                source="test",
            ),
        ) as gc,
        patch.object(agent.anthropic, "AsyncAnthropic", return_value=fake_client) as anth,
        patch.object(agent.openai, "AsyncOpenAI") as oa,
        patch.object(agent, "AnthropicProvider", return_value=fake_provider) as ap,
        patch.object(agent, "OpenAICodexResponsesProvider") as xp,
        patch.object(agent, "OpenAICompletionsProvider") as cp,
        patch.object(agent, "OpenAIResponsesProvider") as rp,
    ):
        result = run_agent(
            "anthropic",
            model="claude-x",
            user_input="hi",
            max_tokens=512,
        )

    gc.assert_called_once_with("anthropic")
    anth.assert_called_once_with(api_key="fake-anthropic-key")
    oa.assert_not_called()
    ap.assert_called_once_with(async_client=fake_client)
    xp.assert_not_called()
    cp.assert_not_called()
    rp.assert_not_called()

    loop_run_sentinel.assert_called_once_with(
        fake_provider,
        agent.TOOLS_BY_NAME,
        "built system",
        "hi",
        "claude-x",
        512,
        permission_gate=None,
        thinking=False,
        effort=None,
    )
    assert result is loop_run_sentinel.sentinel


def test_text_only_headless_agent_passes_full_tool_map_to_loop() -> None:
    fake_provider = object()
    with (
        patch.object(
            agent,
            "get_credential",
            return_value=AuthCredential(
                kind="api_key",
                bearer_token="fake-deepseek-key",
                source="test",
            ),
        ),
        patch.object(agent.openai, "AsyncOpenAI", return_value=object()),
        patch.object(agent, "OpenAICompletionsProvider", return_value=fake_provider),
        patch.object(agent.loop, "arun", new_callable=AsyncMock, return_value=object()) as mock_run,
        patch.object(agent, "build_system_prompt", return_value="built system") as mock_system,
    ):
        run_agent("deepseek", model="deepseek-v4-flash", user_input="hi")

    assert mock_run.call_args.args[1] is agent.TOOLS_BY_NAME
    assert "view_image" in mock_run.call_args.args[1]
    system_tools = mock_system.call_args.kwargs["tools_by_name"]
    assert "view_image" not in system_tools


def test_openai_codex_provider_wires_openai_oauth_bearer(
    loop_run_sentinel: MagicMock,
) -> None:
    fake_provider = object()

    with (
        patch.object(
            agent,
            "get_openai_codex_credential",
            return_value=AuthCredential(
                kind="oauth",
                bearer_token="fake-openai-bearer",
                source="test",
            ),
        ) as gc,
        patch.object(agent.anthropic, "AsyncAnthropic") as anth,
        patch.object(agent.openai, "AsyncOpenAI") as oa,
        patch.object(agent, "AnthropicProvider") as ap,
        patch.object(
            agent, "OpenAICodexResponsesProvider", return_value=fake_provider
        ) as xp,
        patch.object(agent, "OpenAICompletionsProvider") as cp,
        patch.object(agent, "OpenAIResponsesProvider") as rp,
    ):
        result = run_agent("openai_codex", model="gpt-y", user_input="yo")

    gc.assert_called_once_with()
    anth.assert_not_called()
    oa.assert_not_called()
    ap.assert_not_called()
    xp.assert_called_once_with(bearer_token="fake-openai-bearer")
    cp.assert_not_called()
    rp.assert_not_called()

    assert result is loop_run_sentinel.sentinel


def test_openai_completions_provider_wires_openai_vendor_key(
    loop_run_sentinel: MagicMock,
) -> None:
    fake_client = object()
    fake_provider = object()

    with (
        patch.object(
            agent,
            "get_api_key_credential",
            return_value=AuthCredential(
                kind="api_key",
                bearer_token="fake-openai-bearer",
                source="test",
            ),
        ) as gc,
        patch.object(agent, "get_credential") as generic_gc,
        patch.object(agent.anthropic, "AsyncAnthropic") as anth,
        patch.object(agent.openai, "AsyncOpenAI", return_value=fake_client) as oa,
        patch.object(agent, "AnthropicProvider") as ap,
        patch.object(agent, "OpenAICodexResponsesProvider") as xp,
        patch.object(agent, "OpenAICompletionsProvider", return_value=fake_provider) as cp,
        patch.object(agent, "OpenAIResponsesProvider") as rp,
    ):
        result = run_agent("openai_completions", model="gpt-x", user_input="hello")

    gc.assert_called_once_with("openai")
    generic_gc.assert_not_called()
    oa.assert_called_once_with(api_key="fake-openai-bearer")
    anth.assert_not_called()
    xp.assert_not_called()
    cp.assert_called_once_with(async_client=fake_client)
    ap.assert_not_called()
    rp.assert_not_called()

    # Defaults: system=None, max_tokens=4096.
    loop_run_sentinel.assert_called_once_with(
        fake_provider,
        agent.TOOLS_BY_NAME,
        "built system",
        "hello",
        "gpt-x",
        4096,
        permission_gate=None,
        thinking=False,
        effort=None,
    )
    assert result is loop_run_sentinel.sentinel


@pytest.mark.parametrize(
    ("provider_name", "vendor", "base_url", "model"),
    [
        ("deepseek", "deepseek", "https://api.deepseek.com", "deepseek-v4-flash"),
        ("kimi", "kimi", "https://api.moonshot.ai/v1", "kimi-k2.6"),
        ("minimax", "minimax", "https://api.minimax.io/v1", "MiniMax-M2.7"),
    ],
)
def test_openai_compatible_providers_wire_custom_vendor_and_base_url(
    provider_name: str,
    vendor: str,
    base_url: str,
    model: str,
    loop_run_sentinel: MagicMock,
) -> None:
    fake_client = object()
    fake_provider = object()

    with (
        patch.object(
            agent,
            "get_credential",
            return_value=AuthCredential(
                kind="api_key",
                bearer_token=f"fake-{vendor}-key",
                source="test",
            ),
        ) as gc,
        patch.object(agent.anthropic, "AsyncAnthropic") as anth,
        patch.object(agent.openai, "AsyncOpenAI", return_value=fake_client) as oa,
        patch.object(agent, "AnthropicProvider") as ap,
        patch.object(agent, "OpenAICodexResponsesProvider") as xp,
        patch.object(
            agent, "OpenAICompletionsProvider", return_value=fake_provider
        ) as cp,
        patch.object(agent, "OpenAIResponsesProvider") as rp,
    ):
        result = run_agent(provider_name, model=model, user_input="hello")

    gc.assert_called_once_with(vendor)
    oa.assert_called_once_with(api_key=f"fake-{vendor}-key", base_url=base_url)
    cp.assert_called_once_with(async_client=fake_client)
    anth.assert_not_called()
    ap.assert_not_called()
    xp.assert_not_called()
    rp.assert_not_called()
    assert result is loop_run_sentinel.sentinel


def test_openai_responses_provider_wires_openai_vendor_key(
    loop_run_sentinel: MagicMock,
) -> None:
    fake_client = object()
    fake_provider = object()

    with (
        patch.object(
            agent,
            "get_api_key_credential",
            return_value=AuthCredential(
                kind="api_key",
                bearer_token="fake-openai-bearer",
                source="test",
            ),
        ) as gc,
        patch.object(agent, "get_credential") as generic_gc,
        patch.object(agent.anthropic, "AsyncAnthropic") as anth,
        patch.object(agent.openai, "AsyncOpenAI", return_value=fake_client) as oa,
        patch.object(agent, "AnthropicProvider") as ap,
        patch.object(agent, "OpenAICodexResponsesProvider") as xp,
        patch.object(agent, "OpenAICompletionsProvider") as cp,
        patch.object(agent, "OpenAIResponsesProvider", return_value=fake_provider) as rp,
    ):
        result = run_agent("openai_responses", model="gpt-y", user_input="yo")

    gc.assert_called_once_with("openai")
    generic_gc.assert_not_called()
    oa.assert_called_once_with(api_key="fake-openai-bearer")
    anth.assert_not_called()
    xp.assert_not_called()
    rp.assert_called_once_with(async_client=fake_client)
    ap.assert_not_called()
    cp.assert_not_called()

    assert result is loop_run_sentinel.sentinel


def test_run_agent_with_history_returns_response_messages_and_system() -> None:
    response = object()
    captured_messages = [Message(role="user", content=[TextBlock(text="hi")])]

    def fake_loop_run(*args: object, **kwargs: object) -> object:
        messages_out = kwargs["messages_out"]
        messages_out[:] = captured_messages  # type: ignore[index]
        return response

    fake_provider = object()
    with (
        patch.object(
            agent,
            "get_credential",
            return_value=AuthCredential(
                kind="api_key",
                bearer_token="fake-anthropic-key",
                source="test",
            ),
        ),
        patch.object(agent.anthropic, "AsyncAnthropic", return_value=object()),
        patch.object(agent, "AnthropicProvider", return_value=fake_provider),
        patch.object(agent, "build_system_prompt", return_value="built system"),
        patch.object(
            agent.loop,
            "arun",
            new_callable=AsyncMock,
            side_effect=fake_loop_run,
        ) as mock_run,
    ):
        result = agent.run_agent_with_history(
            "anthropic",
            model="claude-x",
            user_input="hi",
            max_tokens=512,
        )

    assert result.response is response
    assert result.messages == captured_messages
    assert result.system == "built system"
    assert mock_run.call_args.kwargs["messages_out"] == captured_messages


def test_xiaomi_token_plan_provider_wires_api_key_and_base_url(
    loop_run_sentinel: MagicMock,
) -> None:
    fake_client = object()
    fake_provider = object()

    with (
        patch.object(
            agent,
            "get_api_key_credential",
            return_value=AuthCredential(
                kind="api_key",
                bearer_token="fake-xiaomi-key",
                source="test",
            ),
        ) as gc,
        patch.object(agent, "get_credential") as generic_gc,
        patch.object(agent.anthropic, "AsyncAnthropic") as anth,
        patch.object(agent.openai, "AsyncOpenAI", return_value=fake_client) as oa,
        patch.object(agent, "AnthropicProvider") as ap,
        patch.object(agent, "OpenAICodexResponsesProvider") as xp,
        patch.object(
            agent, "OpenAICompletionsProvider", return_value=fake_provider
        ) as cp,
        patch.object(agent, "OpenAIResponsesProvider") as rp,
    ):
        result = run_agent(
            "xiaomi-token-plan-sgp",
            model="gpt-5.5",
            user_input="hello",
        )

    gc.assert_called_once_with("xiaomi-token-plan-sgp")
    generic_gc.assert_not_called()
    oa.assert_called_once_with(
        api_key="fake-xiaomi-key",
        base_url="https://token-plan-sgp.xiaomimimo.com/v1",
    )
    cp.assert_called_once_with(async_client=fake_client)
    anth.assert_not_called()
    ap.assert_not_called()
    xp.assert_not_called()
    rp.assert_not_called()
    assert result is loop_run_sentinel.sentinel


def test_unknown_provider_raises_valueerror_with_helpful_message() -> None:
    with pytest.raises(ValueError) as excinfo:
        run_agent("anthropik", model="x", user_input="hi")  # typo

    msg = str(excinfo.value)
    assert "anthropik" in msg
    # All valid choices are listed, sorted.
    for name in (
        "anthropic",
        "deepseek",
        "kimi",
        "minimax",
        "xiaomi-token-plan-sgp",
        "openai_codex",
        "openai_completions",
        "openai_responses",
    ):
        assert name in msg


def test_auth_keyerror_propagates(loop_run_sentinel: MagicMock) -> None:
    with (
        patch.object(
            agent, "get_credential", side_effect=KeyError("no anthropic entry")
        ),
        patch.object(agent.anthropic, "AsyncAnthropic") as anth,
        patch.object(agent, "AnthropicProvider") as ap,
        pytest.raises(KeyError, match="no anthropic entry"),
    ):
        run_agent("anthropic", model="m", user_input="u")

    anth.assert_not_called()
    ap.assert_not_called()
    loop_run_sentinel.assert_not_called()


def test_auth_filenotfounderror_propagates(loop_run_sentinel: MagicMock) -> None:
    with (
        patch.object(
            agent,
            "get_api_key_credential",
            side_effect=FileNotFoundError("missing auth.json"),
        ),
        patch.object(agent.openai, "AsyncOpenAI") as oa,
        patch.object(agent, "OpenAICompletionsProvider") as cp,
        pytest.raises(FileNotFoundError, match="missing auth.json"),
    ):
        run_agent("openai_completions", model="m", user_input="u")

    oa.assert_not_called()
    cp.assert_not_called()
    loop_run_sentinel.assert_not_called()
