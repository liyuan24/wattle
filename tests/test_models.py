from __future__ import annotations

from wattle import models, request_preparation


def test_model_catalog_includes_context_windows() -> None:
    assert {choice.model: choice.context_window for choice in models.MODEL_CATALOG} == {
        "gpt-5.5": 272_000,
        "gpt-5.4": 272_000,
        "gpt-5.4-mini": 272_000,
        "gpt-5.3-codex": 272_000,
        "gpt-5.3-codex-spark": 128_000,
        "gpt-5.2": 272_000,
        "claude-sonnet-4-6": 1_000_000,
        "claude-opus-4-6": 1_000_000,
        "claude-haiku-4-6": 200_000,
        "deepseek-v4-flash": 1_000_000,
        "deepseek-v4-pro": 1_000_000,
        "kimi-k2.6": 262_144,
        "kimi-k2.5": 262_144,
        "MiniMax-M2.7": 204_800,
        "MiniMax-M2.7-highspeed": 204_800,
    }


def test_model_catalog_includes_source_urls() -> None:
    assert {choice.model: choice.source_url for choice in models.MODEL_CATALOG} == {
        "gpt-5.5": models.OPENAI_MODELS_URL,
        "gpt-5.4": models.OPENAI_MODELS_URL,
        "gpt-5.4-mini": models.OPENAI_MODELS_URL,
        "gpt-5.3-codex": models.OPENAI_MODELS_URL,
        "gpt-5.3-codex-spark": models.OPENAI_MODELS_URL,
        "gpt-5.2": models.OPENAI_MODELS_URL,
        "claude-sonnet-4-6": models.ANTHROPIC_MODELS_URL,
        "claude-opus-4-6": models.ANTHROPIC_MODELS_URL,
        "claude-haiku-4-6": models.ANTHROPIC_MODELS_URL,
        "deepseek-v4-flash": models.DEEPSEEK_MODELS_URL,
        "deepseek-v4-pro": models.DEEPSEEK_MODELS_URL,
        "kimi-k2.6": models.KIMI_MODELS_URL,
        "kimi-k2.5": models.KIMI_MODELS_URL,
        "MiniMax-M2.7": models.MINIMAX_MODELS_URL,
        "MiniMax-M2.7-highspeed": models.MINIMAX_MODELS_URL,
    }
    assert all(choice.source_url.startswith("https://") for choice in models.MODEL_CATALOG)


def test_request_preparation_uses_model_catalog_context_windows() -> None:
    for choice in models.MODEL_CATALOG:
        assert request_preparation.context_window_for_model(choice.model) == choice.context_window


def test_unknown_model_context_window_is_unknown() -> None:
    assert models.context_window_for_model("gpt-future-model") is None
    assert request_preparation.context_window_for_model("gpt-future-model") is None


def test_available_model_choices_includes_openai_compatible_vendors(
    monkeypatch,
) -> None:
    configured = {"deepseek", "kimi", "minimax"}
    monkeypatch.setattr(
        models,
        "has_model_auth",
        lambda choice: choice.vendor in configured,
    )

    choices = models.available_model_choices()
    by_model = {choice.model: choice for choice in choices}

    assert by_model["deepseek-v4-flash"].provider == "deepseek"
    assert by_model["kimi-k2.6"].provider == "kimi"
    assert by_model["MiniMax-M2.7"].provider == "minimax"
    assert "gpt-5.5" not in by_model


def test_render_model_choices_empty_mentions_generic_provider_auth() -> None:
    assert models.render_model_choices([], current_model="x") == (
        "No models available. Add provider auth to ~/.wattle/auth.json."
    )
