from __future__ import annotations

from willow import models


def test_available_model_choices_includes_openai_compatible_vendors(
    monkeypatch,
) -> None:
    configured = {"deepseek", "kimi", "minimax"}
    monkeypatch.setattr(models, "has_vendor_auth", lambda vendor: vendor in configured)

    choices = models.available_model_choices()
    by_model = {choice.model: choice for choice in choices}

    assert by_model["deepseek-v4-flash"].provider == "deepseek"
    assert by_model["kimi-k2.6"].provider == "kimi"
    assert by_model["MiniMax-M2.7"].provider == "minimax"
    assert "gpt-5.5" not in by_model


def test_render_model_choices_empty_mentions_generic_provider_auth() -> None:
    assert models.render_model_choices([], current_model="x") == (
        "No models available. Add provider auth to ~/.willow/auth.json."
    )
