"""Model catalog and selection helpers for Wattle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from wattle.auth import get_credential, get_openai_codex_credential

CODEX_PROVIDER = "openai_codex"
CODEX_DEFAULT_MODEL = "gpt-5.5"

XIAOMI_TOKEN_PLAN_SGP_PROVIDER = "xiaomi-token-plan-sgp"
XIAOMI_DEFAULT_MODEL = "mimo-v2.5-pro"
type EffortLevel = Literal["low", "medium", "high", "xhigh", "max"]
type InputModality = Literal["text", "image"]

DEFAULT_EFFORT_LEVELS: tuple[EffortLevel, ...] = ("low", "medium", "high", "xhigh", "max")
DEFAULT_INPUT_MODALITIES: tuple[InputModality, ...] = ("text", "image")
TEXT_ONLY_MODALITIES: tuple[InputModality, ...] = ("text",)
OPENAI_CODEX_EFFORT_LEVELS: tuple[EffortLevel, ...] = ("low", "medium", "high", "xhigh")
XIAOMI_EFFORT_LEVELS: tuple[EffortLevel, ...] = ("low", "medium", "high")


@dataclass(frozen=True)
class ProviderChoice:
    """One supported provider in default-selection order."""

    provider: str
    vendor: str
    description: str


@dataclass(frozen=True)
class ModelChoice:
    """One selectable model, including the Wattle provider that should run it."""

    model: str
    provider: str
    vendor: str
    description: str
    context_window: int | None = None
    # Vendor-documented per-request output ceiling. None means the vendor does
    # not publish one; resolve_max_tokens falls back to a conservative default.
    max_output_tokens: int | None = None
    source_url: str = ""
    effort_levels: tuple[EffortLevel, ...] = DEFAULT_EFFORT_LEVELS
    supported_modalities: tuple[InputModality, ...] = DEFAULT_INPUT_MODALITIES


# Fallback for models without a documented output ceiling. Large enough that
# thinking output (which shares the budget on Chat Completions vendors) does
# not truncate a turn, small enough to stay under every known vendor limit.
DEFAULT_MAX_OUTPUT_TOKENS = 32_768


OPENAI_MODELS_URL = "https://developers.openai.com/api/docs/models/compare"
ANTHROPIC_MODELS_URL = "https://platform.claude.com/docs/en/about-claude/models/overview"
DEEPSEEK_MODELS_URL = "https://api-docs.deepseek.com/quick_start/pricing"
KIMI_MODELS_URL = "https://platform.kimi.ai/docs/models"
MINIMAX_MODELS_URL = "https://platform.minimax.io/docs/guides/text-generation"
XIAOMI_MODELS_URL = "https://token-plan-sgp.xiaomimimo.com/v1/models"


PROVIDER_CATALOG: tuple[ProviderChoice, ...] = (
    ProviderChoice(
        provider=CODEX_PROVIDER,
        vendor="openai",
        description="ChatGPT Plus/Pro Codex OAuth provider.",
    ),
    ProviderChoice(
        provider="anthropic",
        vendor="anthropic",
        description="Anthropic Claude API provider.",
    ),
    ProviderChoice(
        provider="deepseek",
        vendor="deepseek",
        description="DeepSeek OpenAI-compatible API provider.",
    ),
    ProviderChoice(
        provider="kimi",
        vendor="kimi",
        description="Moonshot Kimi OpenAI-compatible API provider.",
    ),
    ProviderChoice(
        provider="minimax",
        vendor="minimax",
        description="MiniMax OpenAI-compatible API provider.",
    ),
    ProviderChoice(
        provider=XIAOMI_TOKEN_PLAN_SGP_PROVIDER,
        vendor=XIAOMI_TOKEN_PLAN_SGP_PROVIDER,
        description="Xiaomi Token Plan SGP OpenAI-compatible API provider.",
    ),
)


MODEL_CATALOG: tuple[ModelChoice, ...] = (
    ModelChoice(
        model=CODEX_DEFAULT_MODEL,
        provider=CODEX_PROVIDER,
        vendor="openai",
        description="Frontier model for complex coding, research, and real-world work.",
        context_window=272_000,
        max_output_tokens=128_000,
        source_url=OPENAI_MODELS_URL,
        effort_levels=OPENAI_CODEX_EFFORT_LEVELS,
    ),
    ModelChoice(
        model="gpt-5.4",
        provider="openai_codex",
        vendor="openai",
        description="Strong model for everyday coding.",
        context_window=272_000,
        max_output_tokens=128_000,
        source_url=OPENAI_MODELS_URL,
        effort_levels=OPENAI_CODEX_EFFORT_LEVELS,
    ),
    ModelChoice(
        model="gpt-5.4-mini",
        provider="openai_codex",
        vendor="openai",
        description="Small, fast, and cost-efficient model for simpler coding tasks.",
        context_window=272_000,
        source_url=OPENAI_MODELS_URL,
        effort_levels=OPENAI_CODEX_EFFORT_LEVELS,
    ),
    ModelChoice(
        model="gpt-5.3-codex",
        provider="openai_codex",
        vendor="openai",
        description="Coding-optimized model.",
        context_window=272_000,
        source_url=OPENAI_MODELS_URL,
        effort_levels=OPENAI_CODEX_EFFORT_LEVELS,
    ),
    ModelChoice(
        model="gpt-5.3-codex-spark",
        provider="openai_codex",
        vendor="openai",
        description="Ultra-fast coding model.",
        context_window=128_000,
        source_url=OPENAI_MODELS_URL,
        effort_levels=OPENAI_CODEX_EFFORT_LEVELS,
    ),
    ModelChoice(
        model="gpt-5.2",
        provider="openai_codex",
        vendor="openai",
        description="Optimized for professional work and long-running agents.",
        context_window=272_000,
        source_url=OPENAI_MODELS_URL,
        effort_levels=OPENAI_CODEX_EFFORT_LEVELS,
    ),
    ModelChoice(
        model="claude-sonnet-4-6",
        provider="anthropic",
        vendor="anthropic",
        description="Balanced Anthropic model for coding and agentic work.",
        context_window=1_000_000,
        max_output_tokens=64_000,
        source_url=ANTHROPIC_MODELS_URL,
    ),
    ModelChoice(
        model="claude-opus-4-6",
        provider="anthropic",
        vendor="anthropic",
        description="Anthropic model for complex coding and deep reasoning.",
        context_window=1_000_000,
        max_output_tokens=128_000,
        source_url=ANTHROPIC_MODELS_URL,
    ),
    ModelChoice(
        model="claude-haiku-4-6",
        provider="anthropic",
        vendor="anthropic",
        description="Fast Anthropic model for lightweight coding tasks.",
        context_window=200_000,
        source_url=ANTHROPIC_MODELS_URL,
    ),
    ModelChoice(
        model="deepseek-v4-flash",
        provider="deepseek",
        vendor="deepseek",
        description="DeepSeek fast model with 1M context and tool calling.",
        context_window=1_000_000,
        max_output_tokens=384_000,
        source_url=DEEPSEEK_MODELS_URL,
        supported_modalities=TEXT_ONLY_MODALITIES,
    ),
    ModelChoice(
        model="deepseek-v4-pro",
        provider="deepseek",
        vendor="deepseek",
        description="DeepSeek stronger model with 1M context and tool calling.",
        context_window=1_000_000,
        max_output_tokens=384_000,
        source_url=DEEPSEEK_MODELS_URL,
        supported_modalities=TEXT_ONLY_MODALITIES,
    ),
    ModelChoice(
        model="kimi-k2.6",
        provider="kimi",
        vendor="kimi",
        description="Moonshot Kimi flagship model for agentic coding.",
        context_window=262_144,
        source_url=KIMI_MODELS_URL,
    ),
    ModelChoice(
        model="kimi-k2.5",
        provider="kimi",
        vendor="kimi",
        description="Moonshot Kimi long-context multimodal model.",
        context_window=262_144,
        source_url=KIMI_MODELS_URL,
    ),
    ModelChoice(
        model="MiniMax-M2.7",
        provider="minimax",
        vendor="minimax",
        description="MiniMax latest OpenAI-compatible coding model.",
        context_window=204_800,
        source_url=MINIMAX_MODELS_URL,
        supported_modalities=TEXT_ONLY_MODALITIES,
    ),
    ModelChoice(
        model="MiniMax-M2.7-highspeed",
        provider="minimax",
        vendor="minimax",
        description="MiniMax M2.7 optimized for faster output.",
        context_window=204_800,
        source_url=MINIMAX_MODELS_URL,
        supported_modalities=TEXT_ONLY_MODALITIES,
    ),
    ModelChoice(
        model=XIAOMI_DEFAULT_MODEL,
        provider=XIAOMI_TOKEN_PLAN_SGP_PROVIDER,
        vendor=XIAOMI_TOKEN_PLAN_SGP_PROVIDER,
        description="Xiaomi MiMo V2.5 Pro model for coding and agentic work.",
        context_window=1_000_000,
        source_url=XIAOMI_MODELS_URL,
        effort_levels=XIAOMI_EFFORT_LEVELS,
        supported_modalities=TEXT_ONLY_MODALITIES,
    ),
    ModelChoice(
        model="mimo-v2.5",
        provider=XIAOMI_TOKEN_PLAN_SGP_PROVIDER,
        vendor=XIAOMI_TOKEN_PLAN_SGP_PROVIDER,
        description="Xiaomi MiMo V2.5 model.",
        context_window=1_000_000,
        source_url=XIAOMI_MODELS_URL,
        effort_levels=XIAOMI_EFFORT_LEVELS,
    ),
)

MODEL_CHOICES_BY_MODEL: dict[str, ModelChoice] = {
    choice.model: choice for choice in MODEL_CATALOG
}


def has_vendor_auth(vendor: str) -> bool:
    """Return whether auth has a usable credential for ``vendor``."""
    try:
        get_credential(vendor)
    except (FileNotFoundError, KeyError, ValueError):
        return False
    return True


def has_model_auth(choice: ModelChoice) -> bool:
    """Return whether auth can run this model's provider."""
    try:
        if choice.provider == CODEX_PROVIDER:
            get_openai_codex_credential()
        else:
            get_credential(choice.vendor)
    except (FileNotFoundError, KeyError, ValueError):
        return False
    return True


def available_provider_choices() -> list[ProviderChoice]:
    """Return supported providers with configured credentials in catalog order."""
    available_providers = {choice.provider for choice in available_model_choices()}
    return [
        choice for choice in PROVIDER_CATALOG if choice.provider in available_providers
    ]


def available_model_choices() -> list[ModelChoice]:
    """Return catalog entries whose vendors are configured in auth."""
    return [choice for choice in MODEL_CATALOG if has_model_auth(choice)]


def first_catalog_model_choice() -> ModelChoice:
    """Return the first supported model in provider/model catalog order."""
    return MODEL_CATALOG[0]


def first_catalog_model_choice_for_provider(provider: str) -> ModelChoice | None:
    """Return the first supported model for ``provider`` in catalog order."""
    return next((choice for choice in MODEL_CATALOG if choice.provider == provider), None)


def first_available_model_choice() -> ModelChoice | None:
    """Return the first credential-backed supported model in catalog order."""
    choices = available_model_choices()
    return choices[0] if choices else None


def context_window_for_model(model: str) -> int | None:
    """Return the max input/context window for a known model."""
    choice = MODEL_CHOICES_BY_MODEL.get(model)
    if choice is not None:
        return choice.context_window
    return None


def max_output_tokens_for_model(model: str) -> int | None:
    """Return the vendor-documented output ceiling for a known model."""
    choice = MODEL_CHOICES_BY_MODEL.get(model)
    if choice is not None:
        return choice.max_output_tokens
    return None


def resolve_max_tokens(model: str, configured: int | None = None) -> int:
    """Resolve the per-request output cap for ``model``.

    An explicit ``configured`` value wins but is clamped to the model's
    documented ceiling (vendors reject values above it). With no explicit
    value, use the model's ceiling so thinking output — which shares this
    budget on Chat Completions vendors — cannot truncate a turn.
    """
    limit = max_output_tokens_for_model(model)
    if configured is not None:
        return configured if limit is None else min(configured, limit)
    return limit if limit is not None else DEFAULT_MAX_OUTPUT_TOKENS


def effort_levels_for_model(model: str) -> tuple[EffortLevel, ...]:
    """Return the reasoning-effort levels Wattle should expose for ``model``."""
    choice = MODEL_CHOICES_BY_MODEL.get(model)
    if choice is not None:
        return choice.effort_levels
    return DEFAULT_EFFORT_LEVELS


def supported_modalities_for_model(model: str) -> tuple[InputModality, ...]:
    """Return input modalities Wattle should send to ``model``.

    Unknown models keep the compatibility default of text and image, matching
    Codex's behavior for model metadata that predates modality declarations.
    """
    choice = MODEL_CHOICES_BY_MODEL.get(model)
    if choice is not None:
        return choice.supported_modalities
    return DEFAULT_INPUT_MODALITIES


def model_supports_modality(model: str, modality: InputModality) -> bool:
    """Return whether ``model`` accepts the given input modality."""
    return modality in supported_modalities_for_model(model)


def find_model_choice(selector: str, choices: list[ModelChoice]) -> ModelChoice | None:
    """Resolve a model id against ``choices``."""
    return next((choice for choice in choices if choice.model == selector), None)


def render_model_choices(
    choices: list[ModelChoice],
    *,
    current_model: str,
) -> str:
    """Render model choices for slash-command output."""
    if not choices:
        return "No models available. Add provider auth to ~/.wattle/auth.json."

    width = max(
        len(choice.model) + (10 if choice.model == current_model else 0)
        for choice in choices
    )
    lines = ["Select Model", ""]
    for choice in choices:
        label = choice.model
        if choice.model == current_model:
            label += " (current)"
        lines.append(f"  {label:<{width}}  {choice.description}")
    return "\n".join(lines)
