# Providers and Models

Wattle has a small provider abstraction and a practical model catalog. The TUI only lists model choices whose credentials are configured.

## Built-in providers

Provider names accepted by `--provider`:

- `openai_codex` - OpenAI Codex OAuth through the Responses API. This is the default provider.
- `openai_responses` - OpenAI Responses API with an API key.
- `openai_completions` - OpenAI Chat Completions API with an API key.
- `anthropic` - Anthropic Messages API.
- `deepseek` - DeepSeek OpenAI-compatible API.
- `kimi` - Moonshot Kimi OpenAI-compatible API.
- `minimax` - MiniMax OpenAI-compatible API.
- `xiaomi-token-plan-sgp` - Xiaomi Token Plan SGP OpenAI-compatible API.

Example:

```bash
wattle --provider anthropic --model claude-sonnet-4-6
```

## Auth file

Wattle reads `~/.wattle/auth.json`.

```json
{
  "anthropic": {
    "api_key": {"api_key": "sk-ant-..."},
    "oauth": {
      "access_token": "...",
      "refresh_token": "...",
      "client_id": "...",
      "expires_at": 1779133973
    }
  },
  "deepseek": {"api_key": {"api_key": "sk-..."}},
  "kimi": {"api_key": {"api_key": "sk-..."}},
  "minimax": {"api_key": {"api_key": "sk-..."}},
  "xiaomi-token-plan-sgp": {"api_key": {"api_key": "tp-..."}},
  "openai": {
    "oauth": {
      "access_token": "...",
      "refresh_token": "...",
      "token_url": "https://auth.openai.com/oauth/token",
      "client_id": "...",
      "expires_at": 1779133973
    },
    "api_key": {"api_key": "sk-..."}
  }
}
```

OpenAI Codex uses the `openai.oauth` credential. `openai_responses` and `openai_completions` use `openai.api_key`. The `anthropic` provider accepts either `anthropic.api_key` or `anthropic.oauth`; if both are present, OAuth is used by default. Refreshable `anthropic.oauth` credentials must include the OAuth `client_id` alongside `refresh_token`; Wattle provides Anthropic's token URL and beta header defaults, but Anthropic does not have a Wattle-wide public client ID default.

## Environment variables

API-key providers can also read credentials from environment variables:

| Provider | Vendor entry | Environment variable |
| --- | --- | --- |
| `openai_responses` | `openai.api_key` | `OPENAI_API_KEY` |
| `openai_completions` | `openai.api_key` | `OPENAI_API_KEY` |
| `anthropic` | `anthropic.api_key` or `anthropic.oauth` | `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` |
| `deepseek` | `deepseek.api_key` | `DEEPSEEK_API_KEY` |
| `kimi` | `kimi.api_key` | `KIMI_API_KEY` |
| `minimax` | `minimax.api_key` | `MINIMAX_API_KEY` |
| `xiaomi-token-plan-sgp` | `xiaomi-token-plan-sgp.api_key` | `XIAOMI_TOKEN_PLAN_SGP_API_KEY` |

`openai_codex` uses OAuth and does not use `OPENAI_API_KEY`. `ANTHROPIC_AUTH_TOKEN` is treated as an OAuth bearer token for Claude and is only used by generic Anthropic credential lookup; API-key-only lookup still requires `ANTHROPIC_API_KEY`. Because environment OAuth tokens cannot be refreshed by Wattle, an `ANTHROPIC_AUTH_TOKEN` with JWT expiry metadata is rejected when expired or too close to expiry.

## Login

Use `/login` in the TUI and follow the provider instructions:

```text
/login
```

Wattle supports OAuth and supported API-key provider login from the TUI. It saves credentials to `~/.wattle/auth.json`. API-key providers can also use the environment variables above.

## Priority rules

Provider and model selection:

- CLI flags win first: `--provider` and `--model`.
- If no CLI flag is set, Wattle reads `provider` and `model` from `~/.wattle/settings.json`.
- If only a catalog model is set, Wattle infers the matching provider.
- If no usable setting is present, Wattle picks the first authenticated catalog model.
- If no authenticated provider is available, the TUI asks you to run `/login`; headless mode exits with an authentication message.
- API-key environment variables do not choose the provider or model; they only make that provider authenticated.

Credential lookup:

- `openai_codex` requires `openai.oauth` in `~/.wattle/auth.json`.
- API-key-only providers prefer a valid `api_key` entry in `~/.wattle/auth.json`.
- If the auth file is missing, the vendor entry is missing, or the vendor has no required method object, Wattle checks the matching environment variable.
- Generic credential lookup uses OAuth before API keys when both methods are configured. This applies to `anthropic.oauth` over `anthropic.api_key` and generic `openai.oauth` over `openai.api_key`.
- `openai_responses` and `openai_completions` explicitly require `openai.api_key` or `OPENAI_API_KEY`.

## Model catalog

Wattle ships with model choices for:

- OpenAI Codex: `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.3-codex`, `gpt-5.3-codex-spark`, `gpt-5.2`
- Anthropic: `claude-sonnet-4-6`, `claude-opus-4-6`, `claude-haiku-4-6`
- DeepSeek: `deepseek-v4-flash`, `deepseek-v4-pro`
- Kimi: `kimi-k2.6`, `kimi-k2.5`
- MiniMax: `MiniMax-M2.7`, `MiniMax-M2.7-highspeed`

Use `/model` to list and select authenticated model choices.

Wattle automatically switches providers when a catalog model belongs to a different provider.

## Reasoning controls

Enable reasoning controls with:

```bash
wattle --thinking --effort high
```

Or inside the TUI:

```text
/effort low|medium|high|xhigh|max|off
/effort off
```

You can also press Shift+Tab in the TUI to cycle through the available reasoning levels.

Supported effort values are `low`, `medium`, `high`, `xhigh`, and `max`. Providers map or clamp these values according to their API capabilities.
