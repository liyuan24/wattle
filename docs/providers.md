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

Example:

```bash
wattle --provider anthropic --model claude-sonnet-4-6
```

## Auth file

Wattle reads `~/.wattle/auth.json`.

```json
{
  "anthropic": {"api_key": {"api_key": "sk-ant-..."}},
  "deepseek": {"api_key": {"api_key": "sk-..."}},
  "kimi": {"api_key": {"api_key": "sk-..."}},
  "minimax": {"api_key": {"api_key": "sk-..."}},
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

OpenAI Codex uses the `openai.oauth` credential. `openai_responses` and `openai_completions` use `openai.api_key`.

## Login

The TUI supports OpenAI Codex OAuth:

```text
/login
```

`/login openai-codex` is equivalent. API-key providers are configured by editing `~/.wattle/auth.json`.

## Model catalog

Wattle ships with model choices for:

- OpenAI Codex: `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.3-codex`, `gpt-5.3-codex-spark`, `gpt-5.2`
- Anthropic: `claude-sonnet-4-6`, `claude-opus-4-6`, `claude-haiku-4-6`
- DeepSeek: `deepseek-v4-flash`, `deepseek-v4-pro`
- Kimi: `kimi-k2.6`, `kimi-k2.5`
- MiniMax: `MiniMax-M2.7`, `MiniMax-M2.7-highspeed`

Use `/model` to list authenticated choices:

```text
/model
/model 2
/model gpt-5.4
```

Wattle automatically switches providers when a catalog model belongs to a different provider.

## Custom model ids

You can set any model id:

```text
/model vendor-model-name
```

When the model id is not in Wattle's catalog, Wattle keeps the current provider and forwards the model string as-is.

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

Supported effort values are `low`, `medium`, `high`, `xhigh`, and `max`. Providers map or clamp these values according to their API capabilities.
