# Quickstart

This page gets you from install to a useful first Wattle session.

## Install

Install the latest published release:

```bash
curl -fsSL https://wattleagent.com/install.sh | bash
```

After install, start Wattle in the repository you want it to work on:

```bash
cd /path/to/project
wattle
```

## Authenticate

Wattle reads credentials from `~/.wattle/auth.json` or provider API-key environment variables. Use the TUI login flow and follow the provider instructions:

```text
/login
```

The TUI handles supported OAuth and API-key provider login, then saves the credential to `~/.wattle/auth.json`.

You can also set an API-key environment variable, such as `ANTHROPIC_API_KEY`, or edit the auth file directly:

```json
{
  "anthropic": {"api_key": {"api_key": "sk-ant-..."}},
  "deepseek": {"api_key": {"api_key": "sk-..."}},
  "kimi": {"api_key": {"api_key": "sk-..."}},
  "minimax": {"api_key": {"api_key": "sk-..."}},
  "xiaomi-token-plan-sgp": {"api_key": {"api_key": "tp-..."}},
  "openai": {"oauth": {"access_token": "...", "refresh_token": "..."}}
}
```

See [Providers and Models](providers.md) for provider names, API-key entries, and model selection.

## First session

Type a request and press Enter:

```text
Summarize this repository and tell me how to run its checks.
```

Wattle can read files, make exact edits, run shell commands, inspect local images, monitor long-running output, and delegate bounded work to subagents. It runs in your current working directory, so use git or another checkpointing workflow when you want easy rollback.

## One-shot mode

Use `-p` for a headless prompt that prints only the final assistant text:

```bash
wattle -p "summarize this repository"
```

Save a headless run into the same session store as the TUI:

```bash
wattle -p "inspect the public API surface" --persist
```

## Update Wattle

The TUI checks for a newer published release at startup. If an update is available, choose **Update** to run the pinned installer and return to your shell, or choose **Skip update** to continue into the TUI.

Trigger the same check manually:

```bash
wattle --upgrade
```

## Choose a provider or model

Pass provider and model flags:

```bash
wattle --provider deepseek --model deepseek-v4-flash
wattle --provider kimi --model kimi-k2.6
wattle --provider minimax --model MiniMax-M2.7
```

Inside the TUI, use `/model` to list authenticated model choices and `/model MODEL_NAME` to switch models.

## Give Wattle project instructions

Add `AGENTS.md` to the project. Wattle loads instructions from `~/.wattle/AGENTS.md` and from `AGENTS.md` or `AGENTS.override.md` files from the git root down to the current directory.

```markdown
# Project Instructions

- Run `uv run pytest` after Python changes.
- Keep edits small and focused.
- Do not modify generated files by hand.
```

Restart Wattle after changing instruction files.

## Next steps

- [Using Wattle](usage.md) - interactive mode, slash commands, and CLI flags.
- [Providers and Models](providers.md) - auth and model catalog.
- [Settings](settings.md) - persistent defaults.
- [Tools](tools.md) - what Wattle can do for the model.
- [Subagents](subagents.md) - parallel investigation and implementation.
