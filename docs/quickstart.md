# Quickstart

This page gets you from a checkout to a useful first Wattle session.

## Install

From a local checkout:

```bash
cd /path/to/wattle
scripts/install.sh
```

From a hosted repository:

```bash
curl -fsSL https://raw.githubusercontent.com/<owner>/wattle/main/scripts/install.sh \
  | WATTLE_REPO_URL=https://github.com/<owner>/wattle.git bash
```

The installer exposes the project script declared in `pyproject.toml`:

```toml
[project.scripts]
wattle = "wattle.cli:main"
```

After install, start Wattle in the repository you want it to work on:

```bash
cd /path/to/project
wattle
```

## Authenticate

Wattle reads credentials from `~/.wattle/auth.json`. The fastest path for the default provider is the built-in OpenAI Codex OAuth flow:

```text
/login
```

The TUI opens or prints an authorization URL and saves the resulting credential under the OpenAI entry in `~/.wattle/auth.json`.

For API-key providers, create the auth file directly:

```json
{
  "anthropic": {"api_key": {"api_key": "sk-ant-..."}},
  "deepseek": {"api_key": {"api_key": "sk-..."}},
  "kimi": {"api_key": {"api_key": "sk-..."}},
  "minimax": {"api_key": {"api_key": "sk-..."}},
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

## Choose a provider or model

Pass provider and model flags:

```bash
wattle --provider deepseek --model deepseek-v4-flash
wattle --provider kimi --model kimi-k2.6
wattle --provider minimax --model MiniMax-M2.7
```

Inside the TUI, use `/model` to list authenticated model choices and `/model next` to cycle through enabled choices.

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
