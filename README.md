<p align="center">
  <img src="docs/assets/wattle_logo.png" alt="Wattle logo" width="180">
</p>

# Wattle

Wattle is a pure-Python coding agent.

## Install

For users, install the latest published release:

```bash
curl -fsSL https://wattleagent.com/install.sh | bash
```

For development, install the current checkout in editable mode:

```bash
scripts/install-dev.sh
```

The installer uses the project script declared in `pyproject.toml`:

```toml
[project.scripts]
wattle = "wattle.cli:main"
```

## Usage

Open the TUI:

```bash
wattle
```

Run one prompt headlessly:

```bash
wattle -p "summarize this repository"
```

Update manually:

```bash
wattle --upgrade
```

Choose a provider and model:

```bash
wattle --provider deepseek --model deepseek-v4-flash
wattle --provider kimi --model kimi-k2.6
wattle --provider minimax --model MiniMax-M2.7
```

Wattle reads API credentials from `~/.wattle/auth.json`:

```json
{
  "anthropic": {"api_key": {"api_key": "sk-ant-..."}},
  "deepseek": {"api_key": {"api_key": "sk-..."}},
  "kimi": {"api_key": {"api_key": "sk-..."}},
  "minimax": {"api_key": {"api_key": "sk-..."}},
  "openai": {"oauth": {"access_token": "...", "refresh_token": "..."}}
}
```
