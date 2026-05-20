# Wattle

Wattle is a pure-Python coding agent.

## Install

From a local checkout:

```bash
scripts/install.sh
```

From a hosted repository:

```bash
curl -fsSL https://raw.githubusercontent.com/<owner>/wattle/main/scripts/install.sh \
  | WATTLE_REPO_URL=https://github.com/<owner>/wattle.git bash
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
