# Willow

Willow is a pure-Python coding agent.

## Install

From a local checkout:

```bash
scripts/install.sh
```

From a hosted repository:

```bash
curl -fsSL https://raw.githubusercontent.com/<owner>/willow/main/scripts/install.sh \
  | WILLOW_REPO_URL=https://github.com/<owner>/willow.git bash
```

The installer uses the project script declared in `pyproject.toml`:

```toml
[project.scripts]
willow = "willow.cli:main"
```

## Usage

Open the TUI:

```bash
willow
```

Run one prompt headlessly:

```bash
willow -p "summarize this repository"
```

Choose a provider and model:

```bash
willow --provider deepseek --model deepseek-v4-flash
willow --provider kimi --model kimi-k2.6
willow --provider minimax --model MiniMax-M2.7
```

Willow reads API credentials from `~/.willow/auth.json`:

```json
{
  "anthropic": {"api_key": {"api_key": "sk-ant-..."}},
  "deepseek": {"api_key": {"api_key": "sk-..."}},
  "kimi": {"api_key": {"api_key": "sk-..."}},
  "minimax": {"api_key": {"api_key": "sk-..."}},
  "openai": {"oauth": {"access_token": "...", "refresh_token": "..."}}
}
```
