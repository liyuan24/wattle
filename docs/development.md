# Development

Wattle is a Python 3.12 project managed with `uv`.

## Setup

```bash
uv sync
```

Run the CLI from the checkout:

```bash
uv run wattle
```

Install or refresh the shell command:

```bash
uv tool install --force .
```

The installed `wattle` command is a separate `uv tool` environment, so dependency changes require both `uv sync` and `uv tool install --force .`.

## Tests

Run the full test suite:

```bash
uv run pytest
```

Run focused tests:

```bash
uv run pytest tests/test_cli.py
uv run pytest tests/test_tui.py
uv run pytest tests/test_tui_pty.py
```

Use the PTY harness for TUI behavior that depends on terminal input, prompt rendering, live redraw, status rows, image anchors, subagent status, and tool status UI.

## Linting

```bash
uv run ruff check .
```

## Project layout

```text
src/wattle/
  agent.py                  headless agent entry points
  cli.py                    command-line parser and provider construction
  tui/                      terminal UI
  providers/                provider adapters
  tools/                    model tool implementations
  session.py                JSONL session persistence
  settings.py               user settings
  auth.py                   credential loading and OpenAI Codex OAuth
  subagents.py              managed subagent runtime
  compaction.py             long-context request compaction
  skills.py                 skill discovery and invocation
tests/
  test_tui_pty.py           PTY-level TUI assertions
  pty_harness.py            terminal harness helpers
scripts/
  install.sh                local and hosted installer
  smoke_e2e.py              smoke test helper
```

## TUI contracts

When editing TUI rendering behavior, preserve:

- three-row submitted user and assistant message blocks
- distinct user/input background
- full-width prompt and status backgrounds
- long-message soft wrapping
- resize behavior
- separate wrapping and background-fill contracts for prompt, transcript, status, and running-status rows

Prefer assertions against actual PTY screen state for rendering regressions.
