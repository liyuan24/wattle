# Development

Wattle is a Python project managed with `uv`.

## Setup

Clone the repository and install the editable developer command:

```bash
git clone https://github.com/liyuan24/wattle.git
cd wattle
scripts/install-dev.sh
```

The developer installer uses `uv tool install --force -e .`, so the shell `wattle` command points at the current checkout.

For local test and lint commands, sync the project environment:

```bash
uv sync
```

You can also run the CLI directly from the checkout:

```bash
uv run wattle
```

The installed `wattle` command is a separate `uv tool` environment, so dependency changes require both `uv sync` and `scripts/install-dev.sh`.

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
  install.sh                hosted release installer
  install-dev.sh            editable checkout installer
  smoke_e2e.py              smoke test helper
```
