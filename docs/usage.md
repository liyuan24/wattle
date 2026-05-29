# Using Wattle

Wattle has two primary modes: the interactive terminal UI and headless one-shot output.

## Interactive TUI

Start the TUI in a project directory:

```bash
wattle
```

Submit an initial prompt while still opening the TUI:

```bash
wattle "review this repository and find the risky parts"
```

The TUI streams assistant output, tool status, subagent lifecycle updates, model and login pickers, session status, and a bottom statusline.

## Headless mode

Use `-p` or `--print` for a single prompt:

```bash
wattle -p "summarize the current diff"
```

Headless mode prints only the final assistant text to stdout. Use `--persist` to save the run:

```bash
wattle -p "write release notes for this diff" --persist
```

## CLI flags

```bash
wattle [PROMPT]
wattle -p PROMPT [--persist]
wattle -r [SESSION]
```

Common options:

- `--provider` - provider name, such as `openai_codex`, `anthropic`, `deepseek`, `kimi`, or `minimax`.
- `--model` - model id forwarded to the provider.
- `--max-tokens` - per-turn output cap.
- `--thinking` - enable provider reasoning controls where supported.
- `--effort low|medium|high|xhigh|max` - set reasoning effort and enable thinking.
- `--yolo` - allow tools without asking. This is the default and only supported permission mode.
- `-r`, `--resume [SESSION]` - resume a saved TUI session, or pick one interactively when no session is supplied.

## Slash commands

Use slash commands in the TUI:

```text
/help
/model
/effort low|medium|high|xhigh|max|off
/compact
/session
/branch
/resume <session-id>
/clear
/exit
```

Available commands:

- `/help` - show commands and current settings.
- `/login` - authenticate a provider.
- `/model [name]` - list or select models.
- `/effort [level]` - show or set reasoning effort. Use `off` to disable.
- `/session` or `/status` - show persistence and session status.
- `/branch` - copy the current conversation into a new session branch.
- `/resume <session-id>` - switch to a saved session id or JSONL path.
- `/compact [notes]` - compact the active request projection.
- `/clear` - reset conversation history.
- `/exit` or `/quit` - exit the TUI.
