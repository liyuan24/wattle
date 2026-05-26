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

The TUI streams assistant output, tool status, permission prompts, subagent lifecycle updates, model and login pickers, session status, and a configurable bottom statusline.

## Headless mode

Use `-p` or `--print` for a single prompt:

```bash
wattle -p "summarize the current diff"
```

Headless mode prints only the final assistant text to stdout. Use `--persist` to save the run:

```bash
wattle -p "write release notes for this diff" --persist
```

`--ask-for-permission` is interactive-only. Use `--yolo` or `--read-only` with `-p`.

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
- `--yolo` - allow tools without asking.
- `--read-only` - allow only read-only tools and safe read-only shell commands.
- `--ask-for-permission` - ask before tool execution in the TUI.
- `-r`, `--resume [SESSION]` - resume a saved TUI session, or pick one interactively when no session is supplied.

## Slash commands

Use slash commands in the TUI:

```text
/help
/model
/model next
/permissions read_only
/effort high
/compact
/session
/branch
/resume SESSION
/statusline off
/clear
/exit
```

Available commands:

- `/help` - show commands and current settings.
- `/login [openai-codex]` - authenticate the OpenAI Codex provider.
- `/model [name|#|next]` - list, select, or cycle models.
- `/model enabled` - show the enabled model filter.
- `/model enable MODEL_OR_NUMBER` - include a model in the TUI cycling list.
- `/model disable MODEL_OR_NUMBER` - remove a model from the TUI cycling list.
- `/effort [level]` - show or set reasoning effort. Use `off` to disable.
- `/permissions [mode]` - show or set tool permission mode.
- `/session` or `/status` - show persistence and session status.
- `/branch` - copy the current conversation into a new session branch.
- `/resume SESSION` - switch to a saved session id or JSONL path.
- `/compact [notes]` - compact the active request projection.
- `/statusline on|off` - toggle the bottom statusline.
- `/clear` - reset conversation history.
- `/exit` or `/quit` - exit the TUI.

## Referencing files and images

Ask Wattle for files by path, or let the model use the `read` tool. The read tool accepts normal paths and `@path`-style paths.

For screenshots and UI issues, Wattle includes `view_image`, which attaches a local PNG, JPEG, WebP, or GIF to the next model turn.

## Context files

Wattle loads instruction files at startup:

- `~/.wattle/AGENTS.md`
- `AGENTS.md` from the project root down to the current directory
- `AGENTS.override.md` in those same project directories

Project roots are detected with `.git`.
