# TUI

Wattle's TUI is a native terminal interface. It is built with Python stdio, raw terminal input, ANSI escape sequences, and a small set of rendering helpers. It does not use curses, Textual, or the alternate screen.

## Main structure

The TUI lives in `src/wattle/tui/`.

```text
src/wattle/tui/__init__.py      app state, input loop, streaming, slash commands, rendering
src/wattle/tui/terminal.py      low-level ANSI row helpers
src/wattle/tui_flowers/         animated running-status glyphs
```

`run_tui()` is the CLI entry point. It resolves resume arguments, builds the provider, creates `WattleApp`, and starts the app.

`WattleApp` owns the durable state:

- current provider and model
- settings, thinking, effort, and statusline fields
- message history and compaction state
- session persistence
- provider streaming
- tool dispatch and tool result rendering
- slash command behavior

`_LiveTerminal` owns live terminal interaction:

- raw-mode input
- prompt buffer and cursor position
- input history
- model and login pickers
- file, command, and skill suggestions
- queued user messages during streaming
- prompt redraws, resize handling, and running-status animation

When stdin or stdout is not a TTY, `WattleApp` uses a simpler line-based loop. Tests and scripted runs use this path.

## Rendering model

The transcript is append-only stdout. Wattle writes completed user messages, assistant text, tool results, panels, and separators into normal terminal scrollback. The live prompt is the only part that is repeatedly redrawn.

This means terminal scrollback remains useful and Wattle does not need to own the whole screen. The tradeoff is that prompt rendering must carefully clear and repaint only the rows it owns.

The live prompt frame can include:

- active running status
- active bash output cell
- visible subagent status
- queued or interrupted user messages
- the input box
- model picker
- login picker
- input suggestions
- statusline

Prompt rows are rebuilt by `_LiveTerminal._build_prompt_frame()` and written by `_write_prompt_frame()`. Low-level row functions in `tui/terminal.py` handle full-width clearing, wrapping, background fill, and ANSI reset behavior.

## Input model

The live TUI puts the terminal in cbreak mode and reads bytes from stdin with `select` and `os.read`.

Important input behaviors:

- Enter submits the current buffer.
- Shift+Enter inserts a newline.
- Tab completes the selected suggestion, or queues a message for the next assistant turn while streaming.
- Shift+Tab cycles the thinking level.
- Arrow keys move picker selection, suggestion selection, or input history.
- Esc interrupts an active streaming turn when possible.
- Bracketed paste is tracked so large pasted text can be represented compactly in the prompt.
- Clipboard image paste stores the image as a session asset and inserts its path.

Escape sequences differ by terminal, so key handling accepts several common encodings for Shift+Enter, Shift+Tab, Ctrl+V, and word movement.

## Turn flow

A normal submitted message follows this path:

1. The user text is expanded if it invokes a skill.
2. File and image references are converted into message content blocks.
3. The message is appended to `self.messages` and persisted when sessions are enabled.
4. `RequestPreparer` builds the provider request, applying compaction when needed.
5. Provider streaming writes assistant text as it arrives.
6. Tool calls are dispatched when the provider returns `tool_use`.
7. Tool results are appended as follow-up user content.
8. The loop continues until the assistant ends the turn.
9. Usage, status, worked duration, and session state are updated.

In the live TUI, the provider turn runs in a worker thread so the prompt can keep accepting input, show running status, and queue follow-up messages.

## Sessions and resume

TUI runs persist sessions by default after authentication is available. Session state includes provider, model, settings, messages, and compaction checkpoints.

Resume can be started from the CLI or from `/resume`. When no exact session is supplied and a TTY is available, Wattle opens a small raw-terminal resume picker with filtering and arrow-key selection.

When a resumed session ends with tool results, Wattle continues the pending provider turn instead of waiting for a new user message.

## Auth and model selection

If no authenticated provider and model are available, the TUI starts but shows a login notice. It does not start a task until authentication is ready.

`/login` opens the provider login picker. `/model` opens the model picker. Pickers are rendered inside the live prompt frame and use the same up/down/enter input path as suggestions.

The available tool list depends on the selected model. For example, image tools are hidden when the current model does not support image input.

## Tools and subagents

Tool dispatch uses the shared Wattle runtime and the same provider/tool abstractions as headless mode. The TUI adds terminal-specific presentation:

- bash output is shown as a compact execution cell while running
- edit results are grouped by file
- research-style command output is summarized
- plan updates get dedicated rendering
- subagent lifecycle events are shown in the prompt while work is active

Tool progress events are drained while the prompt is live so long-running commands can update the screen without blocking input.

## Testing TUI changes

Use focused unit tests for rendering helpers and state transitions:

```bash
uv run pytest tests/test_tui.py
```

Use the PTY harness for behavior that depends on terminal input or screen state:

```bash
uv run pytest tests/test_tui_pty.py
```

Use PTY tests for changes involving raw input, prompt redraws, resize behavior, running status, image anchors, subagent status, and tool status UI.
