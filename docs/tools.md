# Tools

Wattle exposes a compact tool set to the model. The tools are designed around local coding work: inspect first, edit precisely, run checks, monitor long-running output, and delegate bounded side work.

## File tools

### `read`

Reads a UTF-8 text file and returns 1-indexed line numbers. It supports `offset` and `limit`, caps large reads, and gives continuation hints.

### `write`

Creates or overwrites a text file with exact content. Wattle returns a unified diff after writing.

### `edit`

Edits an existing text file by replacing exact `old_text` with `new_text`. Each replacement must match a unique, non-overlapping region. Multiple replacements for one file can be applied in a single call.

`edit` preserves common newline style and rejects ambiguous matches.

## Shell tools

### `bash`

Runs a shell command in the user's default shell from the current working directory. It returns combined stdout and stderr and reports non-zero exit codes in the output.

Capabilities include:

- foreground commands
- PTY mode for commands that require terminal semantics
- background jobs with log files
- output externalization for large output
- timeout handling

Wattle's agent instructions prefer `read`, `write`, and `edit` for file changes instead of shell redirection.

### `monitor`

Runs a shell monitor command and pushes compact runtime events as output lines arrive. Use it for log tails, readiness checks, or watch commands.

Example task:

```text
Monitor the dev server logs and tell me when the app is ready or errors.
```

## Visual tool

### `view_image`

Attaches a local image to the next model turn. Supported media types are PNG, JPEG, WebP, and GIF up to 20 MB.

Use it for screenshots, layout regressions, visual debugging, and generated artifacts.

## Planning tool

### `update_plan`

Maintains a visible task plan with `pending`, `in_progress`, and `completed` steps. Wattle validates that at most one step is in progress.

## Collaboration tools

Wattle includes managed subagent tools:

- `spawn_agent`
- `send_input`
- `wait_agent`
- `close_agent`

See [Subagents](subagents.md).

## Permissions

Permission modes can block or prompt for tool calls. See [Permissions](permissions.md).
