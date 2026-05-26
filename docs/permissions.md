# Permissions

Wattle enforces tool permissions at runtime after the provider asks to use a tool. The permission gate controls execution without changing the model request prefix.

## Modes

Set permission mode from the CLI:

```bash
wattle --yolo
wattle --read-only
wattle --ask-for-permission
```

Or inside the TUI:

```text
/permissions
/permissions read_only
/permissions ask
/permissions yolo
```

## `yolo`

`yolo` allows requested tools to run without confirmation. This is the default mode.

Use it when you trust the task scope and have a rollback path.

## `read_only`

`read_only` allows:

- `read`
- `view_image`
- `update_plan`
- safe read-only `bash` and `monitor` commands

Safe shell commands include simple `pwd`, `ls`, `rg`, and selected `git` reads such as `git status`, `git diff`, `git log`, `git show`, `git branch`, and `git rev-parse`. Commands with shell control characters are blocked.

All mutating tools, including `write`, `edit`, and unrestricted shell commands, are denied.

## `ask_for_permission`

`ask_for_permission` prompts in the TUI before tool execution. The prompt summarizes the requested tool call, such as:

```text
bash: uv run pytest
edit: src/app.py
write: docs/index.md (42 lines)
```

The user can allow, deny, or allow all for the current session.

Headless `-p` mode does not support `--ask-for-permission`; use `--yolo` or `--read-only`.

## Persisting a default

Set a default in `~/.wattle/settings.json`:

```json
{
  "permission_mode": "ask_for_permission"
}
```
