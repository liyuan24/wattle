# Settings

Wattle stores persistent user defaults in `~/.wattle/settings.json`.

## Default settings

When no settings file exists, Wattle uses:

```json
{
  "provider": "openai_codex",
  "model": "gpt-5.5",
  "max_tokens": 4096,
  "thinking": false,
  "effort": null,
  "permission_mode": "yolo",
  "tui": {
    "statusline": ["model", "thinking", "cwd"]
  },
  "compaction_keep_recent_tokens": 20000,
  "git_commit_attribution": true
}
```

CLI flags override settings for that launch.

## Provider and model defaults

Set your normal provider and model:

```json
{
  "provider": "deepseek",
  "model": "deepseek-v4-flash"
}
```

If the selected model is in Wattle's catalog and the matching provider has auth, Wattle can infer the provider from the model.

## Thinking and effort

```json
{
  "thinking": true,
  "effort": "high"
}
```

Effort values: `low`, `medium`, `high`, `xhigh`, `max`.

## Permission mode

Wattle supports only `yolo` permission mode. Tool calls run without runtime confirmation.

```json
{
  "permission_mode": "yolo"
}
```

Legacy `read_only` or `ask_for_permission` settings are ignored and treated as `yolo`.

## TUI statusline

The default statusline fields are:

```json
{
  "tui": {
    "statusline": ["model", "thinking", "cwd"]
  }
}
```

Disable the statusline:

```json
{
  "tui": {
    "statusline": []
  }
}
```

## Environment overrides

Use these environment variables for tests or alternate launchers:

- `WATTLE_SETTINGS_PATH` - override the settings file path.
- `WATTLE_SESSION_DIR` - override the saved session directory.
