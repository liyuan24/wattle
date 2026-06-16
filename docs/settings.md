# Settings

Wattle stores optional persistent user defaults in `~/.wattle/settings.json`.

## Default settings

When no settings file exists, Wattle uses these built-in defaults:

```json
{
  "provider": null,
  "model": null,
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

`provider: null` and `model: null` mean Wattle auto-selects the first authenticated catalog model at launch. If no provider is authenticated, the TUI asks you to run `/login`; headless mode exits with an authentication message.

CLI flags override settings for that launch.

## Fields

- `provider` - default provider name, such as `openai_codex`, `anthropic`, `deepseek`, `kimi`, `minimax`, or `xiaomi-token-plan-sgp`. Use `null` for auto-selection.
- `model` - default catalog model id. Use `null` for auto-selection.
- `max_tokens` - maximum output tokens requested per model turn.
- `thinking` - whether reasoning controls are enabled by default.
- `effort` - default reasoning effort when thinking is enabled: `low`, `medium`, `high`, `xhigh`, `max`, or `null`.
- `permission_mode` - only `yolo` is supported.
- `tui.statusline` - fields shown in the bottom TUI statusline.
- `compaction_keep_recent_tokens` - recent-context budget preserved during compaction.
- `git_commit_attribution` - whether Wattle may include git attribution metadata when creating commits.

## Provider and Model

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

## TUI statusline

Run `/statusline` in the TUI to choose fields interactively. Use the arrow keys to move, press `x` to select or deselect a field, then press Enter to save the selection.

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
