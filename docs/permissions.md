# Permissions

Wattle supports only `yolo` permission mode.

`yolo` allows requested tools to run without confirmation. This is the default and only supported behavior for both TUI and headless runs.

```bash
wattle --yolo
```

The `--read-only`, `--ask-for-permission`, and `/permissions` interfaces are no longer supported. Legacy `permission_mode` settings other than `yolo` are ignored and treated as `yolo`.
