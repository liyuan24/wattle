# Wattle Documentation

Wattle is a lightweight, pure-Python coding agent. It keeps the surface small while covering the core loop: read files, edit code, run commands, inspect images, compact long sessions, persist work, and delegate to subagents.

Wattle only has YOLO mode because that is the only way I use coding agents. It was first built with Codex and is now used to develop itself. Fork it, modify it, and shape it around your own workflow.

## Quick start

Install the latest published release:

```bash
curl -fsSL https://wattleagent.com/install.sh | bash
```

Authenticate providers from the TUI:

```text
/login
```

Or configure API-key providers in `~/.wattle/auth.json`.

Then run Wattle in a project directory:

```bash
wattle
```

For one-shot output:

```bash
wattle -p "summarize this repository"
```

## Start here

- [Quickstart](quickstart.md) - install, authenticate, and run a first useful session.
- [Using Wattle](usage.md) - TUI, headless mode, slash commands, and model switching.
- [Providers and Models](providers.md) - built-in providers, auth file shape, and model catalog.
- [Settings](settings.md) - user defaults in `~/.wattle/settings.json`.
- [Permissions](permissions.md) - `yolo` tool execution mode.
- [Sessions](sessions.md) - persistence, resume, branching, and JSONL storage.
- [Compaction](compaction.md) - long-running sessions without losing the full transcript.

## Agent capabilities

- [Tools](tools.md) - file, shell, image, monitor, planning, and collaboration tools.
- [Subagents](subagents.md) - managed in-process agents for bounded parallel work.
- [Skills](skills.md) - reusable local capabilities loaded from `.wattle/skills`.

## Development

- [Development](development.md) - local setup, tests, TUI harness, and project layout.
