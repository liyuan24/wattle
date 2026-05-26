# Wattle Documentation

Wattle is a pure-Python coding agent for serious terminal work. It gives modern models a small, powerful local runtime: file reading, exact edits, shell execution, image inspection, long-session compaction, persistent sessions, permission controls, and managed subagents that can investigate or implement in parallel.

Wattle is built to feel direct. Start it in a repository, give it the task, and it works against the same files and commands you use.

## Quick start

Install from a local checkout:

```bash
scripts/install.sh
```

Then run Wattle in a project directory:

```bash
wattle
```

For one-shot output:

```bash
wattle -p "summarize this repository"
```

Wattle defaults to the OpenAI Codex provider and `gpt-5.5`. Authenticate from the TUI with:

```text
/login
```

Or configure API-key providers in `~/.wattle/auth.json`.

## Start here

- [Quickstart](quickstart.md) - install, authenticate, and run a first useful session.
- [Using Wattle](usage.md) - TUI, headless mode, slash commands, and model switching.
- [Providers and Models](providers.md) - built-in providers, auth file shape, and model catalog.
- [Settings](settings.md) - user defaults in `~/.wattle/settings.json`.
- [Permissions](permissions.md) - `yolo`, `read_only`, and `ask_for_permission` modes.
- [Sessions](sessions.md) - persistence, resume, branching, and JSONL storage.
- [Compaction](compaction.md) - long-running sessions without losing the full transcript.

## Agent capabilities

- [Tools](tools.md) - file, shell, image, monitor, planning, and collaboration tools.
- [Subagents](subagents.md) - managed in-process agents for bounded parallel work.
- [Skills](skills.md) - reusable local capabilities loaded from `.wattle/skills`.

## Development

- [Development](development.md) - local setup, tests, TUI harness, and project layout.
