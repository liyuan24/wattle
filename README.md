<h1>
  <img src="docs/assets/wattle_logo.png" alt="Wattle logo" width="56" align="center">
  Wattle
</h1>

Wattle is a lightweight, pure-Python coding agent. It keeps the surface small while covering the core loop: read files, edit code, run commands, inspect images, compact long sessions, persist work, and delegate to subagents.

Wattle only has YOLO mode because that is the only way I use coding agents. It was first built with Codex and is now used to develop itself. Fork it, modify it, and shape it around your own workflow.

## Quick Start

Install the latest published release:

```bash
curl -fsSL https://wattleagent.com/install.sh | bash
```

Authenticate providers from the TUI:

```text
/login
```

Run Wattle in a project directory:

```bash
wattle
```

Run one prompt headlessly:

```bash
wattle -p "summarize this repository"
```

Update manually:

```bash
wattle --upgrade
```

## Development

Clone the repository and install the current checkout in editable mode:

```bash
git clone https://github.com/liyuan24/wattle.git
cd wattle
scripts/install-dev.sh
```

Sync the local project environment for tests and linting:

```bash
uv sync
```

Run tests:

```bash
uv run pytest
```

## Official Website

Documentation and install instructions live at [wattleagent.com](https://wattleagent.com).

## License

Wattle is released under the [MIT License](LICENSE).
