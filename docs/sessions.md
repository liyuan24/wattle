# Sessions

Wattle saves TUI conversations as durable JSONL session files. Sessions are plain enough to inspect, while still preserving the provider, model, settings, messages, and compaction checkpoints needed to continue work later.

## Storage

By default, sessions live in:

```text
~/.wattle/sessions
```

Override the directory with:

```bash
export WATTLE_SESSION_DIR=/path/to/sessions
```

Each session is saved as `<session-id>.jsonl`.

## Resume from the CLI

Resume the newest session picker:

```bash
wattle -r
```

Resume a known session id:

```bash
wattle -r 018f2c2d-...
```

Resume a JSONL path:

```bash
wattle -r ~/.wattle/sessions/018f2c2d.jsonl
```

## Resume inside Wattle

```text
/resume SESSION
```

`SESSION` can be a session id or a JSONL path.

## Status

Show the active session:

```text
/session
/status
```

The status panel includes persistence state, session path, permission mode, thinking state, effort, message count, statusline state, and current runtime status.

## Branching

Create a new session branch from the current conversation:

```text
/branch
```

Wattle copies the conversation and compaction state into a new session record and stores the original id as `parent_session_id`. Use this when an investigation reaches a fork in the road and you want both paths preserved.

## Headless persistence

Headless mode does not save by default. Add `--persist`:

```bash
wattle -p "inspect the architecture" --persist
```

Wattle writes the session path to stderr after saving.

## Session format

A session file starts with a header line:

```json
{"type":"session","schema_version":1,"metadata":{...},"settings":{...}}
```

Following lines contain messages and compaction checkpoints. Message blocks preserve text, images, tool calls, tool results, thinking blocks, and token usage fields where available.
