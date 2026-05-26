# Subagents

Wattle can run managed subagents inside the same Wattle runtime. Subagents are not separate CLI processes; they run in-process with their own message history, forked provider state, and a bounded tool set.

Use subagents for independent work that can happen in parallel with the primary task.

## Why use them

Subagents are useful when a task has separable questions:

- one agent inspects a TUI rendering path while the primary agent updates docs
- one agent searches for a regression while another reads tests
- one worker owns a narrow implementation area while another investigates a separate module

Good subagent tasks are bounded, explicit, and non-overlapping.

## Tools

### `spawn_agent`

Starts a subagent:

```json
{
  "task": "Inspect the session persistence code and report the JSONL fields. Do not edit files.",
  "agent_type": "explorer"
}
```

Common `agent_type` values:

- `explorer` - read-only investigation
- `worker` - implementation work
- `default` - normal delegated work

Optional fields include `instructions`, `context`, `model`, `max_tokens`, and `tool_names`.

### `wait_agent`

Waits for a subagent update, completion, or timeout. The subagent keeps running if the wait times out.

### `send_input`

Sends a follow-up message to an idle subagent and continues its existing history.

### `close_agent`

Requests subagent closure.

## Runtime behavior

Each subagent record tracks:

- id
- display name
- role
- task
- model and effort
- workspace
- allowed tool names
- status
- result or error
- turn count

Statuses include `pending`, `running`, `completed`, `failed`, `closing`, and `closed`.

## TUI display

The TUI renders subagent lifecycle events and running status. This makes parallel work visible without forcing the primary agent to block until every delegated task completes.

## Practical guidance

Before spawning a subagent, define:

- what the primary agent will keep doing locally
- what the subagent owns
- whether the subagent may edit files
- when the primary must wait for the result

Wait before depending on delegated findings, editing overlapping files, or producing a final answer that relies on the subagent.
