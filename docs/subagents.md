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

When one or more subagents are active, the TUI shows a vertical selector below the normal statusline:

```text
gpt-5.5 | thinking: off | ~/repos/wattle
▸ input
○ main
○ Hopper explorer running
```

The selector has no `Agents:` heading. It always lists `input` first, then `main`, then active subagents in launch order. Terminal subagents (`completed`, `failed`, or `closed`) disappear from the selector, while their lifecycle/completion notifications remain visible in the main transcript.

Use Up/Down while the selector is visible to move focus between the input box, `main`, and active subagent views. Choosing `main` or a subagent switches the transcript view; choosing `input` returns focus to typing without changing the current transcript target. In a subagent view, the transcript shows the main conversation up to the spawn point, a divider such as `── subagent: Hopper explorer ──`, and that subagent's conversation. Typed input targets the selected subagent; if it is still running, pending, or closing, Wattle rejects the input with a clear message instead of queueing it.

## Practical guidance

Before spawning a subagent, define:

- what the primary agent will keep doing locally
- what the subagent owns
- whether the subagent may edit files
- when the primary must wait for the result

Wait before depending on delegated findings, editing overlapping files, or producing a final answer that relies on the subagent.
