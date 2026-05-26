# Compaction

Wattle keeps the complete session transcript on disk while building a compacted request projection for long model conversations. That gives long-running sessions room to continue without throwing away the durable history.

## Automatic compaction

Wattle tracks provider context usage and model context windows from the model catalog. When the active request is too large, Wattle summarizes the middle of the conversation and keeps recent context intact.

The default recent-context budget is controlled by:

```json
{
  "compaction_keep_recent_tokens": 20000
}
```

## Manual compaction

Compact immediately from the TUI:

```text
/compact
```

Add instructions for the summary:

```text
/compact preserve the current bug hypothesis and changed files
```

Wattle reports the compacted request token estimate when it finishes.

## What is persisted

Session files can include compaction checkpoints with:

- summary text
- first kept message index
- summarized message range
- reason
- before and after token estimates
- read and modified file paths
- creation time

The persisted session remains the source of truth. Compaction changes the request sent to the provider, not the complete saved transcript.

## When to use it

Use `/compact` before switching to a new phase of a large task, after a long debugging trace, or before asking Wattle to continue work that depends on the latest state more than the full raw transcript.
