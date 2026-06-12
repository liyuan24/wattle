# Compaction

Wattle keeps the complete session transcript on disk while building a compacted request projection for long model conversations. That gives long-running sessions room to continue without throwing away the durable history.

## Automatic compaction

Wattle tracks provider-reported input/context usage and model context windows from the model catalog. When provider usage is unavailable, Wattle falls back to a local request-size estimate. When the active request is too large, Wattle summarizes the middle of the conversation and keeps recent context intact.

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

## How compaction works

Compaction changes only the request projection sent to the model. The session transcript on disk stays complete: Wattle keeps the original user messages, assistant messages, tool calls, tool results, thinking blocks, image records, and compaction checkpoints.

What gets compacted: Wattle chooses an older middle range of messages and summarizes that range. This is usually the part of the conversation that is no longer the immediate working tail but still contains useful history.

What is not compacted: the newest messages stay verbatim, so the model still sees the live working state exactly as it happened. That includes the latest user request, recent tool calls, tool results, errors, file paths, and decisions.

How the summary is made: Wattle performs a separate model call for summarization. It asks the provider to preserve the details needed to continue coding work: goals, constraints, completed work, important commands, errors, changed files, open questions, and next steps. If the session was compacted before, Wattle gives the previous summary to the summarizer and asks it to update that summary with the newly compacted messages.

After compaction, the next normal model call receives:

1. The normal system prompt and tool definitions.
2. One synthetic user message containing the compaction summary. This message explicitly says the earlier conversation was compacted and that the summary is prior context, not a new user request.
3. The recent un-compacted tail of the conversation, kept as the original messages.

The model does not receive the full old transcript in that call. It receives the summary plus the recent tail. Wattle can still save and resume the complete session because the durable transcript remains on disk.

Automatic compaction starts when provider-reported input/context usage approaches the model context window. If a provider does not report usage, Wattle uses a local request-size estimate instead. The requested output budget is not part of the trigger. Manual `/compact` forces the same process immediately, optionally with extra instructions for what the summary should preserve.
