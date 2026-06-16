# Changelog

## 0.5.0

### Added

- Added per-model output token limits and model-catalog based `max_tokens` resolution, so unset limits use each model's known ceiling and explicit limits are clamped safely.
- Added runtime deadline context from `WATTLE_RUN_DEADLINE_EPOCH_MS` so provider requests can adapt planning and validation to the remaining wall-clock budget.
- Added foreground and background task stopping support, including cancellation metadata for bash tool results.
- Added TUI rendering support for markdown tables.
- Added headless session persistence during turns, including runtime event capture.

### Changed

- Improved compaction pressure accounting by using provider usage metadata and avoiding compaction triggered only by output caps.
- Bounded provider request and idle timeouts to reduce hanging completions.
- Reduced bash live-output flicker in the TUI.
- Render markdown code fences as raw assistant text in the TUI.
- Tuned subagent delegation and footer status rendering.
- Tightened runtime artifact path discovery.
- Tightened validation contract handling with a compact provider-only checkpoint after tool use.

### Fixed

- Recover from malformed chat tool calls and guide malformed write-call repair instead of failing immediately.
- Clean up foreground bash process groups and escaped child processes more reliably.
- Preserve pasted placeholders when backspacing in the TUI.
