# Changelog

## 0.9.0

### Added

- Added goal continuation hook support, including persisted goal state and update-goal tooling.
- Added goal mode support for headless prompts.
- Added stronger completion validation guidance, including semantic validation, representation-contract checks, verifier-minded contradiction passes, and artifact-first validation evidence.

### Changed

- Start goal mode from the raw objective text instead of rewriting the initial goal prompt.
- Require concrete evidence when updating goals, and reject weak or surface-only completion evidence.
- Move the final audit reminder from per-tool request preparation into the default stop hook.
- Add system prompt guidance to avoid temporary files during normal work unless needed for implementation or validation.
- Improve TUI input word wrapping.

### Fixed

- Fixed dragged image path handling in the TUI.
- Fixed macOS screenshot drag anchors.
- Fixed image anchor backspace handling.
- Isolated Wattle settings during tests.

## 0.8.0

### Added

- Added Intel macOS binary builds to the release workflow.
- Added live TUI shell mode for running commands directly from the interactive interface.
- Added vertical cursor movement in multiline TUI input.

### Changed

- Show full shell-mode command output in the live TUI instead of truncating command results.
- Added a final artifact hygiene reminder in request preparation.
- Clarified the final cleanup reminder in request preparation.

### Fixed

- Fixed subagent selector TUI navigation and documented the updated navigation keys.
- Fixed typed input to an already selected subagent so running subagents reject input instead of silently switching focus.

## 0.7.0

### Added

- Added `/voice` dictation in the live TUI, using OpenAI speech-to-text to insert microphone transcription into the input box while holding Space.
- Added voice dictation documentation, including setup, usage, and troubleshooting.

### Changed

- Added `WATTLE_VOICE_DICTATION_API_KEY` as the required API-key environment variable for voice dictation.

### Fixed

- Report missing and non-working voice dictation API keys with clear `WATTLE_VOICE_DICTATION_API_KEY` guidance.
- Keep the voice dictation reminder inside the live prompt status area instead of leaking repeated rows into scrollback.

## 0.6.0

### Added

- Added Claude OAuth support.
- Added an interactive statusline picker.

### Changed

- Kept deadline status out of system prompts.

### Fixed

- Handle provider server errors gracefully.

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
