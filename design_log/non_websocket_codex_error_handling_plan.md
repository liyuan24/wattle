# Non-WebSocket Codex Error Handling Plan

Date: 2026-05-25

## Goal

Improve Wattle's provider error handling for the HTTPS/SSE Codex and OpenAI
Responses paths by mirroring the useful non-WebSocket classifications found in
`~/repos/codex`, without introducing WebSocket transport behavior.

The main outcome should be fewer opaque `RuntimeError` failures and fewer cases
where an unsuccessful provider response is treated as a normal assistant turn.
Both TUI and headless should benefit through the shared request-preparation and
loop recovery paths.

## Scope

In scope:

- HTTPS request failures.
- SSE stream failures.
- Structured `response.failed` events.
- Structured `response.incomplete` events.
- Context-window recovery.
- Transient overload/rate-limit retry.
- Clear non-retryable messages for quota, unsupported plan, invalid prompt, and
  cyber-policy failures.
- Focused unit tests for providers, shared recovery, TUI retry presentation, and
  headless loop behavior where applicable.

Out of scope:

- Responses WebSocket transport.
- WebSocket fallback to HTTPS.
- WebSocket connection-limit handling.
- Codex app-server protocol changes.
- Large CLI UX redesign.

## Codex Reference Points

Useful non-WebSocket behavior observed in `~/repos/codex`:

- `codex-rs/codex-api/src/sse/responses.rs`
  - Parses `response.failed` error payloads.
  - Classifies:
    - `context_length_exceeded`
    - `insufficient_quota`
    - `usage_not_included`
    - `cyber_policy`
    - `invalid_prompt`
    - `server_is_overloaded`
    - `slow_down`
    - `rate_limit_exceeded` with optional retry delay parsed from text.
  - Treats `response.incomplete` as an error with the explicit incomplete
    reason.
  - Treats stream close before `response.completed` as a stream error.
  - Treats idle timeout waiting for SSE as a stream error.
- `codex-rs/codex-api/src/api_bridge.rs`
  - Maps 503 bodies with `server_is_overloaded` or `slow_down` to a specific
    overload error.
  - Maps 429 usage-limit bodies to usage-limit style errors instead of generic
    transport failures.
- `codex-rs/codex-client/src/retry.rs`
  - Retries 5xx and transport errors by provider policy.
  - Does not blindly retry 429 by default.

## Current Wattle Gaps

Relevant Wattle files:

- `src/wattle/providers/openai_codex.py`
- `src/wattle/providers/openai_responses.py`
- `src/wattle/request_preparation.py`
- `src/wattle/provider_errors.py`
- `src/wattle/tui/__init__.py`
- `src/wattle/loop.py`

Known gaps:

1. `openai_codex._map_event()` handles `response.failed` by raising plain
   `RuntimeError` with only the message.
2. `openai_codex._map_event()` treats `response.incomplete` as a completion
   event, and `_stop_reason_from_response()` only distinguishes
   `max_output_tokens`; other incomplete reasons fall through to `end_turn`.
3. HTTP 429 responses are not retried, which is fine, but the body is not parsed
   into clearer quota/usage/rate-limit classes.
4. Server overload can arrive inside an SSE `response.failed` payload rather
   than as HTTP 503. Wattle currently misses that path.
5. Rate-limit retry delay can arrive in a `response.failed` message such as
   `try again in 2s`; Wattle does not currently parse that for Codex SSE.

## Design Principles

- Keep retry policy conservative.
- Do not retry clearly user-actionable failures such as quota, unsupported plan,
  invalid prompt, cyber policy, or auth failures.
- Prefer structured provider-error classes over string matching at higher layers.
- Preserve existing shared recovery entrypoints:
  - `acomplete_with_recovery()`
  - `astream_with_recovery()`
- Ensure fixes apply to both TUI and headless by putting classification below
  the mode-specific UI layer.
- TUI may keep richer presentation, but core classification must not be TUI-only.

## Proposed Error Types / Mapping

### Reuse Existing Types

Use existing classes where they fit:

- `IncompleteStreamError`
  - stream closed before terminal event
  - malformed/incomplete terminal state that is safe to retry
- `TransientProviderError`
  - overload
  - retryable rate limit
  - retryable HTTP status
  - network/timeout errors

### Add Provider-Facing Non-Retryable Errors If Needed

Consider adding lightweight provider errors in `src/wattle/providers/base.py`, or
local classes if a smaller patch is preferred:

- `ProviderQuotaError`
- `ProviderUsageNotIncludedError`
- `ProviderInvalidRequestError`
- `ProviderPolicyError`

If adding classes feels too broad, a first pass can keep these as `RuntimeError`
with clearer messages, as long as retryable and context-window cases are no
longer plain `RuntimeError`.

## Implementation Plan

### 1. Add Codex SSE Error Payload Parser

In `src/wattle/providers/openai_codex.py`, add helpers to parse the error object
from events shaped like:

```json
{
  "type": "response.failed",
  "response": {
    "error": {
      "code": "server_is_overloaded",
      "message": "...",
      "type": "...",
      "resets_at": 123
    }
  }
}
```

Suggested helpers:

- `_codex_response_error(event: dict[str, Any]) -> dict[str, Any] | None`
- `_raise_for_codex_response_error(error: dict[str, Any]) -> None`
- `_retry_after_from_error_message(message: str | None) -> float | None`

Classification:

- `context_length_exceeded`
  - raise an error that `is_context_length_error()` recognizes.
  - simplest: `RuntimeError("context_length_exceeded")` with clear message.
  - better: dedicated provider context error plus classifier support.
- `server_is_overloaded`, `slow_down`
  - raise `TransientProviderError(message, retry_after=...)`.
- `rate_limit_exceeded`
  - if retry delay is present and reasonable, raise `TransientProviderError` with
    `retry_after`.
  - if no delay is present, decide conservatively:
    - either transient with exponential backoff, or
    - non-retryable clear rate-limit error.
  - Preferred first pass: transient, because Codex maps unknown `response.failed`
    to retryable stream failure.
- `insufficient_quota`
  - non-retryable clear quota/billing error.
- `usage_not_included`
  - non-retryable clear unsupported-plan/upgrade error.
- `cyber_policy`
  - non-retryable policy error with fallback message:
    `This request has been flagged for possible cybersecurity risk.`
- `invalid_prompt`
  - non-retryable invalid request error.
- unknown `response.failed`
  - raise `TransientProviderError` or `IncompleteStreamError` rather than plain
    `RuntimeError`, unless evidence shows these are usually permanent.
  - Include the provider message in the error.

### 2. Fix `response.incomplete` Semantics

Update Codex SSE handling:

- If `response.incomplete` reason is `max_output_tokens`, keep existing behavior
  and return a final response with `stop_reason="max_tokens"`.
- For any other reason:
  - raise `IncompleteStreamError("Incomplete response returned, reason: ...")`,
    or `TransientProviderError` if the reason clearly maps to overload/rate
    limit.

This avoids treating incomplete provider responses as successful `end_turn`.

### 3. Parse HTTP Error Bodies for Clearer Non-Retryable Cases

Enhance `_codex_error_message()` or the HTTPError handler in
`OpenAICodexResponsesProvider._stream_blocking()`:

- Keep retry for `{408, 500, 502, 503, 504}`.
- Parse JSON bodies shaped like `{ "error": { ... } }`.
- For HTTP 503 with `server_is_overloaded` or `slow_down`, raise
  `TransientProviderError` with a clear overload message.
- For HTTP 429:
  - `usage_limit_reached` / `insufficient_quota` -> non-retryable quota/usage
    message.
  - `usage_not_included` -> non-retryable unsupported-plan message.
  - `rate_limit_exceeded` plus retry hint/header -> `TransientProviderError`.
- Preserve 401/403 as non-retryable auth/permission errors.

### 4. Mirror Relevant Classifications in OpenAI Responses Provider

Inspect `src/wattle/providers/openai_responses.py` for SDK response status and
exception payloads.

Add or confirm handling for:

- incomplete response status with non-`max_output_tokens` reason.
- SDK/API errors whose body/code is:
  - `server_is_overloaded`
  - `slow_down`
  - `rate_limit_exceeded`
  - `context_length_exceeded`
  - `insufficient_quota`
  - `usage_not_included`
  - `cyber_policy`
  - `invalid_prompt`

Do not overfit to one SDK exception shape. Reuse a small recursive extractor if
needed, similar to `provider_errors.is_context_length_error()`.

### 5. Extend Shared Classifier Utilities

In `src/wattle/provider_errors.py`, consider adding helpers:

- `is_context_length_error(error: object) -> bool` already exists.
- Add, if useful:
  - `error_values(error: object) -> Iterator[str]` as a public/internal helper.
  - `is_overload_error(error: object) -> bool`
  - `is_rate_limit_error(error: object) -> bool`

Keep these helpers conservative and avoid making quota/auth/policy retryable.

### 6. Tests

Add focused provider tests before or with implementation.

Codex provider tests in `tests/test_provider_openai_codex.py`:

- `response.failed` with `context_length_exceeded` is recognized by
  `is_context_length_error()` or triggers recovery in a loop-level test.
- `response.failed` with `server_is_overloaded` raises `TransientProviderError`.
- `response.failed` with `slow_down` raises `TransientProviderError`.
- `response.failed` with `rate_limit_exceeded` and `try again in 2s` raises
  `TransientProviderError(retry_after=2)`.
- `response.failed` with `cyber_policy` raises clear non-retryable error.
- `response.failed` with `invalid_prompt` raises clear non-retryable error.
- `response.incomplete` with `max_output_tokens` still maps to `max_tokens`.
- `response.incomplete` with another reason is not treated as `end_turn`.
- HTTP 429 quota/usage body gets a clear non-retryable message.
- HTTP 429 retryable rate-limit body/header can become `TransientProviderError`
  if that behavior is chosen.

OpenAI Responses provider tests in `tests/test_provider_openai_responses.py`:

- SDK 503 remains retryable.
- SDK connection/timeout remains retryable.
- structured overload/rate-limit/context/quota/policy/invalid-prompt payloads are
  classified correctly where the SDK exposes them.

Shared loop/recovery tests in `tests/test_loop.py`:

- transient provider error from non-streaming completion retries and succeeds.
- context-length error from provider response triggers forced compaction recovery.

TUI tests in `tests/test_tui.py`:

- transient `response.failed` path shows reconnect status and clears partial
  streamed content, matching existing retry visual contract.
- exhausted transient path keeps prompt readable and preserves history.

Headless/CLI tests in `tests/test_cli.py` or `tests/test_loop.py`:

- headless benefits through shared loop retry for non-streaming provider errors.
- if CLI-level error formatting is added later, assert stderr behavior there.

### 7. Validation Commands

Run focused checks first:

```text
uv run pytest \
  tests/test_provider_openai_codex.py \
  tests/test_provider_openai_responses.py \
  tests/test_loop.py \
  tests/test_tui.py::test_basic_tui_retries_transient_error_and_reports_reconnect \
  tests/test_tui.py::test_basic_tui_exhausted_transient_error_keeps_prompt_readable
```

If TUI rendering behavior changes, also run the relevant PTY harness per
`AGENTS.md`.

## Rollout Order

1. Implement Codex SSE `response.failed` classification.
2. Fix Codex `response.incomplete` semantics.
3. Add HTTP body classification for Codex HTTP errors.
4. Mirror safe classifications in OpenAI Responses provider.
5. Add/adjust shared classifier utilities only where duplication becomes awkward.
6. Improve headless CLI presentation if needed as a follow-up, not as part of
   the provider classification patch.

## Open Questions

- Should unknown `response.failed` be retryable by default, matching Codex's
  broad `ApiError::Retryable`, or should Wattle keep unknown failures
  non-retryable until classified?
- Should Wattle add explicit non-retryable provider error classes, or keep clear
  `RuntimeError` messages for quota/policy/invalid prompt in the first pass?
- Should HTTP 429 `rate_limit_exceeded` without a retry hint be retried using
  exponential backoff, or surfaced immediately?
- Should headless mode catch provider failures and print concise stderr messages,
  or continue propagating exceptions for now?

## Acceptance Criteria

- No `response.failed` Codex SSE event with a known error code is surfaced as an
  opaque generic `RuntimeError`.
- Non-`max_output_tokens` `response.incomplete` is not treated as successful
  `end_turn`.
- Overload and transient rate-limit cases retry through shared recovery in both
  TUI and headless paths.
- Context-length provider errors still trigger compaction recovery.
- Quota, unsupported-plan, invalid-prompt, cyber-policy, and auth failures remain
  non-retryable and have clear user-facing messages.
- Existing retry tests continue to pass.
