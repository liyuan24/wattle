# Provider Error Handling Unification Plan

Date: 2026-05-25

## Goal

Extend the stronger Codex/OpenAI Responses error handling to every supported
Wattle provider while keeping the implementation mostly provider-neutral.

The target outcome is:

- transient overload, timeout, connection, and retryable rate-limit failures
  retry through the existing shared recovery path;
- context-window failures continue to trigger compaction recovery;
- quota, billing, auth, permission, policy, invalid request, and request-size
  failures surface as clear non-retryable user messages;
- adding a new provider should mostly require mapping its SDK exception shape
  into one shared classifier, not rewriting retry policy.

## Official API References Checked

OpenAI:

- Error codes: https://platform.openai.com/docs/guides/error-codes/api-errors
  - OpenAI separates 429 rate-limit errors from 429 quota/usage-limit errors.
  - 500/internal errors and 503 overload/slow-down errors should be retried
    after a wait.
  - Python SDK classes include `APIConnectionError`, `APITimeoutError`,
    `InternalServerError`, `RateLimitError`, `AuthenticationError`,
    `BadRequestError`, `PermissionDeniedError`, `NotFoundError`, and others.
- Rate limits: https://platform.openai.com/docs/guides/rate-limits
  - Rate limits are per organization/project/model family and can be request,
    token, image, daily, or monthly limits.
  - OpenAI recommends retrying rate-limit errors with exponential backoff, but
    notes that failed retries still count toward the per-minute limit.
  - Rate-limit headers include `x-ratelimit-*` limit, remaining, and reset
    values.

Anthropic:

- Errors: https://platform.claude.com/docs/en/api/errors
  - Error responses have a JSON top-level `error` object with `type` and
    `message`, plus a `request_id`.
  - Listed error types include `invalid_request_error`, `authentication_error`,
    `billing_error`, `permission_error`, `not_found_error`,
    `request_too_large`, `rate_limit_error`, `api_error`, `timeout_error`, and
    `overloaded_error`.
  - Anthropic maps overload to HTTP 529 on the Claude Platform.
  - Streaming can fail after a 200 response, so SSE errors still need stream
    classification.
- Rate limits: https://platform.claude.com/docs/en/api/rate-limits
  - 429 rate-limit errors include a `retry-after` header.
  - Anthropic also exposes `anthropic-ratelimit-*` headers for current limit,
    remaining capacity, and reset timing.

These sources match the current Wattle direction: retry transient transport and
overload failures, respect `retry-after` where available, and keep quota/auth/
request-shape failures non-retryable.

## Offline Implementation Notes

This plan should be implementable without live web search. Treat the official
API notes above as the captured source-of-truth summary for the first
implementation pass.

When implementing:

- Do not depend on browsing for provider behavior.
- Verify SDK exception shapes from the installed packages and local tests.
- Prefer structured payload, status-code, and header extraction over exact SDK
  class assumptions.
- Preserve the original provider message in normalized errors, especially for
  OpenAI-compatible providers such as DeepSeek, Kimi, and Minimax.
- Classify conservatively when a provider returns an unknown shape:
  - unknown 5xx and Anthropic 529 are retryable;
  - unknown 4xx is non-retryable;
  - unknown 429 is retryable only when payload/type/message looks like a rate
    limit rather than quota, billing, spend, plan, auth, or permission failure.
- Add regression tests for every new classification rule before widening it.

## Implementation Status

Implemented on 2026-05-25:

- Added normalized non-retryable provider error classes and shared metadata in
  `src/wattle/providers/base.py`.
- Expanded `src/wattle/provider_errors.py` into a shared extraction and
  classification module while preserving `is_context_length_error()` for
  compaction recovery.
- Refactored OpenAI Responses and Codex adapters to raise normalized shared
  errors instead of using duplicated local retry/quota/context classifiers.
- Added shared normalization around OpenAI Chat Completions create calls and
  stream iteration, covering OpenAI-compatible providers through structured
  payload/status-code classification first.
- Added shared normalization around Anthropic create calls, stream setup, and
  stream iteration/finalization.
- Kept retry/recovery policy centralized in `src/wattle/request_preparation.py`;
  provider adapters only normalize and re-raise.
- Added regression tests for shared classifier behavior plus Chat Completions and
  Anthropic create/stream failures.

Validation run:

```text
uv run pytest tests/test_provider_errors.py tests/test_provider_openai_codex.py tests/test_provider_openai_responses.py tests/test_provider_openai_completions.py tests/test_provider_anthropic.py tests/test_loop.py
# 173 passed

uv run ruff check src/wattle/provider_errors.py src/wattle/providers/base.py src/wattle/providers/__init__.py src/wattle/providers/openai_codex.py src/wattle/providers/openai_responses.py src/wattle/providers/openai_completions.py src/wattle/providers/anthropic.py tests/test_provider_errors.py tests/test_provider_openai_completions.py tests/test_provider_anthropic.py
# All checks passed
```

## Current Wattle State

Provider entrypoints:

- `anthropic` -> `src/wattle/providers/anthropic.py`
- `openai` / `deepseek` / `kimi` / `minimax` ->
  `src/wattle/providers/openai_completions.py`
- `openai_responses` -> `src/wattle/providers/openai_responses.py`
- `openai_codex` -> `src/wattle/providers/openai_codex.py`

Shared recovery already exists:

- `src/wattle/request_preparation.py`
  - `acomplete_with_recovery()`
  - `astream_with_recovery()`
  - retries `IncompleteStreamError` and `TransientProviderError`
  - forces compaction when `is_context_length_error()` matches
- `src/wattle/provider_errors.py`
  - currently only exposes context-length detection.

Gaps:

- Codex and OpenAI Responses now have local helpers for retryable HTTP status,
  payload parsing, `retry-after`, and provider-specific error codes.
- Anthropic and OpenAI Chat Completions/OpenAI-compatible providers do little or
  no exception translation, so SDK errors can escape as raw exceptions.
- The same classification vocabulary is duplicated or missing across provider
  adapters.
- The existing retry layer can only act when provider adapters raise the shared
  Wattle exception types.

## Design

### 1. Add Provider-Neutral Error Classes

Add lightweight error classes in `src/wattle/providers/base.py`, next to
`TransientProviderError`:

- `ProviderError`
  - base class for normalized provider-facing failures.
  - fields:
    - `message: str`
    - `kind: ProviderErrorKind`
    - `status_code: int | None`
    - `code: str | None`
    - `retry_after: float | None`
    - `request_id: str | None`
    - `provider: str | None`
- `ProviderQuotaError`
- `ProviderAuthError`
- `ProviderPermissionError`
- `ProviderInvalidRequestError`
- `ProviderBillingError`
- `ProviderPolicyError`
- `ProviderRequestTooLargeError`

Keep `TransientProviderError` for retryable cases because the retry loop already
uses it. Either make it subclass `ProviderError` or keep it separate but give it
the same metadata fields.

Keep `IncompleteStreamError` for incomplete stream/completion states that are
safe to replay.

### 2. Move Classification Into `provider_errors.py`

Expand `src/wattle/provider_errors.py` from context-only matching into the
shared classifier module.

Suggested public/internal helpers:

- `extract_error_payload(error: object) -> ProviderErrorPayload | None`
  - recursively read common SDK shapes:
    - `exc.body`
    - `exc.response`
    - `response.text`
    - `response.content`
    - JSON `{ "error": { ... } }`
    - Anthropic `{ "type": "error", "error": { ... }, "request_id": ... }`
    - OpenAI `{ "error": { "code": ..., "type": ..., "message": ... } }`
- `error_code(payload_or_error) -> str | None`
- `error_message(payload_or_error) -> str | None`
- `status_code_from_error(error: object) -> int | None`
- `headers_from_error(error: object) -> Mapping[str, str] | None`
- `retry_after_from_headers(headers) -> float | None`
- `retry_after_from_message(message) -> float | None`
- `normalize_provider_error(error, provider=None) -> BaseException`

The normalizer should return:

- `TransientProviderError` for:
  - HTTP 408, 500, 502, 503, 504;
  - Anthropic 529 / `overloaded_error`;
  - OpenAI `server_is_overloaded`, `slow_down`;
  - Anthropic `timeout_error`;
  - SDK connection/timeout/internal-server classes;
  - rate-limit errors only when they are actually rate limits, not quota.
- `RuntimeError` or a dedicated context error recognized by
  `is_context_length_error()` for context-window overflow.
- non-retryable provider errors for auth, permission, billing, quota,
  invalid-request, request-too-large, not-found, and policy failures.

Keep status-code fallback conservative:

- Retry `408`, `500`, `502`, `503`, `504`, and `529`.
- Do not retry `400`, `401`, `402`, `403`, `404`, or `413`.
- For `429`, inspect code/type/message:
  - `rate_limit_error`, `rate_limit_exceeded`, or OpenAI SDK
    `RateLimitError` with no quota wording -> retryable with `retry_after` or
    exponential backoff.
  - `insufficient_quota`, `usage_limit_reached`, billing/spend/quota wording ->
    non-retryable quota/billing error.

### 3. Deduplicate OpenAI/Codex Helpers

Move these duplicated helpers out of `openai_codex.py` and
`openai_responses.py` into `provider_errors.py`:

- JSON error body extraction.
- `retry-after` parsing.
- "try again in 2s" parsing.
- retryable HTTP status detection.
- OpenAI-style error-code mapping:
  - `context_length_exceeded`
  - `server_is_overloaded`
  - `slow_down`
  - `rate_limit_exceeded`
  - `insufficient_quota`
  - `usage_limit_reached`
  - `usage_not_included`
  - `cyber_policy`
  - `invalid_prompt`

Then update both providers to call shared helpers.

Provider-local code should only do provider-specific event extraction:

- Codex SSE: extract `response.failed.response.error` and
  `response.incomplete.incomplete_details.reason`.
- OpenAI Responses SDK: extract response status, incomplete reason, SDK body,
  and SDK response headers.

### 4. Add Anthropic Adapter Error Normalization

Update `src/wattle/providers/anthropic.py`:

- Wrap `messages.create()` and `messages.stream()` setup/finalization in a
  small `_raise_normalized_provider_error(exc)` helper.
- For Anthropic SDK exceptions, inspect:
  - exception class name;
  - `status_code`;
  - `body`;
  - `response`;
  - headers including `retry-after`, `request-id`,
    `anthropic-ratelimit-*`, and possibly `x-should-retry`.
- Map:
  - `rate_limit_error` / HTTP 429 -> `TransientProviderError` with
    `retry_after`.
  - `api_error` / HTTP 500 -> `TransientProviderError` unless
    `x-should-retry: false`.
  - `timeout_error` / HTTP 504 -> `TransientProviderError`.
  - `overloaded_error` / HTTP 529 -> `TransientProviderError`.
  - `invalid_request_error` / HTTP 400 -> `ProviderInvalidRequestError`.
  - `authentication_error` / HTTP 401 -> `ProviderAuthError`.
  - `billing_error` / HTTP 402 -> `ProviderBillingError`.
  - `permission_error` / HTTP 403 -> `ProviderPermissionError`.
  - `request_too_large` / HTTP 413 -> `ProviderRequestTooLargeError`.
- For streaming errors after HTTP 200, let the same normalizer handle the raised
  event/SDK exception. If the SDK surfaces a stream close without a final
  message, translate that to `IncompleteStreamError`.

### 5. Add OpenAI Chat Completions / Compatible Adapter Normalization

Update `src/wattle/providers/openai_completions.py`:

- Wrap both `chat.completions.create()` calls and stream iteration.
- Reuse the same OpenAI-style classifier as OpenAI Responses.
- Map official OpenAI SDK classes:
  - `APIConnectionError` -> `TransientProviderError`
  - `APITimeoutError` -> `TransientProviderError`
  - `InternalServerError` -> `TransientProviderError`
  - `RateLimitError` -> retryable unless payload/message says quota or usage
    limit
  - `AuthenticationError` -> `ProviderAuthError`
  - `PermissionDeniedError` -> `ProviderPermissionError`
  - `BadRequestError` -> context error if context-length, otherwise
    `ProviderInvalidRequestError`
  - `NotFoundError` -> non-retryable invalid/not-found error
  - `UnprocessableEntityError` -> `ProviderInvalidRequestError`
- For DeepSeek/Kimi/Minimax and other OpenAI-compatible endpoints:
  - classify by status code and payload first;
  - fall back to SDK class name only when structured payload is absent;
  - preserve original message because compatible providers often expose
    provider-specific validation details.

### 6. Keep Retry Policy in One Place

Do not add retry loops inside provider adapters.

Provider adapters should normalize exceptions and re-raise. The shared retry
policy stays in:

- `RequestPreparer.acomplete_with_recovery()`
- `RequestPreparer.astream_with_recovery()`

This keeps headless, TUI, subagents, and future providers consistent.

Enhance `_retry_delay_for_error()` only if needed:

- prefer `TransientProviderError.retry_after`, capped as today;
- otherwise exponential backoff;
- optionally add jitter later, but avoid changing UX in the first pass.

### 7. TUI Presentation

Keep UI changes small:

- Existing retry status can continue showing retry attempt, max attempts, error,
  and delay.
- For final non-retryable provider errors, include:
  - provider name when known;
  - request ID when known;
  - clear user action for quota, billing, auth, permission, and request-size
    errors.
- Avoid stack traces for normalized provider errors.

### 8. Implementation Order

1. Add provider error classes and shared metadata fields.
2. Expand `provider_errors.py` with extraction and classification helpers.
3. Move OpenAI/Codex duplicate helpers into the shared module while preserving
   existing tests.
4. Update `openai_codex.py` and `openai_responses.py` to use the shared helper.
5. Add `openai_completions.py` normalization and tests.
6. Add `anthropic.py` normalization and tests.
7. Add loop/request-preparation tests proving:
   - transient normalized errors retry;
   - context errors still compact and retry;
   - non-retryable normalized errors do not retry.
8. Add TUI tests for final error presentation if message text changes.

## Test Plan

Shared classifier tests in `tests/test_provider_errors.py`:

- OpenAI 429 rate-limit payload becomes retryable with `retry_after`.
- OpenAI 429 quota/usage payload becomes non-retryable quota error.
- OpenAI 503 overload and slow-down become retryable.
- Anthropic 529 `overloaded_error` becomes retryable.
- Anthropic 429 `rate_limit_error` with `retry-after` becomes retryable.
- Anthropic 402 `billing_error` is non-retryable.
- Anthropic 413 `request_too_large` is non-retryable.
- Context-length payloads are still recognized by `is_context_length_error()`.
- Unknown 5xx is retryable; unknown 4xx is non-retryable.

Provider tests:

- `tests/test_provider_openai_responses.py`
  - existing tests should continue to pass after helper extraction.
- `tests/test_provider_openai_codex.py`
  - existing tests should continue to pass after helper extraction.
- `tests/test_provider_openai_completions.py`
  - add SDK connection/timeout/internal/rate-limit/auth/bad-request cases.
  - add stream iteration failure cases, not only stream creation failure.
- `tests/test_provider_anthropic.py`
  - add 429, 500, 504, 529, 401, 402, 403, 413 cases.
  - add stream event/final-message failure cases if the SDK exposes them in
    tests.

Shared loop tests:

- `tests/test_loop.py`
  - normalized transient provider error retries and succeeds.
  - normalized context error triggers compaction and retries.
  - normalized quota/auth/invalid request errors do not retry.

TUI tests:

- `tests/test_tui.py`
  - final `ProviderQuotaError` shows a concise error and usable prompt.
  - retryable provider errors still clear partial streamed text before retry.

## Acceptance Criteria

- Every provider adapter either returns a normal `CompletionResponse` /
  `StreamComplete` or raises a normalized Wattle provider error.
- Retryable failures use `TransientProviderError` or `IncompleteStreamError`.
- Non-retryable failures do not enter the retry loop.
- OpenAI Responses and Codex behavior does not regress.
- Anthropic overload/rate-limit/timeout failures retry through shared recovery.
- Chat Completions and OpenAI-compatible providers retry transient failures and
  surface quota/auth/request-shape failures clearly.
- Adding a new provider requires only:
  - response/event extraction;
  - optional provider-specific code aliases;
  - no new retry loop.
