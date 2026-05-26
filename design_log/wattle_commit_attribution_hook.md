# Wattle Commit Attribution Hook Plan

Date: 2026-05-25

## Goal

When a commit is created from inside Wattle, GitHub should show the Wattle
account/logo in commit attribution and contributor UI.

Use this co-author trailer:

```text
Co-authored-by: Wattle <287834001+wattle-coding@users.noreply.github.com>
```

## Background

GitHub contributor logos come from normal Git identity resolution. If a commit
author or `Co-authored-by` trailer email maps to a GitHub account, GitHub shows
that account's avatar.

The `wattle-coding` GitHub account already has the desired logo. Its public user
id is `287834001`, so the canonical no-reply email is:

```text
287834001+wattle-coding@users.noreply.github.com
```

## Prior Art Checked

### Codex

Codex does not appear to use a general commit attribution hook for user repo
commits.

Relevant observations:

- Codex uses Git environment injection via `GIT_CONFIG_COUNT`,
  `GIT_CONFIG_KEY_N`, and `GIT_CONFIG_VALUE_N` in sandbox setup.
- That usage is for Git config such as `safe.directory`, not commit trailers.
- Codex has an internal baseline commit message with:

  ```text
  Co-authored-by: Codex <noreply@openai.com>
  ```

  but this is a hardcoded internal baseline commit, not general handling for
  user-created commits.

### Pi

Pi does not appear to use a commit attribution hook either.

Relevant observations:

- Pi relies heavily on policy/instructions around Git workflow.
- It forbids `git commit --no-verify`.
- It asks agents to stage explicit file paths and use safe commit commands.
- Plan-mode checks classify `git commit` as a Git write operation.

## Recommendation

Implement Wattle commit attribution as a Wattle-managed Git `commit-msg` hook
activated only for shell commands executed by Wattle.

Do not permanently edit the user's `.git/hooks` directory.

Do not depend on model prompting alone. Prompting is too easy to forget and does
not catch arbitrary commit commands, scripts, or aliases.

## Implementation Status

Implemented on 2026-05-25:

- Added `src/wattle/git_attribution.py` with a Wattle-managed `commit-msg`
  hook generator and Git environment injection.
- Integrated attribution into `BashTool` foreground, TTY, and background command
  execution.
- Added `git_commit_attribution` to persistent settings, defaulting to `true`.
- Chained existing effective `commit-msg` hooks before adding the Wattle trailer.
- Preserved existing `GIT_CONFIG_COUNT` entries and also set
  `GIT_CONFIG_PARAMETERS` for compatibility with Apple's older system Git.
- Added unit/integration coverage in `tests/test_git_attribution.py` and kept
  existing bash-tool tests passing.

Validation run:

```text
uv run pytest tests/test_git_attribution.py tests/test_bash_tool.py tests/test_tools.py tests/test_cli.py
# 78 passed

uv run ruff check src/wattle/git_attribution.py src/wattle/tools/bash.py src/wattle/settings.py tests/test_git_attribution.py tests/test_bash_tool.py tests/test_tools.py tests/test_cli.py
# All checks passed
```

## Hook Behavior

The hook should append the Wattle trailer if it is missing:

```bash
#!/bin/sh
set -eu

git interpret-trailers --in-place --if-exists doNothing \
  --trailer "Co-authored-by: Wattle <287834001+wattle-coding@users.noreply.github.com>" \
  "$1"
```

Properties:

- Idempotent: existing Wattle trailers are not duplicated.
- Uses Git's trailer parser instead of ad hoc text appending.
- Preserves existing commit message body.

## Activation Model

For every Wattle `bash` command, inject Git config through environment
variables:

```text
GIT_CONFIG_COUNT=<existing count + N>
GIT_CONFIG_KEY_<n>=core.hooksPath
GIT_CONFIG_VALUE_<n>=<wattle-managed-hooks-dir>
```

This follows the same general pattern Codex uses for runtime Git config
injection, but applies it to hook routing.

The managed hook directory should live under Wattle runtime/session state, not
inside the user's permanent Git hook directory. Candidate location:

```text
~/.wattle/runtime/git-hooks/<hash-or-version>/commit-msg
```

or a session-scoped directory if cleanup and per-session isolation are preferred.

## Existing Hook Handling

This is the main design risk.

Setting `core.hooksPath` replaces the repo's normal hook location. If Wattle
does this naively, it can silently bypass project hooks such as lint checks,
commit-msg validation, or secret scanning.

Implementation must handle existing hooks deliberately.

Recommended behavior:

1. Discover the effective user hook path for the command's Git repo:

   ```bash
   git config --path --get core.hooksPath
   ```

   If unset, default to the repo's normal `.git/hooks`.

2. Generate a Wattle `commit-msg` hook that:

   - first runs the existing `commit-msg` hook if it exists and is executable
   - exits with the existing hook's non-zero status if it fails
   - then appends the Wattle trailer

3. If Wattle cannot safely determine or chain the existing hook, choose the
   conservative behavior:

   - do not inject `core.hooksPath`
   - let the commit proceed normally
   - surface a concise warning in the tool output that attribution was skipped
     because existing hooks could not be safely chained

Never silently bypass existing project hooks.

## Scope

Apply to commits created through Wattle-run shell commands:

- TUI bash tool
- headless agent bash tool
- subagent bash tool

Do not affect commits the user makes in their normal terminal outside Wattle.

## Edge Cases

- `git commit --no-verify` bypasses `commit-msg`; Wattle should continue
  discouraging or blocking this command if policy allows.
- Commands may use `git -C <path>` or `cd other/repo && git commit`; the hook
  setup should resolve the actual repo root for the command cwd when feasible.
- Commands may invoke scripts that call Git inside another repo; those may not
  be attributable unless Wattle can inject a hook path that works for that repo.
- Worktrees can use `.git` files instead of directories; repo root and hook path
  resolution must handle both.
- If the user has configured global `core.hooksPath`, Wattle must chain it, not
  bypass it.
- If `git interpret-trailers` is unavailable, skip attribution with a warning
  rather than doing fragile manual edits.

## Proposed Implementation Steps

1. Add a small Git attribution helper module.

   Responsibilities:

   - find Git repo root from a command cwd
   - resolve existing effective hooks path
   - create/update Wattle-managed hook directory
   - generate a chaining `commit-msg` hook
   - return environment additions for `GIT_CONFIG_COUNT`

2. Integrate the helper into `BashTool`.

   Before launching foreground, TTY, or background shell commands:

   - build the subprocess environment from `os.environ`
   - apply attribution hook env when enabled and safe
   - pass that env into `subprocess.Popen`

3. Add a setting.

   Suggested default:

   ```text
   git_commit_attribution = true
   ```

   Provide a way to disable it in settings for users who do not want Wattle
   attribution.

4. Add tests.

   Unit tests:

   - hook appends Wattle trailer
   - hook is idempotent
   - existing executable `commit-msg` is chained
   - existing hook failure prevents commit/trailer append
   - existing global/local `core.hooksPath` is respected
   - env injection preserves existing `GIT_CONFIG_COUNT`

   Integration test:

   - create temp repo
   - run BashTool with `git commit -m "test"`
   - assert commit message contains Wattle trailer

5. Add policy guard if desired.

   Consider rejecting `git commit --no-verify` in Wattle-run bash commands, or
   at least warning that it bypasses commit attribution and project hooks.

## Acceptance Criteria

- A normal `git commit` run inside Wattle receives the Wattle co-author trailer.
- Existing project commit hooks still run and can block the commit.
- Wattle does not modify `.git/hooks` permanently.
- User terminal commits outside Wattle are unaffected.
- The behavior applies consistently to TUI, headless, and subagent shell
  execution.
- Users can disable the feature through settings.
