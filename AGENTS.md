# Wattle Agent Notes

## Git Workflow

For Wattle development, prefer committing directly to the mainline branch (`master`) and pushing it when the user asks to commit or push. Do not create a feature branch or PR by default unless the user explicitly requests that workflow.

## Dependency Updates

When adding or changing runtime dependencies in `pyproject.toml`, update the repo environment and the installed CLI tool environment. Run `uv sync` for the local project, then refresh the shell `wattle` command with `uv tool install --force .` because `~/.local/bin/wattle` is a separate `uv tool` environment and does not automatically pick up new dependencies from `.venv` or `uv build`.

After refreshing the tool, verify the dependency through the interpreter used by the shell command. Inspect `which wattle` and the wrapper if needed, then run a lightweight command such as `wattle --help` or an import check through `/Users/liyuan/.local/share/uv/tools/wattle/bin/python3`.

## TUI Verification

For any change that touches Wattle TUI behavior, run the relevant PTY harness immediately after the focused unit tests. This includes terminal input, prompt rendering, live redraw, queued messages, image anchors, subagent status, and tool status UI.

Prefer reproducing ambiguous TUI bugs through `tests/test_tui_pty.py` instead of relying only on parser/helper tests. For timing-sensitive PTY assertions, wait for the actual user-visible text or state being validated, not an earlier nearby marker.

Before editing TUI rendering behavior, write down the visual contracts the fix must preserve and review the diff against them before finalizing. For prompt/transcript work, preserve three-row submitted user and assistant message blocks, the distinct user/input background, full-width prompt and status backgrounds, long-message soft wrapping, and resize behavior unless the requested change explicitly says otherwise.

TUI regressions should assert both sides of the change: the new bug is fixed and the existing visual contract still holds. Prefer PTY screen-state assertions for row counts, backgrounds, wrapping, and resize behavior; do not update tests merely to match an accidental visual regression.

Keep terminal rendering helpers scoped to their surface: prompt and panel rows, transcript rows, status rows, and running-status rows have different wrapping and background-fill contracts. Do not reuse one helper across these surfaces unless tests prove the contracts match.
