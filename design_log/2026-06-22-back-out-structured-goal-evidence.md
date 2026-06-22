# Back Out Structured Goal Evidence Gate

Date: 2026-06-22

## Context

The structured completion evidence gate required `update_goal(status="complete")`
calls to include `requirements_evidence`, `interface_evidence`,
`semantic_evidence`, and `remaining_risk`. The intent was to make completion
claims harder to make without checking task requirements, downstream interface
details, and semantic correctness.

## Evaluation Result

The Spot-backed GCP eval for commit `d26b5ab` was worse than the prior evaluated
commit:

- Label: `wattle-goal-structured-evidence-gcp-1attempt-20260622`
- Result: 0/7 tasks passed, 0 errors.
- Prior evaluated commit `f540698` passed 1/7 tasks with the same task set.
- `mteb-retrieve` regressed from pass to fail.

The gate improved some surface behavior, especially `sam-cell-seg`, but the
aggregate result showed it was too heavy and did not reliably improve the actual
semantic checks that matter for Terminal-Bench tasks.

## Change

Back out only the hard structured field requirement. Keep the earlier evidence
requirement and the completion-audit prompt language that asks the model to use
authoritative, current-state evidence and representation-contract checks before
marking a goal complete.

## Expected Effect

This restores the better evaluated baseline behavior from `f540698` while
preserving the useful audit guidance. Further changes should be smaller and
evaluated independently rather than adding more mandatory completion bureaucracy.
