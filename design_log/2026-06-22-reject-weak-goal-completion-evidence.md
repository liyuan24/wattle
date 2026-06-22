# Reject Weak Goal Completion Evidence

Date: 2026-06-22

## Context

The latest evaluated baseline, `b458681`, restored the previous 1/7 result after
backing out the structured evidence schema. The remaining failures showed a
repeated pattern: Wattle often marked a goal complete after evidence that was
only about the artifact surface, not the task semantics.

Examples from `wattle-goal-backout-structured-gcp-1attempt-20260622`:

- `raman-fitting` completed after proving only that `/app/results.json` parsed
  and had numeric fields.
- `sam-cell-seg` completed while explicitly saying full runtime execution was
  not possible because dependencies were unavailable.
- `video-processing` and `mteb-retrieve` did run artifacts, but still show that
  completion evidence must stay tied to the actual consumer and source-of-truth
  semantics.

## Change

`update_goal(status="complete")` now rejects evidence that admits required
validation could not run, missing dependencies prevented execution, or the
completed claim rests only on syntax, schema, type, parse, finite-value, or
file-existence checks.

This is intentionally narrower than the reverted structured evidence schema. It
does not require new fields or a fixed evidence format. It only prevents the
model from converting explicitly weak evidence into a completed goal.

## Expected Effect

The model should keep working when its own evidence says a deliverable was not
actually exercised or only superficially checked. This should improve general
task completion behavior for generated scripts, serialized outputs, and
scientific/data artifacts without adding task-specific benchmark knowledge.
