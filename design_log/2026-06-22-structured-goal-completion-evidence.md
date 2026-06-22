# Structured Goal Completion Evidence

## Context

The latest Terminal-Bench eval of commit `f540698` improved one case (`mteb-retrieve`)
but still showed repeated false completion across semantic and contract-heavy tasks.
The agents often produced artifacts and then supplied plausible evidence, but the
evidence did not prove the exact downstream contract or the domain answer:

- `sam-cell-seg` verified argparse positional order while the verifier invoked
  flags.
- `dna-insert` validated one melting-temperature interpretation while the verifier
  used a different primer interpretation.
- `mteb-leaderboard`, `db-wal-recovery`, and `raman-fitting` each had an artifact
  with a plausible self-check but the domain answer was wrong.
- `video-processing` showed that a single example run and visual audit can miss
  frame-convention details.

The previous prompt-only validation guidance was too easy for the model to satisfy
with generic claims. Requiring a single `evidence` string helped but did not force
the agent to separate requirement coverage, interface coverage, semantic correctness,
and residual uncertainty.

## Change

`update_goal(status="complete")` now requires structured completion evidence:

- `requirements_evidence`: how explicit requirements were derived and checked.
- `interface_evidence`: how the exact user/verifier-facing interface and
  representation contract were exercised.
- `semantic_evidence`: how the result was checked against authoritative source data,
  domain rules, or an independent oracle, including relevant alternate
  interpretations.
- `remaining_risk`: what remains unverified; material risk means the goal should
  stay active.

Blocked goals still require only the existing blocker evidence, because the extra
completion fields are not meaningful for a genuine impasse.

## Rationale

This is a general completion-gate improvement, not a task-specific benchmark rule.
It changes the final claim the model must make before Wattle accepts completion.
The goal is to make weak proxy validation harder to pass off as done and to keep
the agent working when exact invocation shape, parsing details, domain semantics,
or alternate interpretations have not been checked.

## Evaluation Plan

Run focused unit tests for the goal tool and TUI goal flow, then the full test suite.
The next Terminal-Bench eval should compare this commit against `f540698` on the same
7-task subset before making another improvement.
