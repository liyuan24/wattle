# Require Goal Completion Evidence

## Context

The targeted GCP eval for the semantic-validation and representation-contract
guidance changes ran under:

`wattle-goal-validation-contract-gcp-1attempt-20260622`

It used Wattle commit `b9ab115` and completed 7 trials after one Spot preemption
and a successful resume. The result was `0/7`, worse than the prior `1/7` goal
run. The important pattern was not just failure; agents still made bare
`update_goal(status="complete")`-style completion decisions after plausible but
weak self-verification. Examples included domain-answer tasks where the agent
claimed independent recomputation but still produced the same wrong answer, and
artifact tasks where schema or compile checks were accepted instead of running
the downstream command path.

Prompt-only guidance did not create enough friction at the actual completion
boundary.

## Change

Make `update_goal` require a non-empty `evidence` argument for both `complete`
and `blocked`. The tool schema now exposes this requirement to the model, and
the runtime refuses missing or empty evidence without closing the active goal.

This is still a general mechanism. It does not inspect benchmark-specific
answers and does not know hidden verifier expectations. It simply prevents the
goal state from closing on a status-only tool call and forces the model to
serialize the current evidence it is relying on when making the completion or
blocked claim.

## Expected Effect

The evidence text cannot prove semantic truth by itself, but it should:

- make completion decisions more deliberate,
- put the claimed verification in the tool-call record for later analysis,
- give the model one more chance to notice missing authoritative evidence before
  closing a goal,
- reduce accidental or reflexive completion calls after weak validation.
