# Semantic Validation Discipline

## Context

Recent Terminal-Bench goal-mode failures shared a common pattern: the agent often
declared work complete after checking only surface properties of the result. Those
checks included file existence, requested schema shape, finite numeric values,
row counts, or consistency with values the agent had already chosen. That was not
enough for tasks where correctness depended on source data, measurements,
domain rules, or an exact artifact contract.

The successful video-processing run was different: the agent caught a bad initial
answer and refined its implementation after comparing the output against the
task's actual timing semantics.

## Change

Strengthen Wattle's model-facing validation guidance in two places:

- The global validation discipline now asks for an independently derived oracle
  from authoritative sources when tasks depend on data, files, measurements, or
  domain rules.
- The persistent goal continuation and update_goal tool instructions now make
  completion require independent semantic evidence for data/artifact/domain
  answers, not just existence, schema, finite values, or self-consistency.

This is intentionally general. It does not name benchmark tasks or prescribe
task-specific answers. The goal is to make Wattle less likely to prematurely
stop on any task whose deliverable can be structurally valid but semantically
wrong.

## Validation Note

Future Terminal-Bench validation should not copy verifier `/tests` directories
into task containers. The intended behavior is to derive the strongest faithful
check available from task-visible source data, public specs, examples, and the
requested artifact contract, while leaving hidden verifier tests hidden.
