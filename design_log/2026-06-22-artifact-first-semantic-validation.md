# Artifact-first semantic validation

Date: 2026-06-22

The verifier-minded contradiction-pass eval produced one pass out of seven. It
improved the `mteb-leaderboard` semantic-answer case and reduced timeouts and
tokens, but several failures still completed after weak checks. The common
pattern was validating intent or surface behavior instead of validating the final
artifact as the downstream consumer/verifier would consume it:

- A generated video analyzer was considered complete because it ran, wrote TOML,
  and produced integer fields, even though its frame indices were semantically
  wrong.
- A mask conversion script was considered complete after AST/interface checks
  and a stubbed runtime, but the real verifier invocation failed.
- Primer validation used intended annealing components rather than proving that
  the final FASTA, parsed as the consumer would parse it, satisfied the
  ground-truth `oligotm` constraints.

This change adds artifact-first validation guidance to both the base system
prompt and goal continuation prompt. For generated scripts and serialized
artifacts, Wattle now tells the model to parse or invoke the final file exactly
as the downstream consumer will, then derive semantic checks from that parsed
artifact. It explicitly says that "script runs + output parses + required
fields" is surface evidence until checked against an oracle, source data,
expected behavior, or faithful domain invariant.

The `update_goal` evidence filter was tightened accordingly. Surface-only
phrases now include common weak completion claims such as AST parsing, script
existence, running on an example, writing an output file, required/integer fields,
TOML parsing, argparse checks, and stubbed validation. The previous semantic
markers `ran` and `executed` were removed because execution alone is not semantic
evidence. Strong evidence still passes when it ties parsing/execution to a
fixture, consumer, source data, independent oracle, expected behavior, or other
authoritative check.
