# Verifier-minded contradiction pass

Date: 2026-06-22

The latest Terminal-Bench goal-mode run improved `video-processing`, but most
remaining failures still came from plausible artifacts that were validated
against weak evidence. Several completions checked that the produced file
parsed, matched values already written, or matched one convenient interpretation
of the source data. Hidden verifiers then failed on semantic details such as WAL
sidecar updates, embedding ranking conventions, Raman fitting windows, primer
interpretation, and literal parsed container types.

This change adds a general contradiction-seeking validation requirement to both
the base system prompt and the goal continuation prompt. Before finalizing a
concrete artifact or answer, Wattle now asks the model to identify plausible
alternate interpretations that would change the result, then eliminate them
using source data, specifications, consumer behavior, or small executable
checks. The examples cover recurring classes of verifier-sensitive ambiguity:
indexing and ranking conventions, ties, units, coordinate systems, model/data
revisions, parser literal types, transaction sidecars, and fitting windows or
baselines.

The `update_goal` evidence gate is also tightened in a narrow way. It now rejects
generic evidence such as "focused tests passed and artifact exists" unless the
evidence names the exercised test, fixture, consumer, source data, or contract.
It also rejects self-referential artifact evidence, such as saying an output
matches the current result file, unless the evidence ties that check to an
authoritative source, consumer, fixture, verifier, or oracle. This is intended to
discourage false completion without requiring benchmark-specific knowledge.
