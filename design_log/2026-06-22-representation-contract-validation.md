# Representation Contract Validation

## Context

The hidden-verifier mismatch failures were not simple missing-file cases. The
artifacts existed and looked close to the requested shape, but the delivered
representation did not match what a downstream consumer or verifier required.
Examples from the failure analysis included semantic interpretation drift and
serialized values that parsed to the wrong runtime type.

This points to a general weakness: agents can over-trust visual inspection,
surface schema checks, or permissive parsing when the actual contract is the
consumer's parsed representation.

## Change

Strengthen model-facing guidance to require representation-contract validation
for serialized outputs and generated artifacts. Wattle now tells the model to
check artifacts the way downstream consumers would parse them, including exact
scalar/container types, field names, ordering, delimiters, quoting, escaping,
units, coordinate conventions, and other literal details that can affect
correctness.

The same idea is included in the persistent goal completion audit and the
update_goal tool description so long-running goal work does not mark completion
based only on a plausible-looking artifact.

## Scope

This is benchmark-neutral. It does not reference hidden verifier code and does
not depend on copying verifier `/tests` directories into task containers. The
intended check should be derived from the task-visible contract, public examples,
and the actual parser or consumer format whenever available.
