# Program 02: Shared-family representation learning

## Object

Families of token actions constrained jointly to a common group
representation or structured assignment manifold. The central question is
relational identifiability: what can be recovered from the family that cannot
be recovered by fitting each action independently?

## Distinct claims

- Joint family retraction can eliminate null directions left by independent
  endpoint fitting.
- Curriculum and compiler experiments test optimization and latent structure
  recovery separately from representational capacity.
- Finite-group results concern controlled A5/Q8-style tasks, not natural
  language or generic memory.

## Nonclaims

- Joint retraction is not intrinsically triality-specific; direct and
  permutation controls must accompany geometric variants.
- Successful endpoint completion does not prove that a model inferred semantic
  labels unless those labels were withheld by construction.
- Table-blind compilation and state-only compilation answer different
  supervision questions.

## Canonical evidence

This program currently lives inside the theorem submodule. Use its
[`program ledger`](../../Spin8-Triality-Research/programs/shared-family-learning/README.md)
for the claim-to-artifact map.

## Next publishable question

Formulate shared-family retraction as a general identifiability principle and
compare it against independently normalized families on several groups and
non-group assignment problems. The paper should not depend on sequence-model
quality claims.
