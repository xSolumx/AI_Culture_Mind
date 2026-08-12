# Latent Cayley Adversarial Audit Contract

## Status

Written after the ten-seed table-blind cohort and Claude's review, but before
running the analyses below. This is a prospective post-hoc stress audit, not a
retroactive preregistration of the completed cohort.

## Corrections to test

1. Dense canonical prefix labels are informationally equivalent to a Cayley
   table once every deterministic transition is covered. The correct claim is
   removal of the explicit table object from the compiler, not weakly
   supervised algebra discovery.
2. Order 60 does not identify A5. The recovered group must be checked directly.
3. The recovery gauge is mathematically arbitrary but the implementation uses
   a deterministic base label and closure order. It may therefore be identical
   across seeds; this must be measured rather than described as independently
   random.

## Fixed audit

For every training seed:

- regenerate only the original-alphabet training batches used by recovery;
- report first-batch transition coverage and the step of full coverage;
- reconstruct the latent group without the true table;
- verify noncommutativity, conjugacy-class sizes, and absence of a nontrivial
  normal subgroup;
- only after recovery, compare the deterministic recovered-element-to-true-
  label map against the known A5 table and require an exact isomorphism;
- assert the original training elements are disjoint from class 33 and that no
  recovered token permutation equals a class-33 token permutation;
- delete one completed transition edge and confirm the current exact recovery
  algorithm refuses to fill it. This is a negative robustness control, not a
  pass claim; and
- load the frozen table-blind checkpoint and evaluate class 33 at every
  multiple of 1024 from L4096 through L16384, one deterministic batch of 256.

The dense interior sweep passes only if every seed remains at least 90% at
every length. No model is retrained and no evaluator parameter is changed after
the sweep begins.

## Results

All algebra and leakage checks pass in all ten seeds. Every recovered group is
noncommutative and simple, has conjugacy-class sizes `1, 12, 12, 15, 20`, and
is exactly isomorphic to the evaluator's A5 table under the recovered
element-to-label map. Training elements and class-33 elements are disjoint,
with zero equal token permutations.

The gauge is deterministic, not seed-random: all ten recover the same
non-identity permutation, with only label 0 fixed. First-batch coverage ranges
from `99.17%` to `100%`; three seeds require a second batch. Removing one edge
is refused in all ten, confirming that partial-table completion remains
unsolved.

All 13 class-33 lengths from L4096 through L16384 score 100% in every seed. The
largest sequential-versus-direct state drift is `0.00624`.

Artifact: `latent_cayley_adversarial_audit_10seeds.json`.
