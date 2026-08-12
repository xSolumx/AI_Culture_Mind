# Learned endpoint-manifold compiler: audited results

Date completed: 2026-08-03.

## Outcome

The frozen zero-additional-query compiler succeeds in all ten seeds. Every
seed accepts an A5-isomorphic 60-state table at the first eligible compilation
attempt, step 850, then jointly retracts the complete four-token action family.

| Outcome category | Seeds |
|---|---:|
| Accepted and post-hoc A5-isomorphic | 10/10 |
| Numerically/structurally rejected as the seed outcome | 0/10 |
| Accepted but wrong structure | 0/10 |
| Dense original and untouched-alphabet gates | 10/10 |
| L4096 and L16384 checkpoint gates | 10/10 |
| 13-point untouched L4096--L16384 sweep | 10/10, every point 100% |

The compiler consumes zero new endpoint labels. It reuses 16,384 L8 examples
from the frozen 512,000-label neural curriculum. It receives no Cayley table,
permutation action, inverse map, token-to-element map, character, irrep branch,
or representative-extension queries.

## Frozen-threshold margins

The cohort is not a collection of threshold grazes. Selected candidates have:

| Metric | Observed range | Frozen acceptance |
|---|---:|---:|
| Alignment RMS | 0.00543--0.01162 | <=0.08 |
| Runner-up gap | 0.53085--0.53813 | >=0.20 |
| Commutator separation | 1.40497--1.42383 | >=0.50 |
| Assignment gap | 0.49218--0.51541 | >=0.10 |
| Maximum product residual | 0.02399--0.03842 | <=0.20 |
| Class consistency RMS | 0.01773--0.03396 | reported, not gated |
| Product RMS | 0.00891--0.01840 | reported, not gated |

After exact joint retraction, compiler invariance RMS is
`2.39e-16`--`2.24e-14` and homomorphism RMS is
`2.79e-16`--`1.13e-15`.

## Complete candidate audit

All four channels at every attempted compilation were persisted before the
formal results were inspected. At step 850 across ten seeds:

- 22/40 channels produce an exact Latin, associative, generated 60-state
  table; every one is post-hoc A5-isomorphic and passes all frozen numerical
  thresholds;
- 18/40 are structurally rejected because nearest-center products do not form
  row permutations;
- zero structurally valid channels recover a wrong group;
- zero seeds require a second attempt.

The weakest nonselected structurally valid channel has assignment gap `0.232`
and maximum product residual `0.1945`, close to the `0.20` residual ceiling.
It is reported rather than hidden, but it never wins alignment. The selected
candidate ranges above are the relevant seed-level margins.

## Behavioral gates

Every seed scores 100% at every original and untouched class-59 dense
L16--L256 checkpoint, at L4096 on both alphabets, and at class-59 L16384.
The separate 13-point class-59 L4096--L16384 audit is recorded in
`endpoint_manifold_long_audit_10seeds.json`: all 130 seed/length cells are
100%, with maximum path-versus-canonical state drift `0.004546`.

## Claim boundary

This is a zero-**additional-query** compiler, not unsupervised group discovery.
Training still uses anonymous exact endpoint classes, knows there are 60
outputs, includes L1/L2 examples, and comes from a hidden A5 environment. The
new result is that the learned action geometry plus already-consumed endpoint
labels suffice to reconstruct multiplication and select a shared exact
representation manifold without a supplied table or a separate membership
query phase.

Artifacts:

- `endpoint_manifold_10seeds_audited.json`
- `endpoint_manifold_audited_checkpoints/`
- `endpoint_manifold_long_audit_10seeds.json`
- `ENDPOINT_MANIFOLD_COMPILER_PREREGISTRATION.md`
