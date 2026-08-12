# Observable A5 Irrep Result

## Status and provenance

This note records a post-hoc mechanistic discovery made after the unchanged
ten-seed coprime multi-scale holonomy run completed. It does not replace the
pre-registered raw-operator gate. The ten-seed run used one configuration for
every seed; the channel audit then reran fixed representative seeds 1, 2, 6,
and 9 with identical training and additional read-only diagnostics.

## Dense reliability result

Dense multiples of the training length from L16 through L256 are now the
default evaluation protocol. The full ten-seed result is:

| seed | L64 | L128 | dense minimum | minimum length |
|---:|---:|---:|---:|---:|
| 0 | 100.00% | 100.00% | 99.61% | 208 |
| 1 | 100.00% | 100.00% | 100.00% | 16 |
| 2 | 100.00% | 99.90% | 92.29% | 240 |
| 3 | 100.00% | 100.00% | 100.00% | 16 |
| 4 | 100.00% | 100.00% | 96.39% | 240 |
| 5 | 100.00% | 100.00% | 98.24% | 32 |
| 6 | 97.46% | 100.00% | 76.37% | 176 |
| 7 | 100.00% | 99.90% | 93.55% | 240 |
| 8 | 100.00% | 100.00% | 98.05% | 176 |
| 9 | 98.93% | 100.00% | 87.50% | 240 |

The gate hierarchy is therefore:

- checkpoint functional gate: **10/10**;
- positive-margin multi-scale state contract: **10/10**;
- dense L16-L256 accuracy floor of at least 90%: **8/10**;
- original full-operator homomorphism RMS at most `1e-3`: **0/10**.

Mean L64/L128 accuracy is 99.64%/99.98%, but the mean dense minimum is 94.20%
and the median dense minimum is 97.22%. Seed 9 is a newly exposed
checkpoint-invisible failure. Thus fixed checkpoints are not an adequate
reliability protocol, and no finite dense sweep is an infinite-horizon
certificate.

One reproducibility caveat was found during the audit. The earlier selected
seed-4 run and the full-cohort seed-4 run have identical configuration and
initial-parameter hashes but dense minima of 99.80% and 96.39%, respectively.
Both pass, so this does not change the gate count, but it shows that seeded CUDA
execution was not bit-deterministic. The harness now enables deterministic
PyTorch algorithms and fixes `CUBLAS_WORKSPACE_CONFIG`; future reliability
claims must use that setting and distinguish seeds from exact reruns.

## Spontaneous irrep specialization

Every seed has one strongly non-commutative channel whose learned joint
geometry matches the exact real 3D icosahedral irrep. The exact oracle values
for this generator pair are:

- generator angles: `2.094395` and `1.256637` radians;
- generator-axis dot product: `0.187592`;
- maximum generator commutator: `1.224745`;
- finite-order cyclic and mixed relators: numerical zero, about `1e-15`.

Across all ten learned strong channels:

- angles lie near `2.077-2.087` and `1.255-1.259`;
- axis dots lie in `0.1831-0.1891`;
- commutators lie in `1.221-1.225`;
- the mean over four cyclic and six mixed finite-order relators is
  `0.0120-0.0267`.

The angles alone are not independent evidence: single-generator periodicity
plus the 2.2-radian cap largely forces the `2*pi/3` and `2*pi/5` branches. The
relative axis orientation and mixed relators are the joint-structure evidence.
The learned residual is still many orders above the oracle and can accumulate
along long paths.

The tempting random-axis coincidence calculation is not a valid p-value. The
channel was selected post hoc by commutator, seeds share data and architecture,
and mixed-word supervision constrains the axes. The oracle match is strong
mechanistic evidence, not an independent-uniform-axis significance test.

## Causal channel audit

The raw 32D homomorphism metric weights four channels equally. The audit asks
whether the oracle-like channel is actually reachable, used by the decoder,
sufficient, and necessary.

| seed | role | strong active hom RMS | decoder energy | orbit energy | full worst | strong only at that length | best one-channel removal |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | uniformly clean | 0.0133 | 90.0% | 82.1% | 100.0% | 100.0% | 100.0% |
| 2 | dense-pass trough | 0.0241 | 89.8% | 93.5% | 92.29% at L240 | 97.75% | 100.0% |
| 6 | dense failure | 0.0171 | 79.4% | 86.1% | 76.37% at L176 | 95.90% | 98.14% |
| 9 | dense failure | 0.0182 | 91.0% | 94.8% | 87.50% at L240 | 98.63% | 100.0% |

For seed 1, the strong channel alone gives 100% canonical accuracy and roughly
99% dense accuracy; removing it collapses dense accuracy to approximately
1-4%. Each weak channel alone is at chance, and removing any weak channel
preserves near-perfect behavior. The discovered irrep is therefore both
sufficient and necessary in the clean run.

For seeds 2, 6, and 9, the strong channel alone is more robust than the full
model at the full model's worst original-generator length. Removing a weak
channel restores 98-100% accuracy there. This establishes harmful
decoder-visible contributions on that distribution, not that every weak
channel is universally a nuisance.

The later changed-generator 15-subset audit falsifies that stronger claim.
Seed 1's full code scores 100.00% while its dominant channel alone scores
97.75%; the best subsets for seeds 2, 6, and 9 also retain one or two auxiliary
channels. The later ten-seed, two-alphabet audit in
`CHANNEL_ENSEMBLE_RESULTS.md` shows a stable decoder ensemble with selective
class-boundary correction, not proportional geometric defect cancellation.

## Correct interpretation

The learned 32D operator is not an exact A5 representation, so the original
raw gate remains 0/10. But equal weighting of unobservable slack obscures the
mechanism. The data support a sharper statement:

> Cross-entropy plus capped multi-scale rotor dynamics reliably discovers an
> approximate 3D icosahedral action inside an overparameterized state. That
> action is the dominant necessary single-channel mechanism, but the full state
> is a redundant learned code whose auxiliary channels can either increase or
> decrease robustness under path-distribution shift.

This is an observability and robust-decoding issue familiar from automata and
control. The dominant irrep gives a candidate minimal realization, but the
current decoder can exploit redundant directions as a stable auxiliary
ensemble. It is not valid to quotient those directions away universally.
This remains an additional post-hoc diagnostic, not a retroactive relaxation
of the raw pre-registered threshold.

## Mathematical consequence

For canonical prototype `p_g`, token action `A_s`, and deviation `delta_t`, the
exact error recurrence is

`delta_(t+1) = A_s delta_t + (A_s p_g - p_(gs))`.

Orthogonality prevents norm explosion but allows rotated local defects to add
coherently or cancel. The later aligned-generator audit in
`A5_IRREP_LIE_AUDIT.md` sharpens the earlier single-plane argument: both anchor
generator defects have nonparallel SO(3) log axes in all ten seeds, giving a
normalized Lie-closure rank of three. Thus the generic risk is exploration of
full SO(3), not merely phase winding on one invariant circle. This remains a
numerical risk diagnostic rather than a formal density proof for the actual
error cocycle. Dense testing describes a practical horizon; exact joint closure
is required for an infinite-horizon claim.

## Joint exact projection result

The trust-region rounding proposal has now been tested in a stronger form; see
`JOINT_A5_ROUNDING_RESULTS.md`. On an untouched macro alphabet that changes the
order-3 generator pair, the frozen learned models all pass the dense L16-L256
gate but all fail L4096. Independently snapping generator angles fixes cyclic
relations and passes dense evaluation, yet still fails three of ten seeds at
L4096. Replacing the anchor by one globally aligned exact A5 action passes
L4096 in all ten seeds, with a `96.88%` population floor and float32
homomorphism RMS below `2.4e-7`.

This is the causal confirmation of the error-cocycle analysis: a nearby exact
joint representation removes the long-horizon failure without changing the
decoder ensemble. It is an oracle-structured post-training intervention, so
the raw learned `1e-3` result remains 0/10.

## Next experiments

1. Add the faithful hard permutation-diagonal ceiling and learned PD-SSM
   baseline under the same dense protocol.
2. Test holonomy transfer to Householder actions.
3. Test changed generators.
4. Replace oracle-structured exact projection with a trainable shared
   representation manifold and differentiable joint retraction. Do not regress
   to independent per-token rounding: it has now been falsified at L4096.

Spin(8), MQAR, and language machinery remain deferred until those controls are
settled.

The changed-generator pilot and ten-seed GA subset audit are now complete; see
`CHANGED_GENERATOR_RESULTS.md` and `CHANNEL_ENSEMBLE_RESULTS.md`. The dominant
GA channel is present in every joint-oracle subset, but one fixed auxiliary
ensemble per seed is more robust across both tested macro alphabets.
Householder passes in the clean seed but fails in the holonomy-rescued seed 3.
The subset result shows why `minimal realization` must not be conflated with
`safe pruning`.
