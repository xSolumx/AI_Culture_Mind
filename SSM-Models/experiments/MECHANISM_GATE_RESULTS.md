# A5 mechanism gate

Date: 2026-08-02

## Purpose

This gate asks whether a recurrent action has learned the group operation, not
merely a useful interpolation of the observed token grammar. The models in
`mechanistic_group_actions.py` have a learned nonzero initial orbit state and
a linear decoder, but no decay, affine write, residual path, feed-forward
network, or history-dependent controller. A token can only apply its fixed
norm-preserving action to the persistent state.

## Pre-registered pass criteria

These criteria were written after the first data-only mechanism run and before
running a relation-aware objective. A candidate passes only if it satisfies all
of the following:

1. At least 99% final-position accuracy on the length-2 held-out generator
   pair. This deterministic composition should not receive partial credit.
2. At least 95% final-position accuracy on held-out-pair-containing sequences
   at the training length, 16.
3. At least 90% final-position accuracy at lengths 64 and 128.
4. Cayley-edge relation RMS, full linear-homomorphism RMS, and identity-word
   state-drift RMS are each at most `1e-3`.
5. Operator orthogonality RMS is at most `1e-5`, the canonical orbit has a
   strictly positive minimum pairwise margin, and all streaming errors are at
   most `1e-5`.
6. After a single-seed mechanism proof, the complete gate must first replicate
   in at least four of five predeclared seeds, then survive a ten-seed
   checkpoint before it supports a comparative reliability claim.

The thresholds are intentionally severe: an exact finite-group algorithm
should not decay gradually with word length.

## Experiment 1: data-only write-free actions

Training used 512,000 length-16 sequences over A5 generators `23145` and
`23451`. The ordered pair `23145 -> 23451` occurred zero times in training and
was forced into every evaluation sequence. All models used four channels of
eight real state values and the same initial state and decoder function.
Parameter counts differ because two Householder reflectors and one Cl(3)
rotor use different raw parameterizations; the report does not call them
parameter matched.

| Family | L2 | L16 | L64 | L128 | edge RMS | homomorphism RMS | identity drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| complex `U(1)^4` | 0.00% | 2.20% | 2.20% | 1.27% | 0.8151 | 1.2719 | 1.2742 |
| Cl(3) rotor | 0.00% | 1.37% | 2.29% | 1.46% | 0.6282 | 0.9171 | 1.1699 |
| two Householder reflections | 0.00% | 1.27% | 1.46% | 1.46% | **0.4699** | **0.6964** | 1.1938 |

All three fail the gate. Their actions remain orthogonal and state-norm error
stays small through length 128, so numerical instability is not the cause.
The models learn nontrivial, partly separable orbits on the restricted
language but not the A5 relations. The noncommutative parameterizations are
more relation-consistent than the commuting complex action, and Householder is
best on the measured relation errors in this run. None handles the forbidden
composition itself.

This negative result rules out the simple hypothesis that removing writes is
sufficient to make cross-entropy training discover a group representation.

## Next controlled experiment

Add an explicitly reported Cayley-relation regularizer after a cross-entropy
warmup. This is a mechanism upper bound, not a claim of learning the operation
from sequence data alone. It asks two narrower questions:

1. Can each action family simultaneously realize the A5 relations and retain a
   linearly decodable faithful orbit?
2. If the Cl(3) rotor reaches the pre-registered gate while the commuting
   complex action cannot, does the expected algebraic separation appear in the
   measured operators?

Only after that upper bound passes should relation information be replaced by
a task-derived consistency objective.

## Experiment 2: relation loss and its trivial-representation trap

A Cayley-edge relation penalty was added after a cross-entropy warmup. Closure
alone has a degenerate global optimum: every token action can remain identity.
In the pilot, increasing relation pressure reduced commutator separation rather
than discovering A5. Adding canonical-orbit classification blocked complete
collapse and learned 59/60 canonical labels plus the held-out length-2 pair,
but full homomorphism RMS remained `0.9458` and long random words failed.

This is a useful negative result. Relation consistency requires an explicit
faithfulness or anti-collapse condition, and canonical-word accuracy alone does
not guarantee agreement among alternate words for the same group element.

## Experiment 3: exact A5 representation upper bound

The harness now constructs a real 3D A5 irrep without hand-entering rotation
matrices:

1. The degree-three A5 character projects the 60D left-regular
   representation onto its rank-nine isotypic component.
2. A symmetric right-regular operator, which commutes with every left action,
   selects one invariant 3D copy.
3. Generator matrices are inverted to match the recurrence's right-product
   convention and converted either to Cl(3) bivector parameters or to two
   Householder reflectors.

With exact actions frozen, both parameterizations meet the mechanism gate.

| Oracle action | L2 | L16 | L64 | L128 | homomorphism RMS | identity drift |
|---|---:|---:|---:|---:|---:|---:|
| Cl(3) rotor | 100% | 100% | 100% | 100% | `5.65e-7` | `9.51e-7` |
| two Householders | 100% | 100% | 100% | 100% | `9.27e-8` | `1.46e-7` |

The rotor remained 100% through length 512. Its fixed-action orbit/decoder
also passed the pre-registered length and mechanism criteria in all five
seeds. These seeds test orbit and decoder optimization; the exact algebra is
the same deterministic character construction in every run.

This establishes sufficiency, not exclusivity. A5 is exactly representable by
the write-free Cl(3) recurrence, but also by a two-reflection orthogonal action.

## Corrected held-out design

The original two-token split had an additional confound. With alphabet
`{a,b}`, forbidding `a -> b` reduces training words essentially to `b* a*`.
It removes most of the language rather than one local transition. The corrected
split uses `{a, a^-1, b, b^-1}` and excludes only `a -> b`:

- all 60 A5 target states occur in training;
- 15/16 input bigrams occur in training;
- the forbidden pair occurs zero times in training and in every evaluation
  sequence.

For a write-free fixed-token action, passing the missing-bigram test is partly
structural: there is no bigram-specific parameter. This experiment therefore
does **not** answer the same question as the original context-dependent model's
falsifier. It asks whether SGD can find useful fixed per-token operators from
the other contexts. The movement from 0% to 100% is not itself a before/after
architectural comparison.

## Experiment 4: data-only inverse-augmented split

Seed 0 separated the commuting model sharply:

| Family | L2 | L16 | L64 | L128 |
|---|---:|---:|---:|---:|
| complex `U(1)^4` | 0.00% | 4.49% | 1.95% | 1.76% |
| Cl(3) rotor | 100% | 100% | 100% | 99.95% |
| two Householders | 100% | 100% | 100% | 100% |

Complex was excluded from the subsequent reliability allocation because this
single seed gave a wide separation. It has **not** been ruled out by a
multi-seed corrected-split experiment.

The noncommutative families were then run for ten seeds under identical data.
The table keeps training-length convergence and long-composition retention
visible as separate outcomes:

| Seed | GA L16 | GA L64 | GA L128 | Householder L16 | Householder L64 | Householder L128 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 100% | 100% | 100% | 100% | 100% | 100% |
| 1 | 100% | 100% | 100% | 100% | 100% | 100% |
| 2 | 16.2% | 2.4% | 2.0% | 100% | 100% | 100% |
| 3 | 100% | 100% | 100% | 100% | 83.2% | 59.1% |
| 4 | 7.4% | 2.3% | 1.7% | 100% | 100% | 100% |
| 5 | 100% | 100% | 99.9% | 100% | 100% | 100% |
| 6 | 100% | 80.9% | 57.4% | 100% | 100% | 100% |
| 7 | 100% | 88.1% | 97.1% | 100% | 57.6% | 38.3% |
| 8 | 100% | 100% | 100% | 100% | 100% | 98.6% |
| 9 | 100% | 79.0% | 77.8% | 100% | 100% | 100% |

These results expose two distinct failure modes:

1. **Optimization/convergence:** GA fails to find a working length-16 solution
   in seeds 2 and 4, so it converges in 8/10 seeds. Householder converges in
   10/10.
2. **Retention after convergence:** of the eight converged GA runs, five pass
   both long-length thresholds. Eight of ten converged Householder runs pass.
   GA seed 7 is non-monotone (`88.1%` at L64, `97.1%` at L128), so not every
   failure is well described by smooth numerical drift.

The complete functional gate therefore passes in 5/10 GA seeds and 8/10
Householder seeds. At this ten-seed checkpoint, two-Householder transitions are
more reliable under this optimizer and task. This is evidence about this
parameterization and training setup, not a universal ordering of orthogonal
transition families.

### GA bimodality diagnostic

Dense trajectories for successful seed 0 and collapsed seeds 2 and 4 rule out
a dead-rotor explanation. All three runs have nonzero action gradients and
learn nontrivial, noncommuting actions. Seed 0 undergoes a sharp coordinated
transition between steps 300 and 400: prefix accuracy rises from `46.4%` to
`98.3%`, final-position accuracy from `21.5%` to `96.1%`, and the median
correct-class margin becomes positive. Seed 2 makes a later partial transition
but never reaches a consistent final-position solution; seed 4 never crosses
positive median margin. One failed seed approaches the rotor chart boundary,
but the other does not. The common failure is therefore a coupled
action-orbit-decoder optimization basin, not vanishing gradients or rotor-angle
saturation alone.

### Optimization interventions

Two interventions were tested first on the predeclared failed GA seeds 2 and
4. Adding another full-weight final-position cross-entropy term made both runs
worse (`3.2%` and `3.1%` at L16), rejecting simple short-prefix loss dominance
as the explanation. Restricting each token rotor to at most 2.2 radians rescued
both runs to 100% at L16. This cap is still above the `2*pi/3` angle required by
the selected A5 generators, and the exact-irrep construction passes unchanged
under it.

Because the two seeds were selected failures, the cap was then replicated over
all ten seeds:

| Family / chart | L16 convergence | complete L64+L128 gate | mean L64 | mean L128 | mean full hom. RMS |
|---|---:|---:|---:|---:|---:|
| GA, `pi` cap | 8/10 | 5/10 | 75.3% | 73.6% | 0.6849 |
| GA, `2.2` cap | **10/10** | 5/10 | 90.0% | 85.2% | 0.6284 |
| two Householders | **10/10** | **8/10** | **94.1%** | **89.6%** | **0.4963** |

The 2.2-radian chart is therefore a real convergence fix, not a complete
mechanism fix. It removes the rotor's bimodal training-length collapse and
improves mean long-length accuracy, but it does not increase the strict gate
count. Learned capped runs settle near 2.08 radians, close to the natural
order-three angle, while state-level relation error remains far above the
pre-registered threshold. The next intervention must target representation
consistency without reopening the trivial-identity solution.

A final seed-2 upper-bound pilot introduced Cayley-edge consistency after step
750 while leaving task loss active. Mean-square closure lowered full
homomorphism RMS from `0.4943` to `0.4538`, yet worsened L128 from `78.8%` to
`55.9%` and left the worst residual unchanged near `1.41`. A tail-sensitive
power-8 objective preserved L128 (`78.5%`) but again left the worst residual
unchanged. Prototype margin and canonical decoded accuracy stayed healthy.
Thus neither mean nor tail one-step closure is a sufficient proxy for
long-word correctness. The next consistency loss must control coherent error
along reachable paths and decoder boundaries, rather than optimizing an
isolated aggregate operator norm.

### Preregistered path-holonomy pilot

The next objective was preregistered in
[`HOLONOMY_PREREGISTRATION.md`](HOLONOMY_PREREGISTRATION.md), including its
all-identity degeneracy. Holonomy is never used alone: supervised task loss
remains active, and a separation term covers both canonical prototypes and
non-canonical alternate paths. Executable tests confirm that identity actions
have zero holonomy but positive separation loss and nonzero supervised rotor
gradients; the exact capped A5 irrep has near-zero holonomy and zero separation
penalty.

The selected capped-GA seed-2 pilot concatenates four shifted length-16 words
into length-64 paths and aligns each path operator with a canonical short-word
operator for the same target element:

| Metric | capped control | path holonomy |
|---|---:|---:|
| L16 final accuracy | 100% | 100% |
| L64 final accuracy | 80.9% | 99.9% |
| L128 final accuracy | 78.8% | 99.1% |
| length-64 path RMS | 1.0345 | 0.6753 |
| alternate target distance | 2.0222 | 0.6719 |
| nearest-negative margin | -3.6769 | +0.2979 |
| orbit-directional hom. RMS | 0.6099 | 0.2694 |

Prototype margin remains `0.8963`, decoded canonical products remain 100%, and
action displacement/commutator separation remain nontrivial. The worst global
operator residual remains near `1.41`. The functional gain therefore comes
from repairing coherent reachable-path alignment and decoder-relative state
geometry, not every possible state-space direction. The preregistered pilot
passes, but it is one selected seed; replication is required before a
reliability claim.

The unchanged objective was then replicated across all ten seeds:

| Family / objective | complete functional gate | positive-margin state contract | mean L64 | mean L128 |
|---|---:|---:|---:|---:|
| GA, 2.2-radian cap | 5/10 | not measured | 90.0% | 85.2% |
| GA, cap + path holonomy | **8/10** | 7/10 | **94.5%** | **97.5%** |
| two Householders | 8/10 | not measured | 94.1% | 89.6% |

These are functional and state-geometry criteria, not the original raw-
operator mechanism gate. Every data-trained holonomy run remains orders of
magnitude above the original `1e-3` homomorphism-RMS threshold: the strict
mechanism result is 0/10, and only the oracle-irrep construction passes it.

The means conceal a sharply bimodal distribution and must remain attached to
the per-seed outcomes:

| Seed | L64 | L128 | alternate margin | functional | state contract |
|---:|---:|---:|---:|:---:|:---:|
| 0 | 100% | 100% | +0.302 | pass | pass |
| 1 | 100% | 100% | +0.276 | pass | pass |
| 2 | 99.9% | 99.1% | +0.298 | pass | pass |
| 3 | 100% | 100% | +0.334 | pass | pass |
| 4 | 96.0% | 75.6% | +0.082 | fail | fail |
| 5 | 100% | 99.8% | +0.230 | pass | pass |
| 6 | 49.3% | 100% | +0.138 | fail | fail |
| 7 | 100% | 100% | -0.046 | pass | fail |
| 8 | 100% | 100% | +0.320 | pass | pass |
| 9 | 100% | 100% | +0.284 | pass | pass |

Median accuracy is 100% at both lengths; seeds 4 and 6 account for most of the
mean reduction and have distinct non-monotone failures. Holonomy
hyperparameters were fixed from seed 2 before inspecting seeds 5, 6, 7, and 9,
then left unchanged for every remaining seed; there was no per-seed tuning.

Holonomy retains 100% L16 accuracy in every seed and matches Householder's
functional pass count while exceeding its mean L128 accuracy. It is not yet
uniformly reliable: seed 4 scores `96.0%/75.6%`, seed 6 scores
`49.3%/100%`, and functionally perfect seed 7 has a slightly negative
alternate-path margin. The fixed length-64 objective has therefore exposed a
new, narrower failure: length-selective phase aliasing. This is a replicated
functional breakthrough, not a claim of exact representation learning.

## What the learned state actually represents

The raw operators are not exact A5 representations. The corrected seed-0
decomposition normalizes every restricted Frobenius error per input-subspace
dimension:

| Family | full-space RMS | centered variation rank | variation-span RMS | common fixed rank / RMS | canonical-direction RMS |
|---|---:|---:|---:|---:|---:|
| GA | 0.6248 | 24/32 | 0.7214 | 8 / `2.0e-7` | 0.7596 |
| Householder | 0.4699 | 32/32 | 0.4699 | 0 / n/a | 0.2223 |

The earlier `orbit_homomorphism_rms` name was too easy to read as an
orthogonal projection. It is a **canonical-orbit directional residual**: the
error operators are applied to the 60 canonical states and normalized by the
initial-state norm. The harness now also reports explicit orthonormal
span/complement metrics and the new unambiguous directional name.

For GA, rotor conjugation fixes the scalar and pseudoscalar coordinate in each
of four channels. Those eight directions have `2.0e-7` relation error. The
centered canonical orbit varies over the remaining 24 dimensions, and
`0.7214 * sqrt(24/32) = 0.6248` to rounding. Thus the full-to-active reversal is
quantitatively explained by invariant-dimension dilution. The additional rise
from `0.7214` to `0.7596` is separate: canonical states weight the active error
directions anisotropically.

Householder has no analogous common fixed sector and its centered orbit spans
all 32 dimensions. Its lower canonical-direction residual therefore reflects
the opposite anisotropy: canonical states put comparatively little energy in
the high-error directions. It is not evidence for an unused linear off-orbit
complement. A direct SVD audit gives common fixed rank `0`; its smallest
fixed-constraint singular value is `1.27e-3`, versus a numerical rank tolerance
of `2.96e-6`, so the no-fixed-subspace result is separated from the threshold
by more than a factor of 400.

The pre-registered `1e-3` representation threshold is missed by orders of
magnitude. Error is present throughout GA's active variation span, while its
seven-dimensional linear complement is invariant and nearly exact;
Householder has no unused linear complement at all. Thus neither result is
explained by arbitrary off-orbit slack. The linear decoder's decision regions
nevertheless identify every generator edge and all 3,600 canonical pair
products correctly. The honest description is a
**decoder-stable quotient of approximate orthogonal dynamics**, not "the model
found the exact group representation."

This also explains occasional long-word failures: small state-level relation
errors can accumulate even when every one-step canonical transition remains on
the correct side of the decoder boundary.

## Revised next step

The representation-targeting intervention has now landed. The preregistered
joint exact projection in `JOINT_A5_ROUNDING_RESULTS.md` leaves the decoder and
auxiliary channels frozen, replaces only the dominant channel by one globally
aligned exact A5 action, and passes the strict float32 mechanism gate plus an
untouched changed-order-3 L4096 behavioral gate in all ten seeds. Independent
angle snapping fails three seeds at L4096 despite exact cyclic relations.

This is a post-training oracle-structured construction, not a retroactive pass
for the learned checkpoints: their original raw `1e-3` count remains 0/10. The
next target is learning or retracting onto the same joint representation
manifold without group-label oracle access.

The coprime multi-scale and causal channel-audit cycle is now complete. See
`OBSERVABLE_IRREP_RESULTS.md` for the full result. In brief: multi-scale GA is
10/10 at L16/L64/L128 and on positive sampled margins, 8/10 on the dense
L16-L256 floor, and 0/10 on the original raw `1e-3` mechanism gate. A post-hoc
four-seed ablation shows that one oracle-matching 3D channel is the dominant
necessary single-channel mechanism. A later changed-generator subset audit
shows that the other channels are not uniformly disposable. A deterministic
ten-seed, two-alphabet follow-up identifies a stable per-seed decoder ensemble
that selectively corrects class boundaries without cancelling the anchor's
full defect vector. The earlier description below is retained as the
historical pre-multi-scale ordering.

1. Treat two-Householder transitions and capped GA plus holonomy as co-leading
   learned baselines: both pass 8/10 functionally, with different residual
   failures and parameterizations.
2. Retain the 2.2-radian rotor cap as a convergence stabilizer; final-position
   emphasis is rejected. The cap is the precondition that lets path holonomy
   address composition without reopening training-length collapse.
3. Replace the single length-64 holonomy distribution with coprime multi-scale
   paths (for example lengths 32, 48, 64, and 80) to break the phase aliases
   observed in seeds 4 and 6, while retaining alternate-path margin.
4. Add accumulated error and decoder-margin curves across dense word lengths;
   L64 and L128 alone demonstrably alias periodic structure.
5. Add a faithful permutation-diagonal baseline with a genuinely hard or
   projected permutation mechanism; do not call a soft dense relaxation PD-SSM.
6. Test changed generator sets and composition curricula. A missing bigram is
   no longer a sufficient structural challenge once token actions are fixed.
7. Continue to defer Spin(8): the current bottleneck is optimization toward a
   stable quotient/representation, not transition dimension.
