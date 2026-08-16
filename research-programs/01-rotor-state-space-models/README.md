# Program 01: Selective rotor state-space models

## Object

A bounded recurrent state in the full eight-coordinate Clifford algebra
`Cl(3,0)`, transported by input-selective `Spin(3)` rotor conjugation and
trained through associative affine scans.

## Claims that currently survive

- A separately maintained Pure Spin(8) v1.0 model now sits alongside Pure
  Rotor v2.1 without changing the lower-dimensional checkpoint contract. Its
  faithful `(8v,8s+,8s-)` cache, associative affine scan, center coverage,
  state bound, gradients, masks, streaming cache, CUDA path, and checkpoint
  roundtrip are tested. On the frozen triality-transport task it reaches L128
  MSE `5.81e-5`--`6.68e-5` and 100% center classification in all three seeds;
  the result is algebra-matched and not generic model superiority.
- The maintained recurrence has a hard state-norm bound under its stated
  finite-input assumptions.
- Its affine transition family is exactly associative in real arithmetic and
  supports constant-state recurrent inference.
- The maintained PyTorch backend has an opt-in Schur-factored prefix scan that
  agrees with direct rotor transport in float64 forward, cache, CUDA, and
  first-order-gradient tests. It is an execution-path experiment, not a new
  state-transition family or a performance result.
- A separate experimental octonion multiplication-operator layer gives a
  lawful associative scan for `h_t=u_t h_(t-1)`. Its exact Lie determinant is
  `-2^49`, its affine state is norm-bounded, and a WSL/Triton fused recurrence
  is 4.96x faster than the work-efficient operator path at L4096 on the local
  RTX 2070 SUPER. On one frozen coordinate-aligned continuous task, its
  72-parameter learned encoder extrapolates from L16 to L128 at MSE `1.85e-12`;
  the invalid collapsed octonion, unfused DeltaProduct, and unfused Mamba-2
  controls remain at `0.216`, `0.124`, and `0.125`. It is not part of v2.1,
  and this aligned one-seed result is not generic task superiority. Its
  three-Haar-basis successor finds each transported law with a 28-parameter
  `SO(8)` gauge and maximum L128 MSE `8.74e-8`; recovered gauges satisfy the
  `G2` automorphism equation to at most `2.17e-4`. The overall frozen cohort
  nevertheless fails two named gates, and direct identification proves that
  the state-matched dense operator is realizable despite its optimizer miss.
  Under terminal-only supervision, fixed L16 training succeeds in only 3/9
  structured runs; an L2/L4/L8/L16 curriculum succeeds in 9/9 structured and
  9/9 dense runs. Because every depth is even, the recovered law is identifiable
  only up to `G2 union -G2`; an L17 parity audit verifies all nine predicted
  signs.
- Learned rotor actions are causally active: post-training identity clamping
  and action shuffling damage prediction.
- At fixed recurrent width in the frozen five-seed v2.1 experiment, rotor
  transport improved prediction loss over identity.

## Negative results and boundaries

- Rotor was not the best state-matched transport; quaternion, commuting phase,
  and larger generic orthogonal rows performed better in the reported table.
- The preregistered associative-recall and Q8 memory gates failed.
- At matched measured eager-PyTorch CUDA cost, the wider identity model was
  substantially better.
- These experiments do not demonstrate a language-model scaling advantage or
  a `Spin(8)`/triality advantage.
- The center-sensitive `2.A5` pilot directly exposes a transport boundary:
  rotor conjugation cannot distinguish `q` from `-q`. The separate Spin
  composition layer succeeds on that benchmark, but it is not yet part of the
  maintained v2.1 recurrence or evidence of broad task quality.
- The preregistered multi-relation/conjugation pilot strengthens that boundary:
  Spin passes all `a^2`, `b^3`, and `(ab)^5` long center gates, whereas Pure
  Rotor, identity, Mamba-2, and an equation-faithful DeltaProduct reference do
  not. It remains a one-initialization symbolic result.
- A separate unit-dual-quaternion motor layer now extends the exact Spin product
  to rotation plus translation. Its numerical `SE(3)`/center/cache/gradient
  gate passes through L4096, but translation is unbounded and eager matrices
  are currently faster. In the learned rigid benchmark, the generic motor
  classifier and every 22k-parameter control fail the strict long-context pose
  gate. The exact state becomes successful only under separately labelled
  local transition identification from every-prefix pose supervision: 9/9
  coordinate/seed runs and all 162 splits pass. A separately frozen signed-pose
  noise audit passes all 15 clean/low/medium runs through 5-degree rotation and
  0.05 translation noise, while the 15-degree/0.15 tier retains the center but
  fails joint pose. This is an identification result conditional on signed
  every-prefix supervision, not an end-to-end learning win.

## Canonical evidence

- [`SSM-Models/FOUNDATIONS.md`](../../SSM-Models/FOUNDATIONS.md)
- [`PURE_V2_1_TRANSPORT_ABLATION_RESULTS.md`](../../SSM-Models/experiments/PURE_V2_1_TRANSPORT_ABLATION_RESULTS.md)
- [`PURE_V2_1_RELATED_WORK_NOTES.md`](../../SSM-Models/experiments/PURE_V2_1_RELATED_WORK_NOTES.md)
- [`PURE_ROTOR_VS_MAMBA2_BENCHMARK.md`](../../SSM-Models/experiments/PURE_ROTOR_VS_MAMBA2_BENCHMARK.md)
- [`PURE_ROTOR_A5_MAMBA2_PROTOCOL.md`](../../SSM-Models/experiments/PURE_ROTOR_A5_MAMBA2_PROTOCOL.md)
- [`PURE_ROTOR_A5_MAMBA2_PILOT200_RESULTS.md`](../../SSM-Models/experiments/PURE_ROTOR_A5_MAMBA2_PILOT200_RESULTS.md)
- [`PURE_ROTOR_A5_MAMBA2_BUDGET1000_RESULTS.md`](../../SSM-Models/experiments/PURE_ROTOR_A5_MAMBA2_BUDGET1000_RESULTS.md)
- [`PURE_ROTOR_2A5_CENTER_PILOT300_RESULTS.md`](../../SSM-Models/experiments/PURE_ROTOR_2A5_CENTER_PILOT300_RESULTS.md)
- [`PURE_ROTOR_2A5_UNSEEN_RELATION_RESULTS.md`](../../SSM-Models/experiments/PURE_ROTOR_2A5_UNSEEN_RELATION_RESULTS.md)
- [`SPIN_2A5_MULTIRELATION_RESULTS.md`](../../SSM-Models/experiments/SPIN_2A5_MULTIRELATION_RESULTS.md)
- [`MOTOR_PATH_DEVELOPMENT_RESULTS.md`](../../SSM-Models/experiments/MOTOR_PATH_DEVELOPMENT_RESULTS.md)
- [`SPIN_MOTOR_RIGID_2A5_RESULTS.md`](../../SSM-Models/experiments/SPIN_MOTOR_RIGID_2A5_RESULTS.md)
- [`OCTONION_OPERATOR_SCAN_RESULTS.md`](../../SSM-Models/experiments/OCTONION_OPERATOR_SCAN_RESULTS.md)
- [`OCTONION_FINAL_ONLY_RESULTS.md`](../../SSM-Models/experiments/OCTONION_FINAL_ONLY_RESULTS.md)
- [`PURE_SPIN8_VS_MAMBA2_RESULTS.md`](../../SSM-Models/experiments/PURE_SPIN8_VS_MAMBA2_RESULTS.md)

## Next publishable question

Separate state efficiency from kernel efficiency. Compare optimized rotor,
quaternion, complex/MIMO, Householder-product, and identity transports at
matched state, parameter, and genuinely optimized hardware budgets. A result
must report all three rather than promoting one matching regime as universal.
The immediate direct Mamba-2 runner is implemented but has only a one-step
smoke artifact; it requires a multi-seed completed cohort before it changes
this programme boundary. The A5 held-out-pair runner now has a checkpointed
three-seed 200-step screen: its one seed-0 rotor success at the unseen direct
pair does not replicate, and every candidate is near chance on long words. The
three-seed 1,000-step calibration makes the short-pair signal stronger but
variable (Pure Rotor passes 2/3) and still leaves all candidates near chance at
L64/L128. The subsequent center-sensitive `2.A5` pilot identifies a more
promising mechanism: direct Spin-element composition passes its registered
center gate, while conjugation-based Pure Rotor does not. The frozen Spin
checkpoints also pass one deterministic relation pair absent from all three
training schedules, reducing—but not eliminating—the local-rewrite
  explanation. The next frozen pilot completes the proposed DeltaProduct/PD
  and held-out-relation comparison: Spin passes every long center gate under
  three conjugated generating sets, while the learned alternatives fail. The
  immediate decision is now fresh-seed validation of the unchanged Spin layer,
  followed by a separately named Spin-plus-affine-write hybrid only if that
  validation replicates. The motor's learned rigid-motion/center benchmark is
  now complete and negative under blind 300-step optimization; the subsequent
  local identifier is perfect across 3 coordinates by 3 schedules. The next
  motor question is no longer another deterministic token lookup. Metric
  signed-pose noise is now bounded experimentally. The next target is a noisy
  continuous encoder with a local group-difference/transition-consistency
  objective, plus a final-only-supervision falsifier where exact adjacent
  target differences are unavailable.

The octonion-operator branch has now completed the final-only gate. The fixed
L16 protocol is negative, whereas the composition-depth curriculum converges
for every structured and dense initialization. Its even-only schedule also
exposes a previously hidden `-G2` parity coset. The next frozen protocol must
mix odd/even training depths, add noisy off-sphere inputs, keep an unseen-basis
split, and separately tune the generic `SO(8)` leaf baseline.

For Pure Spin(8), the next meaningful falsifier is no longer another supplied-
coordinate transport task. Hide the local Lie coordinates behind learned token
embeddings, retain center-paired examples, and report separately matched
parameter, recurrent-state, and measured-compute sweeps against optimized
Mamba/DeltaProduct implementations.
