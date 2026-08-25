# Program 07: Controlled model benchmarks

**Research author:** Hayden Austin

**Status:** legacy supporting track; use
[SUPPORTING_TRACKS.md](../SUPPORTING_TRACKS.md) and the
[current model inventory](../../SSM-Models/MODEL_STATUS.md).

## Object

Matched-path empirical evaluation of proposed Spinor/SSM models against direct,
delta-rule, Mamba-family, and other appropriate baselines.

## Contract

- A run is a result only when its structured artifact is complete and passes
  validation.
- Matching state, parameters, and measured compute answer different questions;
  all must be labeled.
- An out-of-memory event, partial log, or smoke test is operational evidence,
  not a model-quality result.
- The benchmark repository must not retroactively strengthen theorem claims in
  the Spin8 submodule.

## Canonical location

- [`Spin8-SSM-Benchmark/`](../../Spin8-SSM-Benchmark/)
- [`SSM-Models/benchmark_pure_rotor_vs_mamba2.py`](../../SSM-Models/benchmark_pure_rotor_vs_mamba2.py)
  is the distinct maintained-Pure-Rotor comparison runner. Its one-step smoke
  artifact is operational evidence only, and its local Mamba-2 path is the
  unfused Transformers implementation rather than a native `mamba_ssm`
  systems baseline.
- [`SSM-Models/benchmark_pure_spin8_vs_mamba2.py`](../../SSM-Models/benchmark_pure_spin8_vs_mamba2.py)
  is the frozen maintained-Pure-Spin(8) transport comparison. Its complete
  three-seed artifact is model-quality evidence for that synthetic task; its
  unfused references are not a production-kernel throughput comparison.

## Next publishable question

The memory-core matrix is complete: frozen ten-seed routing cohorts and fresh-
process CUDA measurements now compare dense, block, and hard routing plus
direct, native delta, co-moving FLA chunk, and recurrent transported memory.
The next publishable question is the **model-level** transfer: insert the
hierarchical co-moving memory into one controlled sequence model and compare it
against strong attention/delta/SSM baselines with fully validated artifacts,
identical tokens, and separate state/parameter/compute tables. The completed
memory-core benchmark must not be described as that model result.

For the maintained Cl(3,0) model, first complete the existing three-seed
Pure Rotor / identity-transport / Mamba-2 byte-level cohort. It must report the
identity ablation's disabled rotor-controller parameters and distinguish an
unfused architecture-quality comparison from a fused-kernel throughput claim.

The separate maintained Pure Spin(8) v1.0 cohort is complete. All three
24-state/836-parameter models extrapolate to L128 at MSE
`5.81e-5`--`6.68e-5` and classify paired central signs perfectly; the
10,380-parameter Transformers Mamba-2 rows remain at `0.132`--`0.135` MSE and
50% center accuracy. This is a roughly 2,000x MSE gap on a supplied-coordinate
Spin(8) transport task, not evidence that Pure Spin(8) is a generally stronger
language architecture. See
[`PURE_SPIN8_VS_MAMBA2_RESULTS.md`](../../SSM-Models/experiments/PURE_SPIN8_VS_MAMBA2_RESULTS.md).

The separate A5 direct-composition screen now has a complete three-seed,
1,000-update artifact. It is a bounded symbolic mechanism result, not a
substitute for the byte-level cohort: Pure Rotor's short-pair result is variable
and no candidate retains A5 tracking at L64/L128. See
[`PURE_ROTOR_A5_MAMBA2_BUDGET1000_RESULTS.md`](../../SSM-Models/experiments/PURE_ROTOR_A5_MAMBA2_BUDGET1000_RESULTS.md).

The center-sensitive follow-up now has a separately frozen and completed
three-seed artifact. Its paired `2.A5` evaluation holds the A5 projection fixed
while flipping the binary center. The explicit Spin quaternion scan passes the
registered margin gate at L16/L64/L128 in every seed and strongly exceeds Pure
Rotor, identity, and Transformers Mamba-2 on long exact tracking. This is a
parameter-near symbolic mechanism result, not a fused systems result or
state-matched comparison. See
[`PURE_ROTOR_2A5_CENTER_PILOT300_RESULTS.md`](../../SSM-Models/experiments/PURE_ROTOR_2A5_CENTER_PILOT300_RESULTS.md).
The saved cohort also passes one deterministic, zero-training-occurrence
identity/center word pair without retraining; this post-pilot relation result is
explicitly exploratory. See
[`PURE_ROTOR_2A5_UNSEEN_RELATION_RESULTS.md`](../../SSM-Models/experiments/PURE_ROTOR_2A5_UNSEEN_RELATION_RESULTS.md).

The preregistered multi-relation successor is also complete. It simultaneously
withholds `a^2`, `b^3`, and `(ab)^5`, pairs each with an explicit identity-token
block, and uses byte-identical schedules under three conjugated generating
sets. The comparison adds a 29,288-parameter, equation-faithful unfused
DeltaProduct reference and a 120-complex-state exact regular PD oracle. Spin
passes every registered long-retention center split and uniquely wins exact
accuracy in all 18; the other learned candidates fail the center gate.
DeltaProduct is the strongest non-Spin long exact tracker but loses center
preference toward chance. See
[`SPIN_2A5_MULTIRELATION_RESULTS.md`](../../SSM-Models/experiments/SPIN_2A5_MULTIRELATION_RESULTS.md).

This is one initialization across paired coordinate controls, parameter-near
but neither state-matched nor fused-kernel matched. The next controlled step is
a fresh frozen multi-seed validation, then separate state- and kernel-matched
sweeps rather than retroactively tuning this pilot.

The rigid-motor work now has both a numerical gate and a learned comparison. An
eight-scalar unit-dual-quaternion scan matches homogeneous `SE(3)` products
through length 4096 and retains the double-cover center, but the local eager
motor path is about 8.6 times slower than batched 4 by 4 matrices. See
[`MOTOR_PATH_DEVELOPMENT_RESULTS.md`](../../SSM-Models/experiments/MOTOR_PATH_DEVELOPMENT_RESULTS.md).
The frozen translation-plus-center benchmark compares parameter-near
quaternion/motor, Transformers Mamba-2, and DeltaProduct candidates. All four
300-step learned readouts fail the long joint-pose gate. A matched direct-
product state retains center sign while failing translations. A separately
labelled local transition identifier then recovers the seven motors only from
legal supervised prefix differences and passes 9/9 coordinate/seed runs and
all 162 splits. Its frozen signed-pose noise audit passes every clean, 1-degree/
0.01, and 5-degree/0.05 run across five noise seeds; the 15-degree/0.15 stress
tier keeps the central sign but fails joint pose. This remains conditional on
signed every-prefix supervision. See
[`SPIN_MOTOR_RIGID_2A5_RESULTS.md`](../../SSM-Models/experiments/SPIN_MOTOR_RIGID_2A5_RESULTS.md).

This does not satisfy the previously desired faithful Fixed-Point RNN
comparison: official source was found, but its pinned Python/Torch/Triton and
custom scan stack is incompatible with this Windows Python-3.12 environment,
so it was excluded rather than approximated. The next controlled motor test is
noisy continuous increment prediction and final-only supervision; local
identification should remain a named structured estimator, not be merged into
the gradient-trained comparison.

The continuous associator-tracking pilot is now the first controlled task for
the lifted octonion operator. It trains on full ordered `8 by 8` prefix actions
at length 16 and evaluates through length 128. The 72-parameter algebra-matched
encoder reaches L128 MSE `1.85e-12`; the collapsed-octonion ablation,
DeltaProduct reference, and Transformers Mamba-2 score `0.216`, `0.124`, and
`0.125`. The complete target makes this a 64-state operator-identification
problem, not an eight-state streaming comparison. All learned baselines are
unfused and the result is one seed with identity-near initialization, so the
valid claim is synthetic realizability/length extrapolation only. See
[`OCTONION_ASSOCIATOR_TRACKING_PROTOCOL.md`](../../SSM-Models/experiments/OCTONION_ASSOCIATOR_TRACKING_PROTOCOL.md)
and
[`OCTONION_OPERATOR_SCAN_RESULTS.md`](../../SSM-Models/experiments/OCTONION_OPERATOR_SCAN_RESULTS.md).

The frozen successor transports both inputs and full operator targets through
three hidden Haar `SO(8)` bases. Its 28-parameter learned gauge passes the
registered L128 gate in all three bases (`1.54e-9`--`8.74e-8`) and the recovered
ambiguity obeys the `G2` automorphism equations to at most `2.17e-4`. The
overall cohort is still a registered failure: the state-matched dense operator
does not optimize below `1e-3`, and basis 0's float32 oracle is `1.0749e-12`
against a strict `1e-12` cutoff. A labelled post-protocol least-squares audit
identifies the dense leaf map to L128 MSE at most `9.38e-12`, proving its frozen
miss is optimization. See
[`OCTONION_BASIS_TRANSPORT_RESULTS.md`](../../SSM-Models/experiments/OCTONION_BASIS_TRANSPORT_RESULTS.md).

The final-only successor separates optimization from identifiability. Fixed
L16 training is a frozen failure (3/9 structured and 0/9 dense successes),
while an L2/L4/L8/L16 curriculum reaches 9/9 for both families and extrapolates
the structured scan through L1024. Because every curriculum depth is even, the
task cannot distinguish `G2` from `-G2`; a post-protocol L17 audit confirms the
predicted parity in all nine checkpoints. See
[`OCTONION_FINAL_ONLY_RESULTS.md`](../../SSM-Models/experiments/OCTONION_FINAL_ONLY_RESULTS.md).

The next Pure Spin(8) benchmark must conceal local Lie coordinates and compare
at matched parameter, recurrent-state, and measured-compute points. The next
octonion benchmark must mix odd/even depths to remove the signed-coset shortcut.
