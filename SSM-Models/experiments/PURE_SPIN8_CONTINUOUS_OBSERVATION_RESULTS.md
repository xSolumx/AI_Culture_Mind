# Pure Spin(8) noisy continuous-observation results

Primary cohort adjudicated: **2026-08-17T04:27:24.147531+02:00**<br>
Measured-wall continuation adjudicated: **2026-08-17T04:40:35.261975+02:00**

## Result in one sentence

Across three untouched observation charts and schedules, the 930-parameter
shared Pure Spin(8) tracker learned fresh noisy continuous local actions and
composed an excluded adjacent center relation through length 128; it was more
accurate than a capable 957-parameter, exactly state-matched independent
`SO(8)^3` tracker under both equal updates and a separately frozen local-GPU
wall allocation.

This is a controlled synthetic identification result. It is not a theorem that
triality is necessary, a natural-data result, or evidence that this model beats
Mamba in language modelling.

## What was hidden and what was held out

The model receives a unique 12-real noisy observation

`x = tanh(P (u + 0.15 u^3) + b) + epsilon`

of a hidden seven-coordinate Spin(8) increment. The orthonormal projection `P`,
bias `b`, noise, initialization, and schedules change with the seed. Training
supervises every 24-real prefix in `(8v,8s+,8s-)`, includes isolated
half-center actions, and contains zero adjacent half-center pairs. Evaluation
inserts fresh `pi+delta, pi-delta` observations whose actions compose to the
nontrivial center, paired against `beta,-beta` identity controls in byte-equal
surrounding contexts.

The decisive control is not an incapable commutative baseline. Independent
`SO(8)^3` has 957 parameters, the same 24 recurrent scalars, and three separate
orthogonal action families. It can represent the teacher but does not require
the vector and chiral streams to arise from one shared Spin(8) element.

## Frozen equal-update cohort

Every row received the same 800 precomputed updates. Parameters were 930, 957,
931, 960, and 949 for shared Spin(8), independent `SO(8)^3`, Mamba-2,
parameter-near GRU, and observation-only MLP; the separate exact-state GRU had
3,312 parameters. The near cohort's maximum spread was 3.125%.

| Seed | Shared action RMSE | Independent action RMSE | Shared L128 early / late MSE | Independent L128 early / late MSE |
|---:|---:|---:|---:|---:|
| 1 | 0.01334 | 0.02662 | 0.01216 / 0.02306 | 0.03859 / 0.07456 |
| 2 | 0.01440 | 0.02716 | 0.01390 / 0.02814 | 0.03961 / 0.07809 |
| 3 | 0.01468 | 0.03027 | 0.01414 / 0.02622 | 0.04801 / 0.09271 |

All 135 per-seed frozen checks passed. Shared and independent rows both achieved
exactly 100% center/identity classification and row correctness on every L128
split, so the MSE comparison is not created by an incapable structural control.
Every shared L128 split also beat Mamba-2, both GRUs, and the observation-only
row. The aggregate median across the six early/late L128 splits was `0.01860`
for shared Spin(8) and `0.06129` for independent `SO(8)^3`, a `3.295x` ratio.

The learned shared relation-action residual never exceeded `0.01998` RMSE;
the independent control never exceeded `0.02819`. Teacher center/inverse
identities held within `7.76e-8` maximum absolute error.

## Separately frozen measured-wall continuation

The update allocation was fixed from the corrected seed-0 development timing:

| Candidate | Updates |
|---|---:|
| shared Pure Spin(8) | 800 |
| independent `SO(8)^3` | 636 |
| Mamba-2 | 1,134 |
| parameter-near GRU | 5,001 |
| observation-only MLP | 6,604 |
| state-matched GRU | 5,005 |

For each seed, one maximal schedule was precomputed and every candidate received
its corresponding prefix. Schedule generation and the global uniqueness audit
were outside the model-update timer. Each maximal schedule had 3,381,248 unique
observations and zero excluded adjacent half-center pairs. The shared 800-update
row replayed the primary artifact with numeric `max_abs = 0` on training loss,
action diagnostics, and every evaluation metric.

On the RTX 2070 SUPER, all six update times were `12.14--12.66 s`. The maximum
within-seed deviation from shared time was `1.99--2.97%` (median `2.04%`).

| Candidate | Median L128 post-relation MSE | Six-split range |
|---|---:|---:|
| shared Pure Spin(8) | 0.01860 | 0.01216--0.02814 |
| independent `SO(8)^3` | 0.07476 | 0.04684--0.11087 |
| Mamba-2 | 0.13288 | 0.12865--0.13746 |
| parameter-near GRU | 0.13707 | 0.12948--0.14459 |
| state-matched GRU | 0.13181 | 0.12430--0.13631 |
| observation-only MLP | 0.14099 | 0.13702--0.14461 |

Shared Spin(8) beat every row on every early/late L128 split. The independent
control's median was `4.019x` the shared median. This wall allocation is only a
measured local update-time control: it is not FLOP, energy, fused-kernel, or
hardware-independent compute equality. Transformers Mamba-2 used its explicit
naive fallback because the fused selective-update dependencies were absent.

## Interpretation

What the experiment supports:

- the maintained action construction can be trained online from noisy
  continuous observations rather than supplied Lie coordinates or a finite
  token dictionary;
- one shared group element is a materially more sample-efficient inductive bias
  than three independently routed orthogonal actions on this teacher family;
- the advantage is not explained by recurrent-state size, near-cohort parameter
  count, an absent-prefix-state ablation, an incapable independent action class,
  or the measured update wall on this workstation; and
- exact associative action scans convert local identification into stable
  length extrapolation and the otherwise invisible double-cover center sign.

What remains open:

- all 28 Spin(8) tangent coordinates rather than the active seven-coordinate
  family;
- unsigned, partially observed, irregularly labelled, longer-horizon endpoint,
  or natural targets;
- robustness to correlated noise, chart shift, outliers, and noninjective
  observations;
- a fused training kernel and a fused, tuned modern-SSM comparison; and
- language, vision, control, or other natural-task utility.

The correct scientific reading is a replicated structural-prior result for one
well-audited online group-identification problem, not a universal model ranking.

The frozen endpoint-only successor subsequently removed every intermediate
prefix target while leaving the task, architectures, and held-out relation
unchanged. Shared Spin(8) retained median L128 MSE `0.01296`, versus `0.06268`
for the exactly state-matched independent family at equal updates; a separately
pre-frozen update-wall continuation recorded `0.01296` versus `0.09080`. That
result closes the dense-prefix-supervision objection for this signed synthetic
teacher family; see
[`PURE_SPIN8_ENDPOINT_SUPERVISION_RESULTS.md`](PURE_SPIN8_ENDPOINT_SUPERVISION_RESULTS.md).

## Integrity and reproduction

Primary aggregate:
`artifacts/pure_spin8_continuous_observation_validation_seeds1_3.json`<br>
SHA-256:
`34238a1d98fa467e8f8f38b1f90d1a24bc2495cc510934b4327cba81e09ebc6`

Measured-wall aggregate:
`artifacts/pure_spin8_continuous_observation_wall_matched_seeds1_3.json`<br>
SHA-256:
`262be117892cc2b511eaa3edde0ccea6695533dd9a621312d43c42dafaf33a02`

The primary validator strictly rehashed and reloaded 18 checkpoints; the wall
validator independently did the same for its 18 checkpoints. Observation
systems, maximal training schedules, and all 18 evaluation schedules were
distinct across seeds. The corrected development artifact remains excluded
from validation and content-locked by its recorded SHA-256.

Reproduce the primary cohort with the commands in
[`PURE_SPIN8_CONTINUOUS_OBSERVATION_PROTOCOL.md`](PURE_SPIN8_CONTINUOUS_OBSERVATION_PROTOCOL.md).
Run a wall seed with:

```powershell
python benchmark_pure_spin8_continuous_wall_matched.py --seed 1 --device cuda
```

Adjudicate the three wall artifacts with
[`validate_pure_spin8_continuous_wall_matched.py`](../validate_pure_spin8_continuous_wall_matched.py).
