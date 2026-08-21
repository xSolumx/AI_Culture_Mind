# Pure Spin(8) endpoint-only identification results

Equal-update cohort adjudicated: **2026-08-17T05:17:13.461565+02:00**<br>
Measured-wall continuation adjudicated: **2026-08-17T05:40:55.849697+02:00**

## Result in one sentence

Across three frozen fresh observation systems, a 930-parameter shared Pure
Spin(8) tracker identified noisy local actions and composed an excluded center
relation through length 128 when each length-16 training sequence exposed only
its final signed 24-real state; it beat a capable 957-parameter, exactly
state-matched independent `SO(8)^3` action family under both equal updates and
a separately pre-frozen local-GPU wall allocation.

This closes the specific every-prefix-supervision objection to the earlier
continuous-observation result. It is still a synthetic signed-state system-
identification result, not a natural-data result or generic Mamba comparison.

## What changed and what did not

The observation chart, noise, hidden seven-coordinate teacher family,
architectures, initialization rule, held-out adjacent half-center relation, and
early/late L16/L64/L128 evaluations were inherited unchanged from the primary
continuous-observation cohort.

Only the training supervision changed. Each batch contains
`observations`, `endpoint_targets`, `coordinates`, and `events`; it has no
intermediate-target field. Candidate models receive only `observations`, and
the loss is

`MSE(predictions[:, -1], endpoint_targets)`.

Coordinates and events exist only for split auditing. A unit test establishes
exactly zero loss gradient at every nonfinal prediction. Thus each sequence
exposes 24 supervised scalars rather than `16 * 24` prefix scalars.

Each primary seed contained 1,024,000 unique noisy observations, 64,000 final
targets, isolated half-center events, and zero adjacent half-center pairs. The
final signed target retains the double-cover center sign; this experiment does
not infer that sign from unsigned `SO(8)` observations.

## Frozen equal-update cohort

Every row received 2,000 identical precomputed updates. The shared and
independent rows both have 24 recurrent scalars; their parameter counts are 930
and 957. Mamba-2 has 931 parameters and 160 recurrent scalars, the parameter-
near GRU has 960/10, observation-only has 949/0, and the exact-state GRU has
3,312/24.

| Seed | Shared action RMSE | Independent action RMSE | Shared L128 early / late MSE | Independent L128 early / late MSE |
|---:|---:|---:|---:|---:|
| 1 | 0.01167 | 0.07075 | 0.00901 / 0.01817 | 0.05071 / 0.09505 |
| 2 | 0.01248 | 0.03631 | 0.00988 / 0.02014 | 0.04161 / 0.07436 |
| 3 | 0.01116 | 0.06905 | 0.00840 / 0.01604 | 0.05101 / 0.09387 |

All 51 registered checks passed independently in every seed; there was no
median rescue. Both structured action families achieved 100% center/identity
classification and row correctness on every L128 split. The independent
family is therefore a representationally capable control, not a deliberately
commutative or undersized baseline.

| Candidate | Median L128 post-relation MSE | Six-split range |
|---|---:|---:|
| shared Pure Spin(8) | 0.01296 | 0.00840--0.02014 |
| independent `SO(8)^3` | 0.06268 | 0.04161--0.09505 |
| Mamba-2 | 0.12903 | 0.12639--0.13051 |
| parameter-near GRU | 0.12960 | 0.12700--0.13212 |
| observation-only MLP | 0.12813 | 0.12620--0.13055 |
| state-matched GRU | 0.13301 | 0.12969--0.13525 |

The independent-to-shared median ratio is `4.8365x`. Shared Spin(8) beats every
candidate on all six L128 splits. Its maximum learned relation-action residual
is `0.01655`; the independent maximum is `0.05735`, below its frozen `0.08`
cap. All three training schedules, all three observation systems, and all 18
evaluation schedules are distinct.

## Separately frozen measured-wall continuation

The wall allocation was fixed in the protocol from corrected seed-0 model-
update timing before any fresh validation result was inspected:

| Candidate | Updates |
|---|---:|
| shared Pure Spin(8) | 2,000 |
| independent `SO(8)^3` | 1,558 |
| Mamba-2 | 2,811 |
| parameter-near GRU | 11,907 |
| observation-only MLP | 15,482 |
| state-matched GRU | 11,911 |

For each seed, one maximal schedule supplied deterministic prefixes to every
candidate. It contained 7,926,784 unique observations, 495,424 endpoint
targets, no intermediate targets, and zero excluded adjacent pairs. Schedule
generation and uniqueness auditing were outside the model-update timer. The
shared 2,000-update row replayed its primary training loss, action diagnostics,
and every evaluation metric with numeric `max_abs = 0`.

All measured model-update walls were approximately 30--31 seconds. Maximum
within-seed deviation from the shared wall was `1.33--1.97%` (median `1.48%`).

| Candidate | Median L128 post-relation MSE | Six-split range |
|---|---:|---:|
| shared Pure Spin(8) | 0.01296 | 0.00840--0.02014 |
| independent `SO(8)^3` | 0.09080 | 0.06007--0.12079 |
| Mamba-2 | 0.12753 | 0.12602--0.13037 |
| observation-only MLP | 0.12787 | 0.12635--0.13112 |
| parameter-near GRU | 0.15422 | 0.13385--0.28521 |
| state-matched GRU | 0.15049 | 0.13954--0.18867 |

Shared Spin(8) again beats every row on every L128 split. The independent-to-
shared median ratio is `7.0055x`. Extra endpoint updates do not rescue Mamba,
observation-only, or either GRU. The GRUs reduce some training losses but can
extrapolate worse at L128, an honest negative result against interpreting more
endpoint optimization as better group tracking.

The wall result is hardware-specific and descriptive. It is not a FLOP,
energy, fused-kernel, end-to-end-runtime, or hardware-independent compute
match. Transformers Mamba-2 used its explicit naive fallback because fused
selective-update dependencies were unavailable.

## What this supports, weakens, and leaves open

Supported for this bounded teacher family:

- dense every-prefix targets are not necessary to identify the shared action;
- the shared Spin(8) constraint is a materially better finite-data and
  finite-optimization inductive bias than three independently routed orthogonal
  actions, even though both can represent the teacher;
- endpoint-only credit assignment can recover a local transition law that
  extrapolates from length 16 to length 128; and
- the result is not explained by parameter count, recurrent-state size,
  observation-only leakage, an incapable independent control, equal updates,
  or measured model-update wall on this workstation.

Weakened or falsified explanations:

- the earlier continuous result is not merely teacher forcing from every
  prefix;
- it is not merely memorization of repeated observations, because every
  training observation is unique and charts/schedules change across seeds; and
- generic recurrent models are not rescued on this task by allocating many
  more updates to match the shared model's measured local wall.

Still open:

- unsigned, partially observed, noninjective, chart-shifted, irregularly
  labelled, longer-horizon endpoint, or natural targets;
- all 28 Spin(8) coordinates rather than the active seven-coordinate family;
- correlated noise, outliers, unknown initial state, and teacher-family shift;
- a fused training kernel and a fused, tuned modern-SSM comparison; and
- language, vision, robotics, or control utility.

The result demonstrates a replicated structural-prior advantage, not that
triality is mathematically necessary or that Pure Spin(8) is a universal SSM
winner.

The subsequent partial-readout observability cohort sharpens this boundary.
One signed `8s+` or `8s-` endpoint transfers the action into all three views,
but the frozen all-mask aggregate fails because vector-only supervision does
not robustly select an exact lift and one independent supervised control fails
optimization. An exact quotient-input collision separately proves balanced
hidden-lift Bayes MSE `1/8` and accuracy `1/2`; see
[`PURE_SPIN8_ENDPOINT_OBSERVABILITY_RESULTS.md`](PURE_SPIN8_ENDPOINT_OBSERVABILITY_RESULTS.md).

## Integrity and reproduction

Primary aggregate:<br>
`artifacts/pure_spin8_endpoint_supervision_validation_seeds1_3.json`<br>
SHA-256:<br>
`1cf51a4af05303bc3ca9e781478e2352e8dbb077d1c9b367f46af2f384653880`

Measured-wall aggregate:<br>
`artifacts/pure_spin8_endpoint_supervision_wall_matched_seeds1_3.json`<br>
SHA-256:<br>
`538a3bdbddfd76863a5bef5507a6d0019a114b35021d7b5b9d1223d31983ac64`

The primary and wall validators each rehash and strictly reload 18 checkpoints
with endpoint-only metadata. Their aggregate artifacts are content-locked by
tests. The corrected seed-0 development artifact is separately locked at
`87626dd9ac5a4f81695999a8832cb6eb3fb58ef312e86edfe642d1b54163a1d2`
and remains excluded from fresh validation.

The frozen commands and gates are in
[`PURE_SPIN8_ENDPOINT_SUPERVISION_PROTOCOL.md`](PURE_SPIN8_ENDPOINT_SUPERVISION_PROTOCOL.md).
Run one wall seed with:

```powershell
python benchmark_pure_spin8_endpoint_wall_matched.py --seed 1 --device cuda
```

Adjudicate the three wall artifacts with
[`validate_pure_spin8_endpoint_wall_matched.py`](../validate_pure_spin8_endpoint_wall_matched.py).
