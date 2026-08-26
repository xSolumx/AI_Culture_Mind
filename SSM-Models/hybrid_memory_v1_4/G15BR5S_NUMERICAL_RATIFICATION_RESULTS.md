# G15B-R5-S numerical ratification results

**Frozen protocol:**
[`G15BR5S_NUMERICAL_RATIFICATION_PROTOCOL_2026-08-26.md`](G15BR5S_NUMERICAL_RATIFICATION_PROTOCOL_2026-08-26.md)

**Exact artifact:**
[`artifacts/g15br5s_numerical_ratification_sm75_2026-08-26.json`](artifacts/g15br5s_numerical_ratification_sm75_2026-08-26.json)

**SHA-256:**
`3ac514e16e6fa1c720d5ef4244525f5d0f08c233634648e59181c6acfccc3a00`

**Execution commit:** `dde868aba1328396d9fcb3038427d6ef77a6efd2`

**Status:** frozen formal fail; retained-checkpoint repair stops; R5's
performance-positive history attribution remains unchanged

## Bottom line

R5-S failed its prospectively frozen scaled-logit gate in every source/cell:

- `0/45` task/length/checkpoint cells passed;
- `0/135` H/C/B source-cell contracts passed;
- the smallest scaled-logit allowance ratio was `1.171875`;
- the largest was `66.078125`, for seed 2309, needle L1024, bias source.

No threshold is changed after seeing this result. The exact artifact decision
is:

> `stop retained-checkpoint tail repair; R5 numerical stability did not ratify prospectively`

The failure is much narrower than those counts suggest. Every categorical
prediction and aggregate categorical metric remained exact. Maximum BPQ drift
was `1.719827e-7`; state and background-read residuals were
`1.907349e-6` and `2.980232e-6`, both within the frozen `5e-6` engineering
ceiling. Injection summation reached only `5.960464e-8`, source assignment was
exact, every transition replay was bit-identical, and independent FP64 algebra
passed at `5.861978e-14`.

R5-S therefore does **not** show semantic divergence in the memory recurrence.
It shows that the frozen `64 * eps` logit tolerance was not a valid end-to-end
forward-error bound for this checkpoint shell. The small read difference is
passed through an RMS-normalized readout and learned projections; their gain
can turn micro-scale FP32 reduction differences into logits separated by up to
`6.387234e-4` without changing a prediction or measured BPQ materially.

This does not relabel R5 or R5-S. It stops the retained-checkpoint repair route
and supports an exact-by-construction read path in the next fresh model rather
than another post-hoc tolerance repair.

## Exact execution

The evidentiary quality cohort used:

- Python 3.11.16;
- PyTorch 2.9.0+cu128;
- NVIDIA GeForce RTX 2070 SUPER;
- exact compute capability `(7,5)`;
- clean Git status at start;
- retained checkpoint seeds 2309, 2311, and 2333;
- five tasks and lengths 128, 512, and 1,024;
- 512 fresh query decisions per cell;
- 432 scored fresh batches;
- zero optimizer updates;
- `1,155.11` seconds wall time.

All three checkpoint hashes and every R5 helper/model/task source hash matched
the sealed R5 artifact. R5-S reconstructed 3,456 original individual batches,
reproduced all 45 sealed aggregate digests, generated all 432 fresh batches
twice deterministically, and found no individual fingerprint overlap.

## Frozen-gate adjudication

| Contract | Observed maximum or count | Frozen requirement | Result |
|---|---:|---:|---|
| Complete cells | 45/45 | 45/45 | pass |
| Sealed aggregate digests | 45/45 | exact | pass |
| Fresh deterministic/disjoint fingerprints | 45/45 | exact | pass |
| Source/checkpoint hashes | all | exact | pass |
| Independent transition calls | 432/432 | bit-identical | pass |
| Source assignment | 0 residual | exact | pass |
| Injection component sum | `5.960464e-8` | `<= 5e-6` | pass |
| No-reset state sum | `1.907349e-6` | `<= 5e-6` | pass |
| Background read relation | `2.980232e-6` | `<= 5e-6` | pass |
| Query/episode categorical behavior | 135/135 exact | exact | pass |
| BPQ absolute residual | `1.719827e-7` | `<= 1e-6` | pass |
| Independent FP64 algebra | `5.861978e-14` | `<= 1e-10` | pass |
| Scaled-logit allowance | 135/135 above bound | `<= 1.0` ratio | **fail** |

Every source/cell failed only because of its scaled-logit ratio. The ratios
ranged from `1.171875` to `66.078125`. By source, the maxima were:

| Source | Maximum ratio | Maximum absolute logit residual |
|---|---:|---:|
| strict history H | `45.460938` | `4.353523e-4` |
| current token C | `46.492188` | `4.291534e-4` |
| convolution bias B | `66.078125` | `6.387234e-4` |

The corresponding maximum ratios by checkpoint were `66.078125`,
`11.789062`, and `43.888672` for seeds 2309, 2311, and 2333. The effect is
not isolated to one source, task, length, or checkpoint.

## Common-FP64 worst-batch evidence

Each cell selected its worst normalized fresh batch before replaying it against
one common FP64 scan/read reference. Across those 45 retained batches:

| Quantity | Maximum absolute residual |
|---|---:|
| FP32 monolithic state to common FP64 | `1.801424e-6` |
| FP32 decomposed state to common FP64 | `1.836282e-6` |
| FP32 monolithic read to common FP64 | `2.380406e-6` |
| FP32 decomposed read to common FP64 | `2.425473e-6` |
| Independently rebuilt FP64 algebra | `5.861978e-14` |

Both FP32 routes are similarly close to the common reference. The decomposed
route is not hiding a different recurrence; the end-to-end logit check is
amplifying ordinary accumulation differences. R5-S is consistent with FP32
reduction-order effects, but it cannot prove they caused R5's original maxima
because R5 did not retain the original worst-batch locations.

## What remains learned and what does not

R5's separate performance evidence is unchanged:

- the background-free strict-history LWW arm passed all 132 frozen
  performance and bias-separation checks;
- ordinary overwrite reached `0.9424`--`0.9466` mean accuracy;
- the constructed guard reached 1.0;
- current-only and bias-only controls failed.

R5-S did not re-score those gates and cannot promote that arm. Its failure also
does not refute strict-history source sufficiency. It rejects only the proposed
numerical ratification and therefore closes the retained-checkpoint repair
series.

## Architectural decision

Do not run another tolerance sweep and do not keep repairing the retained G15B
checkpoint. The next fresh architecture should make the useful mechanism
explicit:

1. learn a causal pending transaction state from strict convolution history;
2. commit key/value content with separate write and erase/replace control;
3. represent occupancy/reset semantics explicitly;
4. read protected key content without summing a decomposed full state merely
   to reproduce the monolithic path;
5. keep the ordinary monolithic read as the exact residual/reference path;
6. compare identity, fixed-torus, and Spin transport only after the edit law
   learns under parameter/state/token/step-cost-matched controls.

An exact-by-construction implementation can compute the ordinary full read once
and use components only for protected key/background attribution. It should not
assert bitwise-equivalent logits from two different FP32 reduction trees.

The next training protocol must be frozen on fresh seeds and must include
multi-seed natural-text robustness, longer-context recall after ordinary
pretraining, and parameter/compute-matched scaling. R5-S itself authorizes no
training or model promotion.

## Nonclaims

R5-S does not establish optimizer or tokenizer superiority, autonomous
transaction learning, generic association, natural-text improvement,
long-context scaling, efficiency, a Spin benefit, G15C, or model-family
promotion. The `5e-6` and `64 * eps` thresholds were engineering choices, not
derived theorems. The result is a frozen negative numerical gate with a narrow
mechanistic diagnosis.
