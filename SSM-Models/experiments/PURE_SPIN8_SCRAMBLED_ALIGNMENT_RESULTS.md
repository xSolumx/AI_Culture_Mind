# Pure Spin(8) shared-latent scrambled-alignment results

Status: **FAILED**, frozen fresh cohort, no median rescue<br>
Protocol frozen: **2026-08-17T08:40:40.5270068+02:00**<br>
Fresh cohort adjudicated: **2026-08-17T08:54:32.136060+02:00**

## Headline verdict

The preregistered claim that correctly aligned Pure Spin(8) beats a
gradient-route-matched scrambled control in **every** action and L128 view is
false. Seed 7's scrambled control has slightly lower MSE in both directly
supervised vector L128 rows:

| Seed 7 vector row | Pure Spin(8) | scrambled control |
|---|---:|---:|
| early L128 | `0.016321` | **`0.013188`** |
| late L128 | `0.028278` | **`0.027063`** |

That one seedwise gate fails. The frozen aggregate therefore fails even though
35 of 36 seedwise gates pass. No median, average, or post-hoc narrowing rescues
the protocol.

## What was controlled

The scrambled candidate uses the same initialized 12-to-22-to-28 observation
router as Pure Spin(8), the same 24-scalar state, and one common bivector head.
It has 986 trainable parameters versus 930, so it is slightly
over-parameterized. Its positive and negative actions are independently
conjugated by trainable valid Spin(8) actions. This preserves legal orthogonal
composition and the shared-gradient route while removing the supplied
cross-view alignment.

Under adaptive vector-plus-bit loss:

- all 28 common coordinate rows receive gradients;
- the positive alignment receives gradients;
- the hidden negative alignment receives exactly zero data gradient and is
  changed only by AdamW decay.

Under full-triality loss, both alignments receive gradients. The same control
therefore supplies its own capability falsifier.

## Frozen gate audit

Every source, schedule, observation, address, bit, evaluation, checkpoint, and
metric integrity check passes. Seeds 7--9 have distinct 1,024,000-observation
training schedules and 18 distinct evaluation schedules. All 12 checkpoints
strictly rehash, reload, and replay.

| Gate family | Seed 7 | Seed 8 | Seed 9 |
|---|---:|---:|---:|
| shared adaptive absolute gates | pass | pass | pass |
| scrambled observed-channel capability | pass | pass | pass |
| scrambled full-supervision capability | pass | pass | pass |
| shared wins every action view | pass | pass | pass |
| shared wins every L128 view | **fail** | pass | pass |
| full supervision repairs negative view | pass | pass | pass |
| adaptive negative alignment is decay-only | pass | pass | pass |
| full supervision updates both alignments | pass | pass | pass |

## Preserved representation-specific stratum

The failed all-view claim contains a narrower pattern that was not used to
rescue it:

- shared Pure Spin(8) wins all **9/9 action-view** comparisons;
- it wins all **12/12 spinor L128** comparisons;
- it wins all **6/6 completely hidden negative-spinor L128** comparisons;
- it wins only **4/6 directly supervised vector L128** comparisons.

The aggregate adaptive metrics are:

| Candidate / view | median action RMSE | median L128 MSE | full range |
|---|---:|---:|---:|
| shared vector | `0.013996` | `0.017988` | `0.010511--0.028278` |
| scrambled vector | `0.018373` | `0.025031` | `0.011771--0.096804` |
| shared positive | `0.013999` | `0.018058` | `0.010206--0.031203` |
| scrambled positive | `0.052733` | `0.030223` | `0.016770--0.101697` |
| shared negative | `0.013999` | `0.018628` | `0.010219--0.033671` |
| scrambled negative | `0.185325` | `0.157190` | `0.121478--0.229274` |

The shared/scrambled median ratio is about `13.24x` on hidden negative action
RMSE and `8.44x` on hidden negative L128 MSE. These ratios are descriptive,
not protocol-rescuing statistics.

## Capability and causal repair

Full supervision makes the scrambled control competitive:

| Scrambled negative view | adaptive calibration | full triality |
|---|---:|---:|
| median action RMSE | `0.185325` | `0.011859` |
| median L128 MSE | `0.157190` | `0.019852` |

Every fully supervised scrambled center row is exact. Its adaptive negative
center accuracy ranges from `0.5547` to `0.9844`, while full supervision makes
all six rows `1.0`.

The checkpoint trace is exact in every seed:

- adaptive positive-alignment residual from decay-only:
  `0.5246--0.6980`;
- adaptive negative-alignment residual from decay-only: **`0.0`**;
- full positive-alignment residual: `0.6514--0.7858`;
- full negative-alignment residual: `0.6328--1.1737`.

Thus the hidden negative transfer gap is not explained by an incapable model,
a missing shared bottleneck, an unfitted observed target, or a failed
checkpoint. It is exactly tied to whether the correct cross-view alignment is
supplied or supervised.

## Interpretation

- **Disproved:** universal per-view superiority of the aligned model under the
  frozen cohort. The directly observed vector view can tie or favor the
  scrambled control on individual long splits.
- **Empirical, replicated, not headline-rescuing:** the correct supplied
  triality alignment strongly improves both spinor views and especially the
  completely unsupervised negative view across all fresh seeds.
- **Exact for this implementation:** the adaptive loss cannot data-update the
  scrambled negative alignment; full supervision does update it.
- **Bounded causal evidence:** the advantage is localized to cross-view
  representation transfer rather than generic vector action inference.

This is stronger mechanistic localization than the earlier independent
`SO(8)^3` comparison, but it is still a synthetic supplied-prior result. It
does not prove a universal triality advantage, alignment discovery from raw
data, natural-task benefit, or optimized SSM superiority.

## Artifacts

- seed 7 SHA-256:
  `3bc885cbaa5264b3b8eaabe816d71ae36447a73a78a599eb8c08186640fd2992`
- seed 8 SHA-256:
  `fd7a12f19c24da821dfe5d41e74c7cb87425829ddbf794d4c7a46029dec6c5e0`
- seed 9 SHA-256:
  `b56e99b19f980bcff6f810066f14cad099a17cac9283a004c6580bf1253fe03b`
- failed strict aggregate SHA-256:
  `ec6802d9c55f318aa85aaacb9ce4030df697f716158a3bdc5752432394f044a7`

Reproduce with
[`benchmark_pure_spin8_scrambled_alignment.py`](../benchmark_pure_spin8_scrambled_alignment.py)
and adjudicate with
[`validate_pure_spin8_scrambled_alignment.py`](../validate_pure_spin8_scrambled_alignment.py).

## Completed next falsifier

The calibration-rank continuation is now complete. Ordered negative-view basis
probes have exact ranks `0,7,13,18,22,25,27,28,28`. Explicit rational
stabilizer witnesses make the boundary global: zero through six probes cannot
identify the `SO(8)` action, seven do, and eight are redundant. The fresh
empirical all-seed headline nevertheless fails because seed 10's rank-27
residual is too small to satisfy the frozen factor-of-two total-error gate.
All integrity checks pass, and seven probes recover the aligned action and L128
rows exactly in every seed. See
[`PURE_SPIN8_ALIGNMENT_CALIBRATION_RANK_RESULTS.md`](PURE_SPIN8_ALIGNMENT_CALIBRATION_RANK_RESULTS.md).
