# Pure Spin(8) negative-alignment calibration-rank protocol

Status: **FROZEN BEFORE FRESH SEEDS**<br>
Protocol frozen: **2026-08-17T09:52:34.5014289+02:00**<br>
Fresh seeds: **10, 11, 12**

## Question

The earlier shared-latent scrambled control proved that an entirely hidden
negative-spinor alignment receives no data gradient. This protocol asks a more
precise question:

> How many independent external calibration constraints are required to
> identify that 28-parameter alignment locally, and does reaching full rank
> recover the hidden negative sequence behavior?

The protocol is a calibration experiment. It does not infer an alignment from
the vector endpoint or claim that the calibration frame is naturally
available.

## Exact rank contract

Let `T` be the negative-half-spin `SO(8)` action and reveal the ordered basis
probe images

`T e_1, ..., T e_m`.

The differential rank is

`28 - (8-m)(7-m)/2`,

giving the frozen profile

`0, 7, 13, 18, 22, 25, 27, 28, 28`

for `m=0,...,8`. The residual continuous stabilizer is `SO(8-m)`. Seven
probes remove every continuous ambiguity; the eighth is redundant because an
orientation-preserving action is already determined by its first seven basis
images.

Each probe transmits eight scalar values. Thus the raw scalar budgets are
`0,8,...,64`, while the independent ranks are the profile above. The protocol
must report both and must not call eight transmitted values eight independent
constraints.

The certificate is computed exactly over the rationals after multiplying the
maintained generators by two. It also checks the implemented factorized chart
at identity and at each frozen initialization.

## Matched architecture

The observed candidate is the maintained 930-parameter shared Pure Spin(8)
tracker. The intervention wraps the identical initialized router with one
28-parameter negative-only alignment:

- vector action: unchanged;
- positive-spinor action: unchanged;
- negative-spinor action: `T rho_-(g) T^-1`;
- recurrent state: the same 24 scalars;
- total trainable parameters: 958.

The earlier two-alignment control is not reused for this curve. Its trainable
positive alignment changes the gauge seen by the positive lift-bit loss and can
change the common router. Here the observed vector and positive states are
routed directly through the maintained tracker.

The router and hidden alignment use disjoint AdamW optimizers and disjoint
gradient clipping. The router optimizer sees only the unchanged vector plus
adaptive-positive-bit loss. The alignment optimizer sees only the external
basis-probe calibration loss, with weight `1.0`. A strict gate requires the
final router hash to be bitwise identical in the aligned reference and all nine
rank conditions.

No negative sequence endpoint target is transferred to either optimizer.

## Frozen training and evaluation

Every candidate uses:

- 2,000 updates;
- batch size 32;
- length-16 endpoint-only training;
- learning rate `0.003`;
- AdamW weight decay `0.0001`;
- gradient clip `1.0`, separately applied to router and alignment;
- the existing noisy seven-coordinate injective observation system;
- the existing excluded adjacent center relation;
- adaptive positive lift-bit supervision;
- L16, L64, and L128 early/late relation tests with 64 pairs each;
- identical schedules and initialization within each seed.

All probe counts `m=0,...,8` are trained. Fresh seeds are exactly 10, 11, and
12. Development seed 0 selected the architecture, loss weight, and thresholds;
it is not part of the fresh adjudication.

## Frozen seedwise gates

Every gate must pass independently in every fresh seed. No median or aggregate
statistic may rescue a seed.

1. Source, schedule, observation, evaluation, checkpoint, and metric replay
   integrity passes.
2. Shared action RMSE is at most `0.03` in every representation, and final
   adaptive-bit accuracy is exactly `1.0`.
3. Every nonzero probe frame has selected-probe RMSE at most `1e-4`.
4. Six probes/rank 27 retains full-alignment RMSE at least `1e-3`.
5. Seven and eight probes each have full-alignment RMSE at most `1e-4`.
6. Seven-probe negative action RMSE matches the aligned reference within
   absolute `1e-5`.
7. The eighth probe changes negative action RMSE by at most `1e-5`.
8. Seven-probe negative action RMSE is at most half the six-probe value.
9. Seven probes beat six probes on both early and late negative L128
   post-relation MSE.
10. Seven-probe negative L128 MSE matches the aligned reference within `1e-5`
    on both splits.
11. Seven- and eight-probe negative L128 MSE agree within `1e-5` on both
    splits.
12. Every final router hash is bitwise identical to the aligned reference.

The validator also checks the rank and raw scalar budget attached to every
checkpoint, legal orthogonal alignment actions, distinct fresh schedules, and
all 18 distinct fresh evaluation schedules.

## Interpretation boundary

A pass would establish an exact local rank theorem and a replicated empirical
rank-28 recovery threshold for this supplied calibration frame and synthetic
endpoint task. It would not prove global uniqueness of factorized coordinates,
recover a discrete Spin-cover kernel from `SO(8)` probe images, infer the frame
from raw observations, guarantee optimization from arbitrary charts, or show
natural-task or throughput superiority.

A failed gate remains a failed protocol. Any narrower surviving pattern must be
reported separately and cannot replace the frozen headline.
