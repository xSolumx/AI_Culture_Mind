# Pure Spin(8) negative-alignment calibration-rank results

Recorded: **2026-08-17T10:34:43.2866889+02:00**

## Result

The exact geometric threshold is stronger than the development claim, but the
frozen empirical headline **fails**.

- **Exact global theorem:** seven ordered basis probes are necessary and
  sufficient to identify an `SO(8)` alignment action. Zero through six probes
  are globally non-identifying; probe eight is redundant.
- **Implemented chart:** the maintained factorized chart has exact/numerical
  rank profile `0,7,13,18,22,25,27,28,28` at identity and at every frozen
  initialization checked by the cohort.
- **Fresh empirical cohort:** every source, schedule, observation, evaluation,
  checkpoint, reload, and metric replay check passes. Seeds 11 and 12 pass all
  frozen gates. Seed 10 fails two effect-size gates, so no median rescues the
  preregistered headline.
- **Surviving empirical result:** all three seeds fit every revealed frame,
  seven probes recover the full alignment and every aligned sequence metric,
  and the eighth probe changes nothing. The failed statement is the uniform
  size of the rank-27 versus rank-28 task-error gap, not the exact
  identifiability threshold.

## Exact global theorem

For the ordered-probe map

`F_m(T) = (T e_1, ..., T e_m)`, with `T in SO(8)`,

the fiber over a frame is `SO(8-m)`. Its Lie dimension is

`(8-m)(7-m)/2`,

so the information rank is

`28 - (8-m)(7-m)/2`.

This gives:

| probes `m` | transmitted scalars | independent rank | fiber dimension |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 28 |
| 1 | 8 | 7 | 21 |
| 2 | 16 | 13 | 15 |
| 3 | 24 | 18 | 10 |
| 4 | 32 | 22 | 6 |
| 5 | 40 | 25 | 3 |
| 6 | 48 | 27 | 1 |
| 7 | 56 | 28 | 0 |
| 8 | 64 | 28 | 0 |

The certificate goes beyond a Jacobian calculation. For every `m <= 6`, let
`W_m` be identity on the revealed columns and a rational quarter-turn
`[[0,-1],[1,0]]` on two unobserved coordinates. Then `W_m` is an exact,
orientation-preserving orthogonal matrix, fixes all `m` probes, differs from
identity, and has entrywise matrix RMSE `1/4`. Thus fewer than seven probes are
globally insufficient. At seven probes the residual group is `SO(1)={1}`;
equivalently, the eighth column is fixed by orthogonality and orientation.

This is global uniqueness of the **`SO(8)` action matrix**, not of a particular
factorized coordinate vector or of its discrete Spin-cover lift.

## Matched control

The experiment keeps the maintained 930-parameter shared router and its
24-scalar recurrent state unchanged. The control adds one 28-parameter legal
negative-view alignment, for 958 trainable parameters total:

- vector and positive actions are exactly unchanged;
- only the negative action is conjugated;
- the router optimizer receives the existing vector plus adaptive positive-bit
  loss;
- the alignment optimizer receives only external ordered basis probes;
- the optimizers and gradient clipping are disjoint;
- no negative sequence endpoint target is transferred;
- the final router SHA is bitwise identical at all nine probe counts within
  every seed.

This negative-only design replaced a rejected development pilot whose
trainable positive alignment changed the common router gauge. The clean
development seed passed the later-frozen gates, but was not counted in the
fresh cohort.

## Fresh aggregate

The table reports medians across seeds 10--12. L128 combines the early and late
post-relation cells. Full-frame RMSE measures the learned alignment against the
aligned action, not sequence-target error.

| `m` | rank | negative action RMSE, median (range) | full-frame RMSE, median (range) | negative L128 MSE, median |
|---:|---:|---:|---:|---:|
| 0 | 0 | `0.179880` (`0.161043--0.205070`) | `0.266325` (`0.265010--0.332634`) | `0.118969` |
| 1 | 7 | `0.158174` (`0.154130--0.158554`) | `0.261770` (`0.237008--0.280030`) | `0.112321` |
| 2 | 13 | `0.126592` (`0.121758--0.136450`) | `0.214485` (`0.192508--0.226131`) | `0.087859` |
| 3 | 18 | `0.117760` (`0.101397--0.138665`) | `0.190560` (`0.184072--0.213493`) | `0.077848` |
| 4 | 22 | `0.088171` (`0.081846--0.098824`) | `0.162524` (`0.137728--0.170918`) | `0.059228` |
| 5 | 25 | `0.075521` (`0.055023--0.100116`) | `0.118261` (`0.079027--0.167814`) | `0.031982` |
| 6 | 27 | `0.040119` (`0.014759--0.066408`) | `0.077700` (`0.006133--0.131530`) | `0.026551` |
| 7 | 28 | `0.014077` (`0.013059--0.014492`) | `1.410e-23` (`3.316e-28--1.254e-21`) | `0.016215` |
| 8 | 28 | `0.014077` (`0.013059--0.014492`) | `7.637e-27` (`2.980e-28--5.102e-19`) | `0.016215` |

Every nonzero selected frame has RMSE at most `1.681e-7`. Seven-probe
full-frame RMSE is at most `1.255e-21`; eight-probe full-frame RMSE is at most
`5.102e-19`. In every seed the rank-28 rows reproduce the shared negative
action and both L128 cells exactly to the recorded precision.

## Frozen gate adjudication

| seed | passed gates | failed gates |
|---:|---:|---|
| 10 | 12/14 | rank-28 action was not at most half rank-27 action; rank 28 did not beat rank 27 on both negative L128 splits |
| 11 | 14/14 | none |
| 12 | 14/14 | none |

Seed 10 is the direct falsifier. Its rank-27 frame error is nonzero
(`0.006133`), but its total negative-action RMSE is already `0.014759` against
the shared router floor `0.014492`. Rank 28 recovers the aligned value exactly,
yet cannot halve a total error dominated by the common router. On late L128,
rank 27 happens to score `0.023810` versus aligned rank 28 at `0.023877`.

Seeds 11 and 12 show the larger-residual regime. Their rank-27 frame RMSEs are
`0.077700` and `0.131530`, negative-action RMSEs are `0.040119` and `0.066408`,
and rank 28 returns exactly to `0.013059` and `0.014077` respectively.

The preregistered all-seed effect-size claim is therefore false. The result
also diagnoses why: the unidentified `SO(2)` fiber is exact, but the residual
angle left by a random initialization can be small, and total task RMSE mixes
that angle with a common router floor.

## What is established

- Exact, global action non-identifiability from zero through six ordered probes.
- Exact, global `SO(8)` action identifiability from seven ordered probes.
- Exact redundancy of probe eight.
- The same rational differential rank profile in maintained `8v`, `8+`, and
  `8-` generators.
- Numerical agreement of the implemented factorized chart with that profile.
- Replicated optimization to the exact rank-28 action under the supplied
  external calibration frame.
- Bitwise isolation of router learning from alignment calibration.

## What is not established

- The frozen uniform factor-of-two task-error improvement over rank 27.
- Alignment discovery from raw vector observations or sequence endpoints.
- Recovery of the discrete kernel of a Spin-cover representation.
- Global uniqueness of factorized coordinates.
- Natural-task, throughput, state-efficiency, or generic SSM superiority.

The next clean falsifier should not rerun the same noisy total-error gate. It
should ask whether relational observations can *produce* the seven-probe frame,
or use an adversarial exact stabilizer witness when testing worst-case
non-identifiability. Any new metric must be frozen before another cohort.

## Reproduction and artifacts

Run

```text
python SSM-Models/analyze_pure_spin8_alignment_calibration_rank.py
python SSM-Models/benchmark_pure_spin8_alignment_calibration_rank.py --seed 10 --steps 2000 --evaluation-pairs 64 --anchor-counts 0,1,2,3,4,5,6,7,8 --device cuda
python SSM-Models/validate_pure_spin8_alignment_calibration_rank.py --device cuda
```

The benchmark was run separately for seeds 10, 11, and 12 with distinct output
and checkpoint paths. SHA-256:

- frozen protocol: `ba49990666941399e81fba32848e8ff08906c2448480aebb75959a6a2632d2eb`;
- exact certificate: `0a1c6ea0107aa732a0656bbb739b1e1eab650eabf40d64ad6229f42810255124`;
- development source: `681e75597acfe62f8e0458b308a89b3b4e481274a1213c7410bbf274d8d76f66`;
- development replay: `6b614d0dd019aefac2cf69f9618235395049e1b57f4eee9a66f14abb0b667f5f`;
- seed 10: `81091d3cb8a0e1508e74b7136558131ae5efd8b699c25da4d978dc6d40871eb3`;
- seed 11: `9b54e83706c445a36b144cea97727d04758d3dae2c389548a138c22b037db05c`;
- seed 12: `d694e425828184f9ce57c6894f7daf953a3f3b4370e724e6cda31561e8d9928e`;
- failed strict aggregate: `1708d2932cce32f0b1715c1563af35686aa096e7020fe0cbd80ed7f67a2bad2a`.

The frozen protocol is
[`PURE_SPIN8_ALIGNMENT_CALIBRATION_RANK_PROTOCOL.md`](PURE_SPIN8_ALIGNMENT_CALIBRATION_RANK_PROTOCOL.md).
The executable certificate, benchmark, and validator are
[`analyze_pure_spin8_alignment_calibration_rank.py`](../analyze_pure_spin8_alignment_calibration_rank.py),
[`benchmark_pure_spin8_alignment_calibration_rank.py`](../benchmark_pure_spin8_alignment_calibration_rank.py),
and
[`validate_pure_spin8_alignment_calibration_rank.py`](../validate_pure_spin8_alignment_calibration_rank.py).
