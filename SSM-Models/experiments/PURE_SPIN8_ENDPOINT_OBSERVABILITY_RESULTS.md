# Pure Spin(8) endpoint observability results

Exact certificate recorded: **2026-08-17T06:11:01.888654+02:00**<br>
Frozen fresh cohort adjudicated: **2026-08-17T06:39:25.816374+02:00**

## Result in one sentence

The all-mask frozen cohort **failed** without median rescue, while exposing a
sharp and useful boundary: one signed half-spin endpoint is sufficient for the
shared Pure Spin(8) router to recover and transfer the local action across all
three 8D views in every fresh seed, but center-blind vector-only supervision
does not robustly select the exact global lift; if the input itself is reduced
to the vector quotient, an exact collision proves hidden-lift recovery
impossible.

This is not a blanket success report. The preserved aggregate has
`passed = false`.

## Exact observability and impossibility certificate

Twice every maintained triality generator is integral. SymPy rank over the
rationals gives the same infinitesimal probe profile for `8v`, `8s+`, and
`8s-`:

`7, 13, 18, 22, 25, 27, 28, 28`.

The explicit seven-basis-probe witness has rank 28 in each representation, so
full local Lie-algebra observability is generic from seven probes. This is a
local statement; it does not remove discrete kernels globally.

For the tested central element `z=-1`, the maintained representations give

`rho_v(z)=+I`, `rho_+(z)=-I`, and `rho_-(z)=-I`.

The vector endpoint is therefore blind to this sign, while either half-spin
endpoint sees it.

The quotient collision makes the global limitation exact. Plane-0 coordinates
`0` and `2*pi` have the same complete `8v` action matrix (float64 residual
`2.45e-16`) and opposite `8s+` targets (negation residual `1.22e-16`). For
balanced lifts and a unit target, the conditional mean is zero, squared-loss
Bayes MSE is exactly `1/8`, and hidden-lift classification accuracy is at most
`1/2`. No larger neural architecture can recover information absent from its
input.

## Frozen empirical design

Each fresh seed uses 2,000 updates, batch 32, L16, 1,024,000 unique noisy
observations, 64,000 endpoints, and zero adjacent held-out half-center pairs.
Only selected final blocks are sliced on CPU and transferred to the loss; no
prefix targets or hidden endpoint blocks reach the accelerator-side training
step.

The five masks expose 8, 8, 8, 16, or 24 endpoint scalars:
`vector_only`, `positive_only`, `negative_only`, `spinor_pair`, and
`full_triality`. The 930-parameter shared model and 957-parameter independent
`SO(8)^3` control have the same 24 recurrent scalars and use the same schedule
and candidate-specific initialization across masks.

## Preregistered outcome: failed

| Seed | Frozen gates passed | Failures |
|---:|---:|---|
| 1 | 37/40 | vector-only exact hidden-spinor row; positive-only independent supervised capability and center accuracy |
| 2 | 39/40 | vector-only exact hidden-spinor row |
| 3 | 39/40 | vector-only exact hidden-spinor row |

The validator therefore rejected the cohort. Thresholds were not relaxed and
the strong signed-readout rows did not rescue the failed masks.

## What did replicate

### One signed half-spin endpoint transfers to all three views

With only eight final `8s+` scalars supervised, shared action RMSE is nearly
identical in every view:

| View | Median action RMSE | Six L128 MSE median | Six-split range |
|---|---:|---:|---:|
| hidden `8v` | 0.01245 | 0.01564 | 0.00991--0.02278 |
| supervised `8s+` | 0.01245 | 0.01566 | 0.00974--0.02492 |
| hidden `8s-` | 0.01245 | 0.01510 | 0.01058--0.02438 |

All shared spinor center rows are correct in every split. Negative-only
supervision is chirally symmetric:

| View | Median action RMSE | Six L128 MSE median | Six-split range |
|---|---:|---:|---:|
| hidden `8v` | 0.01294 | 0.01460 | 0.01004--0.02328 |
| hidden `8s+` | 0.01295 | 0.01480 | 0.00950--0.02520 |
| supervised `8s-` | 0.01295 | 0.01512 | 0.00932--0.02309 |

The shared model passes every intrinsic action, L128, and center gate for both
single-spinor masks in all three seeds. This is genuine cross-representation
transfer: the independent family learns only its supervised action head, while
its hidden-view action RMSE remains about `0.18--0.22` and its hidden-view L128
MSE remains about `0.22--0.25`.

The positive-only comparative row is not a clean three-seed win because the
independent supervised head failed optimization in seed 1: its L128 `8s+` MSE
was `0.1960/0.2449` and center accuracy `0.4844/0.4922`. Seeds 2 and 3, the
negative-only rows, paired-spinor rows, and full rows demonstrate that the
independent class can fit supervised views. The failed seed is an optimization
failure, not a representational impossibility, and remains attached to the
result.

### Paired-spinor and full readouts are stable

Paired-spinor shared L128 medians are `0.01317--0.01398` across the three views;
full-triality medians are `0.01285--0.01365`. Every preregistered gate for these
two masks passes in every seed.

## What failed and what it means

Vector-only shared training still transfers remarkably well: median action
RMSE is `0.05044--0.05096` across views, and hidden-spinor L128 MSE ranges from
`0.04901` to `0.09256`, far below the independent hidden heads near `0.25`.
But the minimum hidden-spinor center accuracy is `0.984375`, not the required
`1.0`; every fresh seed misses at least one exact vector-only center gate.

This does not contradict the good continuous errors. The injective input chart
still exposes local Lie coordinates, and the shared parameterization strongly
prefers a continuous lift. Yet the supervised vector endpoint factors through
the center, so the objective does not globally fix that lift. The experiment
therefore separates:

- local Lie-algebra/action recovery, which succeeds strongly; from
- exact global double-cover selection, which is not robust without a
  center-visible signal.

The exact quotient collision is sharper still: when the input also factors
through `8v`, the missing lift is not merely weakly trained but formally
unrecoverable under balanced labels.

## Integrity

The independent adjudicator regenerated all three training schedules and all
18 evaluation schedules, reproduced their hashes, corrected the declared
vector-visibility roundoff issue, rehashed and strictly reloaded all 30 fresh
checkpoints, and recomputed every action and per-view metric. All integrity
checks pass; only preregistered scientific gates fail.

Exact certificate SHA-256:<br>
`fa29a9d74a927993c17328b7dffb5f96c7f42f308b2e30450d4f714a9ce89a53`

Fresh source SHA-256 values:

- seed 1: `46ca4c9a3fb66657f99312d4b4adc4ee7ab40bb89bdb40ca20796eb1aa960bca`
- seed 2: `c3591e2a7c98eef45d6dfd053e0d19e900b71bdf3d63149588504444867410e2`
- seed 3: `a1ddab19a2048e2a7b864ce467307be1b93c836d2bb968b3b144ab20018b51cf`

Failed aggregate SHA-256:<br>
`baed378d569391e86c46df731cfc72db4f0c0a24d21883bb17a4604db9e5c987`

The frozen commands and gates are in
[`PURE_SPIN8_ENDPOINT_OBSERVABILITY_PROTOCOL.md`](PURE_SPIN8_ENDPOINT_OBSERVABILITY_PROTOCOL.md).

## Claim boundary and next falsifier

Established here:

- an exact local probe-rank certificate in every 8D view;
- an exact center-visibility classification for the tested `-1` element;
- an exact Bayes lower bound for quotient-input hidden-lift recovery;
- replicated shared transfer from one signed half-spin endpoint to all three
  representations; and
- a replicated failure of exact vector-only lift selection under the frozen
  gate.

Still open are all 28 active coordinates in training, unknown initial state,
noninjective but center-visible sensors, natural data, fused training, and
downstream utility. The clean next experiment is not to demand an impossible
unsigned sign. It is to add the smallest center-visible calibration bit or
single signed probe to the otherwise vector-only endpoint and test whether the
three vector-only failures collapse to exact recovery.
