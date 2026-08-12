# Intertwiner SchurScan equivariant-identification preregistration

- **Date frozen:** 2026-08-10, before implementation or result generation
- **Scope:** controlled identification gate, not a retrieval or language-model result

## Question

When a triangular SchurScan is trained only on a proper coordinate subspace,
does restricting its bilinear drive to the complete one-dimensional
equivariant-intertwiner family provide exact group-orbit extrapolation that an
unrestricted bilinear tensor cannot obtain from the same endpoints?

The gate is deliberately run for both the Spin(8) triality tensor and the
ordinary SO(3) cross product. A shared result is evidence for an intertwiner
prior, not for a triality-specific advantage.

## Frozen recurrence

For each sequence,

\[
u_t=a_t,\qquad v_t=b_t,\qquad
w_t=0.995w_{t-1}+\beta(u_t,v_t),\qquad w_{-1}=0.
\]

This is the maintained triangular SchurScan with zero source actions and a
contractive scalar downstream action. Its endpoint is

\[
w_{L-1}=\widetilde\beta x,\qquad
x=\sum_{t=0}^{L-1}0.995^{L-1-t}(u_t\otimes v_t).
\]

Training therefore uses deterministic float64 least squares rather than an
optimizer. The staged work-efficient scan must independently agree with the
closed-form endpoint and sequential recurrence.

## Instances and data

| Instance | Dimensions | Teacher | Training support | Recurrent state |
|---|---:|---|---:|---:|
| Spin(8) | 8 + 8 + 8 | triality tensor | first 4 coordinates in each source | 24 scalars |
| SO(3) | 3 + 3 + 3 | cross product | first 2 coordinates in each source | 9 scalars |

For seeds 0 through 9:

- use 64 base training endpoints of length 8;
- use fresh Gaussian source tokens confined to the stated training support;
- reserve all evaluation sequences and group elements;
- evaluate 128 fresh sequences at lengths 8, 32, 128, and 512;
- form the orbit split by applying one fresh shared group action to both source
  streams and the teacher output of each sequence;
- give the augmentation control four fresh group transforms per base endpoint.

The Spin(8) actions use the maintained shared vector/positive/negative
representations. The tensor convention is
`beta: negative x positive -> vector`. The SO(3) action is the same rotation in
all three spaces.

## Frozen model rows

1. `oracle_intertwiner`: the supplied teacher tensor; numerical upper bound.
2. `structured_intertwiner`: learn the single scalar multiplying the supplied
   intertwiner from the 64 unaugmented endpoints.
3. `generic_bilinear_restricted`: minimum-norm unrestricted bilinear tensor fit
   to the identical 64 endpoints.
4. `generic_bilinear_augmented`: the same unrestricted tensor family fit to
   four fresh group transforms of each base endpoint.
5. `additive_linear`: minimum-norm linear drive from `[u, v]`, fit to the
   identical unaugmented endpoints. This checks that the task actually needs a
   multiplicative interaction.

All rows use the same recurrent state. Parameter counts are reported, not
matched: the structured row has one fitted scalar, whereas a generic tensor
has `dim(W) * dim(U) * dim(V)` coefficients. The augmented row also receives
four times as many labelled endpoints and is an augmentation control, not a
sample-matched competitor.

## Metrics

- training relative squared endpoint error;
- orbit relative squared endpoint error at every evaluation length;
- mean endpoint cosine, reported beside squared error;
- design rank and singular values for both generic fits;
- exact tensor equivariance residual;
- work-efficient scan versus recurrence and closed-form endpoint residual;
- recurrent-state and fitted-parameter counts.

Relative squared error is computed per example and then averaged, with the
float64 tiny value protecting only a zero target denominator.

## Frozen gates

For both representation families and every seed:

- tensor equivariance maximum absolute error is below `1e-10`;
- structured-intertwiner training and worst orbit mean relative squared error
  are below `1e-16`;
- the restricted generic tensor fits training below `1e-16` but has orbit mean
  relative squared error above `0.10` at length 8;
- the augmented generic design has full input-feature rank and its worst orbit
  mean relative squared error is below `1e-14`;
- the additive row has training mean relative squared error above `0.25`;
- work-efficient scan, sequential recurrence, and the closed-form endpoint
  agree to maximum absolute error below `1e-10`.

The cohort-level gate requires all ten seeds to pass. Thresholds and sample
counts will not be changed after seeing results. A failed gate remains a
result and will be diagnosed without silently resampling.

## Interpretation boundary

Passing would establish a controlled sample/support-extrapolation advantage
for a known one-dimensional equivariant hypothesis class. It would not show
that triality beats another intertwiner, that the representation is discovered
from raw data, that direct-slot or delta memories are inferior, or that the
model improves retrieval, language modelling, or production throughput.

The task is teacher-aligned by construction. Its purpose is to test the exact
empirical consequence of the SchurScan theorem's equivariance clause before a
larger learned-retrieval campaign, not to serve as that campaign.
