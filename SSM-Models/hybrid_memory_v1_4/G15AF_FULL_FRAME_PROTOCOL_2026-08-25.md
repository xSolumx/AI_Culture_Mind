# G15A-F full-frame observability protocol

**Frozen:** 2026-08-25, after the G15A-L failure and its post-hoc
observability diagnostic, and before any G15A-F smoke or quality metric was
produced or inspected

**Required G15A-L evidence SHA-256:**
`7716b75e43964d479bd5fef0cfbd06d0328a315c00a4a0fa107b26f785af1108`
for
[`artifacts/g15al_learned_coordinate_cohort_sm75_2026-08-25.json`](artifacts/g15al_learned_coordinate_cohort_sm75_2026-08-25.json)

**Bound post-hoc diagnostic SHA-256:**
`8c53afd308763c6f25042f702d6fcbd667315c35384483a956119d252e3c8d21`
for
[`artifacts/g15al_observability_diagnostic_sm75_2026-08-25.json`](artifacts/g15al_observability_diagnostic_sm75_2026-08-25.json)

## Question

G15A-L could learn the positive-carrier chart, but its single-query cosine
objective made the vector carrier enter only through a positive scalar. Cosine
normalization removed that scalar. Consequently, `S-broken` could invert its
positive-carrier coordinate permutation and tie `S` while representing a
different vector action.

Does the same minimal token controller recover the exact shared Spin chart
when the loss observes the complete transported association frame, rather than
a quotient of one read? This is an observability repair, not a new architecture
or a post-hoc retry of the failed gate.

## Frozen state law and arms

Each action composition is evaluated against a deterministic bank of four
independent random orthogonal frames

\[
M_0^{(1)},\ldots,M_0^{(4)}\in O(8).
\]

For the ordered action composition, let `V` be the vector carrier and `P` the
positive-spin carrier. The scored state is the actual two-sided fast-weight
transport law used by `SpinDiracFastWeightMemory`

\[
M_T^{(j)}=V_T M_0^{(j)} P_T^\top,
\qquad j=1,\ldots,4.
\]

One matrix alone retains a continuous coupled-carrier gauge. Four generic
probes remove that continuous tangent kernel: unlike the G15A-L scalar read,
neither carrier is removed by cosine normalization. A simultaneous discrete
central sign can remain and is an explicit nonclaim. Training uses raw
elementwise Frobenius MSE. Evaluation reports raw MSE and relative Frobenius
error; for orthogonal probes these differ only by a fixed norm factor, so they
are not counted as independent evidence.

| Arm | Transport |
|---|---|
| `I` | identity |
| `C` | fixed `SO(2)^4` |
| `S` | exact shared vector/positive Spin(8) lift |
| `S-broken` | the frozen non-automorphic positive-carrier coordinate permutation |

There is no readout arm because the complete memory frame is scored directly.
Every arm trains the same `17 x 28` token-to-coordinate table (476 stored
scalars, of which the identity token's 28 scalars are structurally masked).
The sixteen active tokens, seed-specific hidden permutation, coordinate bound
`0.25*tanh(raw)`, exact action magnitude `0.12`, and eight off-torus planes are
unchanged from G15A-L.

## Workload and execution

Training uses fresh ordered action compositions at logical length 16 with two
to six actions. The same seed-specific four-probe bank is repeated for every
composition and paired exactly across arms. Evaluation uses fresh paired
compositions:

| logical sequence length | actions per episode | examples |
|---:|---:|---:|
| 64 | 8 | 80 |
| 256 | 12 | 80 |
| 1,024 | 16 | 80 |

The quality run contains `80 x 4` transported matrices per length, seed, and
arm. A diagnostic-only singleton/inverse table also scores all sixteen signed
primitive tokens and the eight corresponding inverse pairs.

Only real action events are materialized. Identity filler is compiled exactly
to the scalar retention factor `0.999999^(L-1)`, as justified and tested for
G15A-L. No approximate kernel, fallback carrier, or unsupported GPU path is
used.

## Optimizer and budget

- Fresh seeds: `2203`, `2207`, and `2213`.
- FP32 on exact local compute capability 7.5.
- 300 updates, batch size 16, logical length 16.
- `ScalarSecondMomentAdamW`, learning rate `0.05`, no weight decay.
- Loss: mean raw elementwise squared error on `M_T`.
- Gradient clipping at norm 1.
- Evaluation microbatch eight.
- Controller initialization and paired schedules must byte-match across arms.

The optimizer is held fixed because G15A-L exposed an observation-law defect,
not evidence that its scalar second-moment optimizer failed. Changing both the
loss and optimizer would destroy attribution.

## Frozen adjudication

Before training any arm, the exact four-probe bank for each seed must pass:

1. The stacked independent-carrier tangent Jacobian, with shape `256 x 56`,
   has numerical rank 56 at tolerance `1e-10 * sigma_max`.
2. Its `sigma_min / sigma_max` is at least `1e-3`.
3. Every one of the eight target primitive tangents has relative projection
   residual at least `0.05` outside the complete `S-broken` tied-coordinate
   tangent image.
4. An exhaustive integer-scaled Lie-bracket check over all `28 x 28`
   generator pairs records a nonzero mismatch witness for `S-broken`.

Failure stops the cohort before optimization. For every seed and every
evaluation length separately:

1. `S` mean relative Frobenius error must be at most `0.05`, its 95th
   percentile at most `0.10`, and its maximum example error at most `0.20`.
2. Each of `I`, `C`, and `S-broken` must have mean relative Frobenius error at
   least `0.05` worse than `S`.
3. `S-broken` mean relative error must additionally be at least twice `S`.
4. Parameter count, zero initialization hash, paired probe-bank hash, and
   paired training-schedule hash
   must match exactly across all arms.
5. Every result and gradient must remain finite.

All checks must pass for all three seeds. There is no cross-seed averaging
rescue and no threshold change after metric inspection. If `S-broken` still
ties `S`, the shared chart remains unidentifiable even under full-frame
observation. If `S` itself misses the absolute gate, controller learning has
not been solved by observation repair.

## Claim boundary

A pass would establish that a minimal learned token controller can identify
the shared vector/positive Spin chart, up to the unresolved common discrete
center, from a multi-probe full-frame end loss and generalize to longer unseen
compositions under oracle edit timing. It would not establish
learned addressing, learned querying, generic association, language modeling,
long-context factual recall, parameter/compute scaling, fused efficiency, a
negative-spin/Clifford contribution, full triality, or a moving `G2/SU(3)`
frame. A failure remains valuable because it separates insufficient
observability from insufficient controller optimization or representation
identifiability.
