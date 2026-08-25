# G15 Spin-Dirac status and results

**Updated:** 2026-08-25
**Adjudication:** G15A and its frozen conditional controls pass. The shared
vector/positive Spin lift is supported on the designed task; a contribution
from the fixed Clifford/negative-spin read is not. G15A-L then fails to recover
that attribution with a learned coordinate chart. `spin_dirac` is not promoted
to the default model.

This file separates passing algebraic contracts, oracle-controlled mechanism
evidence, and learned-model evidence. The binding protocol is the
[`preregistration`](G15_SPIN_DIRAC_PREREGISTRATION.md), strengthened by the
prospective [`amendment`](G15_SPIN_DIRAC_AMENDMENT_2026-08-25.md).
The later prospective
[`edit-law amendment`](G15_SPIN_DIRAC_EDIT_LAW_AMENDMENT_2026-08-25.md)
repairs the exposed fixed-basis gating defect before training.
The prospective
[`G15A execution protocol`](G15A_EXECUTION_PROTOCOL_2026-08-25.md) now freezes
the previously missing seeds, task support, FP32 budget, aggregation semantics,
optimizer, artifact contract, and retry policy. It was committed before any
G15A runner output was inspected. The later
[`conditional-controls protocol`](G15A_CONDITIONAL_CONTROLS_PROTOCOL_2026-08-25.md)
was likewise frozen after the primary pass but before either attribution
control was run. Both controls are now complete.

## Implementation outcome

The implemented state is an $8\times8$ association matrix per head. It uses
content-addressed outer-product writes, independent erase/write controls,
two-sided vector/positive-half-spin transport, and an optional fixed Clifford
read into the negative-half-spin carrier. It is a 64-state LTV SSM per head,
not the old 24-scalar transported cache and not a geometric Dirac operator.

The primary gate law is now `equivariant_scalar`. This prospective repair was
made before G15 training because channelwise diagonal/Hadamard gating fails
the required shared-frame covariance. `channelwise` is retained only as a
named non-equivariant ablation.

## Integrity ledger

| Contract | Status | Evidence or remaining work |
|---|---|---|
| recurrent versus two-sided parallel scan | pass | float64 equality in `tests/test_spin_dirac_memory.py` |
| arbitrary chunks and token stepping | pass | full-shell replay in `tests/test_model.py` |
| masked matrix and convolution caches | pass | direct memory and full-shell mask tests |
| Clifford equivariance | pass | float64 shared-carrier test |
| all four center signatures | pass | direct backend residual at or below `1e-10` |
| exact $SU(3)\;T^2$ slice | pass | zero total phase, fixed fourth lane, inactive-coordinate checks |
| shared inner-conjugation covariance | pass | state, edit, positive read, negative read, and trained primary-arm replays agree in float64 |
| contractive primary transition and bounded drive | pass | operator-norm and drive-norm tests |
| 4,096-step state falsifier | pass | SM75 FP32 maximum state/ceiling ratios are `7.74e-5` and `8.99e-5`; all state/read/norm outputs finite |
| full-LM gradients | pass | address, value, decay, erase, write, coordinates, output, and residual receive finite nonzero gradients |
| initialization survives shell initialization | pass | neutral coordinates and configured gate biases tested |
| optimizer partition | pass | complete, disjoint grouping; edit/transport controls use the scalar-moment group |
| eight-step optimizer covariance | pass | scalar-moment and SGD mapped parameter/update residuals are below `1.87e-13` in every seed, versus the `1e-10` gate |
| delayed scored-position observability | pass | coordinate and final-query paths exceed the read-change/loss-descent thresholds in all three seeds |
| `S+identity-read` conditional control | complete | ties S at 1.00 in every seed; fixed Clifford second read is not necessary |
| `S-broken` conditional control | complete | scores 0.30/0.20/0.20 versus S at 1.00; shared-coupling margin passes |

The binding machine evidence is
[`artifacts/g15_integrity_sm75_2026-08-25.json`](artifacts/g15_integrity_sm75_2026-08-25.json).
This clears the pre-training implementation gate. It is not a learned
mechanism result.

## G15A learning result

The exact quality cohort started from clean commit `73df687f`, ran in FP32 on
the RTX 2070 SUPER at compute capability 7.5, and passed every frozen condition
in every seed (`2131`, `2137`, and `2141`).

| Metric, in every seed | I | I+C | C | S |
|---|---:|---:|---:|---:|
| supplied-coordinate symmetry macro accuracy | 0.10 | 0.10 | 0.20 | **1.00** |
| learned no-symmetry macro accuracy, L64/256/1,024 | 1.00 | 1.00 | 1.00 | 1.00 |
| trainable parameters | 11,508 | 11,508 | 11,508 | 11,508 |
| recurrent state bytes per FP32 sequence | 256 | 256 | 256 | 256 |

Thus S's per-seed symmetry margins are approximately `+0.90` over I and I+C
and `+0.80` over C, far above the frozen `+0.02` gate, with no learned
no-symmetry regression. Parameter-shape and training-schedule hashes match
exactly across arms. Trained-calibrator inner-conjugation residuals are at most
`5.45e-15`, and the float64 one-hot/overwrite/collision/orthogonal-query ladder
passes with maximum residual `1.12e-16`.

The learned no-symmetry model used the frozen `HarmonicMuonAdamW` optimizer and
reached 100% on the finite eight-class delayed-value support through length
1,024 for all arms. That shows compatibility and basic controller learnability
on this bounded task; it does not show that this optimizer is generally better
than AdamW or that Spin is necessary there.

Evidence:
[`artifacts/g15a_spin_dirac_cohort_sm75_2026-08-25.json`](artifacts/g15a_spin_dirac_cohort_sm75_2026-08-25.json).

This is a positive G15A result, but its symmetry side supplies exact
coordinates and oracle carrier controls. It does not establish learned
coordinate discovery, generic associative memory, ordinary natural text,
long-context recall, scaling, or fused efficiency. G15B, G15C, and G15D have
not run, and v1.4.5's default `gated_delta -> attention` plan is unchanged.

## Conditional attribution result

The conditional run started clean at commit `5fc3d7b`, verified the immutable
primary artifact and unchanged core sources, and completed in FP32 on exact
SM75. It produced the same outcome in every seed:

| Metric | seed 2131 | seed 2137 | seed 2141 |
|---|---:|---:|---:|
| S symmetry accuracy, immutable primary | 1.00 | 1.00 | 1.00 |
| S+identity-read symmetry accuracy | 1.00 | 1.00 | 1.00 |
| S-broken symmetry accuracy | 0.30 | 0.20 | 0.20 |
| S minus S-broken | **0.70** | **0.80** | **0.80** |
| both controls, no-symmetry through L1,024 | 1.00 | 1.00 | 1.00 |

The shared-coupling margin exceeds the frozen `0.02` threshold in all seeds.
The identity-read tie fails the Clifford-read contribution threshold in all
seeds. S-broken's large `0.071--0.078` inner-conjugation residual is reported
as the prospective protocol required and confirms that the named control
actually breaks shared-frame covariance; it was never a pass gate for that
intentionally broken arm.

The strongest supported wording is therefore **shared vector/positive Spin
coupling matters on this supplied-coordinate task**. Because the negative-
spin Clifford read was unnecessary, this is not evidence that all three
triality carriers are useful. The first run exposed a runner-only adjudication
bug that incorrectly promoted the broken arm's diagnostic residual into an
integrity gate. Commit `5fc3d7b` corrected the code to match the already frozen
protocol before the clean evidentiary rerun; no data, seed, margin, or training
setting changed.

Evidence:
[`artifacts/g15a_conditional_controls_sm75_2026-08-25.json`](artifacts/g15a_conditional_controls_sm75_2026-08-25.json).

## G15A-L learned-coordinate result

G15A-L trained only a 476-parameter token-to-coordinate table with the
rotation-covariant `ScalarSecondMomentAdamW`, while holding edit controls and
the final transported query oracle-fixed. The clean SM75 quality run at commit
`0c49f64` failed its frozen per-seed/per-length gate:

| Seed | S mean cosine L64 | L256 | L1,024 |
|---:|---:|---:|---:|
| 2153 | 0.9921 | 0.9878 | 0.9814 |
| 2161 | 0.9957 | 0.9913 | 0.9862 |
| 2179 | 0.9902 | 0.9803 | 0.9726 |

Only seed 2161 at L64 clears S's absolute mean/minimum thresholds. No row
clears the frozen `0.05` margin over all comparators. Most decisively,
S-broken matches S's mean cosine to at most about `6e-8` in every seed and
length, and S+identity-read is exactly equal to S.

The learned tables expose why. After applying S-broken's frozen signed
coordinate permutation, its effective positive-carrier chart matches S's
learned chart within `2.84e-7--3.94e-7`. Their final training losses also agree
to numerical precision. Under the scored observation,

\[
\hat y=r^{L-1}\langle q,Vk\rangle Pv,
\]

cosine normalization cancels the positive scalar containing the vector
carrier. The invertible broken chart can therefore learn the same positive
action. This is a structural observability/gauge failure, not evidence that
more training or a coordinatewise optimizer would recover the shared lift.

Evidence:
[`G15A-L protocol`](G15AL_LEARNED_COORDINATE_PROTOCOL_2026-08-25.md),
[`exact execution amendment`](G15AL_EXECUTION_AMENDMENT_2026-08-25.md), and
[`artifacts/g15al_learned_coordinate_cohort_sm75_2026-08-25.json`](artifacts/g15al_learned_coordinate_cohort_sm75_2026-08-25.json).
The bound
[`post-hoc observability diagnostic`](artifacts/g15al_observability_diagnostic_sm75_2026-08-25.json)
confirms that S and S-broken can have cosine at least `0.99999988` while their
raw predictions differ by as much as `0.284`.

## G15A-F full-frame outcome

The next falsifier changes the observation law, not the failed optimizer. The
prospectively frozen
[`G15A-F protocol`](G15AF_FULL_FRAME_PROTOCOL_2026-08-25.md) scores
`V M_j P^T` for four shared orthogonal probes per action composition with raw
Frobenius loss. One probe retains a continuous coupled-carrier gauge; the
four-probe bank must first pass a rank-56/conditioning certificate. The broken
arm must also exhibit an exhaustive integer-scaled Lie-bracket mismatch and
each primitive target tangent must lie outside its tied-coordinate image.
Only then may the unchanged 476-parameter controller train. This controlled
identifiability gate comes before G15B generic association or any additional
exceptional geometry.

The clean quality run at commit `503fa82` passed every pretraining structural
screen. All three four-probe Jacobians have rank 56, condition ratios
`0.286--0.310`, and minimum primitive projection residuals `0.598--0.611`
outside the broken tangent image. The exhaustive integer-scaled bracket check
finds 474 mismatches among 784 ordered generator pairs, with maximum residual
4. The control is therefore demonstrably non-automorphic and the objective is
locally observable before learning begins.

The frozen quality gate nevertheless **fails**:

| Seed | S mean relative error L64 | L256 | L1,024 |
|---:|---:|---:|---:|
| 2203 | 0.0927 | 0.1201 | 0.1304 |
| 2207 | 0.0912 | 0.1187 | 0.1433 |
| 2213 | 0.0705 | 0.1036 | 0.1392 |

Every row misses the `0.05` absolute mean-error threshold, and the frozen p95
gate fails throughout. However, S beats I, C, and S-broken by at least `0.05`
on every seed and length. Its margin over S-broken is `0.070--0.137`.
Therefore the observation repair succeeds at separating the shared lift, but
the controller does not recover the chart precisely enough. The frozen
requirement that broken error be at least twice S passes only seed 2213 at L64
and L256 and cannot rescue promotion.

Evidence:
[`artifacts/g15af_full_frame_cohort_sm75_2026-08-25.json`](artifacts/g15af_full_frame_cohort_sm75_2026-08-25.json).

## Next learning diagnostic

The next move is not G15B or additional geometry. The retained S tables are
already much closer to the exact chart than S-broken, and singleton errors are
only `0.0166--0.0319`, while long compositions amplify residual coordinate
leakage. A bound post-hoc support/amplitude ablation should first measure how
much error comes from inactive-coordinate leakage versus active-angle bias.
Only then should a new prospective cohort freeze a balanced primitive/inverse
curriculum, a per-token block-scalar second moment, and learning-rate decay.
Changing all three without that diagnostic would forfeit attribution.

The constrained `su3_torus` arm is an additional scientific ablation, not a
replacement for the four frozen primary arms.
