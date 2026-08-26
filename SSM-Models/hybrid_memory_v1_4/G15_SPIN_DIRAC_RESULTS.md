# G15 Spin-Dirac status and results

**Updated:** 2026-08-25
**Adjudication:** G15A and its frozen conditional controls pass. The shared
vector/positive Spin lift is supported on the designed task; a contribution
from the fixed Clifford/negative-spin read is not. G15A-L fails under a
center-blind cosine observation, G15A-F restores observability but misses its
precision gate, G15A-R repairs that precision, and G15A-S passes full
28-generator, held-out-frame, center-sensitive transfer. `spin_dirac` is not
promoted to the default model.

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

## Learning diagnostic and G15A-R freeze

The next move is not G15B or additional geometry. The bound
[`chart-error decomposition`](artifacts/g15af_learning_diagnostic_sm75_2026-08-25.json)
shows that inactive-coordinate leakage is the actual residual. Zeroing only
that leakage while preserving learned active amplitudes reduces mean error to
`0.0051--0.0183` and maximum error below `0.04` across all nine rows. Giving
the failed tables exact active amplitudes while keeping leakage changes almost
nothing. Active-axis MAE is only `0.00189--0.00324`; inactive RMS is
`0.00394--0.00780` and compounds through composition.

The prospectively frozen
[`G15A-R protocol`](G15AR_FIRST_ORDER_PROTOCOL_2026-08-25.md) now isolates
three first-order repairs rather than changing them together: learning-rate
decay, one rotation-covariant second moment per token row, and a balanced
singleton/inverse curriculum. Five paired S-only development arms use fresh
seeds and fixed validation traces. The least intervention clearing every
original absolute gate must then pass untouched four-transport confirmation
seeds. No primitive support is supplied to training.

## G15A-R learning result

The clean exact-SM75 run at commit `eca70f0` passed its complete frozen
development-selection-confirmation protocol. All five development recipes
qualified. The 600-step fixed-LR/random control reached mean error
`0.0155--0.0285`, so LR decay is **not proven necessary**; simply extending
the original stochastic composition training also crosses the absolute gate.

Among selectable arms, the predeclared least-intervention order chose
`G-decay/random`. It changes neither the dense controller, data distribution,
global scalar-second-moment optimizer, nor loss: only 600 updates and staged
LR `0.05 -> 0.01 -> 0.002`. Its development mean errors were
`1.03e-7--1.85e-7`. The block-scalar optimizer and primitive/inverse curriculum
also worked but were unnecessary.

Fresh confirmation then reinitialized the selected recipe on seeds 2251,
2267, and 2273:

| Metric over all seeds and L64/L256/L1,024 | Range |
|---|---:|
| S mean relative error | `1.21e-7--1.89e-7` |
| S p95 relative error | `1.53e-7--2.68e-7` |
| S maximum relative error | `2.00e-7--3.27e-7` |
| I minus S mean-error margin | `0.234--0.355` |
| C minus S mean-error margin | `0.231--0.351` |
| S-broken minus S mean-error margin | `0.153--0.234` |

Every absolute, comparator, broken-two-times, exact-pairing, and finite-value
gate passes in every row. This establishes composition-only learned primitive
coordinates for the shared vector/positive Spin lift under the four-probe
oracle-frame objective and oracle edit timing. It does not establish learned
addressing/querying, generic association, language performance, negative-spin
benefit, or full triality.

Evidence:
[`artifacts/g15ar_first_order_repair_sm75_2026-08-25.json`](artifacts/g15ar_first_order_repair_sm75_2026-08-25.json).

## G15A-S spanning and center-sensitive result

Before any learned metric, the
[`G15A-S protocol`](G15AS_SPANNING_CENTER_PROTOCOL_2026-08-25.md) froze a
57-by-28 table containing both directions of all 28 planes, hidden behind a
seed-specific token permutation. Training retained composition-only final
four-probe frame loss and oracle edit timing. It used 64 training probe banks;
all random evaluation used a disjoint pool of 64 banks. Direct vector and
positive-carrier matrices were evaluated only on unseen structured words,
because the two-sided frame alone is blind to the central element whose two
carrier signs are both negative.

The clean exact-SM75 run from commit `4067926` passed every frozen gate on
fresh seeds 2281, 2287, and 2293:

| Metric over all seeds and L64/L256/L1,024 | Range or worst case |
|---|---:|
| S mean held-out-frame error | `6.89e-7--1.18e-6` |
| S p95 held-out-frame error | at most `1.47e-6` |
| S maximum held-out-frame error | at most `1.94e-6` |
| minimum I/C/S-broken mean-error margin | `0.266` |
| learned active-coordinate maximum error | `7.15e-7` |
| learned inactive-coordinate RMS | `2.10e-8` |
| maximum structured direct-vector error | `2.34e-5` |
| maximum structured direct-positive error | `2.36e-5` |

All 384 training/evaluation banks had independent-carrier tangent rank 56.
The minimum condition ratio was `0.196`, the minimum target residual outside
the broken tied-chart tangent image was `0.506`, and the FP64 oracle center
residual was `5.11e-15`. The structured set covered signed 2-pi and 4-pi loops
in all planes, volume and other-center-coset words, and loop-plus-primitive
continuations. The required frame-blind/direct-carrier witness was observed
for both signed `minus_volume` words.

This establishes a composition-only learned dictionary spanning the full
28-generator shared vector/positive lift, compatible with unseen probe frames
and the hard-coded global Spin center under oracle edit timing. It does not
show discovery of Spin topology from raw data, learned addressing/write/query
behavior, negative-spin or Clifford benefit, generic association, language
quality, scaling, or fused efficiency.

Evidence:
[`artifacts/g15as_spanning_center_sm75_2026-08-25.json`](artifacts/g15as_spanning_center_sm75_2026-08-25.json),
SHA-256 `96e939fa4411e305637961941a565ac26da5a4212b47de3fc198687693b5dbcc`.

The prospectively frozen G15B cohort has now completed on exact SM75 hardware.
It failed the binding gate despite learning near-perfect address classes and
causally using its recurrent matrix. Three-seed mean identity accuracy stayed
near `0.972--0.977` on MQAR/selective-copy and `0.768--0.833` on overwrite,
but overwrite-erase recall was only about `0.506`; no arm/seed passed the
absolute controller gate. Full Spin trailed identity in all nine non-needle
task/length cells, by as much as 17.21 percentage points in a paired seed.

The central repair is architectural, not another optimizer sweep. The frozen
erase label asks whether the present write key appeared arbitrarily earlier,
while the one-block controller sees only the current embedding and a width-4
local convolution. That history bit is not observable. Last-write-wins can
instead erase the addressed key on every valid write, which is harmless for an
empty first-write address and necessary for an overwrite. An independent
collision-aware erase controller would require a causal pre-write occupancy
read. See
[`G15B_INTERLEAVED_CONTROLLER_RESULTS.md`](G15B_INTERLEAVED_CONTROLLER_RESULTS.md).

The subsequent G15B-R0 retained-checkpoint screen rejects the naive tied form
of that repair. Learned heads use a structured one-token write continuation,
so erase-equals-write makes that continuation destructive and exact atomic
timing removes useful learned code. G15B-R1 then keeps that write continuation
bitwise intact and anchors independent erase to every locally observable write
event. It also fails all nine non-needle gates, including a 9.7--11.5 point
overwrite loss. Learned key prototypes have mean absolute off-diagonal cosine
`0.54822` and maximum `0.999934`, making collateral symmetric erase plausible
without proving sole cause. Event-erase training is rejected. G15B-R2 then
keeps the write tail and restricts erase to perfect oracle collision timing.
It improves pre-overwrite recall but makes post-same-key-overwrite recall
10.3--12.1 points worse. G15B-R3 subsequently supplies an oracle per-key
replaceable component: ordinary overwrite gains 12.2--12.8 points and its
constructed guard reaches 1.0, but the frozen saturated-baseline improvement
and FP32 replay-tolerance gates fail. Trainable slot promotion remains blocked
pending an ownership/coupling factorial. That prospective G15B-R4 factorial
is now complete: both value-plus-tail arms pass and both value-only arms fail,
so the frozen decision is still do not train. Background exclusion leaves the
tail-owned arm essentially unchanged, localizing the useful association to a
two-token write continuation rather than to shared background support. See
[`G15BR_CHECKPOINT_REPAIR_RESULTS.md`](G15BR_CHECKPOINT_REPAIR_RESULTS.md) and
[`G15BR1_EVENT_ERASE_RESULTS.md`](G15BR1_EVENT_ERASE_RESULTS.md) and
[`G15BR2_COLLISION_ERASE_RESULTS.md`](G15BR2_COLLISION_ERASE_RESULTS.md) and
[`G15BR3_LOGICAL_COMPONENT_RESULTS.md`](G15BR3_LOGICAL_COMPONENT_RESULTS.md)
and
[`G15BR4_OWNERSHIP_BACKGROUND_RESULTS.md`](G15BR4_OWNERSHIP_BACKGROUND_RESULTS.md).

G15B-R5 then isolates the ambiguous tail's strict-history, current-token, and
bias sources under the unchanged full-token transition. Its background-free
strict-history LWW arm passes all 132 performance and bias-separation checks,
reaching 0.9424--0.9466 mean ordinary overwrite and 1.0 on the constructed
guard. Current-only and bias-only arms fail. R5's formal decision nevertheless
remains failed under its frozen replay/runtime tolerances.

The prospectively frozen R5-S numerical cohort has also completed. It fails all
135 scaled-logit source/cell checks, with allowance ratios
1.171875--66.078125. Predictions, categorical metrics, BPQ, state/read bounds,
bit-exact transitions, fingerprints, source/checkpoint hashes, and independent
FP64 algebra all pass. This closes retained-checkpoint repair without erasing
R5's separate history-source evidence. The next fresh architecture must learn
explicit pending-write/commit semantics and preserve an exact monolithic
residual/read path before any identity/torus/Spin transport comparison. See
[`G15BR5_CAUSAL_TAIL_SOURCE_RESULTS.md`](G15BR5_CAUSAL_TAIL_SOURCE_RESULTS.md)
and
[`G15BR5S_NUMERICAL_RATIFICATION_RESULTS.md`](G15BR5S_NUMERICAL_RATIFICATION_RESULTS.md).

The resulting fresh pivot is G15B-T, not another retained-checkpoint repair.
Its
[`Phase-0 exact-SM75 qualification`](G15BT_PHASE0_QUALIFICATION_RESULTS.md)
passes at clean commit `86372b8`: matched `F/T` arms each contain 38,082 active
parameters and 5,632 batch-2 FP32 state bytes; current-token mutation has zero
effect on the strict-history edit path; the prior-history effect is nonzero;
the maximum transition spectral norm is `0.9995000000000012`; FP64 and FP32
maximum logit residuals are `2.78e-16` and `8.94e-8`; predictions are exact;
and all declared gradients are finite/nonzero. This authorizes only the
prospective Phase-1 constructed screen. It is not learned-memory evidence and
does not revise R5 or R5-S.

The sealed Phase-1 quality cohort has now completed from clean commit
`0c664f3` on exact SM75. Its nine `F/T/T-AUX` reports contain 30,600 updates
and 125,337,600 tokens. Primary `T` fails: it trails `F` on mean overwrite at
every length, seed 2381 collapses to `0.8071--0.8213`, and completed-tail
commit F1 fails every seed/task/length. Diagnostic-only `T-AUX` improves mean
overwrite to `0.9264--0.9308` and mostly learns commit timing, but it never
clears the frozen `+0.05` margin and seed 2381 remains at about 0.8889 commit
F1. Both arms formally fail. The frozen decision stops G15B-T before any
identity/torus/Spin comparison. See
[`G15BT_PHASE1_RESULTS.md`](G15BT_PHASE1_RESULTS.md) and the
[`exact quality artifact`](artifacts/g15bt_phase1_quality_sm75_2026-08-26.json).

The non-geometric successor is the prospectively frozen
[`G15B-E effective-edit test`](G15BE_EFFECTIVE_EDIT_PROTOCOL_2026-08-26.md).
Its exact-SM75
[`Phase-0 qualification`](G15BE_PHASE0_QUALIFICATION_RESULTS.md) passes for
matched full-view product and logit-additive laws at 22,161 parameters and
1,408 FP32 state bytes. The completed exact-SM75 Phase-1 quality cohort then
fails both trained arms. Across 20,400 updates and 83,558,400 tokens, additive
`A` never reaches the frozen `+0.02` overwrite margin over product `P`, misses
absolute all-seed quality, and fails valid-event-only causal sufficiency for
two seeds. All trained numerical boundaries pass with exact predictions. The
frozen decision recommends only a separately frozen residual-delta write-law
test; geometry remains blocked. See the
[`Phase-1 result`](G15BE_PHASE1_RESULTS.md) and
[`quality artifact`](artifacts/g15be_phase1_quality_sm75_2026-08-26.json).

The resulting prospective non-geometric successor is
[`G15B-D coupled residual delta`](G15BD_RESIDUAL_DELTA_PROTOCOL_2026-08-26.md).
It couples erase and write into one channelwise residual correction while
holding the G15B-E control, projections, optimizer, state, data, and future
budget fixed. Its exact clean-SM75
[`Phase-0 result`](G15BD_PHASE0_QUALIFICATION_RESULTS.md) passes from clean
commit `549e6d98d0bebc35fad32daa498486fd075aa906`: `P/D` are matched at 22,161
active parameters and 1,408 FP32 state bytes, direct transition/injection and
coupling residuals are zero, predictions are exact, and all declared
gradients are finite and nonzero. See the
[`artifact`](artifacts/g15bd_phase0_qualification_sm75_2026-08-26.json),
SHA-256
`44a8556b60db7cb8c5e1edc239255dc510b62f03960e4514a49b645e79123921`.
This authorizes only prospective Phase-1 constructed training. No learning
result exists, and geometry remains blocked.

The constrained `su3_torus` arm is an additional scientific ablation, not a
replacement for the four frozen primary arms.

In the parallel completed G16 one-seed SM75 development shootout, official
fused Mamba-2 beat v1.4.5 at every context and by `0.09764` BPRB at L4096.
Local GDN2 and OLMo Hybrid lost; every arm failed the learned-recall gate. This
sets a stronger ordinary-compression reference but does not promote a model
family. The failed G15B cohort blocks G15C and the external-loss-only lane;
G15A-S remains valid, separate oracle-timing geometry evidence. See
[`G16_SM75_FRONTIER_SHOOTOUT_RESULTS.md`](G16_SM75_FRONTIER_SHOOTOUT_RESULTS.md).
