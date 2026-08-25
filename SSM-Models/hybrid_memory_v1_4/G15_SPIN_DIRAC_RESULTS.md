# G15 Spin-Dirac status and results

**Updated:** 2026-08-25
**Adjudication:** G15A passes its frozen three-seed mechanism/observability
gate; conditional attribution controls are frozen and pending; `spin_dirac`
is not promoted to the default model

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
control was run.

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
| shared inner-conjugation covariance | pass at memory-law level | state, edit, positive read, and negative read agree in float64; a trained symmetry-task replay remains pending |
| contractive primary transition and bounded drive | pass | operator-norm and drive-norm tests |
| 4,096-step state falsifier | pass | SM75 FP32 maximum state/ceiling ratios are `7.74e-5` and `8.99e-5`; all state/read/norm outputs finite |
| full-LM gradients | pass | address, value, decay, erase, write, coordinates, output, and residual receive finite nonzero gradients |
| initialization survives shell initialization | pass | neutral coordinates and configured gate biases tested |
| optimizer partition | pass | complete, disjoint grouping; edit/transport controls use the scalar-moment group |
| eight-step optimizer covariance | pass | scalar-moment and SGD mapped parameter/update residuals are below `1.87e-13` in every seed, versus the `1e-10` gate |
| delayed scored-position observability | pass | coordinate and final-query paths exceed the read-change/loss-descent thresholds in all three seeds |
| `S+identity-read` conditional control | implemented and prospectively frozen | authorized by the G15A pass; run pending |
| `S-broken` conditional control | pass implementation contract | orthogonal actions retained while carrier coupling changes |

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

## Next executable gate

Run the two prospectively frozen conditional controls against the immutable
primary artifact:

| Arm | Transport | Second read | Attribution question |
|---|---|---|---|
| S+identity-read | full Spin(8) | identity copy | was the Clifford second read necessary? |
| S-broken | mismatched marginal orthogonal actions | Clifford | was the shared triality lift necessary? |

The runner is
[`g15a_conditional_controls.py`](g15a_conditional_controls.py). If S does not
beat S-broken by the frozen per-seed margin, the allowed interpretation narrows
to richer two-sided orthogonal transport rather than triality-specific
coupling. If S does not beat S+identity-read, the fixed Clifford second read is
not necessary for the observed separation.

The constrained `su3_torus` arm is an additional scientific ablation, not a
replacement for the four frozen primary arms.
