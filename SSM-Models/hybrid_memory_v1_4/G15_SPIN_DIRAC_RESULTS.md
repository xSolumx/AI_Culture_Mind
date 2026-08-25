# G15 Spin-Dirac status and results

**Updated:** 2026-08-25
**Adjudication:** the pre-training implementation integrity artifact passes;
no G15 learning cohort has run; `spin_dirac` is not promoted

This file is intentionally named as the G15 result ledger even while the
result is "not run." It prevents passing algebraic tests from being mistaken
for a trained-model outcome. The binding protocol is the
[`preregistration`](G15_SPIN_DIRAC_PREREGISTRATION.md), strengthened by the
prospective [`amendment`](G15_SPIN_DIRAC_AMENDMENT_2026-08-25.md).
The later prospective
[`edit-law amendment`](G15_SPIN_DIRAC_EDIT_LAW_AMENDMENT_2026-08-25.md)
repairs the exposed fixed-basis gating defect before training.

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
| `S+identity-read` conditional control | implemented by configuration | required only if S passes G15A |
| `S-broken` conditional control | pass implementation contract | orthogonal actions retained while carrier coupling changes |

The binding machine evidence is
[`artifacts/g15_integrity_sm75_2026-08-25.json`](artifacts/g15_integrity_sm75_2026-08-25.json).
This clears the pre-training implementation gate. It is not a learned
mechanism result.

## Learning result

No G15A, G15B, G15C, or G15D training result exists. Therefore:

- there is no evidence that full Spin transport beats identity, the fixed
  torus, or the constrained $SU(3)$ torus;
- there is no triality-specific result against the broken-coupling control;
- there is no generic associative-memory promotion;
- there is no multi-seed natural-text, long-recall, or scaling result; and
- v1.4.5's default `gated_delta -> attention` layer plan remains unchanged.

## Next executable gate

Run the mandatory four-arm G15A comparison now that the deterministic
integrity artifact has passed:

| Arm | Transport | Second read |
|---|---|---|
| I | identity | identity copy |
| I+C | identity | Clifford |
| C | fixed `SO(2)^4` | Clifford |
| S | full Spin(8) | Clifford |

The constrained `su3_torus` arm is an additional scientific ablation, not a
replacement for the four frozen primary arms.
