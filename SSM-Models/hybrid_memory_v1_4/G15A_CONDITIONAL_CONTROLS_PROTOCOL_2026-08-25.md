# G15A conditional attribution-controls protocol

**Frozen:** 2026-08-25, after G15A passed and before either conditional arm
was executed or inspected

**Primary evidence bound by SHA-256:**
`6e2f5e58d8c411ce9c1a594e71197c76327145de00fc3af7023f884e788dd43a`
for
[`artifacts/g15a_spin_dirac_cohort_sm75_2026-08-25.json`](artifacts/g15a_spin_dirac_cohort_sm75_2026-08-25.json)

## Question

G15A established that full Spin(8) transport can solve the finite supplied-
coordinate action task while identity and one fixed commuting torus cannot.
That result does not yet attribute the gain to the fixed Clifford read or to a
shared triality lift. This protocol runs the two controls required by the
prospective G15 amendment without changing the passed primary cohort.

## Conditional arms

| Arm | Transport | Second read |
|---|---|---|
| `S+identity-read` | full factorized Spin(8) | identity copy of the positive read |
| `S-broken` | vector action from the supplied Spin coordinates; positive action from a frozen non-automorphic signed coordinate permutation | fixed Clifford map |

Both controls retain the same trainable tensors, 64-scalar recurrent state,
head-scalar equivariant edit law, bounded values, optimizer, token batches,
state dtype, update counts, and evaluation lengths as G15A. They are compared
only with the already completed `S` arm. The primary I/I+C/C/S cohort is not
rerun or re-adjudicated.

## Frozen execution and pairing

- Seeds are exactly `2131`, `2137`, and `2141`.
- The oracle-controlled symmetry calibration remains 100 AdamW updates at
  learning rate `0.05` with two scalar gains.
- Learned no-symmetry retrieval remains 300 `HarmonicMuonAdamW` updates at
  learning rate `0.003`, weight decay `0.01`, batch size 16, and length 64.
- Evaluation remains 80 balanced examples at lengths 64, 256, and 1,024.
- The core model/task/optimizer source hashes must still match the primary
  artifact. Conditional parameter tensors must byte-match a freshly
  regenerated S initialization under the unchanged deterministic seed path,
  and task schedule hashes must match primary S.
- Every conditional arm must have the same parameter count and parameter-shape
  hash as primary S.
- Quality execution must start at a clean commit on exact compute capability
  7.5 in FP32. Unsupported kernels or compatibility approximations are
  ineligible.

## Attribution rules

For every seed separately, using the same macro accuracy across lengths 64,
256, and 1,024:

1. A **Clifford-read contribution** is supported only if `S` exceeds
   `S+identity-read` by at least `0.02`.
2. A **shared-triality-coupling contribution** is supported only if `S`
   exceeds `S-broken` by at least `0.02`.
3. There is no mean-over-seeds rescue. Equality or a smaller margin is no
   evidence for the corresponding attribution.
4. The conditional arms' no-symmetry scores and inner-conjugation replay
   residuals are reported as diagnostics. They cannot rescue an attribution
   failure.

If `S` does not beat `S-broken` under this rule, the strongest permitted G15A
interpretation is that richer two-sided orthogonal transport solves the
designed task; no triality-specific learning claim is allowed. If `S` does not
beat `S+identity-read`, the fixed Clifford second read is not necessary for the
observed G15A separation.

## Nonclaims

Completion of these controls is still finite oracle-coordinate mechanism
evidence. It does not establish learned coordinate discovery, generic
association, natural-text quality, long-context recall, parameter/compute
scaling, fused efficiency, a moving `G2/SU(3)` frame, or a geometric Dirac
operator.
