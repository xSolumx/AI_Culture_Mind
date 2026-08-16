# Haar-Basis Octonion Operator Replication Results

Protocol frozen: **2026-08-16T21:11:30+02:00**
Execution: **2026-08-16T21:16:23--21:17:32+02:00**
Post-protocol audit: **2026-08-16T21:19:15+02:00**

## Result in one sentence

A 28-parameter `SO(8)` gauge learned three independently Haar-transported
octonion multiplication laws from identity initialization and extrapolated
from length 16 to length 128, with the recovered basis differing from ground
truth only by an approximately `G2` octonion automorphism; however, the frozen
cohort **did not pass every preregistered gate**, and direct least squares shows
that the 512-parameter dense operator control is also realizable.

## Frozen task

For each basis seed, a hidden deterministic Haar matrix `Q in SO(8)` transports
both token and state coordinates:

`x_t = Q u_t`,

`Y_t = Q L_(u_t) ... L_(u_1) Q^T`.

All candidates train on every prefix at length 16 for 300 AdamW updates and
evaluate on fresh fixed sequences through length 128. Complete operator
supervision makes this a 64-scalar recurrent-state task.

## Main three-basis result

L128 MSE:

| Candidate | Parameters | state scalars | basis 0 | basis 1 | basis 2 |
|---|---:|---:|---:|---:|---:|
| Exact transported oracle | 0 | 64 | `1.07e-12` | `4.51e-13` | `3.47e-13` |
| **Learned `SO(8)` basis + octonion scan** | **28** | **64** | **`6.77e-8`** | **`1.54e-9`** | **`8.74e-8`** |
| Dense linear operator scan | 512 | 64 | `0.12555` | `0.12425` | `0.12557` |
| DeltaProduct reference | 13,192 | 256 | `0.12451` | `0.12439` | `0.12443` |
| Transformers Mamba-2 | 12,300 | 1,408 | `0.12565` | `0.12498` | `0.12455` |
| Transported collapsed octonion | 0 | 8 | `0.21604` | `0.21570` | `0.21572` |
| Fixed canonical operator | 0 | 64 | `0.24961` | `0.24900` | `0.24958` |

The learned basis passes its registered `1e-3` L128 gate and beats the fixed
canonical control in all three bases. Its final training losses are
`1.04e-8`, `2.12e-10`, and `1.69e-8`. It begins at the fixed-canonical
function; the hidden Haar matrices are never supplied to the model.

Mamba-2 and DeltaProduct remain around the zero-predictor scale on this budget.
Those paths are unfused architecture controls and the task is deliberately
matched to operator composition. Their result is neither a systems comparison
nor evidence of generic octonion-model superiority.

## The preregistered failures

The top-level artifact says `all_required_checks_passed: false`. This is
intentional and must not be rewritten:

1. The gradient-trained dense linear operator misses its `1e-3` gate in all
   three bases, ending near MSE `0.125`. It still beats the fixed-canonical
   control, but does not identify the transported leaf map in 300 AdamW steps.
2. The float32 exact oracle at basis 0 records L128 MSE `1.0749e-12`, narrowly
   above the frozen strict `1e-12` threshold. Bases 1 and 2 pass. The maximum
   basis-0 absolute discrepancy is `6.58e-6`; this is accumulated float32 QR,
   conjugation, and scan reassociation, not a mathematical counterexample.

No gate or threshold was changed after seeing these values.

## Post-protocol direct-identification control

Every prefix is supervised, so position 1 legally exposes each transported
leaf operator. A post-protocol float64 least-squares diagnostic fits the dense
linear map from the 9,600 training pairs `(x_1,Y_1)` and then scans the
identified leaves without any held-out target access.

| Basis | design rank | condition number | leaf train MSE | L128 MSE |
|---:|---:|---:|---:|---:|
| 0 | 8 | 1.050 | `1.31e-15` | `2.81e-12` |
| 1 | 8 | 1.045 | `1.29e-15` | `9.37e-12` |
| 2 | 8 | 1.040 | `1.33e-15` | `3.76e-13` |

Therefore the dense model's frozen failure is an **optimization result, not an
expressivity result**. The 28-parameter structured model's valid advantage is
that blind gradient training finds the law within the fixed budget while the
unrestricted 512-parameter map does not. Direct supervision can identify both.

## Why the recovered ambiguity is `G2`

Let `Q_hat` be the learned basis and define

`A = Q_hat^T Q`.

Exact equality of the one-step transported actions implies

`A L_u A^T = L_(A u)`.

Together with `A e_0=e_0`, this is the left-action form of preserving octonion
multiplication: `A(uv)=(Au)(Av)`. The stabilizer is the octonion automorphism
group `G2`. The audit evaluates the identity and intertwiner equations on the
complete canonical basis:

| Basis | `max |A e0-e0|` | `max |A L_ei A^T-L_(Aei)|` | orthogonality residual |
|---:|---:|---:|---:|
| 0 | `9.44e-5` | `1.28e-4` | `1.26e-6` |
| 1 | `1.18e-5` | `2.52e-5` | `3.71e-7` |
| 2 | `8.63e-5` | `2.17e-4` | `4.51e-7` |

This explains why the learned-to-true basis trace need not approach eight:
the data identify the transported algebra only up to its `G2` gauge symmetry.
This is an empirical recovery certificate for the implemented model, not a
claim that the classical `G2` stabilizer theorem is new.

## Artifacts and replay

Frozen cohort artifact:
`experiments/artifacts/octonion_basis_transport_replication300.json`
SHA-256:
`b96bed5d0e4c33e229816f6ce2db24d2c42c5b33ae1b978950a2a0bd9960daf5`

Post-protocol audit artifact:
`experiments/artifacts/octonion_basis_identification_audit.json`
SHA-256:
`bb149467ebc4cfe32b8ded8311d6a0b5ec47a7b606d13973d371ee7ae955d5c6`

All basis, training, and evaluation schedule hashes replay. All twelve learned
checkpoints independently rehash and reload. The post-protocol audit passes all
21 per-basis checks; it does not retroactively turn the frozen cohort's failed
gates into passes.

Protocol:
[`OCTONION_BASIS_TRANSPORT_PROTOCOL.md`](OCTONION_BASIS_TRANSPORT_PROTOCOL.md)
Runner:
[`benchmark_octonion_basis_transport.py`](../benchmark_octonion_basis_transport.py)
Audit:
[`audit_octonion_basis_identification.py`](../audit_octonion_basis_identification.py)

## Claim ledger

Established empirically:

- the 28-parameter transported-algebra model crosses its frozen accuracy gate
  in all three Haar bases;
- the invalid eight-scalar collapse remains falsified in all three bases;
- the learned ambiguity satisfies the `G2` automorphism equations to the
  reported numerical tolerances;
- the state-matched dense linear leaf map is exactly identifiable from legal
  first-prefix supervision; and
- the frozen dense AdamW path fails to find that solution in 300 updates.

Not established:

- all frozen cohort gates passing;
- generic superiority over dense operators, Mamba-2, or DeltaProduct;
- a natural-sequence, language, vision, or control benefit;
- recovery under final-only or unsigned supervision;
- recovery of a time-varying basis;
- a production fused comparison; or
- any unrestricted Dirac--Gram/global D-optimality theorem.

## Next falsifier

Remove the direct local-identification shortcut. Freeze a final-only task in
which the model receives a held-out Haar basis and only terminal acted-on-vector
or operator supervision. Compare the 28-parameter gauge model with:

1. the dense operator scan under separately tuned but preregistered optimizer
   budgets;
2. a generic `SO(8)` leaf map with the same 64-scalar state;
3. Mamba-2 and DeltaProduct at matched parameter and measured-compute points;
4. multiple random initializations per basis; and
5. noisy observations that perturb tokens off the unit sphere.

That experiment would test whether the `G2`-quotiented prior helps infer a
latent composition law when the answer is no longer exposed at position 1.
