# Spin(8) triality algebra gate results

Date: 2026-08-03

The prospectively frozen algebra and recurrence gate passed in full. The
machine-readable record is `spin8_triality_algebra_gate.json`; the construction
and reference scan are implemented in `../spin8_triality.py`.

## Algebra result

The fixed octonion/Fano-plane construction produced eight real `16 x 16`
Clifford generators and 28 real `8 x 8` generators in each of the vector,
positive-chiral, and negative-chiral representations.

| Certificate | Result |
|---|---:|
| Octonion basis norm residual | `0` |
| Clifford anticommutator residual | `0` |
| Chirality square residual | `0` |
| Chiral multiplicities | `8 + 8` |
| Generator rank in `V`, `S+`, `S-` | `28, 28, 28` |
| Skew-symmetry residual | `0` |
| Complete `so(8)` commutator-table residual | `0` |
| Triality/Clifford equivariance residual | `0` |
| Random-exponential orthogonality maximum | `4.89e-15` |
| Random-exponential determinant error maximum | `1.11e-14` |
| `2 pi` center-action residual maximum | `2.59e-15` |
| Four center-signature residuals | `0` |

The important triality certificate is not a fitted matrix similarity. The
fixed Clifford multiplication maps satisfy exactly

```text
G-_ij rho(v) - rho(v) G+_ij = rho(J_ij v).
```

Thus one shared bivector controller genuinely drives the three representations
through their invariant trilinear relationship.

## Recurrent result

The three-representation recurrence stores `48` scalars for two channels,
independent of sequence length. In float64:

| Certificate | Result |
|---|---:|
| Affine associativity residual | `3.33e-16` |
| Arbitrary-chunk state/output residual | `0` |
| Token-stream state/output residual | `8.88e-16` |
| Log-depth reference-scan residual | `8.88e-16` |
| Zero-controller tangent gradient norm | `0.9526` |

This closes the earlier recurrent-state defect at the Spin(8) level as well as
the maintained Cl(3) level: inference carries a fixed cache and does not need
to recompute its history.

## Claim boundary

This is a constructive mechanism result, not an empirical superiority claim.
It establishes a correct real triality algebra, distinct center visibility,
associative affine composition, and constant-state streaming. Training and
baseline comparisons remain separate gates.
