# Double-cover Euclidean motor path-development result

**Protocol frozen:** 2026-08-16T19:41:11+02:00
**Authoritative corrected run:** 2026-08-16T19:43:29+02:00 to
2026-08-16T19:44:30+02:00
**Artifact:**
[`motor_path_development_20260816.json`](artifacts/motor_path_development_20260816.json)
**SHA-256:**
`3496e374ddb68d48f3105b41ac39c23a40be3edaa872f76dcf6ce9e06ea8f95a`

## Outcome

The experimental eight-scalar unit-dual-quaternion scan passes every frozen
algebraic/numerical gate through length 4096. It is a correct sign-sensitive
double-cover implementation of orientation-preserving rigid-motion path
composition under the stated float64 tolerances.

This closes an implementation gate, not a learned-model or novelty claim.
Finite-dimensional Lie-group path development and dual-quaternion neural
representations already exist in the literature. The local contribution is a
tested parallel/recurrent/cache-compatible bridge from the repository's
successful Spin(3) prefix state to `Spin(3) semidirect R^3`, with explicit
central-sign retention.

## Longest-length evidence

| Length-4096 check | Maximum error/value |
|---|---:|
| Parallel versus recurrent motor state | `3.62e-14` |
| Chunked versus full motor state | `1.25e-14` |
| Motor-derived versus recurrent homogeneous matrix | `5.82e-14` |
| Real quaternion norm error | `2.22e-16` |
| Study condition `abs(dot(q_r,q_d))` | `6.66e-16` |
| Rotation orthogonality error | `1.78e-15` |
| Rotation determinant error | `2.00e-15` |
| Central-negated state plus original state | exactly `0` |
| Central-negated physical matrix minus original | exactly `0` |
| Maximum accumulated absolute translation | `7.58` |

All values are finite. The translation result is deliberately nontrivial and
unbounded: `SE(3)` is noncompact, so this state does not inherit Pure Rotor
v2.1's affine-memory norm bound.

## What is unified

The implementation now has one exact computational chain:

```text
Spin(3) quaternion scan
        subset: translation = 0
                |
                v
unit dual quaternion motor scan --physical quotient--> SE(3) matrix action
       sign-sensitive state                         sign-blind action
```

- zero translation reproduces the existing quaternion Spin scan within tested
  tolerance;
- motor multiplication matches homogeneous-matrix multiplication;
- a motor and its negative are different recurrent states but induce the same
  physical transformation;
- both recurrent inference and parallel training use the same group product;
- padding is the group identity and chunked inference uses an eight-scalar
  cache per lane.

This is the honest unification supported by code. It does not yet include
conformal inversions/dilations, `SL(2,R)`, arbitrary Clifford groups, or
nonassociative octonion multiplication.

## Local systems diagnostic

Measured on the NVIDIA GeForce RTX 2070 SUPER at length 1024, batch 8, four
lanes, after five warmups and over 20 repeats:

| Eager operation | Mean milliseconds |
|---|---:|
| Motor parallel | 24.13 |
| Motor recurrent | 2,207.47 |
| 4 by 4 matrix parallel | 2.82 |
| 4 by 4 matrix recurrent | 90.50 |

The parallel tree gives the motor implementation about a 91-fold advantage
over its Python recurrent loop. However, the current eager matrix parallel
path is about **8.6-fold faster** than the factorized motor path despite using
twice as many physical coordinates. PyTorch's batched matrix multiplication is
highly optimized; the motor code expands into many small elementwise kernels.

This is a negative result for immediate systems superiority and a clear kernel
target. Neither path is fused, so it is not a production comparison.

## Corrections and provenance

The first generated JSON was rejected before citation because `finished_at`
was evaluated before CUDA timing. The code was corrected so the timestamp is
recorded after all measurements, and the exact frozen command was replayed.
Only the corrected artifact/hash above is authoritative.

The artifact hash and all-gates-pass contract are locked by
[`test_motor_path_development_audit.py`](../test_motor_path_development_audit.py).
Together with [`test_motor_scan.py`](../test_motor_scan.py), 12 tests cover
algebra, homogeneous action, Study normalization, Spin-subgroup reduction,
central sign, tree/recurrence gradients, streaming cache, padding, long scans,
CUDA backward, and the frozen artifact.

## Relationship to current research

- The [Path Development Network](https://arxiv.org/abs/2204.00740) already
  establishes learned products of finite-dimensional Lie-group exponentials as
  sequence features.
- [Dual Quaternion Rotational and Translational Equivariance](https://arxiv.org/abs/2310.07623)
  gives empirical evidence that joint rotation/translation algebra can improve
  physical sequence models.
- [Fixed-Point RNNs](https://arxiv.org/abs/2503.10799) are a necessary modern
  learned state-tracking comparator because they report strong `A5`/`S5`
  results via implicitly dense recurrence.
- The 2026 Lie-embedded dynamics work uses adjoint/Lie-algebra structure for
  stable `SE(3)` dynamics, but solves a different continuous-time equilibrium
  problem: [LieEDNN](https://arxiv.org/abs/2605.26167).

## Limits and next falsifier

- There is no learned rigid-motion result yet.
- Correct representation does not imply sample efficiency or task advantage.
- Translation can drift without bound.
- The timing path is unfused and currently slower than dense matrices.
- Central-sign information is useful only when the downstream target sees the
  double cover rather than only physical pose.
- AHSS, norm-residue theory, and admissible infinite-dimensional
  representations do not supply a tested bridge here.
- Raw octonion multiplication is nonassociative and cannot replace this scan
  operation without an associative operator lift.

The next preregistered benchmark should combine the three withheld binary
relations with nonzero local-frame translations. It must score signed rotation,
physical rotation, translation, long retention, and exact center margin, and
compare equal-state motor/quaternion scans plus Mamba-2, DeltaProduct, and a
faithful Fixed-Point RNN reference. No real-sequence or conformal claim should
precede that falsifier.
