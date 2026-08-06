# Q8 spinor joint-family retraction: seed-0 result

Date completed: 2026-08-03.

## Gate outcome

A deterministic, label-free post-training retraction converts the learned
seed-0 quaternion-spinor recurrence into an exact faithful Q8 mechanism while
preserving every behavioral gate.

The retraction couples all four token actions through one orthonormal generator
frame per channel. It does not normalize, round, search, or repair tokens
independently. The initial recurrent state and decoder are frozen, and there is
no post-retraction optimization.

| Metric | Learned | Jointly retracted |
|---|---:|---:|
| Whole-model linear homomorphism RMS | 0.6330 | `9.98e-8` |
| Whole-model orbit homomorphism RMS | 0.6330 | `9.98e-8` |
| Per-channel homomorphism RMS | 0.0035--1.266 | `9.33e-8`--`1.08e-7` |
| Inverse-pair antipodality RMS | 0.4212 | `4.37e-8` |
| Generator-square-to-`-1` RMS | 0.1910 | `5.81e-8` |
| Generator anticommutator RMS | not previously logged | `6.64e-8` |
| Dense central-pair member/joint floor | 100% / 100% | 100% / 100% |
| Long L4095--L16384 member/joint floor | 100% / 100% | 100% / 100% |

Per-channel action projection RMS is
`[0.00148, 0.49177, 0.00130, 0.00090]`. Thus the intervention barely moves
the three naturally faithful channels, makes the prospectively required large
repair to the nuisance channel, and still leaves the decoder exact. No
channel-specific selection was needed.

Central state separation remains essentially the full sign-flip distance and
streaming state parity remains exact.

## Interpretation

The training-and-retraction pipeline now cleanly separates two jobs:

1. unconstrained tangent optimization discovers the correct Q8 representation
   basin and a decoder that uses it;
2. one coupled algebraic projection removes residual relation error and repairs
   unused channel slack exactly.

This is stronger than behavioral extrapolation alone and stronger than an
oracle initialization: SGD discovers the nearby representation, while the
compiler enforces the exact shared family only after discovery.

## Boundary

This result completes the strict mechanism gate for seed 0. The unchanged
retraction must still pass prospectively over the full ten-seed learned cohort
before becoming a reliability result. It uses the known Q8 token/inverse-pair
contract and is not a table-blind group-discovery algorithm.

Artifacts:

- `q8_spinor_joint_retraction_smoke_seed0.json`
- `q8_spinor_joint_retraction_smoke_long_seed0.json`
- `q8_spinor_joint_retracted_smoke_checkpoints/`
- `Q8_SPINOR_JOINT_RETRACTION_PREREGISTRATION.md`
