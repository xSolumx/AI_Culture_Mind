# Programme 06: Rotor and noncommutative state-space models

## Scope

Bounded recurrent models whose persistent state is transported by
input-selective noncommuting actions. The stable Pure Rotor v2.1 family uses
`Cl(3,0)`/`Spin(3)` rotor conjugation; the separately maintained Pure Spin(8)
v1.0 family uses a faithful `(8v,8s+,8s-)` cache. This is a model-design and
controlled-ablation programme, distinct from the general scan/compiler algebra
and from representation-theorem claims made without a trained model.

## Core Questions

1. Do structured noncommuting transitions improve prediction, memory, state
   efficiency, or measured compute against matched simpler transports?
2. Which stability, equivariance, gradient, and streaming contracts are
   guaranteed by the maintained recurrence?
3. Can optimized kernels change the conclusion reached by eager implementations?
4. Which group center, semidirect-product, and composition-depth effects are
   genuine model mechanisms rather than coordinate-supplied shortcuts?

## Proven / Established Results

- The maintained selective damped-rotor recurrence has persistent state,
  constant-state streaming inference, an associative affine composition law in
  exact arithmetic, and a finite-input state-norm bound under its stated
  assumptions.
- The final-axis tensor contract and zero-angle rotor construction pass the
  maintained shape, gradient, scan/recurrent, and cross-backend checks.
- The coefficient algebra \(\mathrm{Cl}(3,0)\) now has a separate exact
  representation-theoretic embedding into
  \(\mathrm{Cl}^0(1,4)\subset\mathrm{Cl}(1,4)\), constructed inside the
  repository's real Spin(9) matrices. This locates the Euclidean algebra in a
  larger signature hierarchy but does not change the maintained recurrence.
- Post-training action clamping and time shuffling worsen prediction, showing
  that learned actions are causally active in the frozen v2.1 cohort.
- At matched recurrent state in that cohort, rotor transport improves
  prediction loss over identity in all five seeds.
- `pure_spin8_ssm/` v1.0 is a separately maintained PyTorch model with one
  shared 28D controller, vector and both chiral eight-real actions, a faithful
  24-scalar cache, bounded affine writes, work-efficient/recurrent scans,
  masks, gradients, CUDA, and an independent checkpoint schema. It does not
  change Pure Rotor v2.1.
- On the frozen center-sensitive Spin(8) transport task, all three Pure Spin(8)
  checkpoints reach L128 MSE `5.81e-5`--`6.68e-5` and 100% center
  classification. The unfused Mamba-2 reference remains at
  `0.132`--`0.135` and 50%; the comparison is supplied-coordinate and not
  parameter/state/compute matched.
- A direct Spin quaternion scan passes the frozen binary-center `2.A5` gates
  that conjugation-based Pure Rotor cannot represent. A unit-dual-quaternion
  motor extends the state to the double cover of `SE(3)`; its blind learned
  readout fails, while legal adjacent-prefix identification passes all 162
  rigid-pose splits and a separate noise audit records the failure boundary.
- Raw nonassociative octonion multiplication is not scanned. The experimental
  operator lift composes associative `8 by 8` left actions, has exact
  rank-28/`-2^49` `so(8)` closure, and supports bounded affine, recurrent,
  work-efficient, and WSL/Triton paths. Under terminal-only training, an
  L2/L4/L8/L16 curriculum recovers 9/9 structured and 9/9 dense laws; an odd
  holdout exposes the even-depth `G2 union -G2` ambiguity.

## Open Claims

- State-, parameter-, and optimized-compute-matched superiority over quaternion,
  complex/MIMO, Householder-product, delta-rule, and modern SSM baselines.
- Long-context memory and language-model scaling benefits.
- Fused rotor kernels and end-to-end throughput on supported accelerators.
- Broader equivariance beyond the tested `Spin(3)` tensor contract.
- Latent-input and natural-data replication of the Pure Spin(8) result with
  separately matched parameter, recurrent-state, and optimized-compute rows.
- Odd/even, unseen-basis octonion-law recovery without supplied local algebra
  coordinates.

## Dependencies

- Programme 01 provides the general affine-scan and numerical-parity framework.
- The canonical model-only implementation is
  `SSM-Models/pure_rotor_ssm/` in matched PyTorch and JAX backends.
  `SSM-Models/ga_ssm.py` is the JAX experiment/training shell,
  `rotor_ssm_torch.py` is an import-compatibility shell, and `GALib.py` is the
  shared low-level algebra. The original `SpinorModel` is a historical
  prototype and separate evidence track.
- `Spin8-SSM-Benchmark/SpinorDeltaLM` is a separate isolated comparison model,
  not the canonical successor to Pure Rotor SSM v2.1.0.
- This programme does not depend on Spin(8), Spin(9), triality, or the
  Dirac--Gram theorem programme for the validity of Pure Rotor. The separate
  Pure Spin(8) family depends on Programme 04's maintained representation
  identities, not on the open sensing theorem.
- The signature-extension theorem is cross-program mathematical context only.
  The model has not been widened to 16 coefficients, given a Lorentzian state,
  or trained with the \(\mathrm{Cl}(1,4)\) action.

## Non-claims

- Exact algebraic associativity is not bitwise float32 scan equality.
- Active learned rotors are not evidence of rotor-specific superiority.
- Rotor lost the preregistered associative-recall gate and was not the best
  state-matched transport in the reported comparison.
- The wider identity model won the reported matched eager-CUDA-cost comparison.
- No current result establishes language-model or production superiority.
- The approximately 2,000x Pure Spin(8) MSE gap is task-specific and does not
  establish generic Mamba superiority, state matching, or a fused-kernel win.
- Exact motor identification from every-prefix signed poses is not end-to-end
  learning from ordinary `SO(3)` observations.
- Even-depth final-only success does not uniquely identify `G2`; the negative
  parity coset is a proven task symmetry.

## Canonical Evidence

- [Maintained mathematical and implementation contract](../../SSM-Models/FOUNDATIONS.md)
- [Foundations validity audit](../../SSM-Models/FOUNDATIONS_VALIDITY_AUDIT_2026-08-06.md)
- [Pure v2.1 transport ablation](../../SSM-Models/experiments/PURE_V2_1_TRANSPORT_ABLATION_RESULTS.md)
- [Related-work and claim notes](../../SSM-Models/experiments/PURE_V2_1_RELATED_WORK_NOTES.md)
- [Exact \(\mathrm{Cl}(3,0)\) to \(\mathrm{Cl}(1,4)\) embedding](../../Spin-Space-Research/docs/manuscripts/CLIFFORD_SIGNATURE_EXTENSION.md)
- [Pure Spin(8) model contract](../../SSM-Models/pure_spin8_ssm/CONTRACT.md)
- [Pure Spin(8) versus Mamba-2 result](../../SSM-Models/experiments/PURE_SPIN8_VS_MAMBA2_RESULTS.md)
- [Spin/2.A5 multi-relation result](../../SSM-Models/experiments/SPIN_2A5_MULTIRELATION_RESULTS.md)
- [Motor implementation and identification result](../../SSM-Models/experiments/SPIN_MOTOR_RIGID_2A5_RESULTS.md)
- [Octonion operator-scan result](../../SSM-Models/experiments/OCTONION_OPERATOR_SCAN_RESULTS.md)
- [Final-only octonion-law result](../../SSM-Models/experiments/OCTONION_FINAL_ONLY_RESULTS.md)
