# Programme 06: Rotor and noncommutative state-space models

## Scope

Bounded recurrent models whose persistent `Cl(3,0)` state is transported by
input-selective noncommuting actions, especially `Spin(3)` rotor conjugation.
This is a model-design and controlled-ablation programme, distinct from
Spin(8) triality and from general scan/compiler algebra.

## Core Questions

1. Do structured noncommuting transitions improve prediction, memory, state
   efficiency, or measured compute against matched simpler transports?
2. Which stability, equivariance, gradient, and streaming contracts are
   guaranteed by the maintained recurrence?
3. Can optimized kernels change the conclusion reached by eager implementations?

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

## Open Claims

- State-, parameter-, and optimized-compute-matched superiority over quaternion,
  complex/MIMO, Householder-product, delta-rule, and modern SSM baselines.
- Long-context memory and language-model scaling benefits.
- Fused rotor kernels and end-to-end throughput on supported accelerators.
- Broader equivariance beyond the tested `Spin(3)` tensor contract.

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
  Dirac--Gram theorem programme.
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

## Canonical Evidence

- [Maintained mathematical and implementation contract](../../SSM-Models/FOUNDATIONS.md)
- [Foundations validity audit](../../SSM-Models/FOUNDATIONS_VALIDITY_AUDIT_2026-08-06.md)
- [Pure v2.1 transport ablation](../../SSM-Models/experiments/PURE_V2_1_TRANSPORT_ABLATION_RESULTS.md)
- [Related-work and claim notes](../../SSM-Models/experiments/PURE_V2_1_RELATED_WORK_NOTES.md)
- [Exact \(\mathrm{Cl}(3,0)\) to \(\mathrm{Cl}(1,4)\) embedding](../../Spin-Space-Research/docs/manuscripts/CLIFFORD_SIGNATURE_EXTENSION.md)
