# Programme 06: Rotor and noncommutative state-space models

## Scope

Bounded recurrent models whose persistent state is transported by
input-selective noncommuting actions. The stable Pure Rotor v2.1 family uses
`Cl(3,0)`/`Spin(3)` rotor conjugation; the separately maintained Pure Spin(8)
v1.1 family uses a faithful `(8v,8s+,8s-)` cache. This is a model-design and
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
- `pure_spin8_ssm/` v1.1 is a separately maintained PyTorch model with one
  shared 28D controller, vector and both chiral eight-real actions, a faithful
  24-scalar cache, bounded affine writes, work-efficient/recurrent scans,
  masks, gradients, CUDA, and an independent checkpoint schema. It does not
  change Pure Rotor v2.1.
- On the frozen center-sensitive Spin(8) transport task, all three Pure Spin(8)
  checkpoints reach L128 MSE `5.81e-5`--`6.68e-5` and 100% center
  classification. The unfused Mamba-2 reference remains at
  `0.132`--`0.135` and 50%; the comparison is supplied-coordinate and not
  parameter/state/compute matched.
- A parameter-near latent-token continuation withholds the teacher's 28D
  coordinates and excludes the center-producing pair `a,a`. Fresh seeds 1--3
  identify the eight local actions, preserve 100% center/identity correctness
  through L128, and produce six L128 post-relation MSEs in
  `2.66e-5`--`2.87e-4`. Mamba-2 and GRU remain near chance on the central
  distinction. This is finite-dictionary identification under every-prefix
  supervision, not natural-input or prior-matched superiority.
- The v1.1 compiled-token path freezes an identified finite action dictionary
  and scans it with a register-resident Triton recurrence. It preserves the
  24-state faithful cache and the held-out relation while recording a
  30.7x--67.1x local model-forward speedup. It is inference-only, serial in
  sequence depth, workstation-specific, and not a fused-Mamba comparison.
- A separate SM75 hardware audit compiles repeated real eight-dimensional
  isotypic copies to an FP16 Triton `mma.sync` recurrence. PTX proves genuine
  Tensor-Core execution, but the optimized scalar register path wins six of
  eight bounded cells and remains the default for the multiplicity-one
  `(8v,8s+,8s-)` cache. One high-parallelism cell records a 1.147x win, so the
  supported conclusion is shape-dependent hybrid dispatch.
- Compiler v2.1.1 turns that feasibility result into a conservative runtime
  policy and adds a full-gradient continuous-action kernel. Exact isotypic
  type does not by itself imply shared action: the compiler accepts Tensor-Core
  packing only for repeated memories transported by the same action. The new
  profile records one `1.423x` Tensor-Core cell, a bounded `112.62x` fused
  forward/backward gain over a sequential eager oracle, and only `1.059x`
  end-to-end for the self-calibrating layer because action construction now
  dominates.
- Compiler v2.1.2 closes the canonical learned-linear-controller fusion gate:
  all 28 coordinates, three triality factor streams, recurrence, and full
  backward—including controller-table and input gradients—run in custom CUDA.
  The maximally fused witness is slower than staging coordinates once because
  the three representations share them. Profiled auto-dispatch selects the
  staged direct-factor lowering, which records `11.615x`--`14.916x` over
  materialized eager training on the local three-cell grid.
- A frozen noisy continuous-observation continuation removes the finite token
  dictionary. Fresh seeds 1--3 identify unique nonlinear/noisy local actions,
  retain exact center/identity correctness through L128, and give shared
  Spin(8) median L128 post-relation MSE `0.01860`, versus `0.06129` for a
  capable parameter-near, exactly 24-state independent `SO(8)^3` control. A
  separately frozen local update-wall allocation matches all rows within 2.97%
  and changes the independent/shared median ratio from `3.295x` to `4.019x`.
  This is an injective, every-prefix synthetic task with an unfused Mamba
  fallback, not a natural-task or generic-compute result.
- A frozen endpoint-only continuation removes every intermediate target while
  retaining the same noisy observation system and excluded relation. Across
  fresh seeds 1--3, shared Spin(8) has median L128 post-relation MSE `0.01296`,
  versus `0.06268` for the capable exactly state-matched independent family at
  equal updates. The separately pre-frozen local update-wall cohort records
  `0.01296` versus `0.09080`; every structured L128 row remains correct. This
  closes the dense-prefix-target objection for the stated signed synthetic
  teacher, not for unsigned, partial, noninjective, or natural observations.
- The endpoint observability continuation exactly computes rational probe-rank
  profiles `7,13,18,22,25,27,28,28` in every triality view and proves a
  balanced quotient-input hidden-lift lower bound of MSE `1/8` and accuracy
  `1/2`. Empirically, one signed half-spin endpoint transfers through the
  shared action to all three views in every fresh seed. The frozen aggregate
  remains failed: vector-only supervision misses exact lift rows in all three
  seeds, and one independent positive-only control fails optimization.
- A separately frozen adaptive calibration supplies the vector endpoint, a
  three-bit lift-invariant max-coordinate address, and one lift-odd sign bit.
  Exact geometry guarantees selected magnitude at least `1/sqrt(8)`. Across
  untouched seeds 4--6, shared Spin(8) passes every seedwise gate without
  median rescue: action RMSE spans `0.012825--0.013633`, all-view L128 MSE
  spans `0.009198--0.026986`, and all lift/center relation rows are exact. The
  fixed-coordinate sign is worse than vector-only. This is four transmitted
  bits but only one bit of lift information; it is not lift recovery from
  `8v` alone.
- Exact gradient tracing proves that the independent control's unsupervised
  negative-specific head receives zero data gradient and ends exactly at its
  AdamW decay-only counterfactual, while all 28 shared Spin(8) coordinate rows
  receive gradient. A matched same-state shared-latent control then scrambles
  the spinor alignments. Its frozen all-view dominance gate fails on two seed-7
  vector-L128 cells; no aggregate statistic rescues it. Correct alignment still
  wins all `9/9` action, `12/12` spinor-L128, and `6/6` fully hidden negative-
  L128 comparisons, and full supervision establishes scrambled capability.
  This supports bounded cross-view spinor transfer, not universal dominance.
- The negative-only calibration-rank successor proves a sharper exact
  boundary. Ordered basis probes have ranks
  `0,7,13,18,22,25,27,28,28`; explicit rational stabilizers prove global
  `SO(8)` action non-identifiability through six probes, global identification
  at seven, and redundancy at eight. Every fresh rank-28 row recovers the
  aligned action with a bitwise-identical router. The frozen all-seed empirical
  headline still fails two seed-10 effect-size gates, so the exact theorem is
  not reported as a uniform task-error phase transition.
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
- The experimental Dense SO(8) Cayley scan exposes all 28 exact Clifford
  tangent directions from the monomial/golden dense-closure theorem while
  retaining an eight-scalar bounded affine cache. Its float64 structural
  contracts and one WSL2 RTX 2070 SUPER CUDA forward/backward feasibility
  smoke pass. This is neither training evidence nor a speed comparison.

## Open Claims

- State-, parameter-, and optimized-compute-matched superiority over quaternion,
  complex/MIMO, Householder-product, delta-rule, and modern SSM baselines.
- Long-context memory and language-model scaling benefits.
- The canonical learned linear controller and direct triality recurrence are
  fused with full gradients. Fusion of the nonlinear probe-to-Hodge-to-Givens
  self-calibration chart, low-precision/Tensor-Core training, optimizer steps,
  cross-device policy, and matched end-to-end natural-task throughput remain
  open.
- Broader equivariance beyond the tested `Spin(3)` tensor contract.
- Natural-data replication of the Pure Spin(8) result, noninjective/chart-shift
  observation robustness, and fused optimized-compute comparison.
- Deriving a stable calibration address from physically available observations
  rather than supplying it, and global lift consistency under unknown initial
  state or chart-boundary perturbations.
- Deriving the seven-probe negative-view frame from relational or physically
  available observations rather than supplying ordered basis images.
- Odd/even, unseen-basis octonion-law recovery without supplied local algebra
  coordinates.
- Matched-task evidence that the full dense-SO(8) Cayley control chart is more
  useful than restricted, octonion, Givens/Householder, or modern SSM controls.

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
- The latent-token result removes supplied coordinates only for eight fixed
  symbols; the compiler caches those learned actions and does not solve online
  continuous action inference.
- The continuous-observation result does solve online action inference for its
  seven-coordinate injective noisy chart. Its endpoint-only successor shows
  that intermediate prefix targets are unnecessary on this fixed L16 signed-
  state task, but does not cover unsigned/partial observations, irregular or
  longer-horizon labelling, natural inputs, chart shift, or all 28 tangents.
- The partial-readout aggregate is not a pass. Signed half-spin transfer is a
  replicated stratum inside a failed all-mask cohort; vector-only near-perfect
  center accuracy is not exact lift identifiability, and quotient-input lift
  recovery is formally impossible under balanced collisions.
- The adaptive calibration successor is a separate pass, but its four-bit
  interface includes an externally supplied address. It proves one lift-odd
  bit is sufficient inside that chart, not that one total bit or the vector
  endpoint alone determines the trained lift. One independent seed also ends
  below exact bit fit, so the matched-family gap is evidence for the shared
  prior rather than an optimization-independent theorem.
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
- [Pure Spin(8) latent-increment validation](../../SSM-Models/experiments/PURE_SPIN8_LATENT_INCREMENT_RESULTS.md)
- [Pure Spin(8) compiled token scan](../../SSM-Models/experiments/PURE_SPIN8_COMPILED_TOKEN_SCAN_RESULTS.md)
- [Spin(8) isotypic Tensor-Core dispatch audit](../../SSM-Models/experiments/SPIN8_ISOTYPIC_TENSOR_CORE_AUDIT.md)
- [Isotypic-to-silicon compiler v2.1.1](../../SSM-Models/experiments/ISOTYPIC_TO_SILICON_COMPILER_V211.md)
- [Trainable continuous Spin(8) factor compiler v2.1.2](../../SSM-Models/experiments/SPIN8_TRAINABLE_FACTOR_COMPILER_V212.md)
- [Pure Spin(8) noisy continuous-observation validation](../../SSM-Models/experiments/PURE_SPIN8_CONTINUOUS_OBSERVATION_RESULTS.md)
- [Pure Spin(8) endpoint-only continuous identification](../../SSM-Models/experiments/PURE_SPIN8_ENDPOINT_SUPERVISION_RESULTS.md)
- [Pure Spin(8) endpoint observability boundary](../../SSM-Models/experiments/PURE_SPIN8_ENDPOINT_OBSERVABILITY_RESULTS.md)
- [Pure Spin(8) adaptive lift-bit calibration](../../SSM-Models/experiments/PURE_SPIN8_LIFT_BIT_CALIBRATION_RESULTS.md)
- [Pure Spin(8) exact alignment-calibration threshold](../../SSM-Models/experiments/PURE_SPIN8_ALIGNMENT_CALIBRATION_RANK_RESULTS.md)
- [Spin/2.A5 multi-relation result](../../SSM-Models/experiments/SPIN_2A5_MULTIRELATION_RESULTS.md)
- [Motor implementation and identification result](../../SSM-Models/experiments/SPIN_MOTOR_RIGID_2A5_RESULTS.md)
- [Octonion operator-scan result](../../SSM-Models/experiments/OCTONION_OPERATOR_SCAN_RESULTS.md)
- [Dense SO(8) Cayley scan design and CUDA feasibility gate](../../SSM-Models/experiments/DENSE_SO8_CAYLEY_SCAN_DESIGN.md)
- [Final-only octonion-law result](../../SSM-Models/experiments/OCTONION_FINAL_ONLY_RESULTS.md)
