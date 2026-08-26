# Programme 06: Rotor and noncommutative state-space models

**Research author:** Hayden Austin

**Last reconciled:** 2026-08-25

## Scope

Bounded recurrent models whose persistent state is transported by
input-selective noncommuting actions. The stable Pure Rotor v2.1 family uses
`Cl(3,0)`/`Spin(3)` rotor conjugation; the separately maintained Pure Spin(8)
v1.1 family uses a faithful `(8v,8s+,8s-)` cache. This is a model-design and
controlled-ablation programme, distinct from the general scan/compiler algebra
and from representation-theorem claims made without a trained model.

The 2026-08-21 development successor
`SSM-Models/pure_f4_delta_ssm_v1_3/` is intentionally not yet called a
maintained trained model. It supplies the semantic and algebraic machinery for
one 27D Albert carrier with selectable
`Spin(8) / Spin(9) / F4 / E6(-26) / custom` action banks and a generalized
delta memory. Its local audit removed inherited compactness, monotone-schedule,
rank-one, tied-address, and incomplete-cache restrictions. Promotion requires
new matched natural-data and complete-step hardware evidence.

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
- Pure Spin v1.2's post-F14a three-seed Tiny Shakespeare quality gate is
  negative: fused Mamba-2 wins all three seeds and averages 2.4942 bpb versus
  2.7477 for Pure Spin. The local steady-step throughput ordering is unresolved
  at the observed run-to-run variation.
- Pure Exceptional Delta v1.3's fresh five-seed layer-localization gate rejects
  early E6 transport: it wins only 2/5 seeds and is worse by 0.0093 bpb on
  average. Identity remains the supported generic natural-text reference.
- Hybrid Memory v1.4/v1.4.5 is a separate active model workspace. Its G15A
  exact SM75/FP32 primary cohort passes across three seeds: full Spin `S`
  reaches 1.0 symmetry macro versus 0.2 for commuting `C` and 0.1 for `I` and
  `I+C` in every seed, while every arm learns the no-sym delayed control at
  1.0 through L1024. In the frozen conditional cohort, `S+identity-read` ties
  `S` at 1.0 in every seed, so the fixed Clifford/negative-spin read contribution
  is unsupported; `S-broken` falls to 0.3/0.2/0.2, passing the shared-coupling
  control with 0.7/0.8/0.8 margins. Both conditional arms retain 1.0 no-sym
  recall through L1024. The strongest result is shared vector/positive-Spin
  lift on this supplied-coordinate task, not full three-carrier triality, a
  maintained-model promotion, generic association, natural text, or scaling.
- G15A-L then fails autonomous learned-coordinate attribution on clean SM75
  commit `0c49f64` in fresh seeds 2153/2161/2179. `S` mean cosine spans
  0.9902--0.9957 at L64, 0.9803--0.9913 at L256, and 0.9726--0.9862 at L1024;
  only seed 2161 L64 meets the absolute `S` thresholds, and no row meets the
  0.05 comparator gate. Learned `S-broken` matches `S` within about `6e-8`
  across all rows and has an effective positive chart within `<4e-7` of `S`.
  The cosine score cancels positive scalar `<q,Vk>`, making the vector carrier
  unobservable and the broken signed permutation invertible for `P`.
- G15A-F's full-frame repair is a clean-commit `503fa82`, exact-SM75 cohort over
  seeds 2203/2207/2213. All rank-56 screens pass, with condition ratios
  0.286--0.310, minimum primitive projection residual about 0.598, and an exact
  broken Lie-bracket certificate with 474/784 mismatches and maximum integer
  residual 4. The quality gate fails: `S` mean relative Frobenius error is
  0.0927/0.0912/0.0705 at L64, 0.1201/0.1187/0.1036 at L256, and
  0.1304/0.1433/0.1392 at L1024; p95 fails every row and maximum-error gates
  fail cohort-wide. `S` beats `I`, `C`, and `S-broken` by at least 0.05 in every
  row, so the observation repair separates the shared lift, but controller
  precision remains unsolved. The broken-2x check passes only two rows.
- G15A-R passes the first-order repair on exact SM75 clean commit `eca70f0`.
  All five development recipes qualify. The 600-step fixed-LR/random control
  reaches 0.0155--0.0285 mean error, so decay is not proven necessary. Frozen
  least-intervention selection chooses `G-decay/random`, whose development
  means are about `1.03e-7`--`1.85e-7`; staged decay is vastly more precise,
  while block moments and curriculum are unnecessary. Fresh confirmation seeds
  2251/2267/2273 pass all four transports and all L64/L256/L1024 gates: `S`
  means are `1.21e-7`--`1.89e-7`, p95 `<=2.68e-7`, maximum `<=3.27e-7`, and
  every comparator-margin and broken-2x check passes.
- G15A-S separately passes spanning-chart and center-sensitive transfer on
  exact SM75 from clean commit `4067926`, fresh seeds 2281/2287/2293. Its 56
  signed actions span all 28 generators and every train/evaluation frame bank
  has rank 56. On unseen banks, `S` means are `6.89e-7`--`1.18e-6` through
  L1024, p95 `<=1.47e-6`, maximum `<=1.94e-6`; all comparator margins are
  0.266--0.536 and broken-2x passes every row. Structured direct vector/positive
  errors are `<=2.36e-5`, with frame maximum `<=3.54e-5`. This is
  composition-only transfer of a learned signed dictionary to unseen frame
  banks and global center words under oracle edit timing, not learned topology,
  association/address/query learning, negative-spin/Clifford utility, language,
  full triality, or scaling/efficiency.
- G15B completes its exact-SM75 three-seed commissioned-controller cohort from
  clean commit `bd5045a`, with 4,200 updates and 375,360 scored decisions per
  seed/arm. Preflight and integrity pass, but the frozen adjudication fails.
  Identity reaches about 0.972 mean MQAR and 0.975 selective recall, while
  overwrite remains 0.768--0.833. `C` is worse, and `S` is inferior to identity
  in every MQAR/overwrite/selective mean cell, passing noninferiority only on
  needle cells. Address top-1 is near perfect, but overwrite erase recall stays
  about 0.5 and write F1 is below gate. This is a temporal-observability failure:
  collision history defines the erase target but is absent from the token/local-
  convolution controller. G15C/external-only is blocked. The separate G15A-S
  composition result remains intact.
- G16 completed its one-seed parameter-matched SM75 development cohort.
  Official fused Mamba-2 wins all ordinary-compression contexts, reaching
  `1.48571` BPRB at L4096 versus v1.4.5 `1.58335`; local GDN2 and OLMo lose,
  and every arm fails learned recall. This is not a model-family promotion.

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
- G15B has rejected the present token/local-convolution commissioned controller
  and blocked G15C/external-only. G15B-R0 then rejects naive tied delta because
  the learned controllers use a structured one-token write continuation.
  G15B-R1 preserves that continuation but rejects erase at every valid write:
  overwrite loses 9.7--11.5 points and every non-needle gate fails. G15B-R2
  supplies perfect collision timing and preserves the write tail, yet
  post-same-key-overwrite recall falls 10.3--12.1 points. G15B-R3's oracle
  component reset repairs ordinary overwrite by 12.2--12.8 points and reaches
  1.0 on its guard, but fails the frozen saturated-baseline improvement and
  FP32 replay-tolerance gates. G15B-R4's exact-replay factorial then passes only
  the value-plus-tail arms. Value-only overwrite remains below learned at every
  length and collapses for seed 2311; excluding background harms value-only but
  not value-plus-tail behavior. This localizes the retained association to
  ambiguous `t+1` ownership rather than shared background. G15B-R5 then
  isolates strict-history, current-token, bias-only, and exact-residual tail
  sources while preserving the full-token transition. Its background-free
  strict-history arm passes every performance and bias gate, reaching
  0.9424--0.9466 mean ordinary overwrite and 1.0 on the constructed guard.
  The formal quality decision nevertheless fails the frozen BPQ replay and
  FP32 state/read numeric tolerances; discrete replay and learned logits remain
  exact, and the FP64 contract passes. R5-S subsequently fails all 135
  prospective scaled-logit checks despite exact categorical behavior and
  passing BPQ, state/read, transition, fingerprint, provenance, and FP64
  contracts. Retained-checkpoint repair stops. This does not make more generic
  transport geometry the repair. Fresh G15B-T now implements explicit
  strict-history commit/edit control and an exact monolithic residual/read
  path. Its exact clean-SM75 Phase-0 qualification passes with matched 38,082-
  parameter `F/T` arms, exact current/history causality, nonexpansive measured
  transitions, `8.94e-8` maximum FP32 logit residual, exact predictions, and
  finite nonzero declared gradients. The completed exact-SM75 Phase-1 cohort
  then formally fails both primary `T` and diagnostic-only `T-AUX`: `T` trails
  `F` on mean overwrite throughout and misses commit timing, while `T-AUX`
  improves both without reaching the frozen margin or all-seed timing gate.
  G15B-T stops before geometry. R5 and R5-S retain their failed historical
  adjudications.
  Spin remains a specialized prior for
  supplied or coherent frames; learned topology, full three-carrier triality
  utility, natural text, and scaling remain open.

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
- G15A supplies exact task coordinates and oracle carrier controls; its passed
  symmetry separation and shared-coupling control do not establish autonomous
  transport inference, full three-carrier triality, generic association,
  natural-text recall, or scaling.
- G15A-L's high learned-coordinate cosine scores do not rescue attribution:
  `S-broken` is observationally equivalent to `S` under the frozen score, and
  no row passes the comparator gate.
- G15A-F's repaired-observation comparator margins do not rescue its failed
  absolute precision, p95, maximum-error, or broken-2x quality gates and do not
  promote the model.
- G15A-R establishes composition-only minimal-controller chart learning under a
  four-probe oracle frame and oracle edit timing. It does not establish generic
  association, learned addressing/querying, language, full triality, or scaling;
  fixed-LR success means decay is sufficient but not proven necessary.
- G15A-S extends composition to a signed 28-generator dictionary, unseen frame
  banks, and global center words, but retains oracle edit timing. It does not
  establish learned topology, generic association/address/query control,
  negative-spin/Clifford utility, language, full triality, or scaling/efficiency.
- G15B does not establish generic retrieval or Spin-specific utility. Identity
  is stronger in every MQAR/overwrite/selective mean cell, the overwrite edit
  controller fails its frozen gate, and G15C/external-only is blocked. This does
  not invalidate G15A-S's separate composition result under oracle edit timing.
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
- [Current model inventory](../../SSM-Models/MODEL_STATUS.md)
- [Pure Spin v1.2 frontier result](../../SSM-Models/pure_spin_ssm_v1_2/FRONTIER_TRAINING_RESULTS.md)
- [Pure Exceptional Delta layer-localization result](../../SSM-Models/pure_f4_delta_ssm_v1_3/SHAKESPEARE_LAYER_LOCALIZATION_RESULTS.md)
- [Hybrid Memory frontier review](../../SSM-Models/hybrid_memory_v1_4/FRONTIER_REVIEW_2026-08-25.md)
- [Hybrid Spin/torus architecture boundary](../../SSM-Models/hybrid_memory_v1_4/SPIN_TORUS_RESEARCH.md)
- [G15 Spin-Dirac status and result ledger](../../SSM-Models/hybrid_memory_v1_4/G15_SPIN_DIRAC_RESULTS.md)
- [G15A primary cohort artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15a_spin_dirac_cohort_sm75_2026-08-25.json)
- [G15A conditional attribution protocol](../../SSM-Models/hybrid_memory_v1_4/G15A_CONDITIONAL_CONTROLS_PROTOCOL_2026-08-25.md)
- [G15A conditional attribution artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15a_conditional_controls_sm75_2026-08-25.json)
- [G15A-L learned-coordinate protocol](../../SSM-Models/hybrid_memory_v1_4/G15AL_LEARNED_COORDINATE_PROTOCOL_2026-08-25.md)
- [G15A-L learned-coordinate artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15al_learned_coordinate_cohort_sm75_2026-08-25.json)
- [G15A-F full-frame protocol](../../SSM-Models/hybrid_memory_v1_4/G15AF_FULL_FRAME_PROTOCOL_2026-08-25.md)
- [G15A-F full-frame artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15af_full_frame_cohort_sm75_2026-08-25.json)
- [G15A-R first-order repair protocol](../../SSM-Models/hybrid_memory_v1_4/G15AR_FIRST_ORDER_PROTOCOL_2026-08-25.md)
- [G15A-R first-order repair artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15ar_first_order_repair_sm75_2026-08-25.json)
- [G15A-S spanning-center protocol](../../SSM-Models/hybrid_memory_v1_4/G15AS_SPANNING_CENTER_PROTOCOL_2026-08-25.md)
- [G15A-S spanning-center artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15as_spanning_center_sm75_2026-08-25.json)
- [G15B interleaved-controller protocol](../../SSM-Models/hybrid_memory_v1_4/G15B_CONTROL_PROTOCOL_2026-08-25.md)
- [G15B interleaved-controller result](../../SSM-Models/hybrid_memory_v1_4/G15B_INTERLEAVED_CONTROLLER_RESULTS.md)
- [G15B exact-SM75 quality artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15b_interleaved_controller_sm75_2026-08-26.json)
- [G15B-R0 checkpoint-repair result](../../SSM-Models/hybrid_memory_v1_4/G15BR_CHECKPOINT_REPAIR_RESULTS.md)
- [G15B-R0 exact-SM75 artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15br_checkpoint_repair_sm75_2026-08-26.json)
- [G15B-R1 event-erase result](../../SSM-Models/hybrid_memory_v1_4/G15BR1_EVENT_ERASE_RESULTS.md)
- [G15B-R1 exact-SM75 artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15br1_event_erase_sm75_2026-08-26.json)
- [G15B-R2 collision-only erase result](../../SSM-Models/hybrid_memory_v1_4/G15BR2_COLLISION_ERASE_RESULTS.md)
- [G15B-R2 exact-SM75 artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15br2_collision_erase_sm75_2026-08-26.json)
- [G15B-R3 component-replacement result](../../SSM-Models/hybrid_memory_v1_4/G15BR3_LOGICAL_COMPONENT_RESULTS.md)
- [G15B-R3 exact-SM75 artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15br3_logical_component_sm75_2026-08-26.json)
- [G15B-R4 ownership/background result](../../SSM-Models/hybrid_memory_v1_4/G15BR4_OWNERSHIP_BACKGROUND_RESULTS.md)
- [G15B-R4 exact-SM75 artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15br4_ownership_background_sm75_2026-08-26.json)
- [G15B-R5 causal-tail-source result](../../SSM-Models/hybrid_memory_v1_4/G15BR5_CAUSAL_TAIL_SOURCE_RESULTS.md)
- [G15B-R5 exact-SM75 artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15br5_causal_tail_source_sm75_2026-08-26.json)
- [G15B-R5-S numerical result](../../SSM-Models/hybrid_memory_v1_4/G15BR5S_NUMERICAL_RATIFICATION_RESULTS.md)
- [G15B-R5-S exact-SM75 artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15br5s_numerical_ratification_sm75_2026-08-26.json)
- [G15B-T transactional-delta protocol](../../SSM-Models/hybrid_memory_v1_4/G15BT_TRANSACTIONAL_DELTA_PROTOCOL_2026-08-26.md)
- [G15B-T Phase-0 qualification result](../../SSM-Models/hybrid_memory_v1_4/G15BT_PHASE0_QUALIFICATION_RESULTS.md)
- [G15B-T Phase-0 exact-SM75 artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15bt_phase0_qualification_sm75_2026-08-26.json)
- [G15B-T Phase-1 quality result](../../SSM-Models/hybrid_memory_v1_4/G15BT_PHASE1_RESULTS.md)
- [G15B-T Phase-1 exact-SM75 quality artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15bt_phase1_quality_sm75_2026-08-26.json)
- [G16 SM75 frontier-shootout protocol](../../SSM-Models/hybrid_memory_v1_4/G16_SM75_FRONTIER_SHOOTOUT_PROTOCOL_2026-08-25.md)
- [G16 trained-frontier results](../../SSM-Models/hybrid_memory_v1_4/G16_SM75_FRONTIER_SHOOTOUT_RESULTS.md)
- [G16 exact-SM75 runtime qualification artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g16_runtime_qualification_sm75_2026-08-25.json)
- [Local SM75 native-runtime ledger](../../SSM-Models/hybrid_memory_v1_4/SM75_NATIVE_RUNTIME.md)
