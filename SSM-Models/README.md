# GA-SSM experiments

`ga_ssm.py` is the maintained JAX/Flax implementation. Its primary model is
`GASSMLanguageModel`, built from input-selective damped-rotor state transitions
that compose through an associative parallel scan. It contains the model,
deterministic batch generation, train/evaluation steps, checkpoint handling,
and CLI without performing network or profiler work when imported.

`GALib.py` is the shared GA(3, 0) algebra layer. Multivectors use the basis
order `[1, e1, e2, e3, e12, e13, e23, e123]` and always occupy the final array
axis. `GATransformerLM` remains as a corrected attention baseline for ablations.

The state-space layer exposes one `(batch, channels, 8)` recurrent state per
layer. Parallel training, chunked inference, and one-token streaming all use
the same damped-rotor transition and are tested for numerical equivalence.
`sample_text` primes a prompt once, then reuses those fixed-size states rather
than recomputing the context. See [FOUNDATIONS.md](FOUNDATIONS.md) for the
equations, stability/equivariance arguments, GPU ablation, and open questions.

## Spin(8) triality research core

The experimental Spin(8) branch now includes:

- exact vector and two chiral 8D actions from one shared bivector controller;
- a triangular two-stage triality scan with a 24-scalar streaming cache;
- a rank-deficient identifiability gate showing that equivariance reduces the
  cross-representation completion law to one learned scalar;
- orthogonal and tight-frame multiplicity codes with measured capacity laws;
- an exact addressed overwrite recurrence with shared Spin(8) transport.
- blind shared-action completion from partial vector/positive endpoints, with
  ten-seed recovery of the entirely hidden negative-chiral action.
- blind latent-slot completion: a jointly Sinkhorn-retracted address family
  learns collision-free routing from single-key episodes and transfers exactly
  to unseen mixed-key sequences through length 2048 in 10/10 seeds.
- continuous-alias routing without logical key IDs: separate write/query
  encoders plus unlabeled marginal balance pass 10/10 noisier-alias cohorts,
  while independently perfect encoders collide in every seed.
- joint blind-action and continuous-alias completion: one optimization run
  recovers collision-free routing and the held-out negative-chiral action in
  10/10 seeds, while a parameter-richer independent action family fits every
  supplied endpoint but fails off its rank-2 calibration plane.
- a sharp five-probe identifiability boundary: five generic transformed-state
  examples spanning two triality views recover the entirely unobserved third
  action through length 2048 in 10/10 seeds; four mixed-view probes and five
  single-view probes retain an exact three-dimensional stabilizer, while the
  matched independent family retains 55 unconstrained tangent directions.
- active five-query sensing: local Fisher information is exactly independent
  of the unknown orthogonal action, and an exhaustive oracle finds a balanced
  `(2,2,1)` triality sensor with numerical `det(I)=81/1024` and
  `trace(I^-1)=43`. A hard learned selector finds rank-28 designs in 10/10
  untouched seeds, reaches the strict D-optimum in 6/10, and passes noisy
  long-composition recovery in 9/10.
- joint sensor-family continuation: soft learning followed by complete joint
  hard retraction and continuous polish reaches the balanced optimum in 10/10
  fresh seeds, versus 6/10 for a fresh straight-through hard baseline. Every
  unit query contributes an exact rank-seven information projector, proving
  `trace(I)=35` for every five-query design. The balanced optimum additionally
  obeys one prospectively replicated exact degree-28 characteristic polynomial
  that implies `det(I)=81/1024` and `trace(I^-1)=43`.
- the Cayley-spectrum theorem: after fixing the singleton triality view, the
  remaining four probes form a `Spin(7)` four-frame. On the orthonormal orbit,
  the determinant is exactly
  `(1-c^2)^3(9-c^2)^2/1024`, where `c` is its Cayley calibration. Thus the
  information optimum is the Cayley-null orbit `c=0`, while calibrated Cayley
  planes `c=+/-1` are precisely rank-25 failures. Ten fresh allocation sweeps,
  10,000 random frames, and 32 adversarial searches found no global
  counterexample; the unrestricted orthonormal-completion inequality remains
  a clearly labelled conjecture rather than a theorem.
- the Dirac--Gram reduction: every moving query is exactly the graph of a
  scaled isometric seven-frame over the `7+21` Spin(7) split, reducing the
  five-query determinant to
  `2^7 32^-21 det(8 T-S S^T)`. The strengthened Gram-volume bound is proved on
  two complete correlation slices and on a signed four-parameter star family
  with three simultaneous correlations, using independently replayed exact
  rational interpolation and four-dimensional Bernstein positivity. The three
  remaining Cholesky correlations, and hence the global theorem, remain open.

The dynamic-slot stress test matches parallel, recurrent, and symbolic-oracle
execution below 2.3e-15 in float64. The learned-address gate preserves a
64-scalar streaming state and scan parity below 6.7e-16. Logical key identity
has been removed in the controlled alias world, and action/address learning now
passes jointly. The newest gate replaces supplied matrix columns with only five
generic state/action pairs per token and constructively proves the four-probe
ambiguity. Learned hard probe selection now works but retains discrete
allocation traps; joint late query-family retraction removes all four fresh
allocation failures. The spectrum is now proved on the complete orthonormal
balanced orbit, and its nonorthogonal extension is now exact on the signed
star family; the general nonorthogonal completion lemma, cross-allocation
upper bounds, scalable joint retraction, nonorthogonal capacity stress,
endpoint-only action discovery, and naturalistic downstream utility remain
open.
See SPIN8_TRIALITY_EXPERIMENT.md and the Spin8 result files under experiments.

The numbered `GA-SSM-*` scripts are research history. `GA-SSM-3.5.py` is now a
compatibility entry point for `ga_ssm.py`; versions 1-3 remain available for
comparison and old checkpoint investigation, but should not be imported by new
code.

Run the fast local checks from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s SSM-Models -p "test_*.py"
```

Start a training run explicitly (this downloads WikiText-2 if it is not cached):

```powershell
.\.venv\Scripts\python.exe SSM-Models\ga_ssm.py --epochs 10
```

Profiling is opt-in with `--profile-port 9999`. Checkpoints default to
`SSM-Models/ga_transformer_checkpoints`.

On a CUDA-capable PyTorch installation, run the independent recurrent
implementation and matched rotor/identity ablation with:

```powershell
python -m unittest discover -s SSM-Models -p "test_rotor_ssm_torch.py"
python SSM-Models\train_rotor_ssm_torch.py --steps 300 --seed 0 `
  --output SSM-Models\experiments\my_run.json
```

The checked-in `experiments/final_seed*_300.json` reports and
`experiments/final_summary.json` record the final three-seed local-GPU result.

## Recurrence-family harness

`recurrence_families_torch.py` and `compare_recurrences.py` implement a
parameter-aligned experimental ladder over real, complex, quaternion, full
Cl(3,0) rotor, grade-decayed Cl(3,0) rotor, a fixed-width complex/GA direct
sum, and non-selective-rotor recurrences. Every candidate exposes one
fixed-size state per layer and is checked for full/chunk/token equivalence.

Run its correctness suite and CUDA Q8 comparison with:

```powershell
python -m unittest discover -s SSM-Models -p "test_recurrence_harness.py" -v
python SSM-Models\compare_recurrences.py --device cuda --group q8 --steps 1000 `
  --output SSM-Models\experiments\recurrence_ladder_q8.json
```

The harness supports Q8, D4, S3, and A5 through `--group`. Each deliberately
noncommutative task predicts every ordered prefix product and separately
reports accuracy at the final sequence position. The harness records parameter
equality, identical initial behavior, accuracy, throughput, transition
diagnostics, recurrent cache size, and numerical streaming error.

For a sharper compositional test, restrict the input alphabet to group
generators and exclude an ordered pair from training:

```powershell
python SSM-Models\compare_recurrences.py --device cuda --group a5 `
  --input-elements 23145 23451 --held-out-pairs 23145:23451 `
  --steps 2000 --diagnostic-interval 50 `
  --families complex_unitary ga_rotor_selective `
  --output SSM-Models\experiments\heldout_a5.json
```

The report audits that the pair appears zero times in training and in every
evaluation sequence. With `--diagnostic-interval`, it also records gradient,
action-angle, decay, state-norm, and state-spectrum trajectories.

For fixed token actions, use the write-free mechanism harness. The
inverse-augmented alphabet avoids collapsing the two-generator training
language while retaining one genuinely unseen bigram:

```powershell
python SSM-Models\mechanistic_group_actions.py --device cuda --group a5 `
  --input-elements 23145 31245 23451 51234 `
  --held-out-pairs 23145:23451 --steps 1500 `
  --families pure_complex_unitary pure_ga_rotor pure_householder `
  --output SSM-Models\experiments\mechanistic_a5.json
```

This harness has no decay, write, residual, feed-forward path, or contextual
controller. It reports full-operator, centered orbit-variation, orthogonal-
complement, common-fixed-subspace, and canonical-direction relation errors,
plus decoded Cayley accuracy, language coverage, norm preservation, and exact
streaming equivalence. It can also construct the exact real 3D A5 irrep via a
character projector for an explicitly oracle-supervised upper bound; see
`--a5-irrep-init --freeze-actions`.

The Spin(8) experiment has now crossed its constructive algebra gate. The
octonionic implementation builds full-rank vector and two chiral 8D actions
from one shared 28D bivector, verifies the complete `so(8)` table and triality
tensor at zero residual, distinguishes all four center signatures, and exposes
an associative constant-cache recurrence. See
[SPIN8_TRIALITY_EXPERIMENT.md](SPIN8_TRIALITY_EXPERIMENT.md),
[experiments/SPIN8_TRIALITY_ALGEBRA_RESULTS.md](experiments/SPIN8_TRIALITY_ALGEBRA_RESULTS.md),
and `spin8_triality.py`.

Positive-chiral Q8 training reliably fits the short curriculum but its raw
operators drift at long horizons. The first frozen-decoder orbit retraction
passed 8/9 fresh seeds and exposed one important failure: an arbitrary
shortest-word state section can have the wrong decoder gauge. The corrected
path-section compiler estimates endpoint centroids where the raw model is
valid, projects the complete regular-action family, and transports the
observer on the reachable subspace. It passes every dense and L16,384 gate in
9/9 untouched seeds 10--18, with homomorphism RMS below `7.4e-7` and no
post-compilation gradient steps. That first compiler remains explicitly
table-aware. The successor removes the Q8 table, inverse pairs, target labels,
identity label, and group-aware sampler: it reconstructs a regular anonymous
eight-state action from the model's own long-path predictions and passes the
prospective seed-19 smoke plus 9/9 untouched seeds 20--28, all at 100% through
L16,384. See
[experiments/SPIN8_Q8_JOINT_RETRACTION_RESULTS.md](experiments/SPIN8_Q8_JOINT_RETRACTION_RESULTS.md),
[experiments/SPIN8_Q8_PATH_SECTION_VALIDATION_RESULTS.md](experiments/SPIN8_Q8_PATH_SECTION_VALIDATION_RESULTS.md),
[experiments/SPIN8_TABLE_BLIND_COMPILER_RESULTS.md](experiments/SPIN8_TABLE_BLIND_COMPILER_RESULTS.md),
`spin8_q8_joint_retraction.py`, `spin8_q8_path_section_compiler.py`, and
`spin8_table_blind_compiler.py`.

The same table-blind recovery also compiles fresh quaternion spinors in 9/9
seeds and shared four-reflection Householder actions in 8/9. The important
Spin(8) distinction is not Q8 task capacity: its exact regular endpoint section
has rank eight and preserves the complete learned centroid-logit geometry,
whereas both faithful 4D baselines require a rank-four realizability projection.
See
[experiments/TABLE_BLIND_FAMILY_BASELINES_RESULTS.md](experiments/TABLE_BLIND_FAMILY_BASELINES_RESULTS.md)
and `table_blind_family_compiler.py`.

The finite-group suite also includes `pdssm_group_actions.py`, which separates
an exact 60-state regular-action PD ceiling from learned hard column-one-hot and
Hungarian-projected permutation variants. See
`experiments/PDSSM_BASELINE_RESULTS.md`; these runs use dense L16-L256 testing
and never equate a soft dense transition with PD-SSM.

The first controlled CUDA pilot, including the failed sparse-supervision run
and the held-out length curve, is summarized in
[experiments/RECURRENCE_LADDER_RESULTS.md](experiments/RECURRENCE_LADDER_RESULTS.md).
The 1000-step, three-group, three-seed focused replication and subsequent
grade-decay/hybrid trials are summarized in
[experiments/RESEARCH_PHASE_2_RESULTS.md](experiments/RESEARCH_PHASE_2_RESULTS.md).
The literature review, A5 separation, held-out-pair falsification, and revised
Spin(8) decision are in
[experiments/RESEARCH_REVIEW_2026-08-02.md](experiments/RESEARCH_REVIEW_2026-08-02.md).
The pre-registered write-free gate, exact character construction, corrected
held-out design, and five-seed learned results are in
[experiments/MECHANISM_GATE_RESULTS.md](experiments/MECHANISM_GATE_RESULTS.md).
The subsequent deterministic ten-seed audit identifies a stable soft decoder
ensemble around one dominant 3D irrep channel; see
[experiments/CHANNEL_ENSEMBLE_RESULTS.md](experiments/CHANNEL_ENSEMBLE_RESULTS.md).
Its prospective frozen-action intervention trains only three bounded channel
gates and passes an untouched third changed-generator class in all ten seeds;
see [experiments/ROBUST_CHANNEL_GATING_RESULTS.md](experiments/ROBUST_CHANNEL_GATING_RESULTS.md)
and `robust_channel_gating.py`.
The follow-up representation audit distinguishes both real 3D A5 irreps and
finds rank-three aligned anchor defects in all ten seeds; see
[experiments/A5_IRREP_LIE_AUDIT.md](experiments/A5_IRREP_LIE_AUDIT.md).
The resulting preregistered joint-rounding falsifier is the strongest
mechanistic result in the suite: independent angle rounding appears solved
through L256 but fails three seeds at L4096, whereas one globally aligned exact
A5 anchor passes the untouched changed-order-3 alphabet at L4096 in all ten
seeds with a `96.88%` population floor and float32 homomorphism RMS below
`2.4e-7`. See
[experiments/JOINT_A5_ROUNDING_RESULTS.md](experiments/JOINT_A5_ROUNDING_RESULTS.md)
and `joint_a5_rounding.py`.
The subsequent self-compiling experiment removes the supplied irrep matrices,
character values, branch choice, and channel choice. It constructs exact 3D
candidate irreps from the A5 regular permutation action, automatically detects
the nearest learned channel, and continues ambient-gradient training with one
shared conjugacy retraction across the complete token family. All ten seeds
reach 100% on the original and untouched changed-generator class at L4096 with
float32 homomorphism RMS below `2.1e-7`; see
[experiments/SELF_COMPILING_RETRACTION_RESULTS.md](experiments/SELF_COMPILING_RETRACTION_RESULTS.md),
`representation_retraction.py`, and `train_self_compiling_retraction.py`.
The next table-blind experiment removes the explicit Cayley-table object and
token-to-element map, while retaining informationally equivalent dense prefix
labels. It mechanically reconstructs a regular permutation group from those
transitions, derives exact irreps from the recovered
algebra, and reaches 100% in all ten seeds on an untouched fourth generator
family at L16384. See
[experiments/LATENT_CAYLEY_RETRACTION_RESULTS.md](experiments/LATENT_CAYLEY_RETRACTION_RESULTS.md)
and `latent_group_discovery.py`.
The partial-action extension reconstructs all 240 A5 transitions from exactly
120 observed edges in 1,000/1,000 randomly oriented reverse covers, while
equal-budget uniform random masks recover 0/1,000. A 2-SAT adversary shows that
some 120-edge covers admit a wrong inverse matching too, so universal recovery
needs one calibration pair and 121 edges; ambiguous cases are refused. The
result is about joint-family consistency under a guaranteed reverse-edge
cover, not arbitrary half-table completion. See
[experiments/INVERSE_COVER_EXACT_HALF_RESULT.md](experiments/INVERSE_COVER_EXACT_HALF_RESULT.md),
[experiments/INVERSE_COVER_IDENTIFIABILITY_THEOREM.md](experiments/INVERSE_COVER_IDENTIFIABILITY_THEOREM.md),
[experiments/ONE_CALIBRATION_CAYLEY_RETRACTION_RESULTS.md](experiments/ONE_CALIBRATION_CAYLEY_RETRACTION_RESULTS.md),
[experiments/PARTIAL_CAYLEY_RETRACTION_PREREGISTRATION.md](experiments/PARTIAL_CAYLEY_RETRACTION_PREREGISTRATION.md),
and `partial_cayley_supervision_audit.py`.
The next endpoint-only gate removes prefix traces from both compilation and
neural loss. Fixed L16 training fails at chance in three diagnostic seeds, but
an equal-label short-to-long endpoint curriculum passes all dense and long
gates in 10/10 seeds, including a 13-point untouched-alphabet L4096--L16384
sweep at 100%. An exact Markov/information audit attributes the initialization
barrier to random-walk gradient cancellation rather than vanishing recurrent
Jacobians. See
[experiments/ENDPOINT_ONLY_FIXED_LENGTH_RESULTS.md](experiments/ENDPOINT_ONLY_FIXED_LENGTH_RESULTS.md),
[experiments/ENDPOINT_CURRICULUM_RESULTS.md](experiments/ENDPOINT_CURRICULUM_RESULTS.md),
and [experiments/ENDPOINT_MIXING_BARRIER.md](experiments/ENDPOINT_MIXING_BARRIER.md).
The causal controls hold the endpoint-label multiset or fixed-length task
constant: shuffled short/long batches fit a final training batch but never
trigger a faithful channel, while fixed L16 remains at chance through 8,000
updates. This establishes the tested short-to-long schedule, not monotonicity
in general; a prospectively frozen scrambled-block control separates clean
stages from rising difficulty. That `L8 -> L1 -> L16 -> L2 -> L4` control also
fails mechanistically despite fitting the L1/L2/L4 blocks, supporting
incremental depth continuation rather than mere block separation. See
[experiments/ENDPOINT_OPTIMIZATION_CAUSAL_RESULTS.md](experiments/ENDPOINT_OPTIMIZATION_CAUSAL_RESULTS.md)
and [experiments/ENDPOINT_BLOCK_ORDER_RESULTS.md](experiments/ENDPOINT_BLOCK_ORDER_RESULTS.md).
The separate learned-manifold compiler then removes all 1,148 additional
membership queries: in 10/10 seeds it reconstructs an A5-isomorphic table on
the first step-850 attempt using only 16,384 already-consumed curriculum
examples. All selected threshold margins are wide and all dense/long gates
pass. This is zero additional compiler supervision, not unsupervised learning.
See [experiments/ENDPOINT_MANIFOLD_COMPILER_RESULTS.md](experiments/ENDPOINT_MANIFOLD_COMPILER_RESULTS.md).
A separate representation-theoretic audit identifies the next adversarial
gate: rotor sandwich actions erase the central sign of `Spin(n)`, whereas a
left spinor action retains it. On Q8 this gives a proof-level 4-state ceiling
for pure conjugation versus eight distinct states for quaternionic spinor
action. See
[experiments/SPINOR_CENTER_FIDELITY_GATE.md](experiments/SPINOR_CENTER_FIDELITY_GATE.md)
and `spinor_center_fidelity_audit.py`. The Q8 alphabet is bipartite, so its
prospective learned gate uses adjacent odd/even curriculum and evaluation
lengths. A four-reflection O(4) action shared over two state blocks is the
capable generic baseline; the old two-reflection O(8) row is retained as an
equal-raw-parameter but representation-starved control. See
[experiments/Q8_ENDPOINT_MIXING_AUDIT.md](experiments/Q8_ENDPOINT_MIXING_AUDIT.md)
and [experiments/Q8_SPINOR_CENTER_PREREGISTRATION.md](experiments/Q8_SPINOR_CENTER_PREREGISTRATION.md).
The seed-0 learned gate separates the families sharply: quaternion left
action is 100% on every tested central pair through L16384, while both GA
charts and both Householder charts fail extrapolation. A deterministic joint
Q8-frame retraction then preserves 100% behavior while reducing whole-model
homomorphism RMS from `0.633` to `9.98e-8`, including repair of a nuisance
channel, with no decoder update. The complete compiler is now prospectively
validated on fresh seeds 10--19: raw SGD passes 8/10, whereas joint retraction
plus a frozen label-free manifold-distance decoder gate passes 10/10, with all
460 dense and all 40 long seed/length cells at 100% and homomorphism RMS below
`1.7e-7`. See
[experiments/Q8_SPINOR_CENTER_SMOKE_RESULTS.md](experiments/Q8_SPINOR_CENTER_SMOKE_RESULTS.md)
and [experiments/Q8_SPINOR_QUALITY_GATE_VALIDATION_RESULTS.md](experiments/Q8_SPINOR_QUALITY_GATE_VALIDATION_RESULTS.md).
The subsequent Spin(8) path compiler first removes the supplied table and then
removes decoder labels from discovery. Decoder-labeled table-blind recovery
passes 9/9 fresh seeds, but the stricter fixed-cardinality state-only compiler
passes only 7/9 against its frozen 8/9 requirement. The two rejected seeds do
retain exact eight-state actions; they also expose exact two-state character
quotients. This reveals multiple learned state congruences and motivated a
prospective metric-selection rule: select the largest replicated K-means action
only when every other action found by that scan is its homomorphic quotient.
See
[experiments/SPIN8_TABLE_BLIND_COMPILER_RESULTS.md](experiments/SPIN8_TABLE_BLIND_COMPILER_RESULTS.md),
[experiments/SPIN8_STATE_ONLY_COMPILER_RESULTS.md](experiments/SPIN8_STATE_ONLY_COMPILER_RESULTS.md),
[experiments/SPIN8_STATE_CARDINALITY_AUDIT_RESULTS.md](experiments/SPIN8_STATE_CARDINALITY_AUDIT_RESULTS.md),
and
[experiments/SPIN8_STATE_QUOTIENT_LATTICE_RESULTS.md](experiments/SPIN8_STATE_QUOTIENT_LATTICE_RESULTS.md).
That prospective repair passed its frozen behavioral gate: the historically
named finest-congruence compiler selects
`k=8` in all nine untouched seeds 49--57 without receiving state cardinality,
and certifies a nested `Q8/C4 ~= C2` quotient in seven. Three seeds fall below
the old separation floor but pass every algebraic, dense, and L16384 gate;
the cohort is 9/9 with recovered-table homomorphism RMS below `7.2e-7`. See
[experiments/SPIN8_FINEST_CONGRUENCE_RESULTS.md](experiments/SPIN8_FINEST_CONGRUENCE_RESULTS.md).
An exhaustive post-freeze audit over all 4,140 partitions of each recovered
eight-state action subsequently corrected the uniqueness interpretation. Every
seed has the complete Q8 congruence histogram `{1:1, 2:3, 4:1, 8:1}`; the
metric scan omitted the four-state quotient in all nine seeds and some
two-state quotients. Transition closure alone cannot select a semantic quotient
without observations or an explicit prior. See
[experiments/SPIN8_EXACT_CONGRUENCE_LATTICE_RESULTS.md](experiments/SPIN8_EXACT_CONGRUENCE_LATTICE_RESULTS.md).
The missing 28-DOF generic SO(8) baseline is also now closed. The positive
half-spin and standard skew bases are connected by an exact orthogonal
coefficient map: SGD preserves their actions and logits to float64 roundoff,
while AdamW breaks the chart equivalence through coordinatewise adaptation. In
a fresh five-seed Q8 cohort both charts fit the short curriculum in 5/5 and fail
the raw dense gate in 0/5. See
[experiments/SPIN8_SO8_OPTIMIZER_EQUIVARIANCE_RESULTS.md](experiments/SPIN8_SO8_OPTIMIZER_EQUIVARIANCE_RESULTS.md)
and [experiments/SPIN8_SO8_PAIRED_RESULTS.md](experiments/SPIN8_SO8_PAIRED_RESULTS.md).

The 2026-08-03 foundational re-audit found a lower-level expressivity gap in
the maintained Cl(3) layers. `GradeLinear` was Spin(3)-equivariant but spanned
only half of the legal linear commutant because it prohibited mixing scalar
with pseudoscalar and vector with Hodge-dual bivector. The new
`Spin3IsotypicLinear` spans the complete repeated-irrep multiplicity space.
`schur_scan.py` then factors token transitions as multiplicity actions times a
shared group representation, preserving an exact associative affine scan.
The frozen audit finds centralizer dimension 8 versus old rank 4, an exact
capacity witness, and float64 scan/streaming error below `9e-16`; see
`FOUNDATIONAL_REVIEW_2026-08-03.md` and
`experiments/SPIN3_ISOTYPIC_SCHUR_SCAN_RESULTS.md`. This is an architectural
theorem and implementation gate, not yet a language-quality result.
