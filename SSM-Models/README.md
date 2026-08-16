# GA-SSM experiments

> **Archive status.** This folder preserves the development lineage of the
> selective rotor and triality experiments. Dated reports and preregistrations
> state what was known when they were written; later results may supersede their
> interpretation without changing their recorded observations. The canonical
> present-day theorem ledger, correction history, and publication manuscripts
> live in the sibling `Spin8-Triality-Research` repository. The maintained
> Cl(3,0) implementation described below remains in this folder.

> **2026-08-10 programme boundary.** The sibling theorem repository now has a
> completed hierarchical memory-core campaign and an official FLA chunk-kernel
> benchmark for a co-moving transported delta recurrence. The campaign finds
> no extra ordinary overwrite capacity from triality at equal routing and does
> not test this Cl(3,0) language model. `pure_rotor_ssm` does not silently
> inherit that compiler or its approximately fivefold systems result. See the
> [canonical memory report](../Spin8-Triality-Research/docs/experiments/SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md).

> **2026-08-16 maintained-model update.** PyTorch additionally offers the
> opt-in `schur_parallel` scan path; its direct-transport, gradient, padding,
> cache, and CUDA parity checks pass, but it has no production-kernel claim.
> The direct maintained-model comparison runner is
> [`benchmark_pure_rotor_vs_mamba2.py`](benchmark_pure_rotor_vs_mamba2.py).
> Its checked-in one-step artifact is a smoke test, not a quality or systems
> result; this Windows checkout uses Transformers' unfused Mamba-2 path.
> The complementary, preregistered A5 mechanism screen is
> [`benchmark_pure_rotor_a5.py`](benchmark_pure_rotor_a5.py): it compares the
> canonical model, identity transport, and Mamba-2 on a missing-bigram
> non-commutative prefix task. Its completed three-seed 200-update screen
> finds a non-replicated seed-0 short-pair event and chance-level long-word
> behavior, so it is a negative/diagnostic pilot rather than a model claim;
> see [`experiments/PURE_ROTOR_A5_MAMBA2_PILOT200_RESULTS.md`](experiments/PURE_ROTOR_A5_MAMBA2_PILOT200_RESULTS.md).
> The follow-up three-seed 1,000-update screen has a variable short-pair
> signal but still fails L64/L128 retention; see
> [`experiments/PURE_ROTOR_A5_MAMBA2_BUDGET1000_RESULTS.md`](experiments/PURE_ROTOR_A5_MAMBA2_BUDGET1000_RESULTS.md).
> Its execution status, including the completed one-step smoke and invalidated
> full-batch Mamba fallback attempt, is recorded in
> [`experiments/PURE_ROTOR_A5_MAMBA2_EXECUTION_LOG.md`](experiments/PURE_ROTOR_A5_MAMBA2_EXECUTION_LOG.md).

> **2026-08-16 maintained Pure Spin(8) v1.0.**
> [`pure_spin8_ssm/`](pure_spin8_ssm/) is a second maintained family with a
> separate checkpoint schema; it does not rewrite or supersede Pure Rotor
> v2.1. One shared 28-coordinate controller drives the vector, positive-chiral,
> and negative-chiral eight-real actions. The resulting 24-scalar triality
> cache distinguishes every Spin(8) center signature and uses an associative,
> work-efficient affine scan with recurrent streaming. Its model, gradient,
> stability, mask, cache, CUDA, exact-center factorized chart, and checkpoint
> contracts pass. In a frozen three-seed triality-transport cohort, it reaches
> L128 MSE `5.81e-5`--`6.68e-5` and 100% central-sign classification; unfused
> Transformers Mamba-2 scores `0.132`--`0.135` and 50%. This approximately
> 2,000x task-specific MSE gap is an algebra-matched synthetic result, not a
> broad claim that the model beats Mamba. See
> [`PURE_SPIN8_VS_MAMBA2_RESULTS.md`](experiments/PURE_SPIN8_VS_MAMBA2_RESULTS.md)
> and the [`v1.0 contract`](pure_spin8_ssm/CONTRACT.md).

> **2026-08-16 center-sensitive `2.A5` result.** The frozen three-seed runner
> [`benchmark_pure_rotor_2a5.py`](benchmark_pure_rotor_2a5.py) holds the
> projected A5 trajectory fixed while paired binary targets differ by the
> central element. The explicit Spin quaternion product scan passes the
> preregistered center-margin gate at L16/L64/L128 in all seeds and reaches
> `62.20 ± 2.36%` mean exact 120-state accuracy at L128. Pure Rotor v2.1,
> identity, and Transformers Mamba-2 reach `2.29%`, `1.89%`, and `4.56%`
> respectively and are near chance on center metrics. This is a bounded
> mechanism result, not a general sequence-model theorem. See
> [`PURE_ROTOR_2A5_CENTER_PILOT300_RESULTS.md`](experiments/PURE_ROTOR_2A5_CENTER_PILOT300_RESULTS.md).
> The successful primitive is now reusable in
> [`pure_rotor_ssm/spin_scan.py`](pure_rotor_ssm/spin_scan.py), but remains an
> explicit experiment rather than a silent change to v2.1.
> A no-retraining exploratory follow-up selects the shortest cancellation-
> reduced identity/center word pair absent from all three training schedules.
> The same Spin checkpoints retain 100% center margin through L128 and 59.68%
> mean exact L128 accuracy; see
> [`PURE_ROTOR_2A5_UNSEEN_RELATION_RESULTS.md`](experiments/PURE_ROTOR_2A5_UNSEEN_RELATION_RESULTS.md).

> **2026-08-16 multi-relation and external-baseline result.** The separately
> frozen runner
> [`benchmark_spin_multirelation_2a5.py`](benchmark_spin_multirelation_2a5.py)
> excludes `a^2`, `b^3`, and `(ab)^5` simultaneously, adds an identity token,
> and repeats identical token schedules under three inner-conjugate generating
> sets. It compares the existing four candidates with the tested unfused
> [`delta_product_reference.py`](delta_product_reference.py) and an exact
> regular-action PD oracle. Spin's worst registered early-L64/L128 central
> margin is 99.50%, and it uniquely wins exact accuracy in all 18 long splits;
> every other learned candidate fails the center gate. DeltaProduct has the
> strongest non-Spin long exact accuracy but chance-like center retention.
> This is a one-initialization coordinate pilot, not replicated multi-seed or
> fused-kernel evidence. See
> [`SPIN_2A5_MULTIRELATION_RESULTS.md`](experiments/SPIN_2A5_MULTIRELATION_RESULTS.md).

> **2026-08-16 rigid-motor implementation gate.** The separate experimental
> [`pure_rotor_ssm/motor_scan.py`](pure_rotor_ssm/motor_scan.py) lifts the
> sign-sensitive quaternion product to unit dual quaternions, an eight-scalar
> double cover of `SE(3)`. Exact-structure tests and the frozen numerical audit
> establish homogeneous-matrix equivalence, central-state/physical-action
> separation, parallel/recurrent/cache/gradient parity, and valid Study
> constraints through L4096. Translation remains unbounded. On the recorded
> eager CUDA diagnostic, 4 by 4 matrix parallel scan is about 8.6 times faster,
> so no motor-kernel advantage is claimed. See
> [`MOTOR_PATH_DEVELOPMENT_RESULTS.md`](experiments/MOTOR_PATH_DEVELOPMENT_RESULTS.md).

> **2026-08-16 learned rigid-motion and identification result.** The frozen
> [`benchmark_spin_motor_rigid_2a5.py`](benchmark_spin_motor_rigid_2a5.py)
> crosses the three held-out central `2.A5` relations with non-commuting
> body-frame translations and compares parameter-near quaternion, motor,
> Transformers Mamba-2, and DeltaProduct candidates. All four 300-step learned
> readout models fail the strict long-context joint-pose gate; the result is
> negative, including for the motor classifier. A matched 49-parameter direct-
> product state retains the center sign but fails translation. The follow-up
> [`identify_spin_motor_rigid_2a5.py`](identify_spin_motor_rigid_2a5.py)
> instead identifies each token motor from legal supervised prefix differences,
> using no evaluation relation. That 8-scalar state reaches 100% joint signed
> pose and paired double-cover pose accuracy in all 9 coordinate-by-seed runs
> and all 162 splits through L128. This establishes finite deterministic
> identifiability under every-prefix pose supervision, not end-to-end model
> superiority or a continuous-data theorem. A frozen 4-tier by 5-seed noise
> audit then keeps 100% joint/paired accuracy through the 5-degree/0.05 tier;
> at 15 degrees/0.15, all five runs retain the center sign but fall to a worst
> joint/paired accuracy of 46.875%. The noise preserves the signed quaternion,
> so this does not infer an unobserved lift from `SO(3)` poses. See
> [`SPIN_MOTOR_RIGID_2A5_RESULTS.md`](experiments/SPIN_MOTOR_RIGID_2A5_RESULTS.md).

> **2026-08-16 associative octonion-operator lift.** The experimental
> [`pure_rotor_ssm/octonion_operator_scan.py`](pure_rotor_ssm/octonion_operator_scan.py)
> does not scan raw nonassociative octonions. It maps each unit octonion to an
> `8 by 8` multiplication operator, composes those maps with an ordered work-
> efficient tree, and uses the raw parenthesized product only for compact
> eight-scalar streaming. The norm-2 associator remains explicit, and the
> exact seven-generator/21-commutator determinant `-2^49` certifies full
> `so(8)` Lie closure. The bounded layer passes its algebra, scan, gradient,
> cache, CUDA, and L4096 gates. The optional WSL/Triton backend adds a fused
> differentiable recurrence: its L4096 forward median is 1.805 ms versus
> 8.951 ms for the work-efficient operator path, and its L1024 forward/backward
> median is 1.691 ms versus 11.628 ms. This is outside Pure Rotor v2.1 and has
> one separately frozen synthetic task result: at L128 its 72-parameter
> algebra-matched encoder reaches MSE `1.85e-12`, while the invalid collapsed
> octonion, unfused DeltaProduct, and unfused Transformers Mamba-2 controls
> score `0.216`, `0.124`, and `0.125`. This is one-seed coordinate-aligned
> realizability, not generic model superiority. In the three-Haar-basis
> successor, a 28-parameter learned `SO(8)` gauge reaches L128 MSE
> `1.54e-9`--`8.74e-8` and satisfies the recovered `G2` automorphism equation
> to at most `2.17e-4`. The overall frozen cohort remains failed because the
> dense AdamW control misses its registered gate and one oracle crosses its
> strict float32 threshold; post-protocol least squares proves the dense map is
> realizable. See
> [`OCTONION_OPERATOR_SCAN_RESULTS.md`](experiments/OCTONION_OPERATOR_SCAN_RESULTS.md).
> A harder final-only L16 cohort initially recovers only 3/9 structured laws.
> The frozen L2/L4/L8/L16 curriculum recovers 9/9 structured and 9/9 dense
> laws, but its even-only targets identify `G2 union -G2`: an odd L17 audit
> finds four positive- and five negative-coset gauges and verifies the predicted
> sign in all runs. The original unsigned-`G2` audit is retained as failed.
> See
> [`OCTONION_FINAL_ONLY_RESULTS.md`](experiments/OCTONION_FINAL_ONLY_RESULTS.md).

> **2026-08-16 Spin--Dirac ladder.** The separate algebraic gate
> [`spin_dirac_a5_ladder.py`](spin_dirac_a5_ladder.py) now extends the maintained
> Spin(8) gamma system through Spin(9)--Spin(12), while restricting every rung
> to the same embedded Spin(3). It distinguishes vector `A5` from its
> 120-element binary spin lift `2.A5`, verifies the central sign, and records
> the exceptional Spin(8) triality dimension match without extending a
> triality claim to higher dimensions. See
> [`SPIN_DIRAC_A5_LADDER_RESULTS.md`](experiments/SPIN_DIRAC_A5_LADDER_RESULTS.md).
> This is an exact-matrix/numerical-algebra gate, not a trained SSM or model
> benchmark.

> **2026-08-16 exact rigidity upgrade.** The companion
> [`spin_dirac_a5_rigidity.py`](spin_dirac_a5_rigidity.py) replaces the
> float64 binary-group and tangent-rank evidence with exact `Q(sqrt(5))`
> certificates. It enumerates `2.A5`/`A5` as 120/60 elements exactly and proves
> that the `(2,3,5)` relation kernel equals infinitesimal conjugacy for every
> listed rung, hence `H1=0` at the fixed embedding. See
> [`SPIN_DIRAC_A5_RIGIDITY_RESULTS.md`](experiments/SPIN_DIRAC_A5_RIGIDITY_RESULTS.md).
> Global classification and ML consequences remain open. The degree-two status
> from this rigidity stage is superseded by the exact closure immediately below.

> **2026-08-16 cohomology closure.** The exact table-level gate
> [`spin_dirac_a5_cohomology.py`](spin_dirac_a5_cohomology.py) verifies the
> universal low-degree averaging homotopy at all 120 degree-one and 14,400
> degree-two output tuples. It proves `H1=H2=0` for every
> `Q(sqrt(5))`-linear `2.A5` module and shows that the raw presentation
> cokernel is relator-syzygy redundancy, not `H2`. See
> [`SPIN_DIRAC_A5_COHOMOLOGY_RESULTS.md`](experiments/SPIN_DIRAC_A5_COHOMOLOGY_RESULTS.md).
> The global representation-component status from this stage is superseded by
> the exact atlas immediately below.

> **2026-08-16 global component atlas.**
> [`spin_dirac_a5_components.py`](spin_dirac_a5_components.py) reconstructs the
> complete nine-character table from the exact quaternion group, verifies all
> 81 tensor products and Frobenius--Schur types, then enumerates every
> `2.A5 -> Spin(n)` conjugacy component for `n=3,8,9,10,11,12`. The standard
> universal-cover theorem for `2.A5` is explicitly separated from the exact
> table certificate. The result finds 3, 32, 32, 42, 59, and 98 Spin components
> respectively; see
> [`SPIN_DIRAC_A5_COMPONENT_ATLAS_RESULTS.md`](experiments/SPIN_DIRAC_A5_COMPONENT_ATLAS_RESULTS.md).
> This closes the stated global compact-real classification gate through
> dimension 12, but makes no model-quality or triality-beyond-Spin(8) claim.

> **2026-08-16 exact spinor branching.**
> [`spin_dirac_a5_spinors.py`](spin_dirac_a5_spinors.py) derives the spinor or
> half-spinor restriction for all 245 orthogonal types in the global atlas.
> Quaternionic base blocks come from explicit SU(2) weight parity, and every
> result is independently checked against Newton-reconstructed exterior
> characters. All 21 orientation splits exchange distinct chiral characters;
> see
> [`SPIN_DIRAC_A5_SPINOR_BRANCHING_RESULTS.md`](experiments/SPIN_DIRAC_A5_SPINOR_BRANCHING_RESULTS.md).
> “Invariant spinor” is representation-theoretic here, not a geometric Dirac
> zero-mode claim.

`pure_rotor_ssm/` contains the canonical implementation in matched JAX/Flax and
PyTorch backends: algebra, complete Spin(3)-isotypic maps, bounded selective
transitions, associative/recurrent scans, equivariant dropout, fixed-state
streaming, and the decoder. The separate PyTorch-only `spin_scan.py` is an
experimental sign-sensitive composition layer motivated by the completed
`2.A5` screen; it is not part of the v2.1 model contract. The package contains
nothing related to datasets, optimization, checkpoints, profiling, or result
reporting. The canonical recurrence's hard bound and exact scope are in
[pure_rotor_ssm/CONTRACT.md](pure_rotor_ssm/CONTRACT.md).

`ga_ssm.py` is the JAX experiment/training shell and `rotor_ssm_torch.py` is an
import-compatibility shell. Both expose the canonical pure model names. The
remaining trainer and historical classes stay outside the pure package.

The pure architecture is version **2.1.0**. Version 2 introduced the bounded
write law and complete isotypic parameterization; v2.1 expands the default
rotor chart to the full open 180-degree physical range after every v2.0 layer
saturated the old 90-degree cap. The preserved first approximately-4-GiB v2.0
checkpoint and deliberately narrow single-run report are in
[experiments/PURE_ROTOR_SSM_V2_RESULTS.md](experiments/PURE_ROTOR_SSM_V2_RESULTS.md).

The completed preregistered v2.1 ladder gives a qualified answer. At equal
state size and at the nearest live-parameter match, selective rotors beat a
retrained identity transition in all five seeds, and identity-clamping or
time-shuffling trained rotor actions causes substantial confirmation damage.
However, quaternion left action and even commuting complex phases predict
better; identity wins decisively when widths are matched to measured eager-CUDA
time; and rotors do not improve associative recall or Q8 extrapolation. Thus
the current evidence is **prediction benefit versus identity, but no superior
memory or compute efficiency and no Cl(3,0)-specific advantage**. Full paired
tables, confidence intervals, system curves, and limitations are in
[`experiments/PURE_V2_1_TRANSPORT_ABLATION_RESULTS.md`](experiments/PURE_V2_1_TRANSPORT_ABLATION_RESULTS.md).

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

The bullets below record the broader research lineage. For present-day memory
status, the canonical conclusion is: direct and triality-coded slots tie under
the same router; hierarchical routing improves both direct and delta retrieval;
and Spin(8)'s surviving advantage is relational cross-view completion, not a
same-state storage theorem.

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

The trainer accepts `--checkpoint-dir` and `--variant selective_rotor` for a
versioned reloadable checkpoint. Benchmark one with
`benchmark_pure_rotor_ssm.py --checkpoint <path> --output <report.json>`.

The independent model-quality comparison is intentionally separate from that
single-model systems benchmark:

```powershell
python -m pytest SSM-Models\test_pure_rotor_vs_mamba2_benchmark.py -q
python SSM-Models\benchmark_pure_rotor_vs_mamba2.py --steps 20 `
  --validation-batches 1 --seeds 0 --offline `
  --output SSM-Models\experiments\artifacts\pure_rotor_vs_mamba2_smoke.json
```

For a result rather than a smoke test, use independent seeds and retain the
report's state, raw/effective-parameter, decoder-tying, Mamba-backend, memory,
and data-hash fields.

The maintained v2-native falsification ladder is frozen in
[`experiments/PURE_V2_1_TRANSPORT_ABLATION_PREREGISTRATION.md`](experiments/PURE_V2_1_TRANSPORT_ABLATION_PREREGISTRATION.md).
It exchanges only the state transport among identity, real diagonal, four
commuting complex phases, quaternion left action, selective rotor, fixed
rotor, and generic SO(8), then runs state-, effective-parameter-, and measured
CUDA-matched prediction views plus associative-recall and Q8 memory tasks.
Every family/seed is an atomic resumable shard:

```powershell
python SSM-Models\run_transport_ablation_v2.py --phase all
```

The checked-in aggregate is
`experiments/pure_v2.1.0_transport_ablation.json`; regenerate the Markdown
tables with `summarize_transport_ablation_v2.py`.

`transport_ablation_v2.py` deliberately stays outside the pure package: its
non-rotor families are experimental falsification controls, not parts of the
canonical model API.

The checked-in `experiments/final_seed*_300.json` reports and
`experiments/final_summary.json` record the final three-seed local-GPU result
for the superseded pre-pure architecture. They are preserved evidence, not a
quality evaluation of the rewritten model.

## Recurrence-family harness

> **Legacy protocol.** This harness uses the superseded `sqrt(1-d^2)` write
> law, older projections, and tokenwise recurrence. Preserve it for its dated
> group-action artifacts; use `run_transport_ablation_v2.py` for claims about
> the v2 bounded-write architecture.

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
