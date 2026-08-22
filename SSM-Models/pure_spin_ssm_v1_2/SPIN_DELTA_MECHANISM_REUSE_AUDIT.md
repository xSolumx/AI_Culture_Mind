# Spin-Delta mechanism reuse audit

**Date:** 2026-08-22

**Status:** read-only scientific audit completed before the next intervention

**Decision:** test credit assignment and optimizer geometry before adding model
capacity.

## Question and scope

The exact-control Spin-Delta factorial reaches between 91.41% and 100% at 16
writes while using the correct event and slot controls from optimizer step one.
The sign-changing initialization-by-batch-order interaction rules out a simple
bad-seed explanation. This audit asks which mechanisms already tested anywhere
in the repository can diagnose or repair that remaining optimization defect.

The search covers structured-memory, Spin(8), octonion, rotor, v1.2, v1.3,
compiler, and matched-baseline experiments. It does not inherit a claim merely
because two experiments use the same algebra. A mechanism is transferable only
when its intervention acts on the presently open variable.

## Present localization

The following variables are already separated experimentally:

1. **Capacity:** oracle addressing reaches 99.3--100% at 8, 16, and 32 writes.
2. **Final routing:** the causal router identifies every event and slot in every
   measured cell.
3. **Early routing noise:** training a perfect router first and freezing it does
   not remove the bad core basin.
4. **Core trajectory:** the exact-control 3x3 factorial varies by 5.86--8.54
   points at fixed batch order and by as much as 8.59 points at fixed
   initialization.

The open object is therefore the finite-step training geometry of the value
drive, recurrent state, and readout under exact controls. It is not currently
the number of slots, router receptive field, or scan expressivity.

## Repository-wide mechanism ledger

### Address and memory mechanisms

| Mechanism | Existing result | Transfer decision |
|---|---|---|
| Hard/discretized keys | Most robust row in the frozen learned-retrieval campaign; oracle delta is exact | **Keep as control.** The current exact-control gate already supplies the stronger version. |
| Jointly normalized addresses | Sinkhorn/Birkhoff normalization removed relational key collisions and passed 10/10 in the learned-address experiment | **Do not import now.** It repairs a family-level address constraint; current event/slot controls are already exact. |
| Direct versus delta slots | Exactly equal under the same hard routes on the separable two-slot task | **No update-law preference.** Delta capacity is not the present failure. |
| Hierarchical coarse-to-fine routing | Improves direct and delta retrieval under shared routing; selected-block fusion wins locally at inference | **Reserve for larger slot counts.** Two exact slots have no hierarchy problem. |
| Co-moving transported memory | Exact general invertible formula and strong local CUDA result at doubled logical state | **Not first-line.** It changes the frame/compiler problem rather than the observed optimizer basin. |
| More slots or higher rank | v1.3 supplies independent erase/write rank-r memory, but generic exceptional transport did not improve text | **Reject as immediate repair.** More capacity does not address an exact-control two-slot failure. |

### Credit assignment and identification mechanisms

| Mechanism | Existing result | Transfer decision |
|---|---|---|
| Short-to-long curriculum | Fixed-L16 endpoint learning failed 0/3; the frozen monotone curriculum passed 10/10 with the same 512,000 labels and generalized to L16384 | **Highest-priority reuse.** It changes gradient information before changing capacity. |
| Final-only octonion depth curriculum | Fixed L16 reached 3/9 structured and 0/9 dense; L2->L4->L8->L16 reached 9/9 for both families and extrapolated to L128/L1024 | **Independent replication of the same principle.** Include odd and even depths to avoid the discovered sign ambiguity. |
| Dense prefix supervision | Several finite-action experiments learn readily when all prefixes are labelled | **Useful diagnostic, not a fair final task.** It exposes where credit disappears but supplies privileged labels. |
| Local transition/adjacent-prefix supervision | Rigid-motor local identification succeeded where blind endpoint training failed | **Second-line diagnostic.** It can distinguish value encoding from long recurrent credit, but must remain an auxiliary control. |
| Shared representation prior | Learned shared action completed held-out views 10/10 while independently learned views did not | **Not applicable to exact controls.** It addresses cross-view identification, not the present core basin. |

The common causal pattern is an **information homotopy**: begin where the
target recurrent map has coherent local evidence, then lengthen composition
only after the useful map exists. This is stronger local evidence than adding
another nonlinear controller.

### Optimizer and parameterization mechanisms

| Mechanism | Existing result | Transfer decision |
|---|---|---|
| Orthogonal chart audit | Spin(8) and generic SO(8) action coordinates are connected by an exact orthogonal 28x28 map | **Capacity-equivalent control.** Chart labels cannot explain a model-quality difference. |
| SGD versus AdamW covariance | SGD preserved the exact chart relation to about 1e-16; AdamW produced coefficient error 0.0375 and logit error 0.152 after coordinatewise adaptation | **High-priority diagnostic.** The current factorial was run with AdamW and shows trajectory dependence. |
| Identity-start action coordinates | Zero action-controller initialization gives the identity transport and exact baseline pairing | **Retain.** It protects causal attribution and is not itself falsified. |
| Retention-scaled coupling | Won all three Shakespeare seeds by +0.01427 mean bpb, but every systems rescue remained below the frozen 0.90x throughput boundary | **Preserve as quality-positive research evidence.** Do not merge it into the memory-core experiment before isolating credit geometry. |
| Sector-dependent retention/spectrum | Dynamic retention was unstable; static spectrum lost 3/3 | **Closed immediate route.** Do not retry another sector schedule without a new mechanism. |
| Free multiplicity rotations and shared actions | Free recurrence mixing lost; shared-action compression lost 3/3 | **Closed.** Independent Spin heads remain justified. |

AdamW is not wrong in general. The exact audit proves a narrower point: its
coordinatewise second moments can turn an orthogonal change of Lie-algebra
basis into a different trajectory. Therefore any next optimizer experiment
must report which parameter groups use scalar, tensorwise, or coordinatewise
moments. A global optimizer swap would otherwise confound the recurrent core,
input encoder, and readout.

### Value injection and readout mechanisms

| Mechanism | Existing result | Transfer decision |
|---|---|---|
| Bounded drive | The maintained drive is `(1-retention) * write * unit_ball(raw_drive)` and is stable in long scans | **Retain as safety baseline.** The multiplication also couples value magnitude to timescale and is a plausible conditioning variable. |
| Direction-only readout | RMS normalization removes recurrent amplitude | **Known information loss.** It remains maintained because the tested replacement was too small to promote. |
| Triality-invariant amplitude readout | Won 3/3 Shakespeare seeds by +0.00661 mean bpb but missed the +0.010 threshold | **Low-risk secondary control.** Evidence is consistent but too small to be the primary repair. |
| v1.3 amplitude-aware readout | Direction plus trace/log-energy/bounded determinant was retained; projective memory was rejected because amplitude matters | **Supports preserving amplitude, not richer transport.** |
| Algebraically equal contraction reorder | Explicit cubic contraction changed float32 evaluation/gradient order and worsened quality by 0.0160 bpb | **Finite-precision warning.** Exact formulas still require paired gradient and quality tests. |
| Custom SwiGLU replacements | Neither bounded quadratic nor self-gate beat SwiGLU under matched gates | **Closed.** Channel-mixer novelty is not the current core intervention. |

The most specific untested hypothesis in this class is that tying drive size to
`1-retention` creates poor early value gradients. It is mathematically
plausible but has less direct empirical support than curriculum. If tested, it
must preserve bounded state and use a normalized, independently gated value
step rather than an unbounded residual write.

### Representation and algebra mechanisms

| Mechanism | Existing result | Transfer decision |
|---|---|---|
| Full Spin(8) triality transport | Exact algebra, compact recurrent state, compiled backward | **Retain as the model identity.** No claim of generic memory advantage follows. |
| Generic SO(8) chart | Same single-stream transition family as positive half-spin | **Use only as optimizer-coordinate control.** |
| F4/E6 or denser exceptional transport | v1.3 localization preferred identity transport; a fresh early-E6 cohort won only 2/5 with negative mean | **Closed immediate route.** Richer group action is not the missing text or retrieval mechanism. |
| Joint orbit/retraction machinery | Repairs learned families that must share a representation or permutation relation | **Do not cargo-cult.** Current Spin actions are constructed on-group and exact routing has no off-manifold family to retract. |
| Channel ensembles | Bounded gates can combine auxiliary channels, but earlier wins depend on extra boundary models and exact anchor projection | **Reserve.** Adds capacity before the current optimization defect is understood. |
| Spin(9), Clifford, and Dirac-Gram structures | Exact sensing/design and representation certificates | **No direct transfer claim.** They motivate auditable blocks, not a language or memory improvement. |

### Compiler and hardware mechanisms

| Mechanism | Existing result | Transfer decision |
|---|---|---|
| Associative affine prefix scan | Exact semantic closure and recurrent/parallel gradient parity | **Retain.** It changes evaluation order, not memory quality. |
| Staged GEMM plus factorized recurrence | Beats maximal controller fusion by reusing efficient GEMM | **Retain compiler architecture.** Do not rebuild quality experiments around maximal fusion. |
| Raw CUDA hybrid | Faster than the Triton recurrence on the recorded shape; full backward exists | **Retain for matched training.** One shape is not a universal speed claim. |
| Isotypic split and guarded reconstruction | Semantically valid, but neither rescued retention-coupled complete-step throughput | **Do not tune schedules without profiling attribution.** |
| Fused gathered-state kernel | Strong inference result; no backward/controller fusion | **Not a training-backend result.** |
| Tensor-Core materialized action | Useful only where GEMM amortization wins; scalar recurrent work remains serial | **Measure by regime.** Tensor-Core branding is not itself a mechanism. |

Hardware work begins only after a quality mechanism passes. The next core gate
should preserve the current recurrence tensor shapes so the existing raw-CUDA
path remains a valid downstream target.

## Ranked next experiments

### Gate 1: write-depth information homotopy

This is the best-supported next intervention. Compare exact-control training at
the target depth against a monotone short-to-long write curriculum using a
fresh crossed initialization-by-data-order design. Keep model parameters,
optimizer, evaluation set, total optimizer steps, and total presented tokens
explicitly accounted for. Include both odd and even write counts so no parity
symmetry can hide a second solution.

The primary question is robustness, not the best cell: does the curriculum
raise the worst 16-write result and shrink both row and column ranges? A single
100% seed is not a pass.

### Gate 2: local reconstruction credit

Run only if Gate 1 fails or remains unstable. Add a training-only per-slot value
reconstruction loss at writes, while leaving evaluation and recurrence
unchanged. Freeze its coefficient prospectively. This tests whether the core
can encode each supplied value before long-horizon query credit is applied.
Because it uses privileged synthetic state labels, success is a mechanistic
diagnosis, not a natural-language promotion.

### Gate 3: coordinate-covariant optimizer control

If instability remains, compare unchanged AdamW with a parameter-group control
whose Spin action coordinates use a scalar second moment, plus plain SGD as the
exact covariance diagnostic. Do not replace the entire optimizer: that would
confound the action chart with value/readout conditioning. Record mapped action
matrices and gradient covariance in addition to retrieval accuracy.

### Gate 4: decoupled normalized value step

Only after the credit and optimizer controls, test a bounded value injection
whose learned magnitude is not multiplied directly by `1-retention`. Require an
exact baseline embedding, a one-step gradient audit, a long-state bound, and
the fresh crossed robustness grid. This is the first architecture change in
the sequence.

## Routes intentionally not reopened

The evidence does not authorize another router, more slots, shared Spin
actions, free multiplicity angles, sector retention schedules, custom channel
mixers, F4/E6 transport, joint retraction, or a compiler rewrite as the next
Spin-Delta repair. Each either acts on a closed variable, has already failed a
matched gate, or addresses systems performance before model quality.

## Claim boundary

This audit selects falsifiers; it is not a new empirical result. Curriculum is
the strongest prior because it succeeded independently in the finite-group and
octonion programmes, not because success on those tasks guarantees Spin-Delta
success. The optimizer result proves lack of coordinate covariance for the
tested AdamW update, not that SGD will train this model better. The proposed
sequence must still be preregistered and run on fresh seeds.

## Canonical evidence

- [`SPIN_DELTA_PERFECT_CONTROL_FACTORIAL_RESULTS.md`](SPIN_DELTA_PERFECT_CONTROL_FACTORIAL_RESULTS.md)
- [`SPIN_DELTA_CAUSAL_ROUTER_RESULTS.md`](SPIN_DELTA_CAUSAL_ROUTER_RESULTS.md)
- [`SPIN_DELTA_PHASED_ROUTER_RESULTS.md`](SPIN_DELTA_PHASED_ROUTER_RESULTS.md)
- [`../experiments/ENDPOINT_CURRICULUM_RESULTS.md`](../experiments/ENDPOINT_CURRICULUM_RESULTS.md)
- [`../experiments/OCTONION_FINAL_ONLY_RESULTS.md`](../experiments/OCTONION_FINAL_ONLY_RESULTS.md)
- [`../experiments/SPIN8_SO8_OPTIMIZER_EQUIVARIANCE_RESULTS.md`](../experiments/SPIN8_SO8_OPTIMIZER_EQUIVARIANCE_RESULTS.md)
- [`../../Spin-Space-Research/docs/experiments/MATCHED_LEARNED_RETRIEVAL_RESULTS.md`](../../Spin-Space-Research/docs/experiments/MATCHED_LEARNED_RETRIEVAL_RESULTS.md)
- [`../../Spin-Space-Research/docs/experiments/SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md`](../../Spin-Space-Research/docs/experiments/SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md)
- [`../../Spin-Space-Research/docs/experiments/LARGE_SLOT_SEMANTIC_HIERARCHY_RESULTS.md`](../../Spin-Space-Research/docs/experiments/LARGE_SLOT_SEMANTIC_HIERARCHY_RESULTS.md)
- [`TRIALITY_INVARIANT_READOUT_RESULTS.md`](TRIALITY_INVARIANT_READOUT_RESULTS.md)
- [`RETENTION_SCALED_BLOCK_RESULTS.md`](RETENTION_SCALED_BLOCK_RESULTS.md)
- [`FRONTIER_TRAINING_RESULTS.md`](FRONTIER_TRAINING_RESULTS.md)
- [`../pure_f4_delta_ssm_v1_3/CONSTRAINT_AUDIT.md`](../pure_f4_delta_ssm_v1_3/CONSTRAINT_AUDIT.md)
- [`../pure_f4_delta_ssm_v1_3/V1_3_OPTIMIZATION_RESULTS.md`](../pure_f4_delta_ssm_v1_3/V1_3_OPTIMIZATION_RESULTS.md)
