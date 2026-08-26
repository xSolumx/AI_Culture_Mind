# SM75 primitive exceptional transport preregistration

**Frozen:** 2026-08-26, after the first segmented-path development smoke and
before the fused recurrence benchmark

## Question

Can exact F4 or E6(-26) transport be made a cheap component of a trainable
Delta memory on an ordinary Turing GPU, without silently replacing the group
action by an approximation?

The old direct chart evaluates

\[
\exp\!\left(\sum_a \theta_a G_a\right)
\]

as a dense 27 by 27 matrix exponential.  The candidate uses the different,
exact canonical-coordinate chart

\[
\exp(\theta_{F-1}G_{F-1})\cdots\exp(\theta_1G_1)
\exp(\theta_0G_0).
\]

Each factor is evaluated exactly from the generator's disconnected blocks of
size at most three.  Compact factors use Rodrigues' formula; noncompact E6
factors use a fixed real eigensystem.  This is an exact product of subgroup
elements, but it is not numerically identical to the old exponential-of-a-sum
chart away from the origin.

## Fixed hardware and software boundary

- NVIDIA GeForce RTX 2070 SUPER, compute capability exactly 7.5;
- WSL2 Linux and the repository's `sm75-native-2026` Python environment;
- FP32 model state and coordinates for native CUDA measurements;
- no FlashAttention, Mamba, Triton, or other baseline that cannot execute on
  this exact device;
- eager execution unless a separately named compiled arm is qualified.

## Candidate and controls

The primary model candidate is `e6_primitive_event`: one canonical E6 action
every 32 tokens inside the same independent erase/write Delta law.  Controls
are:

1. `identity_safe`, with no value-space action;
2. `e6_safe`, with the old dense direct action at every token;
3. official fused `mamba2_official` from the already qualified local SM75
   environment.

The first regression shape is two layers, `d_model=32`, memory width 4, update
rank 2, batch 4, and sequence length 128.  E6 and Mamba-2 differ by only ten
trainable parameters.  It is a correctness and regression fixture, not the
decisive claim that the path is cheap at a representative model size.

The representative systems shape is frozen prospectively at four layers,
candidate `d_model=126`, memory width 8, update rank 2, and Mamba-2
`d_model=140`.  The candidate/dead arms have 679,866 parameters and official
Mamba-2 has 682,160, a `-0.336%` residual.  The batch-32, length-128 complete
step is the primary representative cost cell.

## Exactness gates

All must pass:

- every packed primitive reconstructs its maintained 27D generator within
  `2e-12` in float64;
- portable full-product outputs and value/coordinate gradients match the
  dense same-chart oracle within the checked float64 tolerances;
- native SM75 full-product output and gradients remain within `1e-4` maximum
  absolute error of the dense FP32 same-chart oracle;
- fused sparse-scan reads, final state, and gradients for retention, write
  key, erase key, write value, initial state, query, and event coordinates
  match the portable recurrence oracle within declared FP32 tolerances;
- whole-sequence and chunked streaming execution use the same absolute event
  schedule and agree numerically.

## Cost gates

The isolated action and the complete model receive separate verdicts.

### Isolated action

- native forward+backward median at least `3x` faster than the dense
  same-chart primitive oracle;
- no per-token dense 27 by 27 action tensor and no `torch.matrix_exp` in the
  candidate path.

### Marginal action-path promotion

At event density `1/32`, after warm-up:

- complete training-step median at most `1.25x` the identity arm;
- peak allocated CUDA memory at most `1.30x` identity;
- complete training-step median at most `0.75x` dense E6;
- peak allocated CUDA memory at most `0.60x` dense E6.

Passing this gate means exceptional transport is cheap relative to its host
memory.  It does not mean the complete model is competitive with Mamba-2.

### Fixed-token shape scaling

The representative model is additionally tested at exactly 4,096 target
tokens per update over

```text
(B,L) = (32,128), (16,256), (8,512), (4,1024), (2,2048), (1,4096).
```

The active E6, exact dead-budget, and official fused Mamba-2 arms run in fresh
processes with the same parameter-matching rule.  Dense E6 is omitted from the
long-context ladder because the regression cell has already established its
cost and materializing it at every ladder point would spend GPU time without
strengthening the cheap-path claim.

The dead-budget control is required to use the same fused recurrence launch,
saved-state layout, event schedule, coordinate controller, backward interface,
and optimizer budget as the active arm.  A native kernel flag skips only the
78 primitive state factors and emits exact zero coordinate gradients.  The
older generic-scan dead arm is not eligible for marginal-cost claims.

This is deliberately called shape scaling, not pure context scaling: batch
concurrency and the number of independently reset streams change with length.
It is still the relevant stress test for whether a persistent recurrent kernel
collapses when long contexts reduce available batch parallelism.

The fixed-token scaling gate requires all of the following:

- clean, identical source revision and exact SM75 in every cell;
- exact active/dead parameter parity and at most 1% absolute residual versus
  Mamba-2;
- no saved `[B,L,27,27]` candidate action tensor;
- active/dead time at most `1.25x` and peak allocation at most `1.30x` in
  every context;
- candidate maximum/minimum median-step spread at most `2.0x` and peak-memory
  spread at most `1.5x` across the fixed-token ladder.

The independent Mamba-competitive ladder gate additionally requires candidate
time and peak allocation at most `1.25x` Mamba-2 in every context.  Failure is
reported as an architecture limitation, not hidden by the smaller regression
fixture.

### Host-model activation repair

If the representative sparse action is fast but the complete model fails the
Mamba memory gate, the frozen first repair is not to weaken the threshold.  It
is to replace autograd's dense-einsum intermediates for the exact Albert Jordan
product with a custom reverse rule that stores only the two 27D operands and
the fixed structure tensor, then recomputes the two vector-Jacobian products
in backward.  Forward remains the same exact dense contraction.  The repair
must preserve output/gradient tests and is judged by a new clean-revision
representative rerun against the untouched official Mamba-2 arm.

If that exact reverse rule still fails the `1.25x` Mamba memory threshold, the
second frozen repair enables non-reentrant per-block activation checkpointing
for training only.  Recurrent state outputs remain explicit, each block is
recomputed during backward, and streaming/inference execution is unchanged.
The checkpointed arm must still satisfy the same complete-step time and memory
thresholds; no memory-only pass is allowed.

### Long-context recurrent repair

**Amendment frozen after the recurrent fixed-token ladder failed and before
measuring the chunked path.**

If the persistent recurrence passes at `(B,L)=(32,128)` but loses fixed-token
throughput as batch concurrency falls, the next repair is the exact separable
two-sided affine scan.  Within each 32-token block, the first 31 Delta edits
compile to a one-sided head-space affine operator.  The final edit plus event
transport forms

\[
S \mapsto A S R^T + B.
\]

These block maps compose associatively because Delta control acts on the head
axis and exceptional transport acts on the 27D value axis.  The implementation
must parallel-scan block operators and reconstruct all within-block token
states.  Dense 27 by 27 actions are permitted only at the scheduled `L/32`
events, never at every token.  The raw scan and complete model must match the
recurrent oracle in outputs, final state, and gradients before a second ladder
run.  The original fixed-token and Mamba thresholds remain unchanged.

### Mamba-competitive systems promotion

This stronger gate requires both complete training-step time and peak CUDA
allocation to be at most `1.25x` official fused Mamba-2.  It is reported
independently and cannot be inferred from the isolated-kernel result.

## Learning gates

Systems qualification precedes a full natural-text cohort.  A promoted quality
claim requires three fresh seeds, 1,000 updates per arm, identical target
digests, fixed validation batches, finite gradients, and checkpoints.  The
candidate must:

- beat its parameter-matched identity control in at least two of three seeds;
- not regress mean validation bits per byte by more than `0.02` versus that
  identity control;
- beat Mamba-2 in all three seeds before any "beats Mamba-2" claim is allowed.

The already passed hidden-coordinate tasks remain mechanism evidence only.

## Fail-closed interpretation

- A fast isolated action plus a slow complete recurrence is a failed
  integration, not a systems pass.
- A cost pass plus a text-quality loss supports cheap group transport, not a
  promoted language model.
- A language win at unmatched parameters, tokens, target streams, or hardware
  is descriptive only.
- Failing generic text does not falsify the exact group action or the existing
  symmetry-aligned learning result.
