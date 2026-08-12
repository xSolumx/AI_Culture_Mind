# Matched learned-retrieval campaign preregistration

- **Date frozen:** 2026-08-10, before implementation or result inspection
- **Programme:** Triality memory and Intertwiner SchurScans
- **Status:** prospective two-task protocol

## Why the campaign is factorial

One all-in-one leaderboard would confound three different resources:

1. the recurrent memory update;
2. address or key inference;
3. completion of a partially observed cross-view action family.

The maintained gauge theorem already proves that direct and triality-bound
hard slots are quality-equivalent when unit keys and the correct actions are
supplied. A benchmark that silently removes an action oracle only from the
direct row would measure a representation prior, not memory capacity.

The campaign therefore has two separately adjudicated tasks. A result from one
task may not be used to claim victory on the other.

## Task A: learned-address overwrite and recall

### Question

Under an identical 64-scalar recurrent-state budget, which update law best
learns continuous addresses and retains values under overwrites, alias shift,
long sequences, and noncommuting supplied value transport?

### World and splits

The task reuses the maintained continuous-alias world:

- eight hidden semantic classes in 24 dimensions;
- eight-dimensional stored values;
- training radii `0.05`, `0.10`, and `0.15`;
- validation radius `0.22`;
- untouched test radii `0.35`, `0.55`, and `0.75`;
- disjoint RNG streams for write aliases, query aliases, values, actions, and
  event schedules;
- no logical key ID exposed to a learned encoder.

The new radii are stress tests, not training curricula. The overwrite cohort
is the full factorial crossing:

- overwrite depths `1`, `2`, `4`, `8`, and `16`;
- test radii `0.35`, `0.55`, and `0.75`;
- no-transport and supplied noncommuting-transport streams;
- clean encoded keys/routes and explicit post-encoding perturbations with
  norms `0.00`, `0.02`, `0.05`, `0.10`, and `0.20`.

The separate length frontier uses lengths `64`, `256`, `1024`, `2048`, and
`4096`, test radius `0.75`, perturbation norms `0.00`, `0.10`, and `0.20`, and
both transport conditions. Its first eight tokens initialize every key; later
tokens overwrite four hot keys while leaving four cold keys untouched. Hot,
cold, and combined query cohorts are reported separately. Batch size is eight,
and positions before token 32 are warm-up, giving exactly 256 reported queries
at length 64 and more thereafter.

All competing rows receive the same event stream, aliases, values, transport
tokens, and query mask for a seed/cell. At least 256 queries are required in
every reported cell.

### Principal memory rows

1. `direct_slot_joint`: jointly balanced categorical routing and eight direct
   eight-vector slots;
2. `triality_slot_joint`: the identical learned route, slot count, and values,
   with triality bind/vector-transport/unbind;
3. `delta_chunk_joint`: learned normalized write/query keys and the standard
   corrective delta update on an `8 x 8` fast-weight state;
4. `delta_chunk_oracle`: exact orthogonal semantic keys, retained as an update
   and capacity oracle;
5. `fast_weight_joint`: additive outer-product writes using the same learned
   key encoder, retained as an overwrite-negative control.

The standard delta update is

\[
S_t=(I-\beta_t k_tk_t^\mathsf{T})S_{t-1}
    +\beta_t k_tv_t^\mathsf{T},
\qquad y_t=q_t^\mathsf{T}S_t.
\]

The primary delta gate is `beta=1` because every write is a commanded
overwrite. A learned gate may be added only as a labelled secondary row; it
may not replace the frozen primary result.

### Chunkwise delta implementation gate

The local implementation must expose:

- a sequential recurrent reference;
- an ordered work-efficient scan of the exact factored affine transition;
- a two-level chunkwise scan with chunk sizes `16`, `32`, and `64`;
- full-prefix and final-state outputs;
- gradients with respect to keys, values, gates, transports, and initial
  state.

All implementations must agree in float64 on irregular lengths and
noncommuting value transports before a retrieval or timing result is read.
One-hot keys must reproduce exact overwrite, and additive fast weights must
retain the expected stale-value counterexample.

This is a faithful mathematical DeltaNet recurrence and a real two-level
chunked execution. It is not described as the production compact-WY Triton
kernel. The official Flash Linear Attention implementation uses a fused
chunkwise kernel; it is an external systems tier, not something an eager
reference can impersonate:
[DeltaNet layer](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/delta_net.py).

The current Windows environment has neither `fla` nor Triton installed. The
absence is recorded before timing. A fused comparison may be run later on a
supported environment without changing Task A quality gates.

## Task B: partial cross-view action and retrieval

### Question

When the learner receives rank-deficient observations of several
representation views, does a shared triality family complete held-out action
directions and preserve retrieval better than independently fitted or generic
bilinear alternatives?

### Frozen foundation

This task extends, rather than rewrites, two maintained controls:

- the restricted-support equivariant-identification gate;
- the blind-action continuous-alias gate with its matched direct path and
  binding-bypass audit.

Training observes only the already frozen partial vector/positive action
columns and rank-two negative endpoints. Evaluation uses the complete
negative complement, fresh aliases, full-dimensional values, noncommuting
action words, and the dense length sweep.

### Principal structural rows

1. shared triality binding and shared triality direct paths;
2. independent-action binding and matched independent direct paths;
3. direct and delta memories supplied with the correct negative action as
   capacity ceilings;
4. delta memory using the independently fitted negative action;
5. the known one-dimensional triality intertwiner;
6. an unrestricted generic bilinear tensor fitted to identical endpoints;
7. the explicitly group-augmented generic tensor, labelled as receiving more
   examples;
8. the SO(3) cross-product control.

The generic bilinear SchurScan belongs here because this task exercises its
triangular bilinear drive. It is not relabelled as an addressed overwrite
memory in Task A.

Binding paths that never consume the learned negative action remain bypass
diagnostics. Only matched direct or delta paths can establish behavioral
completion of that action.

## Matching ladders

No single width is claimed to match state, trainable parameters, scalar FLOPs,
and measured CUDA time simultaneously. These are separate ladders:

### State matched

- direct slots: `8 x 8 = 64` recurrent scalars;
- triality slots: `8 x 8 = 64`;
- delta/fast weight: `8 x 8 = 64`.

The compact three-stream Intertwiner SchurScan has its native streaming size
reported separately. It is not called state matched unless multiplicities are
explicitly adjusted before the cohort.

### Encoder-parameter matched

The primary slot and delta address encoders each use two `8 x 24` linear maps,
for 384 learned scalars. Fixed triality tensors and supplied action generators
are reported as priors rather than hidden trainable parameters.

Sample efficiency is a frozen four-point ladder at `25`, `75`, `150`, and
`300` optimizer steps per curriculum stage. Each budget is trained from scratch
and evaluated at overwrite depth `4`, radius `0.75`, perturbation norm `0.10`,
and both transport conditions. The 300-step policies are reused for the full
grid. The ladder is descriptive; no test-cell checkpoint selection is allowed.

Shared and independent action-family parameter counts remain visible. The
independent action control is intentionally parameter-richer; this is a
stronger falsifier, not a matched-parameter row.

### Training-data matched

Comparable learned rows receive identical examples, optimizer steps, batch
sizes, curricula, and validation policy. Group augmentation is reported as a
separate data-rich row.

### Measured-CUDA matched

Quality is first reported at state-matched width. A separate calibration
ladder chooses the nearest integer width for parameter count and measured
forward/backward CUDA time. No quality claim may substitute theoretical FLOPs
for measured time.

## Seeds, development, and freezing

- development smoke: seed `101`, excluded from every reliability count;
- frozen cohort: seeds `0` through `9`;
- no per-seed restarts;
- no checkpoint selection on test cells;
- a failed implementation smoke may correct code, but any protocol or
  threshold change requires a dated addendum before rerunning the cohort;
- raw failure cohorts remain in the artifact.

## Primary metrics

Report separately:

- mean and minimum retrieval cosine;
- mean and maximum relative squared error;
- query count;
- address collision and clean/OOD agreement;
- within-class key spread and between-class separation;
- overwrite-depth degradation;
- predicted and observed failure length;
- parallel/recurrent/chunkwise discrepancy;
- recurrent-state scalars and trainable parameters;
- optimizer examples and steps;
- forward latency, forward/backward latency, peak allocation, and tokens/s.

Aggregate means never replace per-seed and worst-cell values.

## Decision rules

### Task A

- The maintained gauge theorem predicts an exact direct/triality tie for hard
  one-hot routes. An explicit hard-route depth-16 gate is therefore run for
  every seed and transport condition.
- Explicit route perturbation produces soft cross-slot mixtures, outside the
  hard-slot gauge theorem. A triality/direct difference there is labelled a
  binding-under-soft-routing effect, not an intrinsic memory-update advantage.
- A hard-routing robustness advantage may be reported if direct and triality
  both beat learned delta keys across at least 8/10 paired seeds under frozen
  perturbation cells, while the oracle delta row passes.
- Delta failure is attributed to the update only if the oracle delta row also
  fails. Otherwise key inference and update capacity remain separated.

### Task B

- A triality-specific representation-prior result requires the shared family
  to beat matched independent direct and independent delta paths on held-out
  action directions in at least 8/10 seeds.
- The independent rows must fit all supplied observations before their
  extrapolation failure is informative.
- SO(3) success prevents a generic equivariant-prior effect from being called
  exceptional triality.
- Group-augmented generic success is not a sample-matched loss for the
  structured row.

No competitor is prerequired to lose. Negative, tied, or mixed outcomes are
complete results.

## Claim boundary

The campaign remains synthetic and controlled. It cannot establish a
language-model advantage, universal memory superiority, or production-kernel
throughput. Gated DeltaNet-2 separates erase and write gates and derives a
chunkwise WY algorithm; Erase-then-Delta adds an independent erase address;
DeltaProduct applies several generalized Householder updates per token. Those
are stronger external architecture tiers than a scalar delta rule:

- [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791)
- [Erase-then-Delta Attention](https://arxiv.org/abs/2606.26560)
- [DeltaProduct](https://arxiv.org/abs/2502.10297)

This campaign first establishes a trustworthy local recurrence and causal
task decomposition. Full fused baselines remain mandatory before any
field-wide systems claim.

The unrestricted Dirac--Gram inequality is independent and is not a
prerequisite for this campaign.
