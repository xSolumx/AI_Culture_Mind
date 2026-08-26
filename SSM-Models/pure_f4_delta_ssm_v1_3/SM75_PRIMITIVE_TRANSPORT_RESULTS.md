# SM75 primitive exceptional transport results

**Date:** 2026-08-26

**Hardware:** NVIDIA GeForce RTX 2070 SUPER, compute capability 7.5

**Version:** Pure Exceptional Delta SSM v1.3.2

## Bottom line

The dense exceptional action was an implementation failure, not an unavoidable
cost of F4 or E6(-26).  Exact canonical exceptional transport is now a cheap
SM75 component.  At the representative 0.68M-parameter training cell, active
sparse E6 costs only `1.099x` the exact dead-action control and the complete
checkpointed candidate is `0.955x` official Mamba-2 time and `0.959x` its peak
allocation.  Both frozen representative systems gates pass.

The boundary is equally important.  The original persistent kernel loses
occupancy when a fixed 4,096-token update is reshaped from `(B,L)=(32,128)` to
`(1,4096)`.  The exact parallel chunk scan cuts the hardest development cell
from `563.57 ms` to `172.59 ms`, but a clean all-context Mamba promotion has
not passed.  In true cached inference, active E6 is indistinguishable from its
dead-action control (`0.998x`) and uses only `1.38%` of Mamba-2's recurrent
cache.  Bulk and cache-building prefill pass against Mamba-2, while the
complete eager one-token host is still `1.70x` slower and fails decode
promotion.

This is a successful action, recurrence, memory, and prefill result.  It is not
a complete long-context-training or autoregressive-speed promotion, and the
lean inference host has no quality promotion.  The frozen contract is
[`SM75_PRIMITIVE_TRANSPORT_PREREGISTRATION.md`](SM75_PRIMITIVE_TRANSPORT_PREREGISTRATION.md).

## What changed

The old direct chart constructs

\[
\exp\!\left(\sum_a\theta_aG_a\right)
\]

as a dense 27 by 27 matrix exponential at every token.  The new chart is

\[
R(\theta)=
\exp(\theta_{F-1}G_{F-1})\cdots\exp(\theta_0G_0).
\]

Every factor is an exact one-parameter subgroup action.  The maintained Albert
generators decompose into nine independent real blocks of size three or less:

- compact F4 factors use exact Rodrigues `sin/cos` updates;
- the 26 symmetric E6(-26) factors use fixed eigensystems and real
  exponential/boost updates;
- backward reconstructs primitive states with inverse factors instead of
  storing all 52 or 78 intermediate products.

This is canonical coordinates of the second kind.  It is group-valid and
locally expressive, but it is intentionally not claimed to be a numerically
equivalent acceleration of the old exponential-of-a-sum chart.

The production recurrence is fused as

\[
S_t\leftarrow
\operatorname{retain}
\rightarrow\operatorname{erase}
\rightarrow R_t\text{ on scheduled events}
\rightarrow\operatorname{write},
\qquad y_t=q_t^\top S_t.
\]

The native CUDA forward emits reads and the final streaming state.  Its custom
reverse pass differentiates retention, write keys, erase keys, write values,
initial memory, queries, and event coordinates.  Training saves recurrent
states for reverse mode, but no tensor shaped `[B,L,27,27]` is constructed or
saved.

The 78-coordinate action head is also evaluated only at event tokens.  A
same-parameter `e6_primitive_dead` control retains that head, its backward
path, and its AdamW state while multiplying its contribution by zero.

## Correctness evidence

The current Windows portable suite passes `78` tests with seven native-only
skips.  The canonical WSL SM75 suite passes `85/85` tests.

The checked contracts include:

- reconstruction of every maintained generator from the packed blocks;
- every primitive versus `torch.matrix_exp` in float64;
- full canonical products versus a dense same-chart oracle;
- F4 norm preservation, F4/E6 Albert-cubic preservation, and reverse-product
  inversion;
- native output and value/coordinate gradients versus the dense product;
- fused recurrence reads, final states, and all seven gradient families versus
  the portable recurrence;
- no-event execution and exact chunk/stream event continuation;
- complete native model versus the portable model, including parameter
  gradients and chunked streaming.

For the fused recurrence qualification fixtures, output/final-state absolute
errors are at most approximately `6.6e-9`, and all-input gradient errors are at
most approximately `9e-8`.

## Isolated action result

The source-bound qualified action artifact is
[`artifacts/primitive_action_sm75_qualified_2026-08-26.json`](artifacts/primitive_action_sm75_qualified_2026-08-26.json).
It uses FP32, batch 2, length 16, four state copies, five warm-ups, and twenty
recorded repetitions.

| Action | Native forward | Forward speedup vs dense same-chart oracle | Native forward+backward | F+B speedup |
|---|---:|---:|---:|---:|
| F4 | `0.23 ms` | `294.7x` | `0.41 ms` | `415.4x` |
| E6(-26) | `0.10 ms` | `1035.2x` | `0.35 ms` | `628.5x` |

Maximum FP32 discrepancies from the dense same-chart oracle are:

| Action | Output | Value gradient | Coordinate gradient |
|---|---:|---:|---:|
| F4 | `7.15e-6` | `6.91e-6` | `6.48e-5` |
| E6(-26) | `8.82e-6` | `9.30e-6` | `7.06e-5` |

The speedups are deliberately against the same ordered primitive product, not
against the different direct chart.

## Historical small regression fixture

The qualified systems artifact is
[`artifacts/sparse_action_cost_sm75_qualified_2026-08-26.json`](artifacts/sparse_action_cost_sm75_qualified_2026-08-26.json).
It records three fresh processes per arm, twenty warm-ups and fifty timed
complete training steps per process.  Each step includes zeroing gradients,
forward, cross entropy, backward, gradient clipping, and AdamW.  Inputs and
targets are pregenerated on-device.

The shape is batch 4, length 128, two layers, `d_model=32`, memory width 4,
and update rank 2.  The E6 candidate has 40,858 parameters; official Mamba-2
has 40,848.

| Arm | Median complete step | p10-p90 | Maximum peak allocation |
|---|---:|---:|---:|
| E6 primitive dead budget | `33.15 ms` | `30.14-56.46 ms` | `48,991,744 B` |
| E6 primitive event, `1/32` | **`19.89 ms`** | `18.90-37.00 ms` | **`44,691,968 B`** |
| dense all-token direct E6 | `47.68 ms` | `46.27-71.88 ms` | `143,790,592 B` |
| official fused Mamba-2 | **`13.01 ms`** | `12.02-19.04 ms` | **`34,982,400 B`** |

The candidate ratios are:

- `0.600x` dead-budget time and `0.912x` dead-budget peak;
- `0.417x` dense-E6 time and `0.311x` dense-E6 peak;
- `1.530x` Mamba-2 time and `1.278x` Mamba-2 peak.

This artifact remains useful for candidate-versus-dense regression, but its
active/dead comparison is superseded.  The dead arm used the older one-sided
scan rather than the same fused recurrence launch, so it is ineligible for a
marginal-action claim.  The later representative harness repairs this with a
native `transport_enabled` flag: active and dead share the launch, saved-state
layout, controller, backward interface, parameters, and optimizer state; dead
skips only the 78 primitive factors.

Timing variation is visible on the shared Windows/WSL machine, especially in
the candidate and Mamba tails.  The verdict uses all 150 recorded samples and
remains far inside the dense-E6 thresholds, but Mamba comparison should retain
the recorded p10/p90 context.

## Representative training qualification

The decisive clean artifact is
[`artifacts/sparse_action_cost_representative_checkpointed_sm75_qualified_2026-08-26.json`](artifacts/sparse_action_cost_representative_checkpointed_sm75_qualified_2026-08-26.json).
It binds clean commit `99be9bc`, uses batch 32, length 128, four layers,
candidate width 126, Mamba-2 width 140, memory width 8, rank 2, and three fresh
processes.  The candidate has 679,866 parameters and Mamba-2 has 682,160, a
`-0.336%` residual.  Every step includes forward, loss, backward, clipping,
and AdamW.

The exact Albert Jordan product uses a custom low-memory reverse rule, and
non-reentrant per-block checkpointing recomputes activations during backward.
Neither changes forward mathematics; output, state, and gradient parity are
tested.

| Arm | Median complete step | Maximum peak allocation |
|---|---:|---:|
| exact E6 dead budget | `61.38 ms` | `222,447,104 B` |
| sparse E6 event, `1/32` | **`67.48 ms`** | **`222,710,784 B`** |
| dense all-token E6 | `233.45 ms` | `665,519,616 B` |
| official fused Mamba-2 | `70.64 ms` | `232,300,032 B` |

Candidate ratios are `1.099x` dead time / `1.001x` dead peak, `0.289x` dense
time / `0.335x` dense peak, and `0.955x` Mamba time / `0.959x` Mamba peak.
The cheap-action and Mamba-competitive representative gates both pass.

## Fixed-token long-context boundary

The clean persistent-kernel ladder is
[`artifacts/sparse_context_shape_scaling_checkpointed_sm75_qualified_2026-08-26.json`](artifacts/sparse_context_shape_scaling_checkpointed_sm75_qualified_2026-08-26.json).
It holds 4,096 target tokens per update while changing `(B,L)` through
`(32,128)`, `(16,256)`, `(8,512)`, `(4,1024)`, `(2,2048)`, and `(1,4096)`.
Active/dead time and memory gates pass in every cell, and candidate peak memory
is essentially flat.  Candidate median time nevertheless grows from `67.39`
to `563.57 ms`, an `8.36x` spread, because the persistent CUDA kernel launches
one block per batch stream and exposes too little parallelism at batch 1.
Mamba-2 remains near `43.5 ms` from length 256 onward.  The fixed-token scaling
and all-context Mamba gates fail.

The repaired backend compiles each 32-token block into an exact two-sided
affine map, parallel-scans block maps, and reconstructs token states.  Raw
scan and complete-model output/state/gradient parity pass.  A development
smoke reduced the batch-1/length-4096 cell from `563.57` to `172.59 ms`
(`3.26x`), but it is not a clean qualified ladder artifact and remains slower
than Mamba-2.  The next training systems target is scan composition and launch
fusion, not cheaper exceptional factors.

## True cached inference

The source-bound artifact is
[`artifacts/sparse_inference_2layer_sm75_qualified_2026-08-26.json`](artifacts/sparse_inference_2layer_sm75_qualified_2026-08-26.json).
It binds clean commit `d887e61`, exact SM75, Torch 2.9.0+cu128, and official
`mamba_ssm` 2.3.2.post1.  The hardware-aware pair is a two-layer
SwiGLU/vector-read E6 candidate at width 204 versus cached official Mamba-2 at
width 224: 807,652 versus 808,176 parameters (`-0.065%`).  Batch is 1, prefix
length is 4,096, and each arm runs in three fresh processes with 32 untimed
and 64 timed decode tokens.

| Measurement | Sparse E6 | Dead E6 | Official Mamba-2 | Candidate/Mamba |
|---|---:|---:|---:|---:|
| bulk no-cache prefill | `10.107 ms` | `9.840 ms` | `14.215 ms` | **`0.711x`** |
| cache-building prefill | `10.409 ms` | `9.910 ms` | `27.998 ms` | **`0.372x`** |
| cached one-token decode | `2.392 ms` | `2.397 ms` | `1.408 ms` | `1.700x` |
| recurrent cache | `6,624 B` | `6,624 B` | `481,280 B` | **`0.0138x`** |

Bulk prefill, cache-building prefill, and parameter gates pass.  Streaming
decode fails the `1.25x` Mamba threshold, even though active/dead decode is
`0.998x`.  The exceptional action is therefore not the decode bottleneck;
the unfused eager host around the recurrent state is.  The lean host is a
systems arm only and must receive its own natural-text cohort before any
quality claim transfers to it.

## Failed first integration

The first implementation kept the native primitive action but split the
sequence into four Python-launched associative scans.  In a 40-update
development smoke it was slower than dense E6 (`3.11 s` versus `1.95 s`).
That result is retained in
[`artifacts/primitive_model_sm75_smoke_2026-08-26.json`](artifacts/primitive_model_sm75_smoke_2026-08-26.json).

The lesson is concrete: a fast action kernel is insufficient.  Event action,
Delta edit, recurrent state, reads, and handwritten backward have to execute
as one hardware-aware recurrence.

## Current boundary

Supported:

- exact F4 and E6(-26) canonical-product transport;
- portable CPU/CUDA PyTorch oracle;
- native Linux/WSL FP32 SM75 action and fused recurrence;
- fixed periodic event schedule with exact streaming continuation;
- noncompact E6 log-norm compensation evaluated only at events.

Not yet supported or promoted:

- a learned hard event router;
- native FP16 input with certified FP32 accumulation;
- G2/Spin(7)/Spin(8)/Spin(9) primitive kernels through this interface;
- a clean all-context pass for the parallel chunked training scan;
- a fused complete one-token decode path competitive with Mamba-2;
- natural-text quality superiority over Mamba-2 or a quality result for the
  lean two-layer systems host;
- E7(-25) or E8(-24) carriers.

## Fresh-seed natural-text result

The new cohort uses raw Tiny Shakespeare bytes, three fresh seeds, 1,000
updates, batch 4, length 128, 32 fixed validation batches, eager FP32, AdamW,
and identical target digests within each seed.  Every arm writes an external
checkpoint whose SHA-256 is verified by the fail-closed summary.  All reports
bind clean commit `a62bc63` and the same source hashes.

The summary is
[`artifacts/sparse_quality_summary_sm75_2026-08-26.json`](artifacts/sparse_quality_summary_sm75_2026-08-26.json).

| Arm | Parameters | Mean validation bpb | Geometric mean train bytes/s | Maximum training peak |
|---|---:|---:|---:|---:|
| E6 primitive dead budget | 40,858 | `2.962925` | `14,238.3` | `48,951,296 B` |
| E6 primitive event, `1/32` | 40,858 | **`2.955639`** | **`18,584.5`** | **`44,651,520 B`** |
| dense all-token direct E6 | 40,858 | `2.949676` | `9,559.4` | `145,969,664 B` |
| official fused Mamba-2 | 40,848 | **`2.790171`** | **`36,549.4`** | **`35,990,528 B`** |

Sparse E6 beats its exact dead-budget control in seeds 2677 and 2699, loses in
2683, and improves mean bpb by `0.007286`.  Together with the systems pass,
this satisfies the frozen narrow promotion for **cheap exceptional transport
with positive local text evidence**.

The stronger conclusions fail:

- sparse E6 beats dense E6 in only one of three seeds and is worse by
  `0.005963` mean bpb;
- it loses all three seeds to Mamba-2 and is worse by `0.165468` mean bpb;
- it fails the Mamba quality gate; this historical quality architecture also
  failed its systems gate, while the later representative checkpointed
  systems arm passes at batch 32 / length 128.

The result supports sparse transport as a viable learned component.  It does
not support this complete architecture as a Mamba-2 replacement, does not
establish a learned event router, and does not revive dense all-token
exceptional transport.
