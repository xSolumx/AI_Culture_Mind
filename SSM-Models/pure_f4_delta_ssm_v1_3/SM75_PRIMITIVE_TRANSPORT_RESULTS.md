# SM75 primitive exceptional transport results

**Date:** 2026-08-26

**Hardware:** NVIDIA GeForce RTX 2070 SUPER, compute capability 7.5

**Version:** Pure Exceptional Delta SSM v1.3.2

## Bottom line

The dense exceptional action was an implementation failure, not an unavoidable
cost of F4 or E6(-26).  Exact canonical exceptional transport is now cheap
enough to be a real SM75 component.

At the frozen `1/32` event density, the fused E6 model is `2.36x` faster than
the old all-token dense E6 model and uses only `31.1%` of its peak CUDA
allocation.  It passes the separate cheap-action-path gate.  The complete
model is still `1.41x` slower than official fused Mamba-2 and uses `1.28x` its
peak allocation, so the stronger Mamba-competitive systems gate fails.

This is a successful backend and recurrence result, not yet a natural-text
quality promotion.  The frozen contract is
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

The Windows portable suite passes `66` tests with six native-only skips.  The
canonical WSL SM75 suite passes `72/72` tests.

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
| F4 | `0.12 ms` | `535.0x` | `0.34 ms` | `428.6x` |
| E6(-26) | `0.10 ms` | `953.9x` | `0.34 ms` | `654.4x` |

Maximum FP32 discrepancies from the dense same-chart oracle are:

| Action | Output | Value gradient | Coordinate gradient |
|---|---:|---:|---:|
| F4 | `7.15e-6` | `6.91e-6` | `6.48e-5` |
| E6(-26) | `8.82e-6` | `9.30e-6` | `7.06e-5` |

The speedups are deliberately against the same ordered primitive product, not
against the different direct chart.

## Complete-step SM75 result

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
| E6 primitive dead budget | `31.23 ms` | `30.24-44.37 ms` | `48,991,744 B` |
| E6 primitive event, `1/32` | **`19.77 ms`** | `18.92-36.67 ms` | **`44,691,968 B`** |
| dense all-token direct E6 | `46.60 ms` | `45.69-49.15 ms` | `143,790,592 B` |
| official fused Mamba-2 | **`14.04 ms`** | `11.92-21.23 ms` | **`34,982,400 B`** |

The candidate ratios are:

- `0.633x` dead-budget time and `0.912x` dead-budget peak;
- `0.424x` dense-E6 time and `0.311x` dense-E6 peak;
- `1.408x` Mamba-2 time and `1.278x` Mamba-2 peak.

The primitive arm being faster than its dead-budget control is not evidence
that an action has negative cost.  The fused kernel replaces the generic
Hillis-Steele host scan as well as adding sparse transport; the dead arm keeps
the older one-sided scan so it can isolate the complete active path.

Timing variation is visible on the shared Windows/WSL machine, especially in
the candidate and Mamba tails.  The verdict uses all 150 recorded samples and
remains far inside the dense-E6 thresholds, but Mamba comparison should retain
the recorded p10/p90 context.

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
- representative 1-2M parameter and L4096 cost matrices;
- natural-text quality superiority over the dead-budget control or Mamba-2;
- E7(-25) or E8(-24) carriers.

The immediate quality question is now clean: with cost no longer dominated by
dense matrix exponentials, does sparse exceptional transport learn anything
useful on ordinary text?  That requires the separately summarized fresh-seed
cohort; no answer is inferred from these systems measurements.
