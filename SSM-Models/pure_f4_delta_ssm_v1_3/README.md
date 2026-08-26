# Pure Exceptional Delta SSM v1.3.2

**Research author:** Hayden Austin

This folder is the isolated correctness and evidence boundary for the v1.3
frontier. The model is not a renamed v1.2 and does not inherit its language-
quality or throughput claims.

The implemented 27-dimensional state hierarchy is

\[
G_2\subset\mathrm{Spin}(7)\subset\mathrm{Spin}(8)\subset
\mathrm{Spin}(9)\subset F_4\subset E_{6(-26)},
\]

all acting in one 27-dimensional Albert-algebra carrier. The trace-free
26-dimensional irreducible (F_4) module remains available and tested, but it
is no longer the model ceiling. The built-in hierarchy reaches the full
78-generator (E_{6(-26)}) action; compact (F_4), Spin(9), Spin(8), Spin(7),
G2, identity, arbitrary schedules, and caller-supplied generator banks are
controls. E7(-25) and E8(-24) require new 56D Freudenthal and 248D adjoint
carriers; they are not extra generators on the Albert state. See
[`EXCEPTIONAL_LADDER.md`](EXCEPTIONAL_LADDER.md).

## What is implemented

- the real Albert algebra (H_3(\mathbb O)), its Jordan product, trace, and
  cubic determinant;
- 52 derivations spanning (\mathfrak f_4), plus the 26 traceless left-
  multiplication directions spanning (\mathfrak e_{6(-26)});
- executable G2, Spin(7), Spin(8), and Spin(9) stabilizer restrictions inside
  the same representation;
- direct, polar, and Cartan (KAK) E6 action charts with complete autograd;
- exact canonical-coordinate F4/E6 products built from fixed blocks of size at
  most three, with no dense matrix exponential;
- a native FP32 SM75 canonical-action kernel and fused sparse-event Delta
  recurrence with handwritten backward for all recurrence inputs;
- an event-only exceptional coordinate head and a parameter/optimizer-matched
  dead-action control;
- one or more ordered Lie exponentials per token;
- a configurable rank-(r) Delta memory update with tied safe default and
  independent erase/write controls;
- an independent bounded write-strength gate, normalized queries, long-memory
  retention initialization, and noncompact E6 norm compensation;
- an associative two-sided affine prefix compiler for general invertible value
  actions;
- a bitwise-matched one-sided affine specialization that removes the entire
  27 by 27 value action when identity transport is selected;
- recurrent, direct rank-(r), and logarithmic-depth semantic scans with output,
  state, and gradient parity;
- bounded-memory direct recurrence for long sequences and exact one-token
  streaming;
- complete streaming state, including the causal-convolution cache;
- optional SwiGLU, Albert-Jordan, or no pointwise channel mixer;
- an Albert-invariant readout that keeps the 27D direction while restoring
  trace, energy, and cubic scale information lost by plain RMS normalization;
- optional Riemann-sphere and PGL/Möbius utilities for router controls;
- a custom generator-bank interface that removes the exceptional hierarchy as
  a hard implementation ceiling.

## Status

This package now has both a portable semantic PyTorch implementation and a
source-built Linux/WSL CUDA backend compiled specifically for SM75.  The native
backend is not an Ampere fallback: it implements the F4/E6 primitive action and
the full sparse-event Delta recurrence directly on the RTX 2070 SUPER.  The
canonical WSL SM75 suite passes 72 tests.  The generated audit separates
exact-data, numerical, empirical, and open claims in
[`artifacts/algebra_audit.json`](artifacts/algebra_audit.json).

The new systems result is in
[`SM75_PRIMITIVE_TRANSPORT_RESULTS.md`](SM75_PRIMITIVE_TRANSPORT_RESULTS.md).
At one E6 action per 32 tokens, the complete candidate is 2.36x faster and uses
69% less peak allocation than dense all-token E6.  It passes the cheap-action
gate, but remains 1.41x slower and 1.28x larger in peak allocation than official
fused Mamba-2, so the stronger complete-model systems gate remains failed.

The v1.3.1 repaired model law and matched natural-text results are retained in
[`QUALITY_LEARNING_RESULTS.md`](QUALITY_LEARNING_RESULTS.md). F4 and E6 learn
hidden action coordinates unavailable to their predecessor algebras and
extrapolate from length 4 to length 16 on a controlled composition task. On a
matched three-seed natural-text cohort, however, official fused Mamba-2 wins
every seed and is about 3.63x faster than the strongest exceptional arm. This
is positive mechanism evidence and a failed language-model promotion, not a
claim that exceptional transport generally improves text modeling.

When E6 is selected, the default geometry is now the direct 78-coordinate
chart. Polar and Cartan
charts remain first-class falsifiers rather than defaults: a pinned Tiny
Shakespeare screen found no early quality separation between direct and polar
E6, while direct was cheaper. The no-action control was also essentially tied
at the tested budget, so this is architecture evidence, not evidence of a
language-model advantage. See
[`SHAKESPEARE_DEVELOPMENT_RESULTS.md`](SHAKESPEARE_DEVELOPMENT_RESULTS.md).

For generic Shakespeare language modeling, however, the supported benchmark
reference is identity transport. A post-discovery, prospectively frozen five-
seed gate rejected the apparent `E6 -> identity` improvement (2/5 wins; mean
effect `-0.0093` bpb). Exceptional actions remain available for explicit
symmetry and sparse-transport experiments. See
[`SHAKESPEARE_LAYER_LOCALIZATION_RESULTS.md`](SHAKESPEARE_LAYER_LOCALIZATION_RESULTS.md).

The identity reference is now rigorously optimized. A one-sided associative
scan is bitwise equal to the generic two-sided identity path and improves the
tested RTX 2070 SUPER eager forward+backward median by 1.104x. The cheaper
explicit Albert cubic was rejected as the default after failing a frozen
five-seed quality gate; dense Jordan evaluation remains the accepted default.
The optional fixed-shape CUDA-graph compiler tier is much faster but retains a
small numerical difference and a substantial cold compile cost. See
[`V1_3_OPTIMIZATION_RESULTS.md`](V1_3_OPTIMIZATION_RESULTS.md).

Run from the repository root:

```powershell
$env:PYTHONPATH = "SSM-Models"
python -m pytest -q SSM-Models/pure_f4_delta_ssm_v1_3
python -m pure_f4_delta_ssm_v1_3.audit_algebra `
  --output SSM-Models/pure_f4_delta_ssm_v1_3/artifacts/algebra_audit.json
python -m pure_f4_delta_ssm_v1_3.benchmark_train `
  --device cuda --require-sm75 --variants e6_safe `
  --steps 1000 --seed 2633 `
  --output SSM-Models/pure_f4_delta_ssm_v1_3/artifacts/shakespeare_screen.json
python -m pure_f4_delta_ssm_v1_3.benchmark_exceptional_learning `
  --device cuda --require-sm75 --target f4 --candidates spin9 f4 `
  --steps 1000 --seed 2611 `
  --output SSM-Models/pure_f4_delta_ssm_v1_3/artifacts/f4_learning.json
python -m pure_f4_delta_ssm_v1_3.benchmark_optimization `
  --compile-mode reduce-overhead `
  --output SSM-Models/pure_f4_delta_ssm_v1_3/artifacts/optimization.json
python -m pure_f4_delta_ssm_v1_3.benchmark_primitive_action `
  --output SSM-Models/pure_f4_delta_ssm_v1_3/artifacts/primitive_action.json
python -m pure_f4_delta_ssm_v1_3.benchmark_sparse_action_cost `
  --cycles 3 --warmups 20 --samples 50 --require-sm75 `
  --output SSM-Models/pure_f4_delta_ssm_v1_3/artifacts/sparse_action_cost.json
```

The dense-action cost problem is solved for periodic sparse events.  The next
architecture promotion gate is not "make it larger."  It is quality learning
for the sparse candidate versus the same-parameter dead-action control and
official fused Mamba-2, followed by a genuinely learned hard event router if
periodic transport proves useful.  Dense exceptional action at every token
remains a historical negative control.

See [CONSTRAINT_AUDIT.md](CONSTRAINT_AUDIT.md) for the research-chain audit and
[MATHEMATICAL_DESIGN.md](MATHEMATICAL_DESIGN.md) for the equations and the
projective/exponential-map decision. The dated implementation and experiment
record is [`RESEARCH_LOG_2026-08-26.md`](RESEARCH_LOG_2026-08-26.md).
