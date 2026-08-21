# Pure Exceptional Delta SSM v1.3

This folder is the isolated correctness and evidence boundary for the v1.3
frontier. The model is not a renamed v1.2 and does not inherit its language-
quality or throughput claims.

The implemented state hierarchy is

\[
\mathrm{Spin}(8)\subset\mathrm{Spin}(9)\subset F_4\subset E_{6(-26)},
\]

all acting in one 27-dimensional Albert-algebra carrier. The trace-free
26-dimensional irreducible (F_4) module remains available and tested, but it
is no longer the model ceiling. The built-in hierarchy reaches the full
78-generator (E_{6(-26)}) action; compact (F_4), Spin(9), Spin(8), identity,
arbitrary schedules, and caller-supplied generator banks are controls.

## What is implemented

- the real Albert algebra (H_3(\mathbb O)), its Jordan product, trace, and
  cubic determinant;
- 52 derivations spanning (\mathfrak f_4), plus the 26 traceless left-
  multiplication directions spanning (\mathfrak e_{6(-26)});
- executable Spin(8) and Spin(9) stabilizer restrictions inside the same
  representation;
- direct, polar, and Cartan (KAK) E6 action charts with complete autograd;
- one or more ordered Lie exponentials per token;
- a configurable rank-(r), independent erase/write memory update;
- an associative two-sided affine prefix compiler for general invertible value
  actions;
- recurrent and logarithmic-depth semantic scans with output, state, and
  gradient parity;
- automatic parallel-prefix execution for full sequences and recurrent
  execution for one-token streaming;
- complete streaming state, including the causal-convolution cache;
- optional SwiGLU, Albert-Jordan, or no pointwise channel mixer;
- an Albert-invariant readout that keeps the 27D direction while restoring
  trace, energy, and cubic scale information lost by plain RMS normalization;
- optional Riemann-sphere and PGL/Möbius utilities for router controls;
- a custom generator-bank interface that removes the exceptional hierarchy as
  a hard implementation ceiling.

## Status

This is a semantic PyTorch implementation, not yet an optimized CUDA backend
or a promoted trained model. The algebra and model suite currently has 28
tests. The generated audit separates exact-data, numerical, empirical, and
open claims in `artifacts/algebra_audit.json`.

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

Run from the repository root:

```powershell
$env:PYTHONPATH = "SSM-Models"
python -m pytest -q SSM-Models/pure_f4_delta_ssm_v1_3
python -m pure_f4_delta_ssm_v1_3.audit_algebra `
  --output SSM-Models/pure_f4_delta_ssm_v1_3/artifacts/algebra_audit.json
python -m pure_f4_delta_ssm_v1_3.benchmark_train `
  --device cuda --steps 50 `
  --output SSM-Models/pure_f4_delta_ssm_v1_3/artifacts/shakespeare_screen.json
```

The next promotion gate is not “make it larger.” It is a matched experiment
that independently varies action algebra, update rank, tying, and mixer across
multiple seeds, then compares quality and complete-step cost against v1.2, the
no-action delta control, an unrestricted-action control, and an official fused
modern delta baseline on the same Shakespeare split.

See [CONSTRAINT_AUDIT.md](CONSTRAINT_AUDIT.md) for the research-chain audit and
[MATHEMATICAL_DESIGN.md](MATHEMATICAL_DESIGN.md) for the equations and the
projective/exponential-map decision.
