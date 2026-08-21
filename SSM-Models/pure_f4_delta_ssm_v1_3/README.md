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
is no longer the model ceiling. The default is the full 78-generator
(E_{6(-26)}) action; compact (F_4), Spin(9), Spin(8), arbitrary schedules,
and caller-supplied generator banks are controls.

## What is implemented

- the real Albert algebra (H_3(\mathbb O)), its Jordan product, trace, and
  cubic determinant;
- 52 derivations spanning (\mathfrak f_4), plus the 26 traceless left-
  multiplication directions spanning (\mathfrak e_{6(-26)});
- executable Spin(8) and Spin(9) stabilizer restrictions inside the same
  representation;
- one or more ordered Lie exponentials per token, with complete autograd;
- a configurable rank-(r), independent erase/write memory update;
- an associative two-sided affine prefix compiler for general invertible value
  actions;
- recurrent and logarithmic-depth semantic scans with output, state, and
  gradient parity;
- complete streaming state, including the causal-convolution cache;
- optional SwiGLU, Albert-Jordan, or no pointwise channel mixer;
- optional Riemann-sphere and PGL/Möbius utilities for router controls;
- a custom generator-bank interface that removes the exceptional hierarchy as
  a hard implementation ceiling.

## Status

This is a semantic PyTorch implementation, not yet an optimized CUDA backend
or a promoted trained model. The algebra and model suite currently has 19
tests. The generated audit separates exact-data, numerical, empirical, and
open claims in `artifacts/algebra_audit.json`.

Run from the repository root:

```powershell
$env:PYTHONPATH = "SSM-Models"
python -m pytest -q SSM-Models/pure_f4_delta_ssm_v1_3
python -m pure_f4_delta_ssm_v1_3.audit_algebra `
  --output SSM-Models/pure_f4_delta_ssm_v1_3/artifacts/algebra_audit.json
```

The next promotion gate is not “make it larger.” It is a matched experiment
that independently varies action algebra, update rank, tying, and mixer, then
compares quality and complete-step cost against v1.2, a no-action delta model,
an unrestricted-action control, and an official fused modern delta baseline.

See [CONSTRAINT_AUDIT.md](CONSTRAINT_AUDIT.md) for the research-chain audit and
[MATHEMATICAL_DESIGN.md](MATHEMATICAL_DESIGN.md) for the equations and the
projective/exponential-map decision.
