# Schur-legal multiplicity-query result

**Date:** 2026-08-22

**Verdict:** failed the prospective gate; no readout router remains the
maintained default.

## Complete result

The candidate is a zero-initialized, token-conditioned SO(2) rotation across
the two equivalent triality channels, shared across `8v`, `8+`, and `8-`.
Tests establish exact identity initialization, nonzero finite gradients, and
commutation with the shared Spin(8) action. The protocol was committed and
pushed as `a5eba7c` before training.

| seed | no router bpb | orthogonal query bpb | improvement |
|---:|---:|---:|---:|
| 83 | **2.69083** | 2.69376 | -0.00294 |
| 89 | **2.67697** | 2.68622 | -0.00925 |
| 97 | 2.74228 | **2.72758** | +0.01471 |
| mean | **2.70336** | 2.70252 | +0.00084 |

The candidate won one of three seeds and missed the +0.0100 bpb mean threshold.
It therefore failed two primary criteria while remaining finite and inside the
single-seed safety bound. The candidate adds only 516 parameters, so lack of
capacity is not the confound; the negative result localizes the problem to the
placement of the mechanism.

## Interpretation

A token-dependent channel rotation immediately before a static output
projection is mathematically legal but too late to affect memory formation.
It changes how stored information is viewed, not what the recurrence retains,
forgets, or writes. The result does not refute Programme 1's isotypic compiler;
it falsifies this readout-only use of its commutant layer on the present
language task.

The next serious v1.2 architecture is a coupled isotypic recurrence

\[
H_t=A_t\,M_t\,\rho(g_t)H_{t-1}+D_t,
\]

where a contractive `M_t` acts on multiplicity and commutes with the Spin(8)
factor. That changes memory dynamics and requires a new semantic scan plus a
custom CUDA backward, not another post-scan feature. It should be built as a
separate backend and compared against the unchanged default; neither failed
readout control should be bundled into it.

Exact inputs and the fail-closed summary are in
`artifacts/shakespeare_multiplicity_router_gate_seed{83,89,97}_cu126_f14a.json`
and `artifacts/shakespeare_multiplicity_router_gate_summary_cu126_f14a.json`.
