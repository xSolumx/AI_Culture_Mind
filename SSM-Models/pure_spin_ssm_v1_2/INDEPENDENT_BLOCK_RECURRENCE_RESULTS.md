# Independent-action block-affine recurrence result

**Closed:** 2026-08-22

## Verdict

The independent-action block-affine compiler and full-training CUDA backend are
accepted as correct research machinery. The free per-token `SO(2)` recurrent
coupling is not promoted into maintained Pure Spin v1.2.

| seed | maintained bpb | block-coupled bpb | improvement |
|---:|---:|---:|---:|
| 193 | **2.736944** | 2.751450 | -0.014506 |
| 197 | **2.675457** | 2.711872 | -0.036414 |
| 199 | 2.708134 | **2.704176** | +0.003958 |
| mean | **2.706845** | 2.722499 | -0.015654 |

Positive improvement is maintained minus candidate. The candidate won one of
three seeds, failed the frozen `+0.0100` mean requirement, and had an
unfavorable mean effect. No seed crossed the `0.0500` hard regression limit.
The conditional speed gate was not authorized.

## What the negative result does and does not say

The experiment preserves both independently learned Spin controllers, exactly
the diversity lost by the shared-action compression candidate. Therefore this
is direct evidence against an unrestricted tokenwise multiplicity rotation at
this scale, rather than another consequence of action tying.

It does not invalidate:

- the global Spin-equivariance proof;
- the exact 16-dimensional block-affine associative closure;
- the original contraction bound;
- semantic prefix/recurrent or raw-CUDA gradient parity;
- other couplings whose time discretization is tied to the SSM update scale.

The candidate applied a bounded rotation at every token independently of how
close retention was to identity. In a continuous-time interpretation, both
damping and cross-channel generator flow should scale with the local step
`delta_t`. Since `s_t = exp(-lambda delta_t)`, the bounded observable proxy
`1-s_t` is first-order proportional to that step. A principled successor is
therefore

\[
Q_t=\exp\!\left((1-\bar s_t)\,\kappa_tJ\right),
\]

not another unconstrained angle limit chosen after inspecting this table. That
successor requires a new prospective gate.

The machine-readable decision is
[`artifacts/independent_block_gate_summary.json`](artifacts/independent_block_gate_summary.json).
The theorem and frozen protocol are in
[`INDEPENDENT_BLOCK_RECURRENCE_PREREGISTRATION.md`](INDEPENDENT_BLOCK_RECURRENCE_PREREGISTRATION.md).
