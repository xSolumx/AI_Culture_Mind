# Spin(8) triangular triality lift results

Date executed: 2026-08-03. Gates were frozen in
`SPIN8_TRIANGULAR_TRIALITY_LIFT_PREREGISTRATION.md`. Raw output is
`spin8_triangular_triality_lift.json`.

## Result

Every frozen algebraic and scan gate passed. The 24-scalar staged scan was a
post-gate optimization derived after observing that the 81D lift is redundant
at recurrent inference; its parity number is therefore an exploratory
implementation result, not a preregistered gate.

| Diagnostic | Result |
|---|---:|
| homogeneous polynomial-lift dimension | 81 |
| one-step lift/direct error | `8.88e-16` |
| length-17 lifted parallel/recurrent error | `3.66e-15` |
| lifted/direct triality readout error | `0.0` |
| shared-Spin(8) triality equivariance error | `4.44e-15` |
| staged-scan/direct recurrence error (post-gate) | `4.27e-14` |
| staged streaming cache (post-gate) | 24 scalars |
| staged parallel scan depth (post-gate) | two scan stages |

## The constructive result

Let two independent affine spinor memories evolve as

```text
s+_t = P_t s+_(t-1) + p_t
s-_t = N_t s-_(t-1) + n_t.
```

The Spin(8)-equivariant binding

```text
b_t[i] = s-_t^T rho[i] s+_t
```

is bilinear in the original states. It has two exact parallel realizations.

1. **Single lifted scan.** Carry homogeneous coordinates
   `[1,s+,s-,vec(s+ tensor s-)]`. Each token is one 81x81 linear map, so
   ordinary ordered matrix multiplication gives an associative scan.
2. **Staged scan.** Scan the two independent spinor streams first, evaluate
   `b_t` pointwise, then use it as the drive of a vector-state affine scan.
   This uses two logarithmic-depth scan stages and retains only `s+`, `s-`, and
   `v`—24 scalars—during streaming.

The second implementation is the practical architecture. The first is the
closure proof.

## Exact obstruction

The construction depends on a triangular recurrent dependency graph. If the
bound vector feeds back into both source spinors, polynomial degree doubles in
the generic case:

```text
2, 4, 8, 16, 32, 64, 128, 256, ...
```

No fixed-degree polynomial lift can then remain exact. One must accept a
sequential nonlinear recurrence, truncate/approximate the lift, or impose a
special algebraic closure. Thus “triality is nonlinear but still uses the same
affine scan” is false in general; triangular triality binding is the exact
scan-compatible statement.

## Claim boundary

This establishes an activation-free, Spin(8)-equivariant dynamic binding
primitive with exact constant-cache streaming and parallel training. It does
not establish attention replacement or associative-recall performance. The
next empirical gate must compare it on masked multi-query associative recall
against bilinear fast weights, DeltaProduct/Householder, dense selective SSM,
and an equal-state diagonal baseline.
