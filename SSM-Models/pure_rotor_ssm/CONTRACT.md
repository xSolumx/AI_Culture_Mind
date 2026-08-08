# Pure rotor SSM contract

Model version: **2.1.0**. Version 2 introduced the new state/write law and
parameterization; version 2.1 expands the default physical rotor-angle chart
from `pi/2` to the full open `pi` range. A v2.0 checkpoint remains loadable
because its saved model configuration records the old explicit angle limit.

This directory is the single maintained implementation of the Cl(3,0)
selective rotor SSM. It contains model mathematics only: no dataset access,
optimizer, trainer, checkpoint, profiler, benchmark, or report writer.

Import one backend explicitly:

```python
from pure_rotor_ssm import jax_backend
from pure_rotor_ssm import torch_backend
```

The package initializer intentionally imports neither framework.

## Representation

A multivector is a tensor whose final axis has coefficient order
`[1,e1,e2,e3,e12,e13,e23,e123]`. A layer has `channels` independent recurrent
multivectors. Rotors are even values `[s,0,0,0,b12,b13,b23,0]`; states and
writes are full-algebra values.

Under Spin(3) conjugation, Cl(3,0) decomposes into two trivial and two standard
three-dimensional copies. `Spin3IsotypicLinear` implements the complete real
linear commutant, including scalar/pseudoscalar and vector/Hodge-bivector
mixing. `GradeLinear` is an import-compatibility alias for this complete layer,
not the older grade-diagonal restriction.

## Transition and hard state bound

For each channel and valid token,

```text
h_next = d * Ad(q)(h) + (1 - d) * w * z
```

where `d = exp(-positive_step * positive_rate)`, `w = sigmoid(write_control)`,
`q` is a unit rotor, and

```text
z = projected_input / sqrt(1 + ||projected_input||^2).
```

Thus, over finite real arithmetic, `0 < d < 1`, `0 < w < 1`, `||z|| < 1`, and
`Ad(q)` is orthogonal. The triangle inequality gives

```text
||h_next|| <= d ||h|| + (1 - d),
||h_t|| <= max(||h_0||, 1).
```

This guarantee holds for arbitrary finite input magnitude. It does not claim
that residual-stream activations, losses, parameter gradients, or nonfinite
inputs are bounded. Floating-point executions satisfy the statement only up
to rounding error.

For an invalid/padded token, the transition is exactly `(1, identity, 0)` in
the represented dtype, so the recurrent state is unchanged.

## Controls and gradients

Controllers consume the five per-channel polynomial invariants
`[scalar, pseudoscalar, ||vector||^2, ||dual_bivector||^2,
vector dot dual_bivector]`. They are smooth at zero. Step, write, and rotor
controllers start at zero; half-lives are log-spaced over the configured
range. The bounded bivector chart uses an analytic small-angle branch, giving
a finite nonzero identity-tangent derivative. The raw `normalized_rotor`
helper necessarily has no continuous extension at the zero parameter vector
and uses identity as an explicit fallback.

The previous-state Jacobian has operator norm `d` in exact arithmetic. Long
pure recurrent paths may therefore have vanishing gradients; forward-state
boundedness is not a universal gradient guarantee.

## Scan and cache semantics

An affine rotor transition `(d,q,b)` acts as `d Ad(q)(h)+b`. Applying `a` and
then `b` yields

```text
(d_b d_a, q_b q_a, b_b + d_b Ad(q_b)(b_a)).
```

This is associative in exact arithmetic. JAX uses `lax.associative_scan`.
PyTorch uses a differentiable vectorized Hillis--Steele scan and keeps a
recurrent oracle. Different tree orderings are not bitwise equivalent in
floating point; tests enforce tolerance-based parity.

A model cache is one `(batch, channels, 8)` tensor per layer, independent of
context length. It denotes the state *after* the last supplied token. Supplying
it as `recurrent_states` continues the same recurrence. PyTorch inference must
detach caches or run without gradient tracking for the live-memory claim to
exclude autograd history.

## Equivariance and dropout

Deterministic blocks commute with one shared proper Spin(3) frame action.
Dropout samples one scalar mask for each complete multivector and broadcasts
it across all eight coefficients, so coupled-mask training-time dropout also
commutes with the action. The architecture does not claim reflection, Lorentz,
general Clifford-group, or semantically established linguistic symmetry.

## Backend and performance boundary

Both backends share the basis, formulas, mask behavior, scan semantics, and
streaming API. Primitive outputs are tested cross-backend in float64. Full
models use framework-specific RNGs and initializers, so separately initialized
weights and logits need not match.

The rotor product is specialized as a quaternion product. JAX evaluates the
full-algebra sandwich as two 3D rotations. PyTorch retains both that algebraic
specialization and the dense geometric-product oracle, and dispatches CUDA to
the dense path because the benchmark on the local RTX 2070 SUPER shows that
its fused `einsum` kernels are faster than many small eager elementwise
launches. The PyTorch prefix scan avoids a tokenwise Python loop on CUDA, but
performs `O(L log L)` work; a future fused `O(L)` scan would be a performance
optimization, not a change to the model contract.

## Explicit non-claims

- The five-seed v2.1 byte-model result establishes only a small prediction
  advantage over retrained identity at matched state size and nearest live
  parameter count. Quaternion and commuting-phase transitions perform better
  in that protocol, so it is not a Cl(3,0)-specific advantage.
- The registered associative-recall and Q8 tasks do not establish better rotor
  memory, and the measured-CUDA view favors a wider identity model. The model
  therefore has no demonstrated memory or compute-efficiency advantage.
- The C32/L4 v2.1 retrain is a single seed; its 0.001458-nat improvement over
  v2.0 is descriptive, not an established refinement or scaling result.
- Frozen three-seed GPU artifacts in the parent directory concern the
  superseded grade-linear, `sqrt(1-d^2)` architecture.
- No supplied 3D frame establishes that Spin(3) is a true language symmetry.
- The recurrence is channel-diagonal in state transport; learned channel
  interaction occurs in controls, projections, and the feed-forward path.
