# Pure rotor SSM contract

Model version: **2.1.0**. Version 2 introduced the new state/write law and
parameterization; version 2.1 expands the default physical rotor-angle chart
from `pi/2` to the full open `pi` range. A v2.0 checkpoint remains loadable
because its saved model configuration records the old explicit angle limit.

Documentation last reconciled: **2026-08-21T14:36:14+02:00**. The adjacent
PyTorch-only `schur_parallel` scan mode is execution-compatible and opt-in, and
`spin_scan.py`, `motor_scan.py`, `octonion_operator_scan.py`, and
`dense_so8_cayley_scan.py` are explicitly
experimental companions. The optional `octonion_operator_triton.py` backend is
WSL/Linux CUDA-only. None alters this model version, parameter layout, or
checkpoint contract.

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

PyTorch additionally exposes `scan_mode="schur_parallel"`: an equivalent,
experimental Hillis--Steele implementation in the two trivial plus two
three-dimensional Spin(3) isotypic coordinates. It is guarded against the
direct rotor sandwich in forward and first-order-gradient float64 tests, but
it remains opt-in until matched CPU/GPU throughput and memory measurements
establish a deployment benefit. It does not alter checkpoint parameters or
the recurrence/caching contract, and is not yet implemented by the JAX
backend.

## Experimental sign-sensitive companion

`spin_scan.py` is a PyTorch-only experimental module, not part of the v2.1
checkpoint or recurrence contract. It keeps unit quaternions themselves as
fixed-size state and uses the right-composition update

```text
s_next = normalize(s * q_token).
```

This distinguishes the central pair `q` and `-q`, unlike the canonical
conjugation action `Ad(q)=Ad(-q)`. The module exposes recurrent and
Hillis--Steele scans, identity padding, streaming continuation, compact/Cl(3,0)
coordinate conversion, trainable token increments, and a small classifier
wrapper. Float64 forward and first-order-gradient parity, long unit-norm
stability, cache equivalence, masking, center separation, Clifford-product
orientation, and CUDA backward are tested.

The completed three-seed `2.A5` pilot motivates this component, but does not
establish broad sequence utility. Promoting it into a maintained hybrid needs a
new protocol, state/parameter/compute comparisons, stronger structured-product
baselines, and non-symbolic tasks.

`motor_scan.py` extends this companion state to unit dual quaternions. It
exposes generic token-motor composition/classification plus two direct pose
trackers: the true semidirect motor and a parameter-matched
`Spin(3) x (R^3,+)` ablation. The direct motor emits signed quaternion and
translation coordinates without an MLP. Its algebra, Study projection,
parallel/recurrent/cache/gradient behavior, and homogeneous-matrix action are
tested separately. The rigid `2.A5` benchmark finds that blind 300-step
end-to-end training fails, while local transition identification from every-
prefix pose targets succeeds in 9/9 finite coordinate/schedule replications.
That estimator is not part of the v2.1 training or checkpoint contract and the
result does not establish continuous-data or final-only-supervision utility.

`octonion_operator_scan.py` is a third separate PyTorch experiment. Raw
octonion multiplication remains nonassociative. The layer instead lifts each
unit octonion to a real left/right multiplication matrix, scans those operators
associatively, and evaluates the identical explicitly parenthesized product
with an eight-scalar streaming cache. Its bounded affine recurrence obeys
`||h_t|| <= max(||h_0||,1)`. The exact Lie certificate shows that the seven
imaginary left generators and their 21 commutators span `so(8)`. This does not
make one octonion a parameterization of arbitrary `SO(8)`, nor does one acted-
on eight-vector identify the accumulated operator.

`octonion_operator_triton.py` fuses the same left recurrence and its reverse-
mode derivative in the separately recorded Ubuntu WSL2/Triton environment.
It accelerates chunk recurrence but is not a parallel matrix-prefix kernel or
a native-Windows dependency. Both octonion modules remain outside the v2.1
weights, recurrence, cache, and checkpoint format.

The separately frozen continuous associator-tracking pilot supervises every
complete 64-scalar prefix operator. A 72-parameter identity-near token encoder
plus exact operator scan extrapolates from L16 to L128 near float32 numerical
precision, while collapsed-octonion, unfused DeltaProduct, and unfused Mamba-2
controls do not. This is a one-seed coordinate-aligned realizability result.
Because the target is the complete operator, its recurrent-state accounting is
64 scalars; it does not convert the compact acted-on-vector cache into a full
operator cache or modify the maintained v2.1 contract.

The separately frozen Haar-basis successor learns three transported octonion
laws with a 28-parameter `SO(8)` gauge and satisfies the recovered `G2`
automorphism intertwiner to the recorded tolerances. Its top-level frozen
artifact is explicitly failed: the 512-parameter dense AdamW control misses its
accuracy gate in all bases and one float32 oracle narrowly misses `1e-12`.
A post-protocol legal first-prefix least-squares estimator makes the dense map
numerically exact, so the dense failure is optimization rather than capacity.
None of these experimental weights or 64-scalar full-operator targets changes
the maintained eight-scalar acted-on-state cache contract.

`dense_so8_cayley_scan.py` is a fourth separate PyTorch experiment. It uses
the exact seven grade-one plus 21 grade-two Clifford directions underlying the
mixed monomial--golden dense-\(SO(8)\) theorem as a full 28-dimensional
skew-tangent basis, then applies a Cayley retraction and the same bounded
affine scan pattern. Its eight-scalar state cache represents only an acted-on
vector, not a recovered group element. Float64 algebra, scan, cache, mask,
gradient, and state-bound tests pass; a WSL2 CUDA forward/backward smoke on
the local RTX 2070 SUPER passes at float32 tolerance. It is not part of v2.1
weights or checkpoints, and has no training, benchmark, throughput, or
Tensor-Core performance claim.

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
- The experimental `spin_scan.py` result does not change these canonical v2.1
  claims or retroactively make conjugation center-sensitive.
