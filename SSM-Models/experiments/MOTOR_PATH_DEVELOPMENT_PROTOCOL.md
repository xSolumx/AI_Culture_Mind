# Double-cover Euclidean motor path-development protocol

Protocol frozen **2026-08-16T19:41:11+02:00** (`Africa/Johannesburg`) before
running the full length/timing artifact from
[`audit_motor_path_development.py`](../audit_motor_path_development.py).

**Status at freeze:** 11 algebra/scan/audit tests pass, including CUDA backward;
there is no full audit artifact yet.

**Completed result:** the corrected replay finished at
**2026-08-16T19:44:30+02:00** and passes every frozen gate. See
[`MOTOR_PATH_DEVELOPMENT_RESULTS.md`](MOTOR_PATH_DEVELOPMENT_RESULTS.md). The
freeze statement above is retained as provenance.

## Question

Can the successful sign-sensitive Spin(3) prefix primitive be extended to
orientation-preserving 3D rigid motions without losing:

1. associative recurrent/parallel scan semantics;
2. the nontrivial central sign in the double cover;
3. fixed-size streaming cache;
4. faithful equivalence to conventional homogeneous `SE(3)` matrices; or
5. numerical validity across length 4096?

The proposed state is one unit dual quaternion

\[
\widehat q=q_r+\varepsilon q_d,
\qquad \lVert q_r\rVert=1,
\qquad \langle q_r,q_d\rangle=0,
\]

with multiplication

\[
(p_r+\varepsilon p_d)(q_r+\varepsilon q_d)
=p_rq_r+\varepsilon(p_rq_d+p_dq_r).
\]

The eight-scalar motor is a double-cover representation of `SE(3)`. Simultaneous
negation of both quaternion parts changes the recurrent Spin representative but
not the physical rigid action.

This is a numerical/algebraic implementation gate, not a learned-model result.

## Conventions and correction to avoid

- quaternions are scalar-first `[w,x,y,z]`;
- a motor encodes `x -> R(q_r)x + t` with
  `q_d = 0.5 * [0,t] * q_r`;
- prefix update is `state_next = state * token_motor`;
- motor multiplication must agree with left homogeneous-matrix multiplication;
- normalization acts on `q_r` and projects `q_d` to the Study quadric;
- **Euclidean normalization of all eight coordinates is forbidden**, because it
  changes translation magnitude;
- translation is not bounded: `SE(3)` is noncompact.

## Frozen exact/numerical checks

For deterministic float64 increments at lengths `16,128,1024,4096`, with batch
4 and four motor lanes:

- parallel versus recurrent motor prefixes and final state;
- one-third/two-thirds chunked cache versus full scan;
- parallel versus recurrent 4 by 4 homogeneous-matrix prefixes;
- motor prefixes converted to matrices versus recurrent matrix ground truth;
- real-quaternion unit norm and Study condition;
- rotation orthogonality and determinant one;
- nonzero accumulated translation;
- central-negated motor prefixes equal the state antipodes;
- central-negated physical matrices equal the original matrices;
- all values finite.

Frozen tolerances are `2e-9` for tree/cache/motor-matrix comparisons, `2e-12`
for unit/Study and central-action checks, and `2e-10` for rigid-matrix checks.
These tolerate floating tree-order differences; they are not symbolic
equalities.

## Frozen local timing diagnostic

On CUDA, compare eager PyTorch motor and 4 by 4 matrix scans, recurrent and
parallel, at length 1024, batch 8, four lanes, five warmups, and 20 measured
repeats. Report milliseconds and lane-tokens/s. No candidate uses a fused
kernel, and the result cannot be promoted to a production systems ranking.

## Interpretation

1. Every exact/numerical check must pass for the gate to close.
2. Matrix equivalence plus sign separation establishes a correct double-cover
   implementation, not novelty of dual quaternions themselves.
3. A speed difference is local eager-backend evidence only.
4. Passing licenses a separately frozen **learned** rigid-motion composition
   benchmark. It does not establish learning quality.
5. Failure at long length must be reported; changing normalization or
   tolerance after seeing the result requires a new protocol.
6. Conformal GA, `SL(2,R)`, and other noncompact groups are not inferred from an
   `SE(3)` result.

## Related primary work

This implementation is positioned as a specialized parallel double-cover
realization of finite-dimensional Lie-group path development, not as the first
group-valued sequence layer:

- [Path Development Network](https://arxiv.org/abs/2204.00740) uses products of
  learned Lie-group exponentials for sequential features;
- [Dual Quaternion Rotational and Translational Equivariance](https://arxiv.org/abs/2310.07623)
  demonstrates the value of joint rotation/translation representations;
- [Fixed-Point RNNs](https://arxiv.org/abs/2503.10799) provide a strong modern
  dense-state-tracking comparator for the later learned benchmark.

## Frozen command

```powershell
python SSM-Models\audit_motor_path_development.py --device cuda `
  --lengths 16,128,1024,4096 --batch-size 4 --lanes 4 `
  --timing-length 1024 --timing-batch-size 8 `
  --timing-warmups 5 --timing-repeats 20 `
  --output SSM-Models\experiments\artifacts\motor_path_development_20260816.json `
  --quiet-report
```
