# Dense SO(8) Cayley Scan: Experimental Design

**Updated:** 2026-08-21T14:37:36+02:00
**Status:** implemented structural companion; CUDA feasibility checked; no training or speed claim.

## Why this exists

The exact mixed monomial--golden theorem now establishes that one fixed eleven-generator system is dense in \(SO(8)\). The new `dense_so8_cayley_scan.py` uses the same seven Clifford grade-one and 21 grade-two directions as a 28-dimensional \(\mathfrak{so}(8)\) tangent basis. It is a direct test bed for whether the full closure can be made scan-compatible without confusing mathematical expressivity with model utility.

## Transition contract

For each lane and token, a controller emits \(b_t\in\mathbb R^{28}\), forms a skew matrix \(B_t\), and uses

\[
Q_t=(I-B_t/2)^{-1}(I+B_t/2)\in SO(8).
\]

The bounded affine update is

\[
h_t=d_tQ_th_{t-1}+(1-d_t)w_tz_t,
\]

with \(d_t,w_t\in[0,1]\) and \(\lVert z_t\rVert\le1\). Consequently \(\lVert h_t\rVert\le\max(\lVert h_0\rVert,1)\) in exact arithmetic. Its homogeneous affine transitions compose associatively, so recurrent, work-efficient, and Hillis--Steele scans have the same semantic target.

The streaming cache is eight scalars per lane. It is an acted-on state, not an accumulated \(8\times8\) group element; direct group readout would require a separate 64-scalar operator cache.

## Structural validation

`test_dense_so8_cayley_scan.py` checks the 28-dimensional basis rank, exact skew structure, Cayley orthogonality/determinant in float64, recurrent/parallel/cache/mask parity, first-order gradient parity, the state bound, and streaming layer equivalence.

## WSL CUDA feasibility smoke

On 2026-08-21, Ubuntu WSL2 with PyTorch `2.4.0+cu121` ran the CUDA forward/backward gate on the local NVIDIA GeForce RTX 2070 SUPER:

- batch 4, length 257, lanes 3, float32;
- recurrent vs work-efficient maximum state error: `1.1920928955078125e-07`;
- final-cache error: `5.960464477539063e-08`;
- Cayley orthogonality residual: `4.76837158203125e-07`;
- all tested gradients finite; peak allocated memory: `36.27 MiB`.

This smoke confirms the CUDA execution path and tolerance-scale numerical behavior only. It is not a Tensor-Core kernel result or a comparison against Mamba, delta-rule, octonion, or Rotor baselines.

## Raw execution snapshot

The same WSL2 runtime also measured forward-only execution at batch 8, length 256, eight lanes, float32, after four warmup runs and over 12 synchronized CUDA-event iterations:

| Scan mode | Milliseconds per forward | Peak allocated memory |
|---|---:|---:|
| Work-efficient affine tree | 3.8541 ms | 52.76 MiB |
| Sequential recurrent oracle | 43.8717 ms | 52.76 MiB |

This is an intra-implementation timing snapshot on one RTX 2070 SUPER, not a matched baseline comparison. Dense `8 by 8` solves, controller work, batch/length shape, PyTorch version, and the lack of a fused kernel all materially affect it. The only supported interpretation is that the current parallel path is executable and substantially faster than its own Python-loop oracle at this shape.

## Next decisive experiment

Use matched recurrent-state, parameter, optimizer, sequence-length, and wall-clock rows against the octonion operator scan, Pure Spin(8), Householder/Givens products, and modern SSM baselines. Include dense-28D versus restricted grade-one controls, recurrent versus parallel kernels, and full-operator readout where the task asks for a group element rather than its action on one state.
