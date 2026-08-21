# Spin(8) Isotypic Tensor-Core Dispatch Audit

Execution: **2026-08-21**, WSL2, NVIDIA GeForce RTX 2070 SUPER (`sm_75`)

## Outcome

Programme 01's exact isotypic coordinates give a real hardware compilation
rule. If an irreducible representation (V) occurs with multiplicity (m),
then a shared action on (V\otimes\mathbb R^m) is not (m) unrelated
matrix-vector products. With the copies stored as the rows of
(X\in\mathbb R^{m\times d}), it is the single factored action

\[
X\longmapsto X\rho(g)^T.
\]

The multiplicity coordinate is therefore a legitimate GEMM tile axis. This
identity was checked independently against the dense block-diagonal action
(I_m\otimes\rho(g)), including chronological composition order.

The corresponding Triton FP16 kernel genuinely dispatches Tensor-Core
instructions: its compiled PTX contains two `mma.sync` occurrences, while the
strong register-resident scalar comparator contains none. This is instruction
evidence, not an inference from dtype.

The performance result is deliberately negative for a universal Tensor-Core
policy. On this Turing card the scalar schedule is faster in six of the eight
recorded cells, essentially tied in one low-occupancy cell, and Tensor-Core
faster only at the high-parallelism `(batch=32, length=128, multiplicity=16)`
cell, by **1.147x**. The correct compiler decision is consequently a
device-and-shape dispatch, not “Spin uses Tensor Cores.”

## Hardware measurements

Both kernels use the same frozen learned vector-action dictionary, emit every
prefix, take FP16 actions and states, and round the recurrent state to FP16
after every token. The Tensor-Core dot accumulates in FP32. Kernel timings
reuse preallocated output and are interleaved to reduce clock/order bias.

| Batch | Length | Multiplicity | Scalar, us | Tensor Core, us | TC / scalar speedup | TC max abs vs FP64 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 1 | 127.020 | 123.328 | 1.030x | 0.002030 |
| 1 | 128 | 8 | 71.600 | 80.544 | 0.889x | 0.009517 |
| 1 | 128 | 16 | 71.200 | 79.872 | 0.891x | 0.008647 |
| 1 | 128 | 32 | 71.216 | 80.112 | 0.889x | 0.015470 |
| 1 | 128 | 64 | 71.168 | 80.144 | 0.888x | 0.014953 |
| 8 | 128 | 16 | 71.408 | 79.984 | 0.893x | 0.014107 |
| 32 | 128 | 16 | 92.144 | 80.352 | **1.147x** | 0.014752 |
| 8 | 1,024 | 16 | 276.480 | 343.920 | 0.804x | 0.033309 |

The largest Tensor-Core error against the float64 recurrence from the same
quantized inputs is `0.033309` at length 1,024, below the audit's coarse `0.1`
feasibility ceiling. This is not yet a model-level accuracy or center-sign
certificate. The RTX 2070 SUPER has FP16 Tensor Cores but no TF32 path.

## Compiler consequence

The exact decomposition and the hardware choice should remain separate:

1. the algebra compiler certifies irreducible type, aligned copies, and the
   exact (V\otimes\mathbb R^m) layout;
2. the backend emits both a register-resident matrix-vector schedule and a
   16-column FP16 `mma.sync` schedule;
3. a per-device shape table selects between them under an explicit numerical
   tolerance.

For the maintained Pure Spin(8) faithful cache, each of `8v`, `8s+`, and
`8s-` currently has multiplicity one. Its existing float32 scalar recurrence
therefore remains the honest default. Tensor-Core packing becomes relevant
for widened channels, repeated isotypic copies, or dense controller GEMMs; it
is not automatically useful for a 24-scalar cache.

## Evidence and reproduction

- Runner: [`benchmark_spin8_isotypic_tensor_core.py`](../benchmark_spin8_isotypic_tensor_core.py)
- CPU theorem gates: [`test_spin8_isotypic_tensor_core.py`](../test_spin8_isotypic_tensor_core.py)
- Artifact: [`spin8_isotypic_tensor_core_rtx2070s_20260821.json`](artifacts/spin8_isotypic_tensor_core_rtx2070s_20260821.json)
- Artifact SHA-256: `85accab67560ccdf0f017e31d671ded5518861c89f0cf71550d1a15fed31fefa`
- Source compiled checkpoint SHA-256: `40d71f4e93b957e2ece9b308c02b95c58c46b0e55c9d2f60969e9924a0aa2305`

Run from `SSM-Models/` in the WSL CUDA environment:

```bash
python3 benchmark_spin8_isotypic_tensor_core.py \
  --checkpoint /path/to/compiled_latent_pure_spin8_seed1.pt
```

## Claim boundary

This closes the feasibility question “can exact isotypic multiplicity compile
to actual Tensor-Core instructions?” for the tested real eight-dimensional
block and SM75 toolchain. It does not establish a universal speedup, an
end-to-end trained model improvement, acceptable long-horizon FP16 semantics,
or transfer to complex/quaternionic Schur blocks. The last item is a distinct
layout-and-arithmetic problem and remains open.
