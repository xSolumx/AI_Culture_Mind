# Raw CUDA recurrence comparison

**Status:** historical materialized-forward comparison; training backend now complete

**Hardware:** NVIDIA GeForce RTX 2070 SUPER, compute capability 7.5

**Artifacts:**
[`FP32`](artifacts/raw_cuda_scan_rtx2070s_20260821.json) and
[`FP16`](artifacts/raw_cuda_scan_fp16_rtx2070s_20260821.json)

**Canonical Git-blob SHA-256:** FP32
`f00b3b2c3fbdadc04472f7620ddcd899961e45964bc5d6059d8b829860d98bfa`;
FP16 `dcb5f872087c293e4e1a10466136662c8c4836ad0da33bce3afb86c6059c55fd`

Shape `(batch=8, length=256, channels=16, representations=3, dimension=8)`,
FP32, 20 warmups and 200 timed repetitions using CUDA events on the current
stream:

| dtype | PyTorch oracle | Triton scalar | raw CUDA | raw reduction vs Triton |
|---|---:|---:|---:|---:|
| FP32 | 43,780.5 us | 175.5 us | 145.1 us | 17.3% |
| FP16 | 43,469.8 us | 183.2 us | 121.6 us | 33.6% |

Both fused implementations had maximum absolute error
`1.1175870895385742e-08` against the FP32 PyTorch oracle. In FP16, raw CUDA's
maximum error was `3.0517578125e-05`, versus Triton's `6.103515625e-05`. This is
a local kernel result, not a general ranking; sequence-length, batch, channel,
backward, and end-to-end sweeps remain necessary.

The FP16 kernel does not use Tensor Cores. It performs an 8-by-8 matrix-vector
product with scalar operations and FP32 shared-state accumulation, writing FP16
between time steps. Its FP16 timing is therefore evidence about this CUDA-core
schedule, not a Tensor-Core implementation claim.

This table covers only the original materialized-action forward recurrence.
The source now also includes raw CUDA forward/backward for controller-fused and
coordinate-factorized training. The promoted end-to-end evidence is in
[`FRONTIER_TRAINING_RESULTS.md`](FRONTIER_TRAINING_RESULTS.md).
