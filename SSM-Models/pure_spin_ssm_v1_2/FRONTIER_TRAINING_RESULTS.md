# Pure Spin v1.2 frontier training backend

**Status:** implemented, gradient-validated, and replicated on three natural-data seeds

**Primary artifact:**
[`artifacts/wikitext2_byte_spin_ladder_3seed_summary.json`](artifacts/wikitext2_byte_spin_ladder_3seed_summary.json)

## What is implemented

The training backend now has two raw CUDA schedules:

1. `raw_cuda_controller` fuses coefficient-controller dot products, all 28
   ordered Spin(8) factors, the recurrent update, and exact reverse mode into
   one warp per `(batch, channel)`.
2. `raw_cuda_factorized` lets the dense controller execute as a parallel GEMM,
   then runs the noncommutative ordered factors and recurrence in a raw CUDA
   warp. Its backward reconstructs factor inputs using orthogonality instead of
   storing 28 intermediate states.

The second schedule is faster on the RTX 2070 SUPER. The first schedule placed
only 16 warps on the GPU at the benchmark shape and made each warp compute 28
serial controller dot products. Splitting at the coordinate boundary exposes
the controller to the GPU's dense-matrix machinery and reserves the custom
kernel for the algebra that GEMM cannot express.

Both schedules use PyTorch's current CUDA stream and return gradients for every
trainable input in their scope. Output and full-gradient parity are checked
against the maintained Triton implementations. The coordinate backend accepts
any exact generator subset with between 1 and 28 factors.

At `(B=8, L=256, C=2)`, the isolated 28-coordinate forward+backward kernel took
4,132.5 microseconds versus 12,757.0 for Triton, a 67.6% latency reduction. The
artifact is
[`raw_cuda_factorized_training_rtx2070s_20260821.json`](artifacts/raw_cuda_factorized_training_rtx2070s_20260821.json).
This microbenchmark excludes the controller GEMM; the table below is the
end-to-end model measurement.

## Exact nested-group model

The promoted four-block configuration is

\[
Spin(3)\hookrightarrow Spin(4)\hookrightarrow Spin(6)\hookrightarrow Spin(8),
\]

with respectively `3, 6, 15, 28` plane generators. The implementation selects
all coordinate planes supported on the first `n` axes, so each stage is a
closed Lie subalgebra and an actual subgroup restriction of the same Spin(8)
triality representation. It is not a prefix or a heuristic pruning mask.

The useful special isomorphisms are:

- `SU(1)` is the trivial identity group and supplies no trainable action;
- `Spin(3) ~= SU(2)`;
- `Spin(4) ~= SU(2) x SU(2)`;
- `Spin(6) ~= SU(4)`;
- `Spin(8)` supplies the full `8v + 8+ + 8-` triality action.

All blocks retain the same 24-scalar-per-channel triality state. The residual
stream maps the restricted representations between stages. A dedicated test
proves that the Spin(3) kernel equals the full Spin(8) kernel with the other 25
plane coordinates set exactly to zero, for outputs and gradients.

## Matched natural-data result

Both candidates used the same immutable WikiText-2 UTF-8 byte streams, batches,
optimizer, 300 steps, batch size 8, sequence length 256, RTX 2070 SUPER, Torch
2.10.0+cu130, and seeds 17, 29, and 43. Pure Spin has 626,516 parameters and
official fused Mamba-2 has 623,740, a 0.44% gap.

| seed | Spin ladder bpb | Mamba-2 bpb | Spin tokens/s | Mamba-2 tokens/s | Mamba/Spin |
|---:|---:|---:|---:|---:|---:|
| 17 | 2.718 | 2.500 | 80,967 | 86,264 | 1.065x |
| 29 | 2.732 | 2.470 | 82,596 | 88,597 | 1.073x |
| 43 | 2.747 | 2.500 | 78,529 | 89,111 | 1.135x |
| mean | **2.732** | **2.490** | **80,697** | **87,991** | **1.091x** |

Relative to the first three-seed Triton controller result, mean Pure Spin
throughput rose from 14,987 to 80,697 tokens/s, or 5.38x. Relative to the first
raw controller implementation measured in the final WSL environment at seed
17, the ladder rose from 23,564 to 80,967 tokens/s, or 3.44x.

This does not beat Mamba-2: its mean throughput remains 9.1% lower and its mean
validation result remains worse by 0.242 bits/byte. It also makes no comparison
claim against Jamba, Samba, Nemotron, or any unverified architecture called
"Mamba-4"; those are differently scaled hybrid systems, not matched local
controls.

## Channel-mixer falsification

Two custom non-exponential candidates were implemented:

- `sol_bounded_quadratic`, using a separate value and bounded rational gate;
- `sol_self_gate`, using one projected stream and the map
  `z * (1 + z / sqrt(1 + z^2))`.

Six seed-17 ablations tested fewer parameters, matched parameters, matched
depth, and reallocation of saved gate parameters into model width. None beat
SwiGLU on the matched quality/speed criteria. The closest small model used
30.6% fewer parameters and 8.6% less peak memory, but degraded validation by
0.026 bpb. SwiGLU therefore remains the maintained default.

## Reproduction

```bash
export HF_HOME=/home/local/pure-spin-v12-cache/huggingface
export CUDA_HOME="$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
PYTHONPATH=.. python benchmark.py --offline \
  --steps 300 --batch-size 8 --sequence-length 256 --seed 17 \
  --spin-backend raw_cuda_factorized \
  --spin-group-schedule 3 4 6 8
```

The next speed frontier is a chunk-parallel associative affine scan. The
current kernel still assigns one warp to an entire sequence, so sequence length
remains serial inside each `(batch, channel)`. The next quality frontier is
controller and readout design; the hardware result does not erase the observed
0.242 bpb gap.
