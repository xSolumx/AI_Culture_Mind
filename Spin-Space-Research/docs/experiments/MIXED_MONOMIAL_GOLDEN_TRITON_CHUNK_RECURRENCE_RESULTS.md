# Register-Resident Exact Compiled Chunk Recurrence

**Validated implementation plus empirical CUDA benchmark — 2026-08-17**

**Status:** a one-program-per-sequence Triton recurrence now walks the exact
labelled `N-H-N` dictionary, keeps the eight-state in registers, emits every
causal prefix, and implements an initial-state reverse recurrence. It won all
recorded forward and backward cells against both the optimistic pre-gathered
parallel scan and the hybrid parallel scan with fused local expansion.

**Implementation:**
[`mixed_monomial_golden_triton_chunk_recurrence.py`](../../src/mixed_monomial_golden_triton_chunk_recurrence.py)

**Benchmark:**
[`benchmark_mixed_monomial_golden_triton_chunk_recurrence.py`](../../src/benchmark_mixed_monomial_golden_triton_chunk_recurrence.py)

**Artifact:**
[`mixed_monomial_golden_triton_chunk_recurrence_benchmark_20260817.json`](../../artifacts/mixed_monomial_golden_triton_chunk_recurrence_benchmark_20260817.json)

**Artifact SHA-256:**
`b6897f60a011f2dcb0788fccc32195277a54c600f87199d8d13f75b043f37a7d`

## Algorithm

For one batch sequence, the kernel initializes `state = x0` and executes

\[
y_c=P[\ell_c,h_c,r_c]x_c,
\qquad
x_{c+1}=y_c[16:24],
\]

for `c = 0,...,C-1`. The 24 outputs are stored as the three chronological
states `Rx`, `HRx`, and `LHRx`. One CUDA program owns the entire sequence, so
the eight-state remains register-resident and the Python endpoint-tree launch
sequence disappears.

The reverse kernel walks chunks backward. If `g_c` is the direct 24-output
cotangent and `u_{c+1}` is the future-state cotangent, it computes

\[
u_c=P_c^T g_c + (P_c[16:24])^T u_{c+1}.
\]

This yields the initial-state gradient without saving every recurrent state.

## Parallel-depth tradeoff

This kernel is deliberately not called a parallel prefix scan. It has serial
depth `C` inside each program and GPU parallelism across batch elements. The
maintained two-stage control instead has a logarithmic-depth endpoint tree but
launches and materializes several eager PyTorch stages. The benchmark asks
which tradeoff wins for dimension eight and bounded chunks on the reference
GPU; it does not assert an asymptotic replacement.

## Controls and protocol

The synchronized RTX 2070 SUPER grid covers all three views, batches 1, 64,
and 1,024, and 4, 16, and 64 chunks (12, 48, and 192 primitive steps). Five
paths are compared:

1. parallel endpoint scan with pre-gathered local operators;
2. parallel endpoint scan with eager indexed local expansion;
3. parallel endpoint scan with the fused local Triton kernel;
4. sequential indexed eager recurrence; and
5. the register-resident Triton recurrence.

Forward and forward plus initial-state backward are timed separately. Every
backward cell has at least ten synchronized repeats and five warmups.

## Results

Against the strongest deliberately optimistic parallel control—whose local
operators are already gathered outside the timed region—the candidate gave:

| Phase | Minimum | Median | Maximum | Winning cells |
|---|---:|---:|---:|---:|
| Forward | `4.709x` | `8.344x` | `19.861x` | 27/27 |
| Forward + initial-state backward | `2.213x` | `2.755x` | `6.465x` | 27/27 |

Against the hybrid parallel path with fused local expansion:

| Phase | Minimum | Median | Maximum | Winning cells |
|---|---:|---:|---:|---:|
| Forward | `4.838x` | `9.075x` | `20.023x` | 27/27 |
| Forward + initial-state backward | `2.299x` | `2.833x` | `5.721x` | 27/27 |

The fused candidate's median complete-call time across all cells was
`102.05 us` forward and `618.3 us` for forward plus initial-state backward.
The recorded ranges were `94.8-174.25 us` and `579.75-665.4 us`.

Maximum disagreement from the parallel preselected oracle was
`2.384185791015625e-6` forward and `1.4901161193847656e-8` for the initial
gradient. All declared float32 parity gates passed.

## Interpretation

For these fixed eight-dimensional dictionary transitions and lengths through
64 chunks, launch elimination and register-resident state dominate the lost
logarithmic depth. The increasing large-length forward gains show that the
kernel is not merely removing a constant table-lookup overhead.

This is the strongest systems result on the mixed monomial/golden line so far,
but it is not yet a model result. It establishes a fast associative operator
execution path for known discrete transitions. It does not establish that a
network can infer those transitions from data, that the dictionary improves
prediction quality, or that the layer beats Mamba in an end-to-end task.

## Boundary and next falsifier

- The Triton path treats the exact table as frozen and differentiates only the
  initial state. A trainable table uses the eager oracle.
- Discrete labels are supplied; selector gradients and learned continuous
  transitions are absent.
- Serial depth may lose for much longer sequences, larger state dimension, or
  very different hardware. The tested maximum is 64 chunks.
- No optimizer step, parameter gradient, full model, or SSM accuracy is timed.
- The next model-level gate is a matched task with a learned label/router
  front-end, identical emitted-prefix supervision, and Mamba/GRU/parallel-scan
  controls under measured compute—not another kernel-only aggregate.

## Replay

From `Spin-Space-Research`:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q `
  tests/test_mixed_monomial_golden_triton_chunk_recurrence.py `
  tests/test_mixed_monomial_golden_triton_chunk_recurrence_benchmark.py
python src/benchmark_mixed_monomial_golden_triton_chunk_recurrence.py `
  --batch-sizes 1,64,1024 `
  --chunk-counts 4,16,64 `
  --output artifacts/mixed_monomial_golden_triton_chunk_recurrence_benchmark_20260817.json
```
