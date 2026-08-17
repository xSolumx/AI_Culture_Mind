# Fused Indexed Local-Prefix Expansion

**Validated implementation plus empirical CUDA benchmark — 2026-08-17**

**Status:** one Triton program now fuses exact labelled-table selection with
the `24x8` local-prefix matrix-vector product. Forward and incoming-state
backward pass CUDA parity tests. The isolated kernel usually improves the
realistic indexed eager path, but the eager endpoint scan absorbs most of that
gain in the complete two-stage pipeline.

**Implementation:**
[`mixed_monomial_golden_triton_local_prefix.py`](../../src/mixed_monomial_golden_triton_local_prefix.py)

**Integrated scan:**
[`mixed_monomial_golden_parallel_chunk_scan.py`](../../src/mixed_monomial_golden_parallel_chunk_scan.py)

**Benchmark:**
[`benchmark_mixed_monomial_golden_triton_local_prefix.py`](../../src/benchmark_mixed_monomial_golden_triton_local_prefix.py)

**Artifact:**
[`mixed_monomial_golden_triton_local_prefix_benchmark_20260817.json`](../../artifacts/mixed_monomial_golden_triton_local_prefix_benchmark_20260817.json)

**Artifact SHA-256:**
`b124288e9e68d0513e66f39e5fd20c5c831b49d824be28d73ae76305040c6f17`

## Kernel contract

For a frozen exact table

\[
P[\ell,h,r]\in\mathbb R^{24\times8}
\]

and one incoming state per chunk, the kernel loads the selected operator and
computes

\[
P[\ell,h,r]x=
\begin{bmatrix}
Rx\\HRx\\LHRx
\end{bmatrix}.
\]

One program owns one `(batch, chunk)` pair. It loads three integer labels, the
selected 192 float32 coefficients, and the incoming eight-vector. The reverse
kernel applies the selected transpose to an output cotangent. The public
dispatcher uses eager PyTorch on CPU, unsupported dtypes, or whenever the
operator table requires gradients.

## Controls and protocol

The synchronized RTX 2070 SUPER grid covers all three triality views, batches
1, 64, and 1,024, and chunk counts 4, 16, and 64. Every cell compares:

1. optimistic eager application after the `24x8` operators have already been
   gathered;
2. realistic eager indexed lookup plus application; and
3. fused Triton indexed lookup plus application.

Both the isolated local operation and the complete two-stage endpoint-scan
pipeline are timed for forward and forward plus state backward. The complete
pipeline keeps the endpoint tree identical between indexed controls.

## Results

The 27 cells per phase gave:

| Scope | Phase | Speedup vs indexed eager | Median | Winning cells |
|---|---|---:|---:|---:|
| Isolated local expansion | Forward | `0.950-3.238x` | `1.245x` | 26/27 |
| Isolated local expansion | Forward + incoming backward | `0.958-1.409x` | `1.051x` | 25/27 |
| Full two-stage scan | Forward | `0.979-1.380x` | `1.003x` | 16/27 |
| Full two-stage scan | Forward + initial backward | `0.949-1.254x` | `1.003x` | 14/27 |

Against the pre-gathered eager lower bound, the local kernel won only 5/27
forward and 5/27 backward cells; the full pipeline won 5/27 and 3/27. That is
not a defect in the control: it isolates the cost that lookup fusion can
remove and prevents a claim that the matvec itself was universally faster.

At batch 1,024, the median complete-pipeline speedup over realistic indexed
eager was `1.112x` forward and `1.012x` backward. At smaller batches the
complete pipeline stayed near break-even because the eager endpoint tree and
autograd launch overhead dominated.

Maximum disagreement from the preselected eager oracle was
`4.76837158203125e-7` forward and `3.725290298461914e-9` for the measured state
gradient.

## Interpretation

This gate implements the previously missing fused indexed local expansion. It
also falsifies the stronger systems hypothesis that local lookup fusion alone
is sufficient: most isolated gains disappear behind the multi-launch endpoint
tree. That negative localization directly motivates the register-resident
[compiled recurrence continuation](MIXED_MONOMIAL_GOLDEN_TRITON_CHUNK_RECURRENCE_RESULTS.md).

## Boundary

- The exact dictionary is frozen in the Triton path. A trainable table falls
  back to eager autograd.
- Label selection is discrete and has no gradient.
- The endpoint prefix tree is not fused by this kernel.
- Timed backward covers incoming state locally and initial state in the full
  pipeline, not table gradients or optimizer steps.
- This is a kernel benchmark, not an end-to-end SSM or accuracy comparison.
- Timing conclusions are local to the recorded Windows/CUDA/Turing stack.

## Replay

From `Spin-Space-Research`:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q `
  tests/test_mixed_monomial_golden_triton_local_prefix.py `
  tests/test_mixed_monomial_golden_triton_local_prefix_benchmark.py
python src/benchmark_mixed_monomial_golden_triton_local_prefix.py `
  --batch-sizes 1,64,1024 `
  --chunk-counts 4,16,64 `
  --output artifacts/mixed_monomial_golden_triton_local_prefix_benchmark_20260817.json
```
