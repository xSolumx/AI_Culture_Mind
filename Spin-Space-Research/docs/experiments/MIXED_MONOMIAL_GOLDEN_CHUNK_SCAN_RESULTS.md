# Exact Every-Prefix Chunk Compiler and Recurrent Benchmark

**Exact compiler plus empirical CPU/CUDA recurrence benchmark — 2026-08-17**

**Status:** one exact $24\times8$ labelled operator emits all three causal
prefix states of an $N$–$H$–$N$ chunk. Across recurrent sequences of 3–192
primitive steps, the compiled every-prefix path beat three primitive
applications in all 72 recorded view/device/batch/length cells, although the
large-batch CPU margin approached break-even.

**Exact compiler source:**
[`mixed_monomial_golden_chunk_compiler.py`](../../src/mixed_monomial_golden_chunk_compiler.py)

**Benchmark source:**
[`benchmark_mixed_monomial_golden_chunk.py`](../../src/benchmark_mixed_monomial_golden_chunk.py)

**Exact artifact:**
[`mixed_monomial_golden_chunk_compiler_20260817.json`](../../artifacts/mixed_monomial_golden_chunk_compiler_20260817.json)

**Exact artifact SHA-256:**
`ed1ae7e8ac98c5e037be4e45d10f22ec3236e7d6f8337fbc2b9f9a499e13e5de`

**Empirical artifact:**
[`mixed_monomial_golden_chunk_benchmark_20260817.json`](../../artifacts/mixed_monomial_golden_chunk_benchmark_20260817.json)

**Empirical artifact SHA-256:**
`35aef11f6e2577ac5848c800d5afb1dcbd38def2db3dcb5f3deb4dc820793f74`

## Exact causal construction

For the matrix word

\[
W=LHR
\]

acting on a column state $x$, the causal primitive order is $R$, then $H$,
then $L$. Define

\[
P(L,H,R)=
\begin{bmatrix}
R\\
HR\\
LHR
\end{bmatrix}
\in\mathbb Q(\sqrt5)^{24\times8}.
\]

One matrix-vector product returns

\[
P(L,H,R)x=
\begin{bmatrix}
x_1\\x_2\\x_3
\end{bmatrix}
=
\begin{bmatrix}
Rx\\HRx\\LHRx
\end{bmatrix}.
\]

Thus the compiler does not discard or approximate the two interior prefixes.
The exact replay checks every labelled triple and verifies blockwise that:

1. rows 0–7 equal $R$;
2. rows 8–15 equal $HR$; and
3. rows 16–23 equal the independently compiled exact endpoint matrix $LHR$.

## Table sizes

Unlike endpoint matrices, the complete prefix paths do not deduplicate: each
labelled triple has a distinct $24\times8$ operator.

| View | Prefix operators | FP32 table | FP16 table |
|---|---:|---:|---:|
| Vector | 867 | 665,856 bytes | 332,928 bytes |
| Positive half-spin | 1,156 | 887,808 bytes | 443,904 bytes |
| Negative half-spin | 1,156 | 887,808 bytes | 443,904 bytes |

Every FP32 table is below one decimal megabyte. The FP16 byte counts are exact
storage arithmetic only; FP16 does not preserve the
$\mathbb Q(\sqrt5)$ identities or spectral certificates.

## Recurrent benchmark

The benchmark uses the same Windows 11, single-thread CPU, RTX 2070 SUPER,
Python 3.12.10, PyTorch 2.12.0+cu130, and synchronized CUDA environment as the
endpoint compiler benchmark. It covers:

- batch sizes 1, 64, and 1,024;
- 1, 4, 16, and 64 chunks;
- primitive sequence lengths 3, 12, 48, and 192;
- all three triality views; and
- CPU and CUDA.

For each cell it times two distinct recurrent paths:

- **endpoint-only:** three primitive $8\times8$ applications per chunk versus
  one compiled endpoint application;
- **every-prefix:** three primitive applications with every state retained
  versus one compiled $24\times8$ application with all three outputs retained.

The compiled state at prefix three becomes the input of the next chunk, so the
long rows test recurrent numerical error rather than independent blocks.

## Results

Across all 72 cells:

| Output contract | Minimum speedup | Maximum speedup |
|---|---:|---:|
| Endpoint only | $1.565\times$ | $2.688\times$ |
| Every primitive prefix | $1.007\times$ | $2.394\times$ |

Every recorded median favored compilation. Grouped across views and sequence
lengths, the every-prefix ranges were:

| Device | Batch | Every-prefix speedup range |
|---|---:|---:|
| CPU | 1 | $1.37$–$1.63\times$ |
| CPU | 64 | $1.19$–$1.37\times$ |
| CPU | 1,024 | $1.01$–$1.09\times$ |
| CUDA | 1 | $1.19$–$2.38\times$ |
| CUDA | 64 | $1.76$–$2.39\times$ |
| CUDA | 1,024 | $1.74$–$2.39\times$ |

The maximum recurrent float32 disagreement was

\[
2.1457672119140625\times10^{-6}
\]

through 192 primitive steps, below the declared $10^{-4}$ acceptance limit.

## Interpretation

The every-prefix objection is no longer a structural blocker for this fixed
discrete distribution: a stacked local operator can emit all interior states
and won every recorded median. The mechanism is straightforward. A
$24\times8$ multiply performs essentially the same scalar arithmetic as three
$8\times8$ multiplies, but it reduces table gathers, Python dispatch, and CUDA
kernel launches.

That explanation also predicts the observed limit. At CPU batch 1,024, dense
matrix arithmetic is already efficient and dispatch overhead is amortized, so
the compiled advantage shrank as low as $1.007\times$. This is too close to
call a robust large-batch CPU win without replication or a lower-level kernel.
CUDA retained a much larger advantage because launch-count reduction remained
valuable.

## Remaining boundary

This benchmark is a sequential chunk recurrence, not a parallel prefix scan.
It does not include backward passes, parameter gradients, token-to-label
decoding, continuous learned transitions, or end-to-end model quality. The
tables apply only to the fixed finite $N$–$H$–$N$ alphabet.

The natural scan architecture is now two-stage:

1. scan the compiled $LHR$ endpoint matrices across chunks to obtain every
   chunk's incoming state; and
2. apply all selected $24\times8$ local-prefix operators in parallel.

The eager two-stage algorithm and initial-state backward benchmark are now
implemented in the
[parallel chunk continuation](MIXED_MONOMIAL_GOLDEN_PARALLEL_CHUNK_SCAN_RESULTS.md).
They pass, including all-input gradient parity in float64, but remain unfused.
The next hard test is the custom CUDA/Triton kernel and full table/model
backward. Near-break-even rows remain explicit rather than averaged away.

That custom-kernel continuation is now recorded in
[Fused Indexed Local-Prefix Expansion](MIXED_MONOMIAL_GOLDEN_TRITON_LOCAL_PREFIX_RESULTS.md)
and
[Register-Resident Exact Compiled Chunk Recurrence](MIXED_MONOMIAL_GOLDEN_TRITON_CHUNK_RECURRENCE_RESULTS.md).
The latter accelerates frozen discrete labels and initial-state backward; full
table/model backward and learned transitions remain open.

## Replay

From `Spin-Space-Research`:

```powershell
$env:PYTHONPATH = "src"
python src/mixed_monomial_golden_chunk_compiler.py `
  --output artifacts/mixed_monomial_golden_chunk_compiler_20260817.json
python src/benchmark_mixed_monomial_golden_chunk.py `
  --devices cpu,cuda `
  --batch-sizes 1,64,1024 `
  --chunk-counts 1,4,16,64 `
  --output artifacts/mixed_monomial_golden_chunk_benchmark_20260817.json
```
