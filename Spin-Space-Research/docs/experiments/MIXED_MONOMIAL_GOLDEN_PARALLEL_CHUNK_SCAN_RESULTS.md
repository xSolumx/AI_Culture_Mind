# Two-Stage Parallel Chunk Scan: Forward and Backward

**Validated implementation plus empirical CPU/CUDA benchmark — 2026-08-17**

**Status:** the compiled scan now uses the maintained ordered work-efficient
parallel tree. It scans one endpoint per three-letter chunk, then applies all
selected $24\times8$ local-prefix operators in parallel. Float64 unit tests
match the sequential recurrence and all input gradients. On the final
workstation grid, compiled forward and forward-plus-initial-state-backward beat
the primitive parallel scan in every recorded cell.

**Implementation:**
[`mixed_monomial_golden_parallel_chunk_scan.py`](../../src/mixed_monomial_golden_parallel_chunk_scan.py)

**Benchmark:**
[`benchmark_mixed_monomial_golden_parallel_chunk_scan.py`](../../src/benchmark_mixed_monomial_golden_parallel_chunk_scan.py)

**Artifact:**
[`mixed_monomial_golden_parallel_chunk_scan_benchmark_20260817.json`](../../artifacts/mixed_monomial_golden_parallel_chunk_scan_benchmark_20260817.json)

**Artifact SHA-256:**
`73816cbcf8733ad9e2a4be8a87376ebfa96e369be64b3aa4093b3f0cafa93900`

## Algorithm

For $C$ chunks, the primitive chronological sequence has length $3C$:

\[
R_0,H_0,L_0,R_1,H_1,L_1,\ldots.
\]

The primitive parallel control scans all $3C$ matrices. The compiled algorithm
uses the exact endpoint

\[
T_c=L_cH_cR_c
\]

and local-prefix operator

\[
P_c=
\begin{bmatrix}
R_c\\H_cR_c\\L_cH_cR_c
\end{bmatrix}.
\]

It then performs:

1. an ordered inclusive scan over $T_0,\ldots,T_{C-1}$;
2. a one-position shift of those endpoint prefixes to obtain each chunk's
   incoming state; and
3. all products $P_c x_{3c}$ in parallel, reshaped to the $3C$ causal outputs.

Both paths use the same work-efficient Blelloch-style tree maintained by the
Pure Rotor octonion-operator scan. Chronological composition is
`later @ earlier`; all updates are out of place for autograd.

## Exact operation-count reduction

Excluding the final local-prefix applications, the maintained tree uses

\[
3P-2
\]

matrix products for nontrivial length, where $P$ is the next power of two.

| Chunks $C$ | Primitive length $3C$ | Primitive tree products | Compiled tree products |
|---:|---:|---:|---:|
| 4 | 12 | 46 | 10 |
| 16 | 48 | 190 | 46 |
| 64 | 192 | 766 | 190 |

The compiled tree therefore removes about three quarters of the expensive
$8\times8$ scan compositions before local expansion.

## Implementation parity

Independent float64 tests cover chunk counts 1, 3, 5, and 8 and verify:

- sequential recurrence equals the primitive work-efficient scan;
- sequential recurrence equals primitive Hillis–Steele;
- both equal the compiled two-stage scan;
- precompiled endpoint/local tables equal differentiable on-the-fly
  compilation; and
- gradients with respect to $L$, $H$, $R$, and the initial state agree with
  the sequential oracle.

These tests use arbitrary stable dense matrices rather than relying on finite
group identities, so scan correctness is not limited to exact orthogonal test
points.

## Workstation benchmark

The artifact covers all three views on single-thread CPU and synchronized RTX
2070 SUPER CUDA, with:

- batches 1, 64, and 1,024;
- chunks 4, 16, and 64;
- primitive lengths 12, 48, and 192;
- float32 forward; and
- forward plus gradient with respect to the initial state.

Every backward cell has at least ten synchronized timed repeats and five
warmups. This strengthened protocol replaced a preliminary low-repeat screen;
only the final artifact is maintained.

## Results

Across all 54 cells:

| Path | Minimum speedup | Maximum speedup |
|---|---:|---:|
| Forward | $1.160\times$ | $3.984\times$ |
| Forward + initial-state backward | $1.012\times$ | $3.392\times$ |

Ranges across views and chunk lengths were:

| Device | Batch | Forward range | Forward + backward range |
|---|---:|---:|---:|
| CPU | 1 | $1.30$–$1.41\times$ | $1.01$–$1.10\times$ |
| CPU | 64 | $2.00$–$2.79\times$ | $1.26$–$2.05\times$ |
| CPU | 1,024 | $2.30$–$3.69\times$ | $1.83$–$2.79\times$ |
| CUDA | 1 | $1.19$–$1.43\times$ | $1.06$–$1.15\times$ |
| CUDA | 64 | $1.16$–$1.45\times$ | $1.01$–$1.11\times$ |
| CUDA | 1,024 | $1.71$–$3.98\times$ | $1.05$–$3.39\times$ |

The maximum forward discrepancy from the sequential recurrence was

\[
2.384185791015625\times10^{-6},
\]

and the maximum initial-state gradient disagreement between primitive and
compiled parallel scans was

\[
7.450580596923828\times10^{-9}.
\]

## Interpretation

This is the first maintained result on this line that simultaneously retains
every causal prefix, uses a parallel associative scan, validates gradients,
and shows a local forward/backward speed benefit. The gain comes from reducing
the scan tree from $3C$ leaves to $C$ leaves, while the exact local operator
restores the three states per chunk.

The minimum backward speedup of $1.012\times$ is effectively near break-even.
It prevents a broad claim that eager compilation robustly accelerates every
shape or device. The stronger gains occur when endpoint scan composition is a
meaningful fraction of runtime; small or launch-bound backward cases leave
less room.

## Remaining boundary

This is still eager PyTorch composition. It is not a fused Triton or raw-CUDA
kernel, and the benchmarked backward scope is the gradient with respect to the
initial state. The unit tests validate gradients through differentiable
$L,H,R$ compilation, but no timed discrete-label gradient, learned continuous
compiler, optimizer step, or full model training path is claimed.

The next implementation target is precise:

- fuse the endpoint work-efficient tree and incoming-state formation;
- fuse indexed $24\times8$ local expansion;
- benchmark full table/state backward, not only the initial-state gradient;
- retain the eager path as the semantic oracle; and
- reject the kernel if the near-break-even cells regress or register pressure
  destroys the large-shape gains.

## Later continuation

The indexed local expansion was subsequently implemented and benchmarked in
[Fused Indexed Local-Prefix Expansion](MIXED_MONOMIAL_GOLDEN_TRITON_LOCAL_PREFIX_RESULTS.md).
That kernel usually won in isolation but exposed the eager endpoint tree as
the dominant full-pipeline cost. The follow-on
[Register-Resident Exact Compiled Chunk Recurrence](MIXED_MONOMIAL_GOLDEN_TRITON_CHUNK_RECURRENCE_RESULTS.md)
therefore removes the tree for the bounded streaming path and wins every
recorded CUDA cell. It has serial chunk depth and initial-state-only backward,
so it complements rather than supersedes this parallel semantic control.

## Replay

From `Spin-Space-Research`:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q tests/test_mixed_monomial_golden_parallel_chunk_scan.py
python src/benchmark_mixed_monomial_golden_parallel_chunk_scan.py `
  --devices cpu,cuda `
  --batch-sizes 1,64,1024 `
  --chunk-counts 4,16,64 `
  --output artifacts/mixed_monomial_golden_parallel_chunk_scan_benchmark_20260817.json
```
