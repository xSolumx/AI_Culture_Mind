# Exact Macro Compiler and Workstation Benchmark

**Exact compiler plus empirical CPU/CUDA benchmark — 2026-08-17**

**Status:** the symmetric $N*H*N$ distribution has been compiled into exact
finite matrix dictionaries, and both compiled layouts beat online three-letter
construction at every recorded batch size on the reference CPU and RTX 2070
SUPER. The direct labelled table is consistently faster than the more compact
deduplicated layout.

**Exact compiler source:**
[`mixed_monomial_golden_macro_compiler.py`](../../src/mixed_monomial_golden_macro_compiler.py)

**Benchmark source:**
[`benchmark_mixed_monomial_golden_macro.py`](../../src/benchmark_mixed_monomial_golden_macro.py)

**Exact artifact:**
[`mixed_monomial_golden_macro_compiler_20260817.json`](../../artifacts/mixed_monomial_golden_macro_compiler_20260817.json)

**Exact artifact SHA-256:**
`9595578918bd46387ce5773607d1ded3d6117b11cb2b2353b586a1c6fc0cd438`

**Empirical artifact:**
[`mixed_monomial_golden_macro_benchmark_20260817.json`](../../artifacts/mixed_monomial_golden_macro_benchmark_20260817.json)

**Empirical artifact SHA-256:**
`93103d6fa5b4c36a43c89d8cac012d22deaa41a020c516d4c2b9ad9fc2bd8add`

## Exact compilation result

The sandwich theorem samples a labelled triple

\[
(n_1,h,n_2)\in S_N\times S_H\times S_N
\]

uniformly and applies $n_1hn_2$. Exact enumeration over
$\mathbb Q(\sqrt5)$ gives:

| View | Labelled triples | Distinct exact matrices | Distinct/triples | Maximum multiplicity |
|---|---:|---:|---:|---:|
| Vector | 867 | 530 | $530/867$ | 12 |
| Positive half-spin | 1,156 | 394 | $197/578$ | 29 |
| Negative half-spin | 1,156 | 394 | $197/578$ | 29 |

The complete multiplicity histograms are:

\[
\begin{aligned}
\text{vector}:&\quad
1^{274},2^{223},3^1,4^{30},12^2,\\
\text{half-spin}:&\quad
1^{148},2^{116},4^{82},8^{38},9^4,12^2,13^2,29^2.
\end{aligned}
\]

Here $m^k$ means that $k$ distinct matrices have multiplicity $m$; it is not a
group-power notation.

For every view the exact compiler verifies:

- all source steps are in $SO(8)$;
- the lookup covers every labelled triple;
- multiplicities sum to 867 or 1,156;
- matrix transposition pairs inverses with the same multiplicity; and
- multiplicity-weighted dictionary averaging equals $M_NM_HM_N$ exactly.

Uniform sampling over the **distinct** matrices would discard these
multiplicities and define a different probability measure. None of the
sandwich spectral bounds may be transferred to that flattened measure without
a new certificate.

## Storage layouts

Two runtime layouts were tested:

1. **Deduplicated:** one matrix per exact value plus a labelled-triple-to-matrix
   lookup.
2. **Labelled:** one matrix per labelled triple, retaining duplicates and
   eliminating the second lookup.

| View | Deduplicated FP32 table + uint16 lookup | Full labelled FP32 table |
|---|---:|---:|
| Vector | 137,414 bytes | 221,952 bytes |
| Either half-spin | 103,176 bytes | 295,936 bytes |

The exact compiler proves that every deduplicated index fits in 16 bits. The
PyTorch benchmark itself uses `int64` indices because advanced indexing expects
that runtime dtype; the byte table above describes a deployable custom layout,
not the temporary PyTorch tensor footprint.

## Benchmark design

Recorded platform:

- Windows 11;
- Intel Family 6 Model 158 CPU, with PyTorch intra-op and inter-op threads both
  fixed to one;
- NVIDIA GeForce RTX 2070 SUPER, compute capability 7.5;
- Python 3.12.10, PyTorch 2.12.0+cu130, CUDA 13.0;
- `float32`, deterministic seed 20260817;
- batches 1, 64, 1,024, and 16,384; and
- explicit CUDA synchronization around every timed call.

Two operations were measured:

- **transition materialization:** two online batched matrix products versus one
  compiled matrix gather;
- **final-state application:** three online batched matrix-vector products
  versus compiled gather plus one batched matrix-vector product.

Warmups and 30–500 timed repeats were used according to batch size. The
artifact stores minimum, median, p10, and p90 latency rather than one best-case
number.

## Empirical result

The following ranges span the vector and both half-spin views for the direct
labelled layout:

| Device | Batch | Transition speedup | Final-state speedup |
|---|---:|---:|---:|
| CPU | 1 | $1.92$–$2.42\times$ | $1.60$–$1.90\times$ |
| CPU | 64 | $2.66$–$3.01\times$ | $1.92$–$2.14\times$ |
| CPU | 1,024 | $4.57$–$4.81\times$ | $2.65$–$2.72\times$ |
| CPU | 16,384 | $5.65$–$5.68\times$ | $2.78$–$2.82\times$ |
| CUDA | 1 | $2.14$–$3.15\times$ | $2.25$–$2.31\times$ |
| CUDA | 64 | $3.14$–$3.17\times$ | $2.25$–$2.40\times$ |
| CUDA | 1,024 | $3.10$–$3.31\times$ | $2.30$–$2.35\times$ |
| CUDA | 16,384 | $8.06$–$8.07\times$ | $1.98$–$3.14\times$ |

Across all 24 view/device/batch cells, direct labelled transition lookup ranged
from $1.916\times$ to $8.071\times$, and labelled final-state application from
$1.597\times$ to $3.135\times$. The deduplicated layout also won every
transition comparison and all final-state comparisons, but one CUDA cell was
only $1.043\times$; its extra indirection therefore buys memory efficiency,
not peak speed.

The maximum sampled float32 disagreement between online construction and
either compiled layout was

\[
4.76837158203125\times10^{-7},
\]

below the preregistered $2\times10^{-6}$ acceptance threshold.

## What this means for a scan

For a discrete three-token block where only the block endpoint is needed, the
direct labelled table is the clear local implementation choice on this
machine. It keeps the exact labelled probability law, occupies less than
300 kB per view in FP32, eliminates online matrix composition, and wins every
recorded microbenchmark.

It is **not** yet an end-to-end SSM acceleration result. A causal layer that
must emit every intermediate prefix cannot replace three transitions with one
macro without separately reconstructing the two interior states. The benchmark
also excludes token decoding, full parallel scan scheduling, backward passes,
training, and a custom packed-uint16 CUDA kernel. Results are local to the
recorded workstation and software stack.

That next implementation is now complete in the
[every-prefix chunk continuation](MIXED_MONOMIAL_GOLDEN_CHUNK_SCAN_RESULTS.md).
A stacked `24x8` operator emits all three interior states and won every recorded
median through 192 primitive steps, although large-batch CPU rows were nearly
break-even. Fused parallel scan and backward remain unmeasured.

## Replay

From `Spin-Space-Research`:

```powershell
$env:PYTHONPATH = "src"
python src/mixed_monomial_golden_macro_compiler.py `
  --output artifacts/mixed_monomial_golden_macro_compiler_20260817.json
python src/benchmark_mixed_monomial_golden_macro.py `
  --devices cpu,cuda `
  --batch-sizes 1,64,1024,16384 `
  --output artifacts/mixed_monomial_golden_macro_benchmark_20260817.json
```
