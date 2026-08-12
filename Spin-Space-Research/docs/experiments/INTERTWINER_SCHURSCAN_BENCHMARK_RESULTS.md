# Intertwiner SchurScan: work-efficient scan and benchmark results

> **Current systems pointer, 2026-08-10.** This remains the eager benchmark for
> the triangular intertwiner scan. The later official FLA measurement applies
> to the co-moving transported delta recurrence, not to a fused intertwiner
> binder. End-to-end triality-fusion remains open.

- **Date:** 2026-08-07
- **Status:** maintained implementation result; eager PyTorch, not a fused kernel
- **Implementation:** [`intertwiner_schurscan.py`](../../src/intertwiner_schurscan.py)
**Benchmark:** [`benchmark_intertwiner_schurscan.py`](../../src/benchmark_intertwiner_schurscan.py)

## Result in one paragraph

The original SchurScan harness used a Hillis--Steele prefix tree.  It had
logarithmic dependency depth, but performed \(O(N\log N)\) affine
compositions.  The maintained implementation now also contains an ordered
Blelloch-style tree with \(O(N)\) composition work, supports noncommuting
actions at arbitrary positive sequence lengths, and preserves PyTorch
autograd.  On the local RTX 2070 SUPER at batch 8 and length 4,096, the
homogeneous work-efficient backend reproduced a median forward time of
6.744 ms, versus 24.196 ms for Hillis--Steele: a 3.59-fold speedup.  The
corresponding composition counts are 12,286 and 45,057, a 3.67-fold arithmetic
ratio.  This is an implementation speedup over the earlier eager scan, not a
claim of superiority over fused production SSMs.

## 1. The ordered work-efficient tree

For chronological transformations \(M_0,M_1,\ldots,M_{N-1}\), define

\[
\operatorname{compose}(L,R)=RL.
\]

The order matters: the later, right-hand segment acts after the earlier,
left-hand segment.  An upsweep stores \(RL\) at every parent.  During the
downsweep, a node with exclusive prefix \(P\) sends

\[
P_L=P,
\qquad
P_R=LP
\]

to its two children.  A leaf \(M_t\) then obtains the inclusive prefix

\[
M_tP_t=M_tM_{t-1}\cdots M_0.
\]

This convention was tested with dense, noncommuting matrices, so the test does
not accidentally pass because scalar or diagonal actions commute.

Let \(P=2^{\lceil\log_2N\rceil}\).  For \(N>1\), the implemented tree uses

\[
(P-1)+(P-1)+P=3P-2
\]

compositions: upsweep, downsweep, and exclusive-to-inclusive conversion.  The
reference Hillis--Steele tree uses

\[
\sum_{2^k<N}(N-2^k)
\]

compositions.  Both have \(O(\log N)\) dependency depth, but the constants
differ:

\[
d_{\mathrm{HS}}=\lceil\log_2N\rceil,
\qquad
d_{\mathrm{WE}}=2\lceil\log_2N\rceil+1.
\]

Thus the work-efficient tree performs less total arithmetic but has a longer
critical path.  A crossover should be expected rather than assumed.

## 2. Affine semidirect product

An affine transition is the pair \((A,b)\) acting by \(x\mapsto Ax+b\).
Composition is

\[
(A_2,b_2)\circ(A_1,b_1)
=
(A_2A_1,\;A_2b_1+b_2).
\]

The code exposes two work-efficient representations:

1. **specialized affine:** stores \((A,b)\) directly, minimizing arithmetic
   and memory;
2. **homogeneous:** stores
   \(\begin{psmallmatrix}1&0\\b&A\end{psmallmatrix}\), increasing arithmetic
   but reducing eager-PyTorch launch count.

The homogeneous backend is now the default because it wins the maintained
long-sequence CPU and CUDA timing gates.  The direct affine tree remains the
preferred target for a future fused kernel: its extra eager launches, not its
algebra, are the present bottleneck.

## 3. Correctness gates

The following maintained gates pass:

- noncommutative ordered prefixes at lengths
  \(1,2,3,5,7,9,17,31,64,127\), in float32 and float64;
- affine action and drive gradients versus a sequential recurrence, to
  float64 roundoff;
- end-to-end SchurScan gradients through all actions, drives, initial states,
  and the bilinear tensor;
- a contractive length-2,048 recurrence;
- Hillis--Steele, direct affine tree, homogeneous tree, lifted proof scan, and
  recurrent agreement;
- malformed tensor and empty-sequence rejection.

The generic SO(3) diagnostic now reports:

| Gate | Maximum absolute discrepancy |
|---|---:|
| default work-efficient scan vs recurrence | \(3.55\times10^{-14}\) |
| Hillis--Steele vs recurrence | \(2.84\times10^{-14}\) |
| direct affine tree vs recurrence | \(2.84\times10^{-14}\) |
| 19D homogeneous proof lift vs recurrence | \(4.26\times10^{-14}\) |
| SO(3) cross-product equivariance | \(1.78\times10^{-15}\) |

These discrepancies are floating-point reassociation effects.  The recurrences
are algebraically identical; they are not claimed to be bitwise identical.

## 4. Benchmark protocol

Inputs were constructed on CPU from a canonical seed and moved to the target
device before timing.  Every action was a contractive orthogonal transition,

\[
A_t=0.99\exp(0.018K_t),\qquad K_t^\mathsf{T}=-K_t,
\]

and the bilinear map was the maintained \(8\times8\times8\) triality tensor.
Timing excluded input generation.  CUDA measurements used events and explicit
synchronization.  Reported values are medians after warmup; raw artifacts also
retain minima, means, standard deviations, and 20th/80th percentiles.

The host was limited to six PyTorch threads:

- CPU: Intel Core i7-9700K class host, float64, batch 1;
- GPU: NVIDIA GeForce RTX 2070 SUPER, compute capability 7.5, float32,
  batch 8;
- PyTorch: 2.12.0+cu130;
- GPU TF32: disabled;
- fused Triton kernel: absent.

### CUDA forward sweep

Times are milliseconds.  `WE-hom` is the homogeneous work-efficient tree and
`WE-aff` is the direct affine tree.

| Length | Hillis--Steele | WE-aff | WE-hom | HS / WE-hom |
|---:|---:|---:|---:|---:|
| 16 | 2.656 | 5.608 | 3.033 | 0.88x |
| 64 | 3.547 | 7.631 | 3.780 | 0.94x |
| 256 | 4.287 | 9.722 | 4.531 | 0.95x |
| 512 | 4.506 | 10.710 | 4.886 | 0.92x |
| 1,024 | 6.017 | 11.284 | 5.065 | 1.19x |
| 2,048 | 11.295 | 12.657 | 7.763 | 1.45x |
| 4,096 | 24.148 | 13.288 | 6.745 | 3.58x |

The reversed-order replication used 20 repeats at the three longest lengths
and is the primary long-length timing record:

| Length | Hillis--Steele | WE-aff | WE-hom | HS / WE-hom |
|---:|---:|---:|---:|---:|
| 1,024 | 6.053 | 11.776 | 5.233 | 1.16x |
| 2,048 | 12.110 | 12.684 | 5.537 | 2.19x |
| 4,096 | 24.196 | 13.461 | 6.744 | 3.59x |

The replication corrected a non-monotone point in the first sweep: the first
length-2,048 `WE-hom` row was 7.763 ms, whereas the reverse-order value was
5.537 ms.  Since the replicated sequence is monotone

\[
5.233\ \mathrm{ms}\;(N=1024)
<5.537\ \mathrm{ms}\;(N=2048)
<6.744\ \mathrm{ms}\;(N=4096),
\]

the first 2,048 value is retained as an order/runtime-state-sensitive outlier,
not interpreted as a hardware scaling law.  The 4,096 speedup itself is stable
across both orders: 3.58x and 3.59x.

At length 4,096 the homogeneous tree processed 4.86 million tokens/s.  Its
p20--p80 interval was 6.720--7.389 ms.  The direct affine tree used 56.2% of
the incremental CUDA memory of Hillis--Steele; the homogeneous tree used
79.3%.

The worst same-dtype relative discrepancy anywhere in the CUDA sweep was
\(1.07\times10^{-6}\).  The float64 CPU maximum was
\(2.42\times10^{-15}\).

### CPU forward sweep

| Length | Hillis--Steele | WE-hom | HS / WE-hom |
|---:|---:|---:|---:|
| 64 | 1.049 | 1.363 | 0.77x |
| 256 | 1.657 | 1.785 | 0.93x |
| 512 | 2.394 | 2.204 | 1.09x |
| 1,024 | 3.157 | 2.684 | 1.18x |
| 2,048 | 4.410 | 3.461 | 1.27x |

### Full-gradient timing

The backward gate differentiates actions, drives, all three initial states,
and the triality tensor.  It is more demanding than a forward-only inference
test.

| Device | Length | Hillis--Steele | WE-hom | HS / WE-hom |
|---|---:|---:|---:|---:|
| RTX 2070 SUPER | 512 | 17.112 ms | 19.976 ms | 0.86x |
| RTX 2070 SUPER | 1,024 | 26.849 ms | 21.593 ms | 1.24x |
| i7-9700K | 512 | 10.830 ms | 10.522 ms | 1.03x |
| i7-9700K | 1,024 | 14.742 ms | 13.429 ms | 1.10x |

At CUDA length 1,024, the homogeneous tree used 49.4 MB of incremental
full-gradient memory versus 166.3 MB for Hillis--Steele (29.7%).  The direct
affine tree used 37.7 MB (22.7%) but remained slower in eager execution because
its algebra is split across more launches.  This memory gate measures the
autograd graph, intermediates, and gradients above already-allocated leaf
inputs.

## 5. What was learned

The main engineering result is not merely that one Python function is faster
than another.  It is the measured separation of three resources:

\[
\text{total compositions},\qquad
\text{dependency depth},\qquad
\text{kernel-launch count}.
\]

Hillis--Steele minimizes tree depth but repeats work.  The direct affine tree
minimizes arithmetic and memory but launches both matrix products and
matrix-vector products at each level.  Homogeneous packing performs more
scalar arithmetic per composition, yet wins at long lengths because eager
PyTorch expresses each composition as one batched matrix product.  A fused
affine kernel should combine the arithmetic advantage of `WE-aff` with the
launch behavior of `WE-hom`; that is the next performance gate.

No claim is made here about DVFS, cache hierarchy, SIMD/SM utilization, or wave
quantization.  Those mechanisms require clock telemetry and a hardware
profiler.  The benchmark establishes the crossover and arithmetic correlation,
not a unique microarchitectural cause.  CUDA Events bound execution on the
active stream; cached-intermediate allocation remains part of the eager tensor
program being measured.

## 6. Triton decision

The NVIDIA page originally suggested for installation describes
[NVIDIA Dynamo with Triton Inference Server](https://developer.nvidia.com/dynamo-triton),
which is deployment software, not the Triton GPU kernel language used by
PyTorch Inductor.  It was therefore not installed.

The local GPU is Turing (`sm75`).  The Windows Triton fork documents Turing
support only through Triton 3.2, whereas the installed PyTorch 2.12 line is
paired with Triton 3.7.  Installing an unsupported global combination would
weaken reproducibility.  A custom Triton kernel should instead be developed in
an isolated, pinned environment (or on a supported Linux/Ampere-or-newer
machine) and added as a new backend, never substituted silently into these
eager baselines.

## 7. Claim boundary

Established:

- the work-efficient tree preserves noncommutative order and autograd;
- it reduces composition work from \(O(N\log N)\) to \(O(N)\);
- a reproducible long-sequence crossover occurs on both maintained CPU and
  CUDA systems;
- the length-4,096 CUDA speedup over the previous eager tree replicates.

Not established:

- superiority over fused Mamba, DeltaNet, linear-attention, or vendor scan
  kernels;
- end-to-end model quality, sample efficiency, or state-efficiency advantage;
- a fused-kernel throughput claim;
- bitwise equality between differently associated floating-point scans.

## 8. Artifacts

| Artifact | SHA-256 |
|---|---|
| `artifacts/intertwiner_schurscan_20260807.json` | `1279ccee575f2642c54fabdec5fa0923f0b617e6bb57fbbf8cce58bc696c6d15` |
| `artifacts/intertwiner_schurscan_cuda_rtx2070s_20260807.json` | `2609ba04ff2547fcda518ed124923be70c9dc5d23368789fb718c4b88a4bff8f` |
| `artifacts/intertwiner_schurscan_cuda_long_replication_20260807.json` | `81f7234afe6a369d53affcd49c726e3e6f67f496aec08afa73d87c7f3ee71caa` |
| `artifacts/intertwiner_schurscan_cpu_i7_9700k_20260807.json` | `6c7a41600f1d9a700cd906b3deb5ee6b8574edf7f0d506ae69f034d890d73105` |
| `artifacts/intertwiner_schurscan_cuda_training_rtx2070s_20260807.json` | `940231579aa9c62df49e4321debe48aba1103d6f9cdbd384214d766a5fbec093` |
| `artifacts/intertwiner_schurscan_cpu_training_i7_9700k_20260807.json` | `b1f2ac9c54c6a7ffddac1ccaf5a698fcc8040d3ac70e09ce06dccf2eea484297` |

## Replay

```powershell
$env:PYTHONPATH='src'

python -m pytest -q `
  tests/test_intertwiner_schurscan.py `
  tests/test_benchmark_intertwiner_schurscan.py

python -m benchmark_intertwiner_schurscan `
  --device cuda --dtype float32 --batch 8 `
  --lengths 16 32 64 128 256 512 1024 2048 4096 `
  --warmup 5 --repeats 15 --backward-max-length 256 `
  --lift-max-length 32 --threads 6 `
  --output artifacts/intertwiner_schurscan_cuda_rtx2070s_20260807.json
```
