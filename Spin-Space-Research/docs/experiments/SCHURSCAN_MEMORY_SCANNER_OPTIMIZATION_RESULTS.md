# SchurScan local-algebra optimization results

> **Scope after the full-transport campaign.** These eager homogeneous-slot
> timings remain valid for this scanner family. The later co-moving FLA result
> benchmarks a different transported delta core and does not retroactively
> make this SchurScan implementation fused. See
> [`SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md`](SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md).
> A still later 64-slot campaign measures a different inference-only fused
> gathered-state kernel; see
> [`LARGE_SLOT_SEMANTIC_HIERARCHY_RESULTS.md`](LARGE_SLOT_SEMANTIC_HIERARCHY_RESULTS.md).

- **Date:** 2026-08-10
- **Status:** maintained eager-PyTorch systems result on the named local host
- **Frozen addendum:**
  [`SCHURSCAN_MEMORY_SCANNER_BENCHMARK_PREREGISTRATION.md`](SCHURSCAN_MEMORY_SCANNER_BENCHMARK_PREREGISTRATION.md#optimization-addendum-local-homogeneous-slot-blocks)
- **Implementation:**
  [`spin8_triality_memory.py`](../../src/spin8_triality_memory.py) and
  [`benchmark_schurscan_memory_scanners.py`](../../src/benchmark_schurscan_memory_scanners.py)

## Verdict

The best maintained local scanner is now a device- and length-dependent
hybrid. There is still no universal backend winner.

- On the six-thread i7-9700K CPU, packed local homogeneous blocks win at
  lengths 64 and 256. The lower-arithmetic structured representation wins at
  1,024 and 2,048.
- On the RTX 2070 SUPER, packed local homogeneous blocks win at every tested
  length. Hillis--Steele wins at 64 and 256; the ordered work-efficient tree
  wins at 1,024 through 4,096.
- The long-length decisions reproduce when the length order is reversed.
- All rows execute the same 64-scalar recurrence. This is a scanner result,
  not evidence that triality improves memory quality.

The useful algebraic optimization is to preserve the eight independent Schur
blocks while packing each affine eight-vector slot into a local `9 x 9`
homogeneous matrix. This avoids both extremes in the first sweep: several
small eager operations per structured composition and one dense `65 x 65`
matrix per token.

## Local homogeneous factorization

For slot `h`, the recurrence

\[
M_t[h]=r_t[h]A_tM_{t-1}[h]+b_t[h]
\]

is represented exactly by

\[
H_t[h]=
\begin{pmatrix}
1&0\\
b_t[h]&r_t[h]A_t
\end{pmatrix},\qquad
\binom{1}{M_t[h]}=H_t[h]\binom{1}{M_{t-1}[h]}.
\]

Products of the `H_t[h]` are associative and never mix slots. Eight local
matrices use 648 scalar entries per token, versus 4,225 for the dense global
homogeneous lift. The implementation flattens batch and slot axes so each tree
composition becomes one batched matrix multiplication.

This is a representation change, not a recurrence approximation. Irregular
lengths, both tree schedules, the end-to-end packing path, and gradients agree
with the structured recurrence in maintained tests.

## CPU benchmark

Batch 1, float64, six threads, 15 repeats. Times are median milliseconds.
`Local` rows exclude packing; `Local E2E` includes it.

| Length | Structured HS | Structured WE | Local HS | Local WE | Best local E2E | Winner |
|---:|---:|---:|---:|---:|---:|---|
| 64 | 0.699 | 1.159 | 0.582 | 0.488 | 0.607 WE | Local WE |
| 256 | 1.234 | 1.696 | 1.273 | 0.827 | 1.061 WE | Local WE |
| 1,024 | 2.567 | 2.581 | 18.643 | 4.598 | 6.151 WE | Structured HS |
| 2,048 | 3.982 | 3.722 | 46.786 | 10.305 | 12.283 WE | Structured WE |

At length 256, prepacked local WE is `1.49x` faster than the best structured
control. Packing reduces the gain to `1.16x`, so the end-to-end boundary is
reported rather than hidden. At long CPU lengths, `9 x 9` matrix arithmetic
dominates and the factored scalar-plus-action form is better.

The reversed 20-repeat sweep reproduced all three overlapping decisions:

| Length | Structured HS | Structured WE | Local WE | Winner |
|---:|---:|---:|---:|---|
| 256 | 1.284 | 1.669 | 0.825 | Local WE |
| 1,024 | 2.562 | 2.666 | 5.092 | Structured HS |
| 2,048 | 4.784 | 4.292 | 11.201 | Structured WE |

## CUDA benchmark

RTX 2070 SUPER, batch 2, float32, TF32 disabled, 15 repeats. Times are median
milliseconds.

| Length | Structured HS | Dense global homogeneous WE | Local HS | Local WE | Best local E2E | Winner |
|---:|---:|---:|---:|---:|---:|---|
| 64 | 2.351 | 1.007 | 0.960 | 1.110 | 1.119 HS | Local HS |
| 256 | 3.175 | 1.379 | 1.177 | 1.292 | 1.319 HS | Local HS |
| 1,024 | 3.864 | 2.683 | 3.825 | 1.794 | 1.837 WE | Local WE |
| 2,048 | 4.186 | 4.970 | 7.212 | 2.352 | 2.501 WE | Local WE |
| 4,096 | 4.413 | 9.811 | 15.714 | 3.749 | 4.132 WE | Local WE |

The strongest primary improvements over the best original scan-only control
are `1.50x` at length 1,024 and `1.78x` at length 2,048. At length 4,096 the
prepacked gain is `1.18x`; after packing, the local row is still `1.07x`
faster. The winner emits about 2.19 million tokens/s at batch 2.

The reversed 20-repeat sweep is stable:

| Length | Structured HS | Dense global homogeneous WE | Local WE | Winner |
|---:|---:|---:|---:|---|
| 1,024 | 3.886 | 2.690 | 1.708 | Local WE |
| 2,048 | 4.213 | 5.032 | 2.365 | Local WE |
| 4,096 | 4.418 | 9.897 | 3.780 | Local WE |

## CUDA incremental memory at length 4,096

Bytes are allocations above already-constructed scan-only leaves. End-to-end
rows include packing.

| Backend | Incremental memory | Relative to dense global homogeneous scan |
|---|---:|---:|
| Structured HS | 17.69 MB | 3.64% |
| Structured WE | 19.79 MB | 4.07% |
| Local homogeneous WE scan-only | 96.71 MB | 19.90% |
| Local homogeneous WE E2E | 117.94 MB | 24.26% |
| Dense global homogeneous WE scan-only | 486.07 MB | 100% |
| Dense global homogeneous WE E2E | 758.73 MB | 156.10% |

The structured scanner remains the memory winner. The packed local backend
spends about `5.47x` its incremental memory to reduce CUDA latency, while still
using only one fifth of the dense global scan allocation.

## Whole-local-repository method audit

The optimization followed a workspace-wide search rather than treating the
first scanner in isolation.

| Local method | What transfers | Result for this memory scan |
|---|---|---|
| [Schur/isotypic factorization](../../../SSM-Models/schur_scan.py) | Keep irreducible or multiplicity blocks separate instead of materializing a Kronecker/global action | **Used.** It is the reason eight local affine matrices suffice. |
| [Mamba-3 augmented affine state](../../../Spin8-SSM-Benchmark/mamba3_reference.py) | Turn an affine recurrence into a small homogeneous matrix product | **Used.** Applied independently inside each Schur block. |
| [Precomputed GA tables and GEMM packing](../../../Spin8-SSM-Benchmark/spinor_delta_ssm.py) | Fixed algebra tables and regular matrix contractions can beat larger eager einsums | **Supports the implementation choice.** It does not change the recurrence or establish a rotor advantage. |
| Walsh codes, tight frames, and the Welch bound | Orthogonal codes give exact retrieval for `K <= H`; tight frames minimize average code-correlation energy for `K > H` | **Quality-side only.** They do not accelerate the prefix scan or evade the rank bound. |
| Signed-set/Walsh Dirac reductions | Parseval and character support control aggregate signed-sector energy | **No memory theorem transfers.** Aggregate energy is not a pointwise sign or worst-query guarantee. |
| Rotor and Householder parameterizations | Compile a supplied structured orthogonal action before scanning | **Same scanner.** Parameterization does not create more state or a better prefix law. |
| Low-rank delta/DeltaProduct ideas | A faithful chunkwise kernel can exploit rank-one or short-product structure | **Now measured separately.** Official FLA compact-WY and fused-recurrent DeltaRule kernels decisively beat the eager scanners on the transport-free recurrence; the standard operator does not implement the programme's noncommuting value transport. |

The signed-set boundary is especially important. For `K > H`,

\[
\frac{1}{K}\sum_q\sum_{k\ne q}\langle c_q,c_k\rangle^2
\ge \frac{K-H}{H}.
\]

This equals expected relative squared retrieval error only for independent,
zero-mean isotropic stored values. It is not a deterministic per-query bound
for correlated or adversarial values. Truncated Walsh tight frames can attain
the average bound while having coherence one; simplex/equiangular designs are
better candidates when worst-case crosstalk matters. None permits exact linear
storage of arbitrary `K > H` values.

At the present eight-slot width, an FWHT is not the scan bottleneck and was not
added. It becomes worth benchmarking only if code transforms grow well beyond
eight channels or enter the timed token path.

## Updated best-by-regime answer

| Regime | Best supported local method | Boundary |
|---|---|---|
| Supplied hard keys and correct actions | Direct addressed slots | Triality slots are an exact orthogonal gauge, so direct is simpler with identical quality. |
| Incomplete cross-view action observations | Shared triality representation family | This is the one demonstrated triality-specific identification advantage; it is not a capacity win. |
| Known bilinear equivariant coupling on restricted support | One-parameter intertwiner SchurScan | SO(3) reproduces the extrapolation effect, so the mechanism is general equivariance. |
| CPU prefix execution on this host | Local homogeneous WE through 256; structured HS at 1,024; structured WE at 2,048 | Hardware- and shape-specific eager result. |
| CUDA prefix execution in eager PyTorch on this host | Local homogeneous HS through 256; local homogeneous WE from 1,024 to 4,096 | Fastest measured local eager scanner, not the overall kernel winner. |
| CUDA transport-free DeltaRule on this host | Official FLA fused recurrent through 2,048; official FLA compact-WY chunk at 4,096 | Frozen three-process result; not a benchmark of the noncommuting cross-view action. |
| Overcomplete linear codebook | Tight frame for expected isotropic MSE; simplex/equiangular design for more uniform crosstalk | Neither defeats the rank bound or guarantees adversarial retrieval. |
| Learned soft distributed addressing | Maintained learned-key delta loses to hard/discretized slots in the frozen quality campaign | The official fused kernel has not yet been extended to the programme's noncommuting transported-value task. |

## Dirac--Gram dependency

The unrestricted Dirac--Gram global proof is not a prerequisite. It concerns
global sensing/design optimality. The block factorization, gauge equivalence,
equivariant identification gate, signed-code rank bound, and scanner benchmark
are independent local algebra and systems statements.

## Artifacts

| Artifact | SHA-256 |
|---|---|
| `artifacts/schurscan_memory_scanners_cpu_optimized_20260810.json` | `90d12b46e936ca1c4643ce39b742a56f562cf6bba4e57e9e460b3d9601cc8ed9` |
| `artifacts/schurscan_memory_scanners_cpu_optimized_long_replication_20260810.json` | `4e900b2b81f1f3488fb4553ff62b08a3b1a04d4c28154bebe1203c16cec7fccc` |
| `artifacts/schurscan_memory_scanners_cuda_optimized_20260810.json` | `128852f9a5fed55fd6d4d6e5d61fd7b4adb35defc7ea78b8b8949ac05bdee731` |
| `artifacts/schurscan_memory_scanners_cuda_optimized_long_replication_20260810.json` | `2bde60213f4d4d6a0d49adde522bf85f41298843fb73d145be5fea2ee9c9f9f5` |

## Claim boundary and next gate

This result ranks eager tensor programs on one workstation. A subsequent
frozen benchmark against the official FLA DeltaRule kernels is reported in
[`MATCHED_LEARNED_RETRIEVAL_RESULTS.md`](MATCHED_LEARNED_RETRIEVAL_RESULTS.md):
FLA wins the transport-free systems comparison. Neither experiment proves
hardware-general thresholds, triality-specific memory quality, or superiority
on a recurrence with the programme's noncommuting cross-view value action.

The next high-value systems gate is therefore a fused operator that preserves
that transported-value law, or a proof that it can be compiled into supported
fused primitives without changing the recurrence. The next scientific gate is
orthogonalized/discretized address learning plus explicit Task B action replay.
Global Dirac--Gram work can proceed in parallel; it does not block either gate.

> **Later resolution:** the co-moving transported FLA compiler, prospective
> Task B paired-action replay, overlapping-semantic large-slot routing, and real
> fused gathered-block inference kernel are now complete. The next scientific
> gate is a matched recent-window/selected-block/compressed-global model. See
> [`SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md`](SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md)
> and
> [`LARGE_SLOT_SEMANTIC_HIERARCHY_RESULTS.md`](LARGE_SLOT_SEMANTIC_HIERARCHY_RESULTS.md).
