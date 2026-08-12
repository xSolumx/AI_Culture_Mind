# SchurScan memory-scanner benchmark results

> **Current systems pointer.** This phase-1 eager record is unchanged. The
> broader 2026-08-10 report adds hierarchical learned routing and a separately
> derived co-moving FLA transported-delta benchmark; it does not reclassify
> these rows as fused timings.

> **Phase-1 record.** The original rows and interpretation below are preserved.
> The current optimized winner-by-regime result is
> [`SCHURSCAN_MEMORY_SCANNER_OPTIMIZATION_RESULTS.md`](SCHURSCAN_MEMORY_SCANNER_OPTIMIZATION_RESULTS.md),
> which adds the prospectively frozen local `9 x 9` homogeneous block backend.

- **Date:** 2026-08-10
- **Status:** maintained eager-PyTorch systems result; not a fused-kernel or
  universal architecture ranking
- **Protocol:** [`SCHURSCAN_MEMORY_SCANNER_BENCHMARK_PREREGISTRATION.md`](SCHURSCAN_MEMORY_SCANNER_BENCHMARK_PREREGISTRATION.md)
- **Implementation:** [`benchmark_schurscan_memory_scanners.py`](../../src/benchmark_schurscan_memory_scanners.py)

## Verdict

There is no single local scanner winner across devices and lengths.

On the six-thread CPU, the original structured Hillis--Steele slot scan wins
through length 1,024, while the new work-efficient structured tree crosses over
at length 2,048. On the RTX 2070 SUPER, dense homogeneous packing wins through
length 1,024, but the original structured slot tree wins at lengths 2,048 and
4,096. The long-length winners reproduce when the length order is reversed.

The systems result is separate from memory quality. Direct and triality-bound
slots use the identical structured transition monoid; Householder-, rotor-, or
other supplied orthogonal transports also collapse to the same scanner after
their token action is compiled. Triality has no scanner advantage by itself.

## Exact recurrence

Every row executes all prefixes of the identical 64-scalar recurrence

\[
M_t[h]=r_t[h]A_tM_{t-1}[h]+b_t[h]
\]

for eight slots of eight coordinates. Maximum relative discrepancies against
sequential execution are below `7.2e-16` on CPU and `5.4e-7` on CUDA. The new
structured work-efficient tree separately passes irregular-length and
full-gradient parity tests.

## CPU forward timing

Batch 1, float64, six threads. Times are median milliseconds over 15 repeats.

| Length | Slot HS | Slot WE | Dense affine | Dense homogeneous | Dense affine E2E | Dense homogeneous E2E | Winner |
|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 0.746 | 1.159 | 1.583 | 0.994 | 1.827 | 1.494 | Slot HS |
| 256 | 1.230 | 1.617 | 6.497 | 5.370 | 6.999 | 7.518 | Slot HS |
| 1,024 | 2.490 | 2.501 | 26.137 | 24.071 | 32.729 | 32.704 | Slot HS |
| 2,048 | 4.061 | 3.607 | 57.973 | 46.800 | 56.299 | 66.001 | Slot WE |

The reversed long-length replication used 20 repeats:

| Length | Slot HS | Slot WE | Winner |
|---:|---:|---:|---|
| 1,024 | 2.528 | 2.631 | Slot HS |
| 2,048 | 4.148 | 3.577 | Slot WE |

Thus the work-efficient CPU crossover is stable but modest: `1.13x` in the
primary sweep and `1.16x` in replication. Fewer compositions matter only after
the longer Blelloch critical path is amortized.

## CUDA forward timing

RTX 2070 SUPER, batch 2, float32, TF32 disabled. Times are median milliseconds
over 15 repeats.

| Length | Slot HS | Slot WE | Dense affine | Dense homogeneous | Dense affine E2E | Dense homogeneous E2E | Winner |
|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 2.482 | 3.336 | 2.364 | 1.010 | 2.622 | 1.383 | Dense homogeneous |
| 256 | 3.172 | 4.091 | 4.922 | 1.389 | 3.273 | 1.748 | Dense homogeneous |
| 1,024 | 3.956 | 4.993 | 4.037 | 3.088 | 4.286 | 3.713 | Dense homogeneous |
| 2,048 | 4.225 | 5.686 | 5.321 | 4.982 | 5.510 | 6.091 | Slot HS |
| 4,096 | 4.463 | 5.964 | 7.574 | 9.841 | 7.788 | 11.872 | Slot HS |

The reversed long-length replication used 20 repeats:

| Length | Slot HS | Slot WE | Dense homogeneous | Winner |
|---:|---:|---:|---:|---|
| 1,024 | 3.929 | 5.209 | 2.682 | Dense homogeneous |
| 2,048 | 4.227 | 5.779 | 5.033 | Slot HS |
| 4,096 | 4.430 | 6.109 | 9.866 | Slot HS |

The length-4,096 structured winner processes 1.84 million tokens/s. Its
advantage over the prebuilt dense homogeneous scan is `2.20x` in the primary
sweep and `2.23x` in replication. Including dense materialization increases
the primary advantage to `2.66x`.

## CUDA incremental memory

Bytes below are allocated above already-constructed scan-only leaves. E2E rows
also include structured-to-dense materialization.

| Length 4,096 backend | Incremental memory | Relative to dense homogeneous scan-only |
|---|---:|---:|
| Slot HS | 17.56 MB | 3.61% |
| Slot WE | 19.79 MB | 4.07% |
| Dense affine scan-only | 479.18 MB | 98.58% |
| Dense homogeneous scan-only | 486.07 MB | 100% |
| Dense affine E2E | 613.40 MB | 126.20% |
| Dense homogeneous E2E | 758.73 MB | 156.09% |

The structured state law, not only the tree schedule, is the decisive memory
optimization. It avoids materializing an eight-block `64 x 64` operator.

## Work, depth, and launch behavior

At length 4,096, Hillis--Steele uses 45,057 structured compositions and depth
12; the ordered work-efficient tree uses 12,286 compositions and depth 25.
The CUDA work-efficient slot row is slower despite `3.67x` fewer compositions.
Its structured composition launches several small tensor operations at each
tree level, and its dependency path is longer. The benchmark establishes that
resource tradeoff; it does not uniquely attribute it to a specific hardware
microarchitectural mechanism.

Dense homogeneous packing repeats much more scalar arithmetic but expresses a
composition as one larger matrix product. That wins at short and medium CUDA
lengths. At long length, the dense arithmetic and materialization dominate,
and the structured Hillis row crosses over.

## Best memory by regime

The broader local evidence now supports a conditional answer rather than an
absolute winner.

| Regime | Best supported local result | Triality-specific? |
|---|---|---|
| Supplied hard addresses and correct actions | Triality slots, direct slots, and oracle delta overwrite are exact quality ties at 64 state scalars | No |
| Learned continuous aliases in the frozen eight-key task | Jointly balanced hard slots pass 10/10; direct and triality tie. The tested learned delta row passes 0/10 and additive fast weights pass 0/10 | No; the win is hard joint routing |
| Incomplete cross-view action observations | One shared triality representation family completes held-out action directions that independently fitted actions miss | Yes, as a representation-completion prior; not a generic capacity win |
| Restricted-support bilinear identification | Known intertwiner SchurScan extrapolates exactly; same-endpoint generic tensor fails, and group augmentation rescues it | No; SO(3) reproduces the effect |
| Local eager 64-scalar prefix scanning | Phase 1: structured slot scan at long CPU/CUDA lengths; dense homogeneous packing at medium CUDA lengths. Superseded for the current winner by the local-block optimization result linked above | No; direct/triality/compiled orthogonal actions share the scanner |

This makes direct slots the simplest default when hard addresses and the value
action are already known. Triality is justified when the application actually
uses cross-view equivariance or incomplete shared-action identification. Delta
memories remain essential for soft distributed addressing, but the local
learned-key failure is not evidence against modern Gated DeltaNet,
Erase-then-Delta, Q-Delta, or DeltaProduct kernels.

## Dirac--Gram dependency

The unrestricted Dirac--Gram global inequality does not need to be proved
before any result in this document. That theorem concerns global sensing/design
optimality. The slot monoid, triality/direct gauge equivalence, equivariant
identification gate, and scanner timings use only already-established local
algebra and executable recurrences.

Proving Dirac--Gram remains valuable for the geometric paper, but it neither
licenses nor blocks a claim about memory quality or scan throughput.

## Artifacts

| Artifact | SHA-256 |
|---|---|
| `artifacts/schurscan_memory_scanners_cpu_i7_9700k_20260810.json` | `e7ff43b32a452ebc844f60e90f68efda1efd56c1f7a6b533d0c19f017fde082f` |
| `artifacts/schurscan_memory_scanners_cuda_rtx2070s_20260810.json` | `5c17323609658b845ddfa41998686db14164ff0dd8bc0e0580adf448c8ebb3df` |
| `artifacts/schurscan_memory_scanners_cpu_long_replication_20260810.json` | `96bdfd2e744aa06c62459dc5b85846963e28432b55d2834bff5512d694a099c8` |
| `artifacts/schurscan_memory_scanners_cuda_long_replication_20260810.json` | `ad8e3c87aadea5576a9669a5cbabafb3e0bb4c4bf6715cc81f2f8408d6d5dadc` |

## Replay

```powershell
$env:PYTHONPATH='src'
python -m pytest -q `
  tests/test_benchmark_schurscan_memory_scanners.py `
  tests/test_spin8_triality_direct_memory_equivalence.py

python -m benchmark_schurscan_memory_scanners `
  --device cpu --dtype float64 --batch 1 `
  --lengths 64 256 1024 2048 --warmup 5 --repeats 15 --threads 6 `
  --output artifacts/schurscan_memory_scanners_cpu_i7_9700k_20260810.json

python -m benchmark_schurscan_memory_scanners `
  --device cuda --dtype float32 --batch 2 `
  --lengths 64 256 1024 2048 4096 --warmup 5 --repeats 15 --threads 6 `
  --output artifacts/schurscan_memory_scanners_cuda_rtx2070s_20260810.json
```
