# V1.3 rigorous optimization result

**Date:** 2026-08-21  
**Hardware:** NVIDIA GeForce RTX 2070 SUPER, WSL2, PyTorch 2.10.0+cu130  
**Status:** accepted semantic fast path; rejected determinant shortcut; optional
fixed-shape compiler tier

## Decision

The default identity-transport reference now uses the one-sided affine scan
and the dense Jordan-trace determinant. This is the strongest optimization
that cleared both the semantic and empirical gates.

Two tempting changes were not promoted:

- the explicit `H_3(O)` cubic is algebraically correct and much cheaper, but
  its changed float32 evaluation order failed the frozen five-seed quality
  noninferiority rule;
- the 531-entry sparse Albert product is correct in exact arithmetic, but its
  GPU reduction order perturbs float32 gradients, increases eager peak memory,
  and gave no useful eager end-to-end speedup at the tested shape.

Both remain explicit research controls. Neither is the default.

The frozen seed-102--106 artifacts call the then-default combined candidate
`identity_delta`; their recorded source hashes resolve that name to the
explicit determinant. After the decision, current source assigns the rejected
control the unambiguous name `identity_explicit`, while `identity_delta` means
the accepted one-sided/Jordan default.

## Algebraic reduction that passed

For general exceptional transport the memory transition is

\[
S_t=A_tS_{t-1}R_t^{\mathsf T}+B_t.
\]

Identity transport has `R_t=I`, hence

\[
S_t=A_tS_{t-1}+B_t,
\qquad
(A_2,B_2)\circ(A_1,B_1)
=(A_2A_1,B_2+A_2B_1).
\]

The specialized compiler therefore removes the materialized 27 by 27
identity action, its stored prefix, and every value-axis identity matrix
multiply. It retains the same associative scan and chronological convention.
The model-level float32 output and every parameter gradient are bitwise equal
to the old generic path; recurrent, parallel, and chunk-streaming tests cover
the new transition type.

## Prospective quality falsifier

The determinant shortcut was tested exactly as frozen in
`V1_3_OPTIMIZATION_PREREGISTRATION.md`: seeds 102--106, 300 AdamW updates,
batch 2, length 64, two 32-wide layers, and 16 fixed validation batches on the
pinned disjoint Tiny Shakespeare split.

| Seed | Legacy bpb | Explicit candidate bpb | Candidate - legacy bpb | Throughput speedup |
|---:|---:|---:|---:|---:|
| 102 | 3.49250 | 3.50298 | +0.01048 | 1.123x |
| 103 | 3.58600 | 3.62147 | +0.03547 | 1.089x |
| 104 | 3.69813 | 3.70845 | +0.01031 | 1.125x |
| 105 | 3.57582 | 3.58575 | +0.00994 | 1.121x |
| 106 | 3.56510 | 3.57908 | +0.01398 | 1.214x |
| **Mean / geometric mean** | **3.58351** | **3.59955** | **+0.01604** | **1.134x** |

The maximum single-seed regression was +0.03547 bpb, inside the +0.0500
limit. The mean regression exceeded the preregistered +0.0100 limit, so the
combined explicit-determinant candidate **failed**. Its mean peak allocation
was 24,129,024 bytes versus 29,908,480 for legacy, but speed and memory do not
override the quality gate.

The accepted one-sided/Jordan path was then replayed for a complete seed-107
training trajectory. Legacy and optimized runs had identical step-1,
step-150, and step-300 losses, identical maximum pre-clip gradient norm, and
identical final 3.7040653361 bpb. Throughput rose from 3,426.23 to 3,984.26
tokens/s (1.163x). This replay is a consistency audit, not an extra tuned
quality seed; the stronger reason for accepting the scan is exact path parity.

## Matched systems result

The systems runner uses interleaved CUDA-event samples, 5 warmups, 25 timed
repeats, identical weights/tokens, fullgraph compilation, and separate cold
compile time. The fixed shape is batch 2, length 64, `d_model=32`, two layers,
memory width 4, rank 2, float32.

| Execution | Inference median | Forward+backward median | Incremental peak allocation |
|---|---:|---:|---:|
| Generic legacy eager | 9.880 ms | 29.858 ms | 11,371,520 B |
| Accepted one-sided eager | 8.166 ms | 27.040 ms | 7,607,296 B |
| Accepted `torch.compile(default)` | 4.684 ms | 13.004 ms | 6,742,528 B |
| Accepted `torch.compile(reduce-overhead)` | 0.815 ms | 4.606 ms | 525,312 B* |

The accepted eager reduction is 1.210x for inference and 1.104x for
forward+backward, with 33.1% less incremental peak allocation. The
`reduce-overhead` row is 12.36x and 6.52x faster than generic eager at this
fixed shape, but requires a 19.7-second first forward+backward compile. Its
reported memory is incremental *after* CUDA-graph capture, not total process
memory.

Compiled output differs from eager legacy by relative L2 `1.72e-7`, and its
parameter-gradient vector by `2.24e-7`. Consequently the compiler tier is
opt-in fixed-shape systems evidence, not part of the eager quality promotion
and not a hardware-general result. Dynamic shapes, long contexts, and full
compiled multi-seed training remain open.

## Reproduction artifacts

- `artifacts/shakespeare_optimization300_seed102_rtx2070s_20260821.json`
  through seed 106: immutable paired quality inputs;
- `artifacts/shakespeare_optimization300_fresh_seed_summary_rtx2070s_20260821.json`:
  mechanically validated decision and input hashes;
- `artifacts/shakespeare_safe_identity300_seed107_rtx2070s_20260821.json`:
  exact safe-trajectory replay;
- `artifacts/optimization_safe_identity_b2_l64_default_rtx2070s_20260821.json`;
- `artifacts/optimization_safe_identity_b2_l64_reduce_overhead_rtx2070s_20260821.json`.

`summarize_optimization.py` refuses missing or duplicate seeds, non-eager
artifacts, or mismatched dataset, configuration, and source hashes. The
artifact records its own source hash as well.

## Honest boundary

This result optimizes the no-action Shakespeare reference, not the expensive
Spin(8)/Spin(9)/F4/E6 action construction. It does not establish superiority
to Mamba-2 or any other architecture. The next valuable compiler work is a
shape-aware dispatch study over longer contexts and batches, followed by
action-controller/exponential fusion only where an exceptional-action task
first demonstrates a quality reason to pay that cost.
