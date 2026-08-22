# Retention-scaled block kernel rescue: frozen systems gate

**Frozen:** 2026-08-22, after the first systems gate failed and before timing
the isotypic-split implementation.

## Motivation and fixed implementation

The retention-scaled recurrence passed its prospective quality gate but its
first packed-warp CUDA schedule reached only 76.42% of maintained v1.2
throughput. At batch 8, that forward schedule exposed eight resident scan
warps, whereas maintained `raw_cuda_hybrid` exposed 48 by separating channel
and triality representation.

The mathematical transition remains block diagonal across the inequivalent
`8v`, `8+`, and `8-` representation spaces. The rescue implementation therefore
changes only the forward schedule:

- one warp owns one triality representation and both multiplicity copies;
- the learned `2x2` left action still couples only those two copies;
- three warps are launched per batch item instead of one;
- the existing packed, reduction-free backward is unchanged;
- model parameters, recurrence values, and the quality artifacts are unchanged.

Six targeted WSL/cu126 raw-CUDA tests passed before this document was frozen,
including semantic output and full-gradient parity at subgroup factor counts 3,
4, 6, and 8. This is a compiler-schedule optimization, not a new model.

## Frozen timing protocol

- RTX 2070 SUPER, WSL PyTorch 2.10.0+cu126;
- fresh seed 241, batch 8, sequence length 256;
- maintained `raw_cuda_hybrid` versus retention-scaled `raw_cuda_block`;
- complete forward, backward, gradient clipping, and AdamW update;
- fixed synthetic input and target tensors per model;
- 10 untimed warmup steps;
- five CUDA-event windows of 10 steps per model per cycle;
- four cycles with alternating execution order;
- aggregate by median of the four per-cycle throughput medians.

The rescue passes only if candidate throughput is at least 90% of maintained
throughput. Otherwise it remains a quality-positive research control and is
not promoted. No confirmatory repeat is authorized after a failure. A pass
authorizes one repeat with fresh seed 251 under the identical protocol; default
promotion requires both runs to pass.

## Claim boundary

The timing excludes data loading, host-to-device copies, and validation. It is
a fixed-shape complete-GPU-step comparison on one Turing device, not a
convergence, end-to-end, multi-device, or Mamba-2 performance claim.
