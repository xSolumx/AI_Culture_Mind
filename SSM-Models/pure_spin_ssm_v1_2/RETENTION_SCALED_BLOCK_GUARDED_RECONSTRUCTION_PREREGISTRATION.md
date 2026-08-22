# Retention-scaled block guarded reconstruction: frozen systems gate

**Frozen:** 2026-08-22 after semantic testing and before any timing of this
implementation.

## Fixed compiler change

For the block recurrence,

```text
state_t = L_t rotated_t + drive_t,
L_t = diag(scale_t) Q_t.
```

The previous backward unconditionally replayed both independently controlled
Spin actions to recover `rotated_t`. The fixed CUDA change instead computes

```text
rotated_t = inverse(L_t) (state_t - drive_t)
```

when `abs(det(L_t)) > 1e-7`. If the determinant is at or below that guard, it
uses the original factor-by-factor replay. The ordinary model path has positive
sigmoid retentions and orthogonal `Q_t`; the fallback preserves the semantic
definition for zero retention, rank-deficient controls, and direct kernel use.

The forward schedule, model parameters, optimizer, quality artifacts, and
mathematical recurrence are unchanged. The change removes one complete
two-channel Spin factor traversal per token from the ordinary backward path.

Before freezing this document, seven targeted WSL/cu126 tests passed: the four
subgroup/full-factor output-and-gradient parity cases, full-model unit and
retention-scaled gradient parity, and a new alternating rank-deficient-left
fallback test. No performance timing of this code has been observed.

## Frozen timing protocol

- RTX 2070 SUPER, WSL PyTorch 2.10.0+cu126;
- fresh seed 257, batch 8, sequence length 256;
- maintained `raw_cuda_hybrid` versus retention-scaled `raw_cuda_block`;
- complete forward, backward, gradient clipping, and AdamW update;
- fixed synthetic input and target tensors per model;
- 10 untimed warmup steps;
- five CUDA-event windows of 10 steps per model per cycle;
- four cycles with alternating execution order;
- aggregate by median of the four per-cycle throughput medians.

The gate passes only if candidate throughput is at least 90% of maintained
throughput. A failure ends this optimization route without a repeat. A pass
authorizes one identical confirmatory run at fresh seed 263; default promotion
requires both runs to pass.

## Claim boundary

This is fixed-shape complete-step GPU timing on one Turing device. It excludes
data loading, host-to-device copies, and validation, and it is not a quality,
convergence, end-to-end, multi-device, Tensor-Core, or Mamba-2 speed claim.
