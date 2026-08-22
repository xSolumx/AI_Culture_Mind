# Retention-scaled block coupling result

**Quality gate closed:** 2026-08-22

## Quality verdict

The retention-scaled independent-action block recurrence passes its frozen
Tiny Shakespeare quality gate.

| seed | maintained bpb | retention-scaled bpb | improvement |
|---:|---:|---:|---:|
| 211 | 2.758037 | **2.743302** | +0.014736 |
| 223 | 2.760831 | **2.749814** | +0.011017 |
| 227 | 2.684770 | **2.667704** | +0.017065 |
| mean | 2.734546 | **2.720273** | +0.014273 |

The candidate wins all three seeds, exceeds the frozen `+0.0100` mean
requirement, never regresses, and keeps the parameter increase to 516
(0.0824%). This authorizes—but does not predetermine—the systems gate.

The result is empirical at this model size, dataset, and 300-step horizon. It
supports the retention-scaled coupling rule over maintained v1.2; it does not
establish superiority over Mamba-2 or a general theorem about continuous-time
discretization.

## First systems gate: failed

The prospectively frozen packed-warp kernel reached 64,101 tok/s versus 83,881
tok/s for maintained v1.2, or a throughput ratio of 0.7642. Its 23.58%
regression failed the maximum 10% allowance, so no confirmatory repeat was run
and the model was not promoted.

The order-balanced four-cycle result is recorded in
[`artifacts/retention_scaled_block_speed.json`](artifacts/retention_scaled_block_speed.json).
It identifies a scheduling bottleneck: the candidate exposes only one
sequential scan warp per batch item, while maintained v1.2 splits its forward
work across channel and triality blocks.

## Frozen kernel-rescue protocol

The transition is block diagonal across `8v`, `8+`, and `8-`. An
isotypic-split forward therefore launches three independent representation
warps per batch item while retaining both coupled multiplicity copies in each
warp. This changes scheduling, not recurrence values, and retains the existing
packed backward. Six targeted semantic and full-gradient CUDA tests pass.

Before timing this rescue, the conditional systems campaign is fixed as
follows:

- RTX 2070 SUPER, maintained WSL PyTorch 2.10/cu126 runtime;
- batch 8 and sequence length 256;
- complete forward, backward, gradient clipping, and AdamW update;
- fresh seed 241 and fixed synthetic token/target tensors per model;
- 10 untimed warmup steps;
- five CUDA-event windows of 10 steps per model per cycle;
- four cycles with alternating model order;
- aggregate by the median of each cycle median;
- pass only if candidate throughput is at least 90% of maintained throughput.

The protocol compares `raw_cuda_hybrid` maintained v1.2 with the isotypic
forward / packed backward `raw_cuda_block` retention-scaled coupling. Data
loading, copies, and validation are excluded. A pass still requires one fresh
seed-251 confirmatory repeat before default promotion. A failure retains the
model as a quality-positive research control.

The quality decision is
[`artifacts/retention_scaled_block_quality_summary.json`](artifacts/retention_scaled_block_quality_summary.json).
The prospective mechanism is documented in
[`RETENTION_SCALED_BLOCK_PREREGISTRATION.md`](RETENTION_SCALED_BLOCK_PREREGISTRATION.md).
The compiler-only rescue is frozen separately in
[`RETENTION_SCALED_BLOCK_KERNEL_PREREGISTRATION.md`](RETENTION_SCALED_BLOCK_KERNEL_PREREGISTRATION.md).

## Isotypic kernel-rescue verdict: failed

The representation-split schedule improved the candidate from 0.7642x to
0.8049x maintained throughput, but still missed the frozen 0.90x boundary:

| implementation | maintained tok/s | candidate tok/s | ratio |
|---|---:|---:|---:|
| packed forward | 83,881 | 64,101 | 0.7642 |
| isotypic forward | 81,816 | 65,856 | 0.8049 |

The rescue therefore failed and no confirmatory repeat was run. The artifact is
[`artifacts/retention_scaled_block_isotypic_speed.json`](artifacts/retention_scaled_block_isotypic_speed.json).

The remaining asymmetry is backward reconstruction. Maintained v1.2 recovers
the rotated pre-affine state from its scalar retention, whereas the block
kernel replays both complete forward Spin actions because a general learned
left map may be singular. Here the left map has the stronger form
`diag(scale) @ Q`; a guarded analytic `2x2` inverse can take the ordinary path
and retain exact replay only near singularity. That is the next bounded
compiler optimization, not evidence for default promotion.
