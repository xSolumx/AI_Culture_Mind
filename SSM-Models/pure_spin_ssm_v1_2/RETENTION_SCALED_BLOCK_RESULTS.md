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

## Frozen speed protocol

Before timing, the conditional systems campaign is fixed as follows:

- RTX 2070 SUPER, maintained WSL PyTorch 2.10/cu126 runtime;
- batch 8 and sequence length 256;
- complete forward, backward, gradient clipping, and AdamW update;
- fixed seed 239 and fixed synthetic token/target tensors per model;
- 10 untimed warmup steps;
- five CUDA-event windows of 10 steps per model per cycle;
- four cycles with alternating model order;
- aggregate by the median of each cycle median;
- pass only if candidate throughput is at least 90% of maintained throughput.

The protocol compares `raw_cuda_hybrid` maintained v1.2 with
`raw_cuda_block` retention-scaled coupling. Data loading, copies, and validation
are excluded. A pass would still require a confirmatory repeat before default
promotion. A failure retains the model as a quality-positive research control
and makes block-kernel optimization the next systems frontier.

The quality decision is
[`artifacts/retention_scaled_block_quality_summary.json`](artifacts/retention_scaled_block_quality_summary.json).
The prospective mechanism is documented in
[`RETENTION_SCALED_BLOCK_PREREGISTRATION.md`](RETENTION_SCALED_BLOCK_PREREGISTRATION.md).
