# Pure rotor SSM v2.0.0: first CUDA training and systems benchmark

Date: 2026-08-06

> **Historical-result pointer, reconciled 2026-08-16T16:05:27+02:00.** This
> one-seed v2.0 report is preserved unchanged. The later v2.1 transport cohort
> is the current local transport evidence; the new Pure Rotor / Mamba-2 smoke
> runner has not produced a quality result and does not supersede this report.

## Status

This is the first checkpoint of the rewritten pure architecture. It establishes
that the implementation trains, uses its rotor path, reloads, streams, and runs
efficiently through the parallel scan. It is one seed without a matched
baseline and therefore is not evidence of architectural superiority, a scaling
law, or statistical reliability.

Machine-readable artifacts:

- `pure_v2.0.0_base_seed0_500.json`
- `pure_v2.0.0_base_seed0_500_benchmark.json`
- `../checkpoints/pure_rotor_ssm_v2.0.0_selective_rotor_seed0_c32_l4_ctx256_step500.pt`

Checkpoint SHA-256:
`ae287e884e6c9d7f7d25699c2f9d36b5386c60c57222db30a6258647ea29f819`.

## Training protocol

The run used an NVIDIA GeForce RTX 2070 SUPER with PyTorch 2.12.0+cu130.
The model has 407,840 parameters, 32 multivector channels, four layers,
expansion two, byte vocabulary 256, and context 256. Batch size was 60,
giving 15,360 byte-tokens per step and 7.68 million sampled byte-tokens over
500 AdamW steps at learning rate 0.003. Dropout was disabled.

WikiText-2 raw UTF-8 bytes were loaded from the existing local Hugging Face
cache. The exact arrays have hashes:

- train: `1bfd06255cddf4dae742db40e827e824e8054f67ca0ff6f6e7d9d57bf6623cd1`
- validation: `52b6d7dd2f63bfa0e29ad7c665117e3a6cad8a1a1bbd3cdfbe13d447d491fa8a`

The trainer samples random training windows with replacement. Validation is
the fixed first 20 batches, not a whole-split sweep.

| Measurement | Value |
|---|---:|
| Initial validation loss | 5.543009 nats |
| Final validation loss | 1.761575 nats |
| Final validation bits/byte | 2.541415 |
| Final train loss | 1.764778 nats |
| Mean last-20 train loss | 1.754637 nats |
| Training time | 309.73 s |
| Training rate | 1.614 steps/s |
| Peak CUDA tensor allocation | 4,109.17 MiB |

All four layers learned nonzero rotor controls. Mean rotor angles were
`[1.2009, 1.3923, 1.3372, 1.3133]` radians. These values establish activity,
not that rotation caused the loss reduction.

The saved CPU checkpoint reloaded with strict key matching. On a deterministic
two-sequence, context-257 check, maximum logit disagreement was `5.96e-6`
between parallel and recurrent evaluation and `5.25e-6` between full and
chunked evaluation. Maximum state disagreements were `1.88e-6` and `1.01e-6`;
the largest final per-channel state norm was `0.9276`, inside the proved unit
ball. These are float32 numerical checks, not exact equalities.

## Trained-checkpoint benchmark

The benchmark used eager float32 execution, batch eight, three warmups, and ten
timed iterations. Forward/backward excludes an optimizer step. Throughput is
aggregate tokens per second.

| Context | Scan | Inference ms | Inference tok/s | Fwd+bwd ms | Fwd+bwd tok/s |
|---:|:---|---:|---:|---:|---:|
| 64 | parallel | 56.945 | 8,991 | 164.263 | 3,117 |
| 64 | recurrent | 204.138 | 2,508 | 478.059 | 1,071 |
| 256 | parallel | 67.288 | 30,436 | 198.261 | 10,330 |
| 256 | recurrent | 752.396 | 2,722 | 1,689.218 | 1,212 |

At context 256, the vectorized prefix path was 11.18 times faster for
inference and 8.52 times faster for forward/backward than the tokenwise
recurrent oracle in this configuration. The recurrent path remains the
streaming/semantic reference, not the preferred full-sequence training path.

The algebraically specialized two-vector rotor sandwich took 0.931 ms on the
kernel benchmark, while the dense two-`einsum` geometric product took 0.768 ms.
Their maximum absolute float32 disagreement was `7.15e-7`. PyTorch therefore
dispatches CUDA to the measured 1.21-times faster dense path; JAX retains the
specialized expression for XLA fusion.

## Interpretation boundary

This run is not directly comparable with the historical 22,968-parameter,
context-64, 300-step experiment: architecture, parameter count, context,
batch, data exposure, write law, and linear commutant all changed. A credible
quality claim now requires frozen scalar-SSM, identity-rotation,
non-geometric, and attention baselines at matched effective capacity, followed
by multiple untouched seeds and long-context retrieval tests.
