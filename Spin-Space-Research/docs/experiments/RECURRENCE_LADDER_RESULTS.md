# Recurrence ladder: first controlled Q8 result

This is a one-seed mechanism pilot, not a model-selection conclusion. The
authoritative machine-readable report is
`recurrence_ladder_q8_prefix_lengths_seed0_300.json`.

## Protocol

- Device: NVIDIA GeForce RTX 2070 SUPER
- Task: ordered Q8 prefix-product tracking
- Train sequence length: 16
- Steps: 300
- Batch size: 256
- Model: 2 layers, 4 channels, 8 real state values per channel
- Parameters: 13,512 for every family
- Recurrent cache: 64 real scalars per sequence for every family
- Same initial parameters, initial function, batches, optimizer, and head

An earlier final-target-only pilot remained at chance for all families. That
negative result is retained in `recurrence_ladder_q8_seed0_300.json`. Dense
prefix supervision was introduced to separate transition learning from opaque
long-horizon credit assignment.

## Result

Final-position accuracy by evaluated sequence length:

| Family | L=2 | L=4 | L=8 | L=16 | L=32 | Steps/s |
|---|---:|---:|---:|---:|---:|---:|
| Real selective | 100.00% | 45.70% | 19.41% | 12.62% | 12.38% | 64.26 |
| Complex/unitary | 100.00% | 57.52% | 35.47% | 14.99% | 12.87% | 24.88 |
| Quaternion/even | 100.00% | 53.00% | 18.12% | 12.35% | 12.70% | 12.04 |
| Selective GA rotor | 100.00% | **62.35%** | **43.55%** | **19.85%** | 12.57% | 17.06 |
| Static GA rotor | 100.00% | 44.58% | 13.48% | 12.30% | 11.79% | 17.10 |

On the main validation set at length 16, selective GA also has the best prefix
accuracy (46.74%) and validation loss (1.1393). Complex/unitary is second at
41.17% and 1.3142.

Full-sequence, chunked, and single-token recurrent execution remain close;
the largest reported state error is `9.54e-6`. Every family uses a fixed cache
independent of context length.

## Interpretation

The selective rotor learns this ordered noncommutative task faster than its
matched controls and substantially outperforms the static rotor. This is
evidence that token-selective rotation is useful in this setting. It is not yet
evidence of an exact noncommutative algorithm:

- all families collapse to chance at twice the training length;
- Q8 is not a neutral benchmark for quaternionic/geometric models;
- only one seed and one training budget were run;
- the fastest real recurrence is about 3.8 times faster than the explicit GA
  recurrence in this unfused PyTorch implementation.

Next experiments should add multiple non-abelian groups, five seeds, a length
curriculum, dense orthogonal and Householder-product baselines, and a fused or
compiled implementation before any architectural claim is promoted.
