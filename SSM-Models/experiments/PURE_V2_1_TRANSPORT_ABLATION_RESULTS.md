# Pure rotor SSM v2.1.0 transport-ablation results

Date: 2026-08-06.

This is the outcome report for the prospectively frozen transport ladder. The raw JSON retains every run, seed, timing, intervention, hash, and numerical-parity measurement.

## Refined-model retrain

The standalone approximately-4-GiB v2.1 rotor retrain changed only the open rotor-chart limit from `pi/2` to `pi`; its measured peak was 4109.2 MiB, slightly above 4096 MiB.

| Version | Validation nats | Bits/byte | Peak MiB | Train seconds |
|---|---:|---:|---:|---:|
| v2.0.0 | 1.761575 | 2.541415 | 4109.2 | 309.7 |
| v2.1.0 | 1.760117 | 2.539312 | 4109.2 | 314.4 |

The 0.001458-nat single-seed difference from v2.0 is descriptive, not an established refinement gain.

### Standalone v2.1 scan benchmark

Batch 8, PyTorch eager float32 on the recorded local CUDA device. Times are forward/backward medians.

| Context | Parallel ms | Recurrent ms | Speedup | Parallel peak MiB |
|---:|---:|---:|---:|---:|
| 64 | 183.04 | 526.35 | 2.88x | 130.7 |
| 128 | 214.98 | 949.46 | 4.42x | 265.2 |
| 256 | 200.96 | 1857.80 | 9.24x | 579.7 |
| 512 | 245.67 | 3546.08 | 14.43x | 1223.8 |

## Cohort decision gates

- Cohort completeness: **PASS** (105/105 prediction, 70/70 memory runs).
- Preregistered state-matched prediction rule: **PASS**.
- Preregistered memory rule by task: `associative_recall` **FAIL**, `q8_ordered_product` **FAIL**.
- Preregistered compute-efficiency rule: **FAIL** (CUDA-matched rotor minus identity loss +0.4076 nats).

## Direct answer

- **Prediction: qualified yes versus identity.** Rotor improves state-matched loss by 0.0204 nats and wins all five seeds. It is not the best transport: quaternion improves by 0.0446, commuting phases by 0.0284, and the larger state-matched SO(8) row by 0.1268.
- **Parameter efficiency: qualified yes versus identity, not versus the best alternatives.** Rotor retains a 0.0204-nat advantage, but quaternion and commuting phases remain better.
- **Memory: no.** Rotor loses to identity on the mean associative-recall result at every length and Q8 remains at chance-scale noise.
- **Compute efficiency: no.** At matched measured CUDA budget, rotor is 0.4076 nats worse than the wider identity model.

## Prediction: State matched

Loss and paired deltas are mean [95% t interval] over five seeds; negative paired loss favors the family over retrained identity.

| Family | C | Effective params | Confirmation nats | Loss minus identity | Wins | Train tok/s | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Identity | 8 | 25,848 | 2.4513 [2.4373, 2.4653] | 0.0000 [0.0000, 0.0000] | 0 | 208,435 | 170.0 |
| Real diagonal | 8 | 31,096 | 2.4424 [2.4239, 2.4610] | -0.0089 [-0.0199, 0.0022] | 5 | 187,031 | 230.0 |
| Complex phases | 8 | 28,472 | 2.4229 [2.4022, 2.4435] | -0.0284 [-0.0560, -0.0009] | 4 | 136,858 | 229.0 |
| Quaternion left | 8 | 27,816 | 2.4067 [2.3788, 2.4347] | -0.0446 [-0.0666, -0.0225] | 5 | 72,208 | 216.1 |
| Cl(3,0) rotor | 8 | 27,528 | 2.4310 [2.4163, 2.4456] | -0.0204 [-0.0311, -0.0096] | 5 | 85,480 | 308.4 |
| Fixed rotor | 8 | 25,896 | 2.4626 [2.4364, 2.4888] | 0.0113 [-0.0079, 0.0305] | 1 | 90,994 | 289.8 |
| Generic SO(8) | 8 | 67,832 | 2.3246 [2.3018, 2.3473] | -0.1268 [-0.1493, -0.1043] | 5 | 56,740 | 1353.7 |

## Prediction: Effective-parameter matched

Loss and paired deltas are mean [95% t interval] over five seeds; negative paired loss favors the family over retrained identity.

| Family | C | Effective params | Confirmation nats | Loss minus identity | Wins | Train tok/s | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Identity | 8 | 25,848 | 2.4513 [2.4373, 2.4653] | 0.0000 [0.0000, 0.0000] | 0 | 208,403 | 170.0 |
| Real diagonal | 7 | 25,701 | 2.4782 [2.4634, 2.4930] | 0.0269 [0.0071, 0.0466] | 0 | 182,413 | 208.0 |
| Complex phases | 8 | 28,472 | 2.4229 [2.4022, 2.4435] | -0.0284 [-0.0560, -0.0009] | 4 | 133,395 | 229.0 |
| Quaternion left | 8 | 27,816 | 2.4067 [2.3788, 2.4347] | -0.0446 [-0.0666, -0.0225] | 5 | 72,500 | 216.1 |
| Cl(3,0) rotor | 8 | 27,528 | 2.4310 [2.4163, 2.4456] | -0.0204 [-0.0311, -0.0096] | 5 | 84,595 | 308.4 |
| Fixed rotor | 8 | 25,896 | 2.4626 [2.4364, 2.4888] | 0.0113 [-0.0079, 0.0305] | 1 | 90,298 | 289.8 |
| Generic SO(8) | 5 | 30,791 | 2.4576 [2.4192, 2.4961] | 0.0063 [-0.0337, 0.0463] | 3 | 88,671 | 858.0 |

## Prediction: Measured-CUDA matched

Loss and paired deltas are mean [95% t interval] over five seeds; negative paired loss favors the family over retrained identity.

| Family | C | Effective params | Confirmation nats | Loss minus identity | Wins | Train tok/s | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Identity | 60 | 628,996 | 2.0233 [2.0000, 2.0467] | 0.0000 [0.0000, 0.0000] | 0 | 87,738 | 957.0 |
| Real diagonal | 50 | 655,006 | 2.0512 [2.0191, 2.0834] | 0.0279 [-0.0208, 0.0766] | 2 | 86,372 | 1134.4 |
| Complex phases | 37 | 323,895 | 2.1224 [2.0761, 2.1687] | 0.0991 [0.0389, 0.1592] | 0 | 84,445 | 850.3 |
| Quaternion left | 8 | 27,816 | 2.4067 [2.3788, 2.4347] | 0.3834 [0.3341, 0.4328] | 0 | 75,529 | 216.1 |
| Cl(3,0) rotor | 8 | 27,528 | 2.4310 [2.4163, 2.4456] | 0.4076 [0.3739, 0.4413] | 0 | 88,157 | 308.4 |
| Fixed rotor | 19 | 90,411 | 2.2490 [2.2272, 2.2708] | 0.2256 [0.1849, 0.2663] | 0 | 83,692 | 596.9 |
| Generic SO(8) | 5 | 30,791 | 2.4576 [2.4192, 2.4961] | 0.4343 [0.3795, 0.4890] | 0 | 91,134 | 857.4 |

## Trained-rotor interventions

- Identity clamping changed confirmation loss by +0.1532 nats on average.
- Time-shuffling rotor actions changed confirmation loss by +0.1773 nats on average.

## Memory: Associative recall

| Family | L=64 | L=128 | L=256 | L=512 | Longest minus identity |
|---|---:|---:|---:|---:|---:|
| Identity | 13.98% | 10.70% | 8.41% | 7.32% | +0.00 pp |
| Real diagonal | 13.87% | 10.85% | 8.30% | 7.17% | -0.15 pp |
| Complex phases | 10.66% | 8.95% | 7.29% | 6.84% | -0.48 pp |
| Quaternion left | 10.17% | 8.00% | 7.26% | 6.70% | -0.62 pp |
| Cl(3,0) rotor | 13.80% | 10.12% | 7.94% | 6.93% | -0.40 pp |
| Fixed rotor | 13.69% | 9.72% | 7.85% | 6.99% | -0.33 pp |
| Generic SO(8) | 11.70% | 9.07% | 7.90% | 6.77% | -0.55 pp |

## Memory: Q8 ordered product

The registered even Q8 evaluation lengths have four-label support, so a supported-class chance predictor is 25%.

| Family | L=32 | L=64 | L=128 | L=256 | Longest minus identity |
|---|---:|---:|---:|---:|---:|
| Identity | 23.83% | 23.62% | 23.47% | 23.82% | +0.00 pp |
| Real diagonal | 23.84% | 23.62% | 23.45% | 23.92% | +0.10 pp |
| Complex phases | 22.51% | 22.46% | 22.31% | 22.54% | -1.28 pp |
| Quaternion left | 21.31% | 21.15% | 21.30% | 21.53% | -2.29 pp |
| Cl(3,0) rotor | 23.66% | 23.76% | 23.57% | 23.76% | -0.06 pp |
| Fixed rotor | 23.84% | 23.59% | 23.57% | 23.92% | +0.10 pp |
| Generic SO(8) | 23.75% | 23.47% | 23.43% | 23.87% | +0.05 pp |

## Width calibration

The parameter column is the integer width nearest the C=8 rotor parameter count. CUDA residual is relative to the C=8 rotor forward/backward target at batch 64, context 128.

| Family | Parameter C | CUDA C | CUDA ms | Residual |
|---|---:|---:|---:|---:|
| Identity | 8 | 60 | 93.220 | -0.5% |
| Real diagonal | 7 | 50 | 93.927 | +0.3% |
| Complex phases | 8 | 37 | 92.506 | -1.2% |
| Quaternion left | 8 | 8 | 105.467 | +12.6% |
| Cl(3,0) rotor | 8 | 8 | 93.650 | +0.0% |
| Fixed rotor | 8 | 19 | 95.239 | +1.7% |
| Generic SO(8) | 5 | 5 | 95.705 | +2.2% |

## CUDA systems result (state matched, batch 8, context 256)

| Family | C | Inference ms | Forward/backward ms | Train tok/s | Peak MiB |
|---|---:|---:|---:|---:|---:|
| Identity | 8 | 10.877 | 38.975 | 52,547 | 54.2 |
| Real diagonal | 8 | 11.685 | 44.775 | 45,740 | 70.2 |
| Complex phases | 8 | 17.684 | 88.899 | 23,037 | 70.2 |
| Quaternion left | 8 | 33.454 | 124.441 | 16,458 | 67.3 |
| Cl(3,0) rotor | 8 | 34.435 | 126.789 | 16,153 | 92.4 |
| Fixed rotor | 8 | 31.981 | 97.846 | 20,931 | 87.2 |
| Generic SO(8) | 8 | 17.708 | 63.123 | 32,444 | 376.4 |

## Interpretation boundary

A prediction win establishes an empirical advantage under this small byte-model protocol, not a universal language-model advantage. A Q8-only memory win supports ordered algebraic composition, not general recall. A systems result applies to PyTorch eager float32 on the recorded local GPU. Exact recurrence closure and norm bounds are mathematical properties; finite-precision scan parity, optimization, generalization, and throughput remain empirical.

## Evidence integrity

- Frozen protocol SHA-256: `6766b91bfa2741a0fbc498f182ecfffaf8b8613deb4ef7f7f2dc22a08a5e0923`.
- Raw aggregate SHA-256: `9641ed494df60961321f4fb81fd27f33f3af3d7a4945344e8072a53f05d1a4c9`.
- WikiText-2 train-byte SHA-256: `1bfd06255cddf4dae742db40e827e824e8054f67ca0ff6f6e7d9d57bf6623cd1`.
- WikiText-2 validation-byte SHA-256: `52b6d7dd2f63bfa0e29ad7c665117e3a6cad8a1a1bbd3cdfbe13d447d491fa8a`.
- Standalone v2.1 checkpoint SHA-256: `406b8bc19cbcc072bfa0dd15bb747c36d72eab52b4de42c78ca332a0781352c3`.
- Maximum prediction parallel/recurrent or full/chunked logit discrepancy: `7.6293945e-06`.
- Maximum prediction-run peak allocation: `1353.68 MiB`; memory-run peak: `623.74 MiB`.
- Summed measured training-loop time across 175 runs: `1.722396 hours` (excludes calibration and evaluation).
