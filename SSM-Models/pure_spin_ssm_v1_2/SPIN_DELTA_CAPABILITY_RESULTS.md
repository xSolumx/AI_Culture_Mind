# Spin-Delta overwrite capability results

**Decision:** absolute capability failed its frozen robustness rule;
differential advantage failed.

## Frozen results

Every artifact passed finite execution, implementation/task identity, common
parameter equality, and the `2e-6` initial-logit pairing bound. Accuracy is for
retrieving the latest value written to one of two repeatedly overwritten keys.

| Seed | Model | W=8 | W=16 | W=32 |
|---:|---|---:|---:|---:|
| 401 | maintained v1.2 | 85.55% | 87.06% | 85.89% |
| 401 | Spin-Delta | **97.09%** | **97.29%** | **97.49%** |
| 409 | maintained v1.2 | **95.92%** | **95.24%** | **95.43%** |
| 409 | Spin-Delta | 85.94% | 85.28% | 85.23% |
| 419 | maintained v1.2 | **86.84%** | **87.70%** | **88.06%** |
| 419 | Spin-Delta | 75.83% | 75.71% | 76.10% |

Across seeds, maintained versus Spin-Delta mean accuracy was:

| Writes | Maintained | Spin-Delta | Delta minus maintained |
|---:|---:|---:|---:|
| 8 | 89.44% | 86.29% | -3.15 points |
| 16 | 90.00% | 86.09% | -3.91 points |
| 32 | 89.79% | 86.27% | -3.52 points |

## Frozen decisions

Candidate capability required every seed to reach at least 90% at `W=8` and
75% at `W=16`. The length-16 requirement passed, but seeds 409 and 419 failed
the trained-length requirement. Therefore robust candidate capability failed.

Differential advantage required a five-point `W=16` win in at least two seeds
and a five-point mean improvement. Spin-Delta won only seed 401 and averaged a
3.91-point regression. Differential advantage failed.

The aggregate machine decision is preserved in
`artifacts/spin_delta_capability/summary.json`.

## What the experiment establishes

Seed 401 is important negative-result context: Spin-Delta reached roughly 97%
at all three lengths, including four times the training-write count. Thus this
implementation can learn a stable two-key overwrite/retrieval computation; it
is not structurally incapable or broken.

However, that capability is optimization-sensitive and unnecessary here.
Maintained v1.2 already learned the task with 89--90% mean accuracy and beat
Spin-Delta in two of three seeds. Neither architecture showed material decay
from `W=8` to `W=32`, so the extra slot did not uniquely enable length
extrapolation.

The correct conclusion is therefore:

1. empirical existence of a high-accuracy Spin-Delta solution in one seed;
2. no robust absolute capability under the frozen cohort;
3. no retrieval advantage over maintained v1.2;
4. no rescue of the failed Shakespeare claim.

## Systems diagnostics

Fixed-order training took 18.00--18.44 seconds for Spin-Delta versus
12.53--13.70 seconds maintained. Peak allocated CUDA memory was 122,172,416
bytes versus 107,750,400 bytes. These are commissioning diagnostics, not an
order-balanced speed gate.

Further tuning of this same two-head/two-slot construction would be post-hoc.
Any follow-on must pose a new hypothesis—for example supplied orthogonal
addresses versus learned addresses—and use new seeds. The verified compiler
remains reusable independently of the closed model claim.

The prospectively frozen oracle intervention subsequently localized the
failure. Supplying causal write/query events and exact binary slots raised mean
accuracy to 99.55--99.58% across all lengths and passed every seed. See
`SPIN_DELTA_ORACLE_ADDRESS_RESULTS.md`. This proves recurrence capacity on the
named task, while leaving autonomous address inference open.
