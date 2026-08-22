# Spin-Delta exact-control write-curriculum results

**Decision:** the write-depth information homotopy passed every frozen robust
core-repair condition.

## Frozen paired cohort

All 18 WSL/cu126 runs completed from the prospectively committed protocol.
Initial model digests and initial evaluation metrics agree across both arms and
all data orders for each initialization. Every artifact shares the same
implementation hashes and fixed evaluation rows.

Sixteen-write accuracy was:

| Init | Data | Fixed | Curriculum | Change |
|---:|---:|---:|---:|---:|
| 541 | 563 | 97.75% | 100.00% | +2.25 points |
| 541 | 569 | 99.46% | 99.61% | +0.15 points |
| 541 | 571 | 99.71% | 99.80% | +0.10 points |
| 547 | 563 | 99.90% | 100.00% | +0.10 points |
| 547 | 569 | 100.00% | 99.76% | -0.24 points |
| 547 | 571 | 98.29% | 99.41% | +1.12 points |
| 557 | 563 | 99.37% | 100.00% | +0.63 points |
| 557 | 569 | 100.00% | 99.37% | -0.63 points |
| 557 | 571 | 93.51% | 100.00% | +6.49 points |

Aggregate results across all nine cells were:

| Metric | Fixed depth | Curriculum |
|---|---:|---:|
| Mean accuracy, 8 writes | 98.68% | **99.86%** |
| Mean accuracy, 16 writes | 98.67% | **99.77%** |
| Mean accuracy, 32 writes | 98.46% | **99.83%** |
| Minimum accuracy, 8 writes | 94.73% | **99.46%** |
| Minimum accuracy, 16 writes | 93.51% | **99.37%** |
| Minimum accuracy, 32 writes | 93.12% | **99.51%** |
| Robustness threshold | failed | **passed** |

The mean paired 16-write improvement is `+0.0110677`, or +1.11 percentage
points. The worst paired change is -0.63 points, inside the frozen two-point
regression allowance. The worst-cell improvement is +5.86 points, exceeding
the required +2 points.

## Variance contraction

The fixed arm independently reproduced the previously detected instability.
Its largest fixed-data initialization range at 16 writes was 6.20 points, and
its largest fixed-initialization data-order range was 6.49 points.

The curriculum reduced those maxima to 0.59 and 0.63 points respectively:

| Held fixed | Fixed-depth ranges | Curriculum ranges |
|---|---|---|
| Data seeds 563 / 569 / 571 | 2.15 / 0.54 / 6.20 points | **0.00 / 0.39 / 0.59** points |
| Init seeds 541 / 547 / 557 | 1.95 / 1.71 / 6.49 points | **0.39 / 0.59 / 0.63** points |

Thus the intervention did not merely improve the favorable cells. It removed
the large bad basin on this fresh factorial while preserving near-perfect
cells.

## Information and compute accounting

Both arms used 800 AdamW updates, batch 128, and 102,400 training examples.
The fixed arm saw 2,662,400 tokens. The 2/3/5/8-write curriculum saw 2,009,600
tokens, only 75.48% as many. It nevertheless improved the mean, minimum, and
both sensitivity measures.

This supports the proposed credit-assignment mechanism: short overwrite chains
provide coherent evidence for value injection and slot state before the same
parameters must solve deeper composition. It does not prove that the ordering
alone caused the effect, because shorter examples also change the distribution
of recurrent depths and reduce token exposure.

## Decision and consequence

All five frozen promotion clauses pass:

1. every curriculum cell passes the 8/16/32-write absolute thresholds;
2. the worst 16-write cell improves by at least two points;
3. both factorial sensitivity maxima fall below five points;
4. mean paired 16-write improvement is nonnegative;
5. no paired cell regresses by more than two points.

The exact-control Spin-Delta core therefore has a robust training procedure on
this synthetic task. This closes the immediate need for per-slot auxiliary
reconstruction or an optimizer intervention: those remain fallback diagnostics,
not the next default experiment.

The next falsifier is transfer through the already-perfect causal router. A
phase-separated router/core cohort should keep the router frozen and replace
only its fixed eight-write core phase with the same 2/3/5/8 curriculum. That
tests whether the learned straight-through controls preserve the newly robust
core geometry without returning to privileged oracle controls.

## Boundaries and artifacts

This is a finite synthetic result, not natural-language quality, a population
variance theorem, or a maintained-model promotion. It supports a training
schedule for one exact-control recurrence and does not establish that more Spin
algebra improves learning.

Canonical artifacts are in
[`artifacts/spin_delta_write_curriculum/`](artifacts/spin_delta_write_curriculum/).
The summary SHA-256 is
`807a02e9d0335ccf8e004103682998ac15107d8ae4b6e96b0bc9d3b60c7e400a`.
The summarizer reproduces that artifact byte-for-byte from the 18 cell files.
The four focused gate suites pass 9 tests, the complete maintained WSL/cu126
suite passes 112 tests, and Ruff passes on the runner, summarizer, and tests.
