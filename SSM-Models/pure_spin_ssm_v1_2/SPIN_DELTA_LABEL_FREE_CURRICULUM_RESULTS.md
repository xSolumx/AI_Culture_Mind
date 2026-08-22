# Spin-Delta label-free curriculum results

**Decision:** learning autonomy failed. The no-label contract passed, but both
retrieval autonomy and gauge-correct router identification failed.

## Frozen result

Sixteen-write accuracy was:

| Init | Data | Fixed | Curriculum | Change |
|---:|---:|---:|---:|---:|
| 653 | 673 | 63.67% | 28.08% | -35.60 points |
| 653 | 677 | 78.32% | 5.18% | -73.14 points |
| 653 | 683 | 74.76% | 2.88% | -71.88 points |
| 659 | 673 | 65.92% | 44.04% | -21.88 points |
| 659 | 677 | 97.36% | 98.19% | +0.83 points |
| 659 | 683 | 46.92% | 4.59% | -42.33 points |
| 661 | 673 | 31.49% | 3.61% | -27.88 points |
| 661 | 677 | 73.10% | 51.86% | -21.24 points |
| 661 | 683 | 33.98% | 70.31% | +36.33 points |

Aggregate retrieval was:

| Metric | Fixed depth | Curriculum |
|---|---:|---:|
| Mean accuracy, 8 writes | **62.09%** | 35.99% |
| Mean accuracy, 16 writes | **62.84%** | 34.30% |
| Mean accuracy, 32 writes | **62.75%** | 34.32% |
| Minimum accuracy, 8 writes | **30.52%** | 6.59% |
| Minimum accuracy, 16 writes | **31.49%** | 2.88% |
| Minimum accuracy, 32 writes | **30.52%** | 3.03% |
| Maximum accuracy, 16 writes | 97.36% | **98.19%** |
| Robustness threshold | failed | failed |

Mean paired 16-write change was `-0.2853190`, or -28.53 percentage
points. The worst paired regression was -73.14 points. The curriculum's
initialization and data-order ranges reached 93.02 and 93.60 points, so neither
the worst-cell rescue nor variance-contraction decision passed.

## Router identification failed after gauge correction

One global binary slot permutation was selected per trained model and applied
jointly to its write and query slots at all lengths. This legitimate gauge
correction did not rescue identification: the minimum curriculum router metric
was zero.

Across all curriculum cells and all six readiness lengths:

- write-event F1 ranged from 0.00 to 0.78;
- query-event F1 was exactly 0.00 everywhere;
- raw write-slot accuracy ranged from 0.00 to 0.72;
- raw query-slot accuracy remained near chance, ranging from 0.47 upward.

The most revealing counterexample is initialization 659, data order 677. It
reached 98.19% retrieval at 16 writes while query-event F1 remained zero and
the query-slot decision remained near chance. High retrieval therefore does
not identify the explicit router grammar in the present architecture.

## Mechanism diagnosis: an optimization bypass

The routed query used by the core is

```text
q = event * routed_query + (1 - event) * internal_query.
```

The router initializes the hard query event off with bias -3. While that hard
event is zero, the core's internal continuous query controller remains a viable
retrieval path and the routed query-slot branch is multiplied by zero. Final
retrieval loss can therefore improve without crossing the event threshold or
training the explicit query address.

The write path has a related redundancy: token-conditioned internal drive and
erase machinery can partly absorb event mistakes. Thus router and core are not
an identifiable factorization under final retrieval supervision. The short
curriculum, which repairs a core given correct controls, instead encourages
several shortcut basins when the discrete controller itself must be discovered.

This is not evidence that information homotopy is generally harmful. It shows
that the proven core curriculum does not transfer through a hard, redundant,
jointly learned factorization.

## Decision and next falsifier

The frozen decisions are:

- label-free/no-oracle contract: **passed**;
- retrieval autonomy: **failed**;
- router identification modulo global slot permutation: **failed**;
- learning autonomy: **failed**.

More curriculum stages are not justified. The next low-cost result should be a
one-step gradient-topology certificate at initialization, separating event and
slot heads under hard versus soft gates. Only after that audit should one
isolated repair be frozen:

1. remove the redundant query fallback and make the address router
   authoritative; or
2. use a soft-to-hard event continuation so slot gradients are live before
   discretization.

Those interventions answer different questions and must not be combined in the
first gate.

## Boundaries and artifacts

The nine-cell result is finite synthetic evidence. It does not refute all
label-free routing methods, all curricula, or Spin-Delta capacity. The earlier
exact-control and supervised-router curricula remain valid positive results.

Canonical artifacts are in
[`artifacts/spin_delta_label_free_curriculum/`](artifacts/spin_delta_label_free_curriculum/).
The summary SHA-256 is
`b682277ebd8d6f1ab9d2cf7b7aa8f39fa98748bedf1d7b98f13687ab7e009b0b`;
the summarizer reproduces it byte-for-byte from the nine cell files.
Nine focused curriculum/router tests pass, the complete WSL/cu126 suite passes
119 tests, and Ruff passes on the label-free runner, summarizer, and tests.
