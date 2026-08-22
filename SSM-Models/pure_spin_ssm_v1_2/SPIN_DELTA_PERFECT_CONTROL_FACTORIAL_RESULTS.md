# Spin-Delta perfect-control factorial results

**Decision:** perfect-control robustness failed; both initialization and
minibatch-order sensitivity were detected.

## Frozen 3x3 result

Every cell used exact causal controls from optimizer step one and the same
fixed evaluation set. Rows are core initializations; columns are training-batch
orders. Entries are 16-write retrieval accuracy.

| Init \ data | 509 | 521 | 523 | Row range |
|---:|---:|---:|---:|---:|
| 491 | 99.51% | 99.95% | 94.14% | 5.81 points |
| 499 | 91.65% | 91.41% | 100.00% | 8.59 points |
| 503 | 100.00% | 96.68% | 99.32% | 3.32 points |
| Column range | 8.35 points | 8.54 points | 5.86 points | |

Across all nine cells, mean accuracy was 96.98%, 96.96%, and 97.02% at 8,
16, and 32 writes. The minima were 91.26%, 91.41%, and 91.36%; the maxima were
100% at every length. Because several cells missed the frozen thresholds,
perfect-control robustness failed.

Initialization sensitivity crossed five points for every fixed data order.
Data-order sensitivity crossed five points for initializations 491 and 499.
The exact decision is in
`artifacts/spin_delta_perfect_control_factorial/summary.json`.

## Mathematical and optimization conclusion

The Spin-Delta map has sufficient representational capacity but its finite
training dynamics have multiple outcome basins. Neither initialization nor
batch order carries a globally good label: data order 523 hurts initialization
491 yet takes initialization 499 from about 91.5% to exactly 100%. This sign
reversal is direct finite evidence of an initialization-by-trajectory
interaction, not a single bad seed class.

The correct current claim is therefore:

- exact routing and the recurrence can realize near-perfect, length-stable
  overwrite/retrieval;
- a causal router can identify the synthetic grammar perfectly;
- the present AdamW training geometry does not reach that solution robustly,
  even with exact controls from the first step.

The next intervention should target core conditioning rather than addressing:
an identity-preserving residual parameterization, normalized value injection,
or an explicit per-slot reconstruction objective. It must be tested on this
crossed grid or a fresh equivalent, because single-seed improvements are now
known to be unreliable.

## Boundaries

This is a nine-cell finite audit, not a convergence theorem or population
variance estimate. It uses privileged synthetic controls and does not promote
the model on natural data.
