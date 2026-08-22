# Spin-Delta phase-separated router results

**Decision:** router readiness passed; phase-separated capacity and the
co-adaptation bottleneck decision failed.

## Frozen results

Both schedules began from bitwise-identical complete models and saw identical
phase-specific batches. The phased schedule proved by tensor equality that its
core remained untouched during 100 router-only steps, then froze a perfect
router while training the core for 800 steps.

| Seed | Schedule | W=8 | W=16 | W=32 |
|---:|---|---:|---:|---:|
| 467 | joint | 97.12% | 97.66% | 97.56% |
| 467 | phased | **100.00%** | **99.98%** | **99.98%** |
| 479 | joint | **99.83%** | **99.95%** | **99.83%** |
| 479 | phased | 98.61% | 98.66% | 98.14% |
| 487 | joint | **96.00%** | **96.46%** | **96.95%** |
| 487 | phased | 94.97% | 93.85% | 95.24% |

Three-seed means were 97.65%, 98.02%, and 98.11% for joint training and
97.86%, 97.49%, and 97.79% for phased training at 8, 16, and 32 writes.
Phase separation changed 16-write mean accuracy by -0.53 points.

Every phased router metric was exactly 1.0 before core training at every seed
and length. Nevertheless seed 487 missed the frozen 95% thresholds at 8 and
16 writes. The exact decision is preserved in
`artifacts/spin_delta_phased_router/summary.json`.

## What this closes

Early routing noise is not a complete explanation for Spin-Delta's seed
variance. Removing it produces a very strong mean and two high-capacity seeds,
but does not guarantee the frozen absolute threshold. It also does not
outperform joint training: the sign changes by seed and the mean is slightly
negative.

Combined with the preceding gates, the causal chain is now:

1. oracle controls prove high recurrence capacity on the original cohort;
2. the learned causal router identifies the grammar perfectly;
3. perfect final identification does not ensure robust joint optimization;
4. perfect routing before core training still does not ensure robust core
   optimization on a new cohort.

The remaining object is therefore the optimization distribution of the
perfect-control recurrence itself—not router architecture or co-adaptation
alone. A factorial audit that separates core initialization from minibatch
order is now more informative than another router modification.

## Boundaries

The phased curriculum remains synthetic and label supervised. It is a useful
high-performing procedure, not a promoted robust theorem or natural-data
model. No throughput claim is authorized.
