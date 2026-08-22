# Spin-Delta query-event continuation results

**Decision:** mechanism repair failed and learning-autonomy promotion failed.
Restoring query-slot gradients was real but insufficient; the continuation
worsened aggregate retrieval and did not identify the query-event grammar.

## Frozen paired result

Each row began from a shared model state and replayed the same minibatches in
the hard-event and linear-continuation arms.

| Init | Data | Hard W16 | Continuation W16 | Change |
| ---: | ---: | ---: | ---: | ---: |
| 719 | 739 | 51.90% | 59.33% | +7.42 points |
| 719 | 743 | 64.36% | 11.47% | -52.88 points |
| 719 | 751 | 4.30% | 41.02% | +36.72 points |
| 727 | 739 | 63.48% | 49.76% | -13.72 points |
| 727 | 743 | 59.18% | 52.83% | -6.35 points |
| 727 | 751 | 59.42% | 42.97% | -16.46 points |
| 733 | 739 | 2.29% | 8.40% | +6.10 points |
| 733 | 743 | 42.14% | 3.37% | -38.77 points |
| 733 | 751 | 7.96% | 4.15% | -3.81 points |

Aggregate retrieval was:

| Metric | Hard event | Linear continuation |
| --- | ---: | ---: |
| Mean accuracy, 8 writes | **41.82%** | 32.08% |
| Mean accuracy, 16 writes | **39.45%** | 30.37% |
| Mean accuracy, 32 writes | **39.81%** | 29.52% |
| Minimum accuracy, 16 writes | 2.29% | **3.37%** |
| Maximum accuracy, 16 writes | **64.36%** | 59.33% |

Mean paired 16-write change was `-0.0908203125`, or -9.08 points. The
worst paired regression was -52.88 points. Only three of nine cells improved.
The continuation's largest initialization and data-order ranges remained
50.93 and 47.85 points, so it did not contract the unstable optimization
geometry.

## Router result

Over all nine models and all 2/3/5/8/16/32-write readiness evaluations:

| Metric | Hard range | Continuation range |
| --- | ---: | ---: |
| Query-event F1 | 0.000--0.000 | 0.000--0.127 |
| Gauge-correct query-slot accuracy | 0.348--0.871 | 0.484--1.000 |
| Write-event F1 | 0.000--0.326 | 0.000--0.952 |
| Gauge-correct write-slot accuracy | 0.494--0.659 | 0.494--1.000 |

The continuation increased the minimum query-slot accuracy by 13.62 points,
and one cell learned the query slot perfectly at every length. This confirms
that the gradient restoration measured in the one-step audit had a real
optimization effect. It nevertheless left the minimum query-event F1 at zero;
only weak event identification appeared in isolated cells.

Every prospectively frozen mechanism component failed: mean retrieval gain,
paired-regression limit, worst-cell rescue, query-event repair, and query-slot
repair. The no-label/no-oracle contract passed. The stronger robustness,
variance, and identification promotion decisions all failed.

## What the failure establishes

The implication

```text
nonzero query-slot derivative  =>  autonomous controller recovery
```

is false for this architecture and task. The derivative was necessary for the
named slot head to learn, but not sufficient to make the six-field router an
identifiable factorization of final retrieval loss.

Inside each Spin-Delta block, query controls are applied only after the slot
state scan: they select a read from `raw_slot_states` but do not modify that
block's recurrent transition. In the two-block model, early reads can still
influence the next block through the residual stream, so non-final query events
are not claimed to be exactly unobservable in the full stack. Their route to
the final loss is, however, indirect. At the final token, hard event zero still
selects the viable internal query. These two paths leave many controller
behaviours compatible with the same final retrieval target.

The sign-changing paired outcomes show that a longer or differently shaped
annealing schedule would be another optimizer search, not a justified
mechanism repair.

## Next falsifier

Before another training cohort, compute a temporal observability map of final
loss with respect to every query-event and query-slot logit:

1. separate one-block and two-block models;
2. separate current hard fallback, soft event, and routed-query authority;
3. report per-position Jacobian norms at write, delimiter, payload, and final
   query tokens;
4. verify the one-block structural zero for non-final read controls;
5. quantify the cross-block indirect path rather than assuming it is useful.

If final retrieval cannot observe the desired event grammar, the architecture
must stop demanding that grammar from this loss. The clean alternatives are an
always-active query-address distribution, a task with differentiated losses at
actual query positions, or explicit auxiliary supervision. These answer
different questions and must be tested separately.

## Boundaries and replay

This is finite synthetic evidence on the RTX 2070 SUPER. It does not refute all
continuation methods or all label-free routing. It does refute this frozen
linear continuation as the selected repair for the current task.

Canonical artifacts are in
[`artifacts/spin_delta_query_continuation/`](artifacts/spin_delta_query_continuation/).
The deterministic summary SHA-256 is
`fd5ed62532f67f7c227d826b9bf445a6ca4753fd64268c9eaa4f8512da452f87`.
It replayed byte-for-byte from the nine raw artifacts under WSL2, PyTorch
`2.10.0+cu126`, CUDA 12.6, and the RTX 2070 SUPER. The 18 training arms consumed
the frozen 102,400 examples and 2,009,600 tokens each; their recorded training
time totaled 487.04 seconds and peak allocated CUDA memory was 140,992,000
bytes.
