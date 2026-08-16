# Final-Only Composition-Homotopy Protocol

Protocol frozen: **2026-08-16T21:33:30+02:00**

## Motivation

The frozen fixed-length-16 final-only cohort has a basis-dependent basin: all
three initializations solve Haar basis 1, while all initializations for bases 0
and 2 remain near the fixed-canonical loss. This successor tests a single
preregistered remedy without altering that negative artifact.

## Intervention

Train only on terminal transported operators, but continue composition depth:

| optimizer steps | sequence length |
|---:|---:|
| 1--250 | 2 |
| 251--500 | 4 |
| 501--750 | 8 |
| 751--1,000 | 16 |

There is no length-1 stage and no intermediate-prefix supervision. Batch size,
optimizer, learning rate, total updates, hidden Haar bases, candidate sizes,
and evaluation lengths remain as in the fixed-L16 protocol. Each candidate for
a basis sees the identical curriculum schedule and terminal targets.

Run three initializations for the 28-parameter learned-basis operator and the
512-parameter dense linear operator. Retain one Transformers Mamba-2 and one
DeltaProduct reference per basis under the same curriculum.

## Frozen gates

For each of bases 0, 1, and 2:

- exact transported oracle maximum MSE below `2e-12`;
- collapsed-octonion L128 MSE above `1e-2`;
- every learned-basis initialization L128 MSE below `1e-3`;
- every learned-basis initialization beats its matched dense initialization at
  L128; and
- all learned metrics finite and all saved checkpoints reload/rehash.

No dense, Mamba-2, or DeltaProduct absolute-accuracy gate is registered. A
failure remains a failure; no post-run curriculum, threshold, or optimizer
change belongs to this protocol.

## Interpretation boundary

Passing would show that a composition-depth homotopy removes the observed local
basin on three fixed synthetic Haar transports. It would not be a global
optimization theorem, natural-task result, triality-specific result, or fused
systems comparison. The curriculum is an extra supervision distribution and
must be compared against the failed fixed-L16 run as such.
