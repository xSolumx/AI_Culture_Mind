# Task-B prospective paired-action replication preregistration

- **Frozen:** 2026-08-10, after the historical replay failure was diagnosed
  and before any replication result was run
- **Development seed:** `102`, excluded
- **Frozen seeds:** `20` through `29`, one durable process per seed
- **Arithmetic:** CPU float64
- **Training budget:** `500` Adam steps per curriculum stage plus `150` LBFGS
  steps for each action family

## Motivation

The first Task-B delta replay proved exact direct/delta equivalence but failed
its strict historical-metric reproduction gate because the old artifact did
not retain learned parameters from an underidentified independent action
family. This replication removes that dependency rather than weakening the
failed threshold.

## Paired design

For each untouched seed, train both original families on identical frozen
partial observations and optimizer budgets:

1. one 28-coordinate shared Spin(8) action family;
2. three independently fitted 28-coordinate representation actions.

Each family learns its own write/query alias encoders. The observation design,
rank-two negative calibration plane, continuous-alias world, action-word
generator, values, and dense length ladder remain unchanged from the original
Task-B protocol.

## Evaluation rows

All learned routes are hardened by `argmax` relative to their collision-free
learned permutation. Report:

1. shared-action direct and standard delta memory with the shared router;
2. independent-action direct and standard delta memory with the independent
   router;
3. independent action with the shared router, isolating action completion from
   routing;
4. correct negative action with each learned router as two capacity ceilings.

Every direct/delta pair receives identical event masks, writes, rotations,
queries, values, and one-hot keys. The maintained factored delta transitions
and work-efficient scan must be called directly.

## Replay integrity improvement

Every per-seed artifact must retain:

- teacher, shared, and independent action matrices;
- shared and independent action coordinates;
- shared and independent write/query router weights;
- training reports, software versions, and all evaluation metrics.

A verifier reconstructs the frozen policies from those weights and reruns at
least one short paired sequence plus the scan parity gate. This prevents a
future metric-only replay failure.

## Frozen gates

Implementation passes per seed only if:

- all direct/delta state and prediction differences are at most `1e-10`;
- direct and delta parallel/recurrent errors are below `1e-9`;
- each router is collision-free, cross-encoder consistent, and at least `0.99`
  stable on fresh aliases;
- both correct-action delta ceilings have mean cosine at least `0.995` and
  minimum cosine at least `0.98` at every length;
- shared and independent families each fit supplied observations below `1e-6`
  MSE.

The representation-prior decision passes only if, in at least `8/10` seeds:

- shared-action delta has length-2048 mean cosine at least `0.995`;
- independent-action delta has length-2048 mean cosine at most `0.90`;
- the routing-matched independent-action row also has mean cosine at most
  `0.90`.

Direct/delta equality is expected and cannot be called a delta advantage.
SO(3) already demonstrates that the value of a correct equivariant prior is
not exceptional to triality. No result here establishes generic memory
capacity, language quality, or production throughput.

## Planned artifacts

- `artifacts/task_b_paired_action_replication_seed20.json` through `seed29.json`;
- `artifacts/task_b_paired_action_replication_seeds20_29.json`;
- `artifacts/task_b_paired_action_replication_dev102.json`.

This preregistration is not a result.
