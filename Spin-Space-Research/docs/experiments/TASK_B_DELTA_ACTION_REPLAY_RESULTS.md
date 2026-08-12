# Task-B independent-action delta replay results

- **Date:** 2026-08-10
- **Protocol:**
  [`TASK_B_DELTA_ACTION_REPLAY_PREREGISTRATION.md`](TASK_B_DELTA_ACTION_REPLAY_PREREGISTRATION.md)
- **Frozen seeds:** `0` through `9`
- **Verdict:** exact memory equivalence passes; strict historical replay gate
  fails; the original Task-B empirical decision is therefore not declared
  closed by this cohort

## What passed

The actual standard factored delta implementation and the matched direct-slot
memory agree exactly on every hard-routed event in every seed:

- maximum direct/delta state error: `0.0`;
- maximum direct/delta prediction error: `0.0`;
- direct and delta scan discrepancies: below the preregistered tolerances;
- minimum write/query route stability: `1.0`;
- minimum oracle-delta query cosine: `0.9999999999999996`.

The independently fitted negative action fails at length 2048 in all ten new
fits. Mean cosine ranges from `0.447079` to `0.578860`, with cross-seed mean
`0.500178`. The stored shared-family direct row remains essentially exact.
Thus the auxiliary mechanism evidence agrees with the theorem: delta does not
repair an incorrect held-out value action, and it is not worse than direct
memory when both receive the identical hard route and action.

## Why the strict gate failed

The preregistration required the historical soft independent-direct mean
cosines to replay within `1e-12`. They do not. The original artifact retained
metrics but not learned action matrices or router weights. Re-running the
original joint-then-independent program now reproduces the new action fit, not
the stored fit. Because the independent family has 21 unconstrained tangent
directions, both fits satisfy the supplied observations while choosing
different held-out complements.

This is not a delta discrepancy: direct and delta are exactly equal inside the
new cohort. It is a provenance/replay defect in the historical learned-
parameter artifact. The frozen aggregate therefore reports:

- implementation passes under the strict protocol: `0/10`;
- auxiliary representation-prior wins: `10/10`;
- `task_b_decision_rule_fully_empirically_closed: false`.

The failed threshold is not relaxed after inspection.

## Consequence

A new prospective paired replication must train shared and independent action
families together on untouched seeds, evaluate both in the same artifact, and
retain the learned actions, coordinates, and router weights. That experiment
can close Task B without pretending that unavailable historical weights were
replayed.

That prospective replication is now complete and passes `10/10`; see
[`TASK_B_PAIRED_ACTION_REPLICATION_RESULTS.md`](TASK_B_PAIRED_ACTION_REPLICATION_RESULTS.md).
The failed historical-replay verdict in this document remains unchanged.

## Artifacts

- `artifacts/task_b_delta_action_replay_seed0.json` through `seed9.json`;
- `artifacts/task_b_delta_action_replay_seeds0_9.json`;
- `artifacts/task_b_delta_action_replay_dev101.json` (excluded development
  smoke).
