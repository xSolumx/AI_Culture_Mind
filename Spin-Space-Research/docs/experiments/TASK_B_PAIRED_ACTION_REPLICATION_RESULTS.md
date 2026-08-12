# Task-B prospective paired-action replication results

- **Date:** 2026-08-10
- **Protocol:**
  [`TASK_B_PAIRED_ACTION_REPLICATION_PREREGISTRATION.md`](TASK_B_PAIRED_ACTION_REPLICATION_PREREGISTRATION.md)
- **Development seed:** `102`, excluded
- **Frozen seeds:** `20` through `29`
- **Verdict:** implementation `10/10`; representation-prior decision `10/10`;
  Task B is empirically closed under the frozen synthetic protocol

## Main result

At length 2048, the shared Spin(8) action family remains exact while the
parameter-richer independent action family fails on the held-out negative-
chiral complement:

| Row | Mean across seeds | Worst/best boundary |
|---|---:|---:|
| Shared-action delta | `0.999999999999994` | worst `0.999999999999987` |
| Independent-action delta | `0.542924` | range `0.453707–0.620403` |
| Independent action with shared router | `0.542924` | range `0.453707–0.620403` |

The routing-matched independent row is identical seed by seed to the
independent row. Both routers are collision-free and perfectly stable on the
frozen aliases. The deficit is therefore the held-out value action, not router
quality.

## Memory-law control

Every hard direct/delta pair receives the same one-hot key, write, action,
query, and event stream. Across all five evaluation rows, eight lengths, and
ten seeds:

- maximum direct/delta state or prediction error: `0.0`;
- maximum delta parallel/recurrent error: `7.77e-16`;
- minimum correct-action oracle query cosine: `1.0`;
- one principal row contains `581,946` query events across the cohort.

Thus standard delta memory reproduces direct slots exactly in the hard-key
regime. It neither causes nor repairs the independent action-completion
failure. This closes the missing empirical row without creating a delta or
triality overwrite-capacity claim.

## Observation and complement controls

Both families fit every supplied observation:

- maximum shared observed-column MSE: `6.23e-17`;
- maximum independent observed-column MSE: `1.36e-16`.

Held-out negative-complement cosine is at least
`0.999999999999999` for the shared family. The independent family averages
`0.912322` and ranges from `0.880243` to `0.924657` despite using three times
as many action coordinates (`336` versus `112`). Both rows use the same
384-parameter router size.

## Replay correction and retained parameters

The preceding historical replay cohort correctly failed its strict
preregistered metric-reproduction gate because the old metric-only artifact
did not retain parameters from an underidentified family. This prospective
replication repairs the evidence design rather than relaxing that failure.

Every new per-seed artifact stores teacher/shared/independent actions, action
coordinates, router weights, training reports, and software versions. The
aggregate reconstructs every policy, reconstructs both learned action families
from their retained coordinates, and reruns short sequence plus scan gates:

- maximum retained-parameter replay difference: `0.0`;
- replay passes: `10/10`.

## Claim boundary

Supported: when partial observations truly arise from one shared Spin(8)
action, the shared representation prior completes the unobserved view and
preserves long-horizon retrieval more reliably than independently fitted
actions.

Not supported:

- extra storage capacity from triality;
- a superior delta update law;
- an exceptional advantage over every other correct equivariant prior (the
  separate SO(3) cross-product intertwiner experiment already reproduces the
  structured-prior effect; it is a control family, not a standalone model);
- natural-language or production-model superiority.

The next blocker was no longer Task B. It has now been executed as the
[`LARGE_SLOT_SEMANTIC_HIERARCHY_RESULTS.md`](LARGE_SLOT_SEMANTIC_HIERARCHY_RESULTS.md)
campaign: 64 overlapping keys, shared-three-view versus independent routers,
actual gathered state, and a separately frozen fused kernel. That later result
does not revise this cohort's verdict or its synthetic action-completion scope.

## Artifacts

- `artifacts/task_b_paired_action_replication_seed20.json` through `seed29.json`;
- `artifacts/task_b_paired_action_replication_seeds20_29.json`;
- `artifacts/task_b_paired_action_replication_dev102.json` (excluded);
- `artifacts/matched_retrieval_campaign_synthesis_task_b_closed_20260810.json`.
