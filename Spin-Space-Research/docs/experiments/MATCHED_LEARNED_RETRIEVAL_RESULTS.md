# Matched learned-retrieval campaign results

> **Completion notice.** This document preserves the first matched campaign
> and its open-row wording. The 2026-08-10 follow-on completes hierarchical
> direct/delta routing, the adversarial key-noise frontier, Spin(9) boundary,
> and frozen full-transport FLA measurements. Current synthesis:
> [`SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md`](SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md).
> The last independent-action delta row was later closed prospectively on
> untouched seeds `20`--`29`; see
> [`TASK_B_PAIRED_ACTION_REPLICATION_RESULTS.md`](TASK_B_PAIRED_ACTION_REPLICATION_RESULTS.md).
> Its clean close does not erase the failed strict historical replay in
> [`TASK_B_DELTA_ACTION_REPLAY_RESULTS.md`](TASK_B_DELTA_ACTION_REPLAY_RESULTS.md).
> The later 64-slot overlapping-semantic and fused gathered-state campaign is
> reported in
> [`LARGE_SLOT_SEMANTIC_HIERARCHY_RESULTS.md`](LARGE_SLOT_SEMANTIC_HIERARCHY_RESULTS.md).
> It supersedes this document's remaining routing/kernel roadmap, not this
> original cohort or its quality verdict.

- **Protocol frozen:** 2026-08-10, before implementation and result inspection
- **Frozen reliability cohort:** seeds `0` through `9`
- **Excluded development seed:** `101`
- **Historical status at this report:** Task A complete locally; Task B
  synthesis complete with one explicit empirical row still open

## Verdict

The best current **local** memory scanner is hard or discretized semantic
routing backed by the optimized 64-scalar slot scan. Direct and triality-bound
slots are quality-equivalent when routes are hard. They remain much more robust
than learned continuous delta keys on this frozen overwrite campaign, but the
advantage belongs to address geometry, not to a triality-specific update law.

Triality's supported role is different: a shared Spin(8) action family completes
held-out cross-view action directions that independently fitted actions do not.
That is a constrained representation-prior result. The separate SO(3)
cross-product intertwiner experiment shows that the broader
structured-equivariance effect is not exceptional to triality; it is not a
standalone SO(3) model.

The unrestricted Dirac--Gram inequality is not a prerequisite for either
result. It remains an independent proof programme.

## Task A implementation gates

The new delta backend implements the exact factored affine recurrence

\[
S_t=(I-k_tk_t^\mathsf{T})S_{t-1}R_t^\mathsf{T}+k_tv_t^\mathsf{T}
\]

as a sequential reference, an ordered work-efficient scan, and a real two-level
chunked scan. It is an eager mathematical implementation, not compact-WY or a
fused Triton kernel.

| Gate | Frozen result |
|---|---:|
| Seeds passing the implementation gate | 10/10 |
| Oracle delta passing all 1,800 quality cells | 10/10 |
| Hard-route direct/triality gauge gate | 10/10 |
| Maximum chunk/recurrent absolute error | `2.4314e-14` |
| Maximum hard-route direct/triality prediction gap | `1.4322e-14` |
| Maximum oracle-delta relative squared error | `7.6476e-28` |
| Total reported queries | 3,901,440 |

Unit tests cover irregular lengths, chunk sizes `1`, `4`, `16`, `32`, and `64`,
noncommuting value transport, one-hot overwrite, stale additive values, prefix
objects, and recurrent/chunked gradients.

## Task A quality

All rows have 64 recurrent scalars. Direct and triality slots share the same
two `8 x 24` route encoders; delta and fast-weight rows share the same-sized
normalized-key encoders. `delta_chunk_oracle` receives ideal orthogonal semantic
keys so update capacity is not confused with address inference.

The table reports the mean query cosine over all seeds and cells in a cohort.

| Cohort | Direct slots | Triality slots | Learned delta | Oracle delta | Additive fast weight |
|---|---:|---:|---:|---:|---:|
| Overwrite, clean | 1.000000 | 1.000000 | 0.938717 | 1.000000 | 0.430487 |
| Overwrite, corrupted | 0.990631 | 0.990637 | 0.921749 | 1.000000 | 0.425222 |
| Stream, clean | 1.000000 | 1.000000 | 0.602608 | 1.000000 | 0.281744 |
| Stream, corrupted | 0.645217 | 0.645078 | 0.551786 | 1.000000 | 0.251321 |
| Stream hot keys, corrupted | 0.987647 | 0.987627 | 0.904633 | 1.000000 | 0.137792 |
| Stream cold keys, corrupted | 0.302787 | 0.302529 | 0.198938 | 1.000000 | **0.364851** |

The final row is important. Additive fast weights are a bad overwrite memory,
but because they never erase, they can preserve untouched cold keys longer than
corrective memories under severe corrupted-address leakage. Reporting only a
global mean would hide this tradeoff.

### Paired robustness result

Both slot variants beat learned delta in every corrupted overwrite cell in
10/10 seeds. The same is true for combined stream quality and hot-key stream
quality. On cold keys, both slots win in the per-seed aggregate for 10/10 seeds,
but not in every individual cell: all methods approach chance after thousands
of repeated corrupted writes.

This satisfies the preregistered hard-routing robustness rule. It does **not**
show that the delta update lacks capacity: oracle delta is exact everywhere.
The learned row's loss is therefore assigned to nonorthogonal key inference and
its interaction with repeated erase updates.

### Direct versus triality

Clean direct and triality rows tie exactly in all 400 clean overwrite and stream
cells. Under explicit soft-route corruption:

| Cohort | Mean triality minus direct | Maximum absolute cell difference | Triality wins / cells |
|---|---:|---:|---:|
| Overwrite | `+5.84e-6` | 0.00350 | 583/1,200 |
| Combined stream | `-1.39e-4` | 0.03317 | 92/200 |
| Hot stream | `-1.98e-5` | 0.00275 | 96/200 |
| Cold stream | `-2.58e-4` | 0.06601 | 93/200 |

These small, sign-unstable differences arise after routes are perturbed into
cross-slot mixtures, outside the hard-slot gauge theorem. They do not support a
triality-specific memory-law advantage.

### Long cold-key frontier

Cold keys are initialized during the first eight tokens, then never overwritten;
four hot keys continue to receive writes. Values still undergo the supplied
noncommuting actions.

| Perturbation | Length | Direct | Triality | Learned delta | Oracle delta | Fast weight |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 64 | 1.0000 | 1.0000 | 0.6875 | 1.0000 | 0.7284 |
| 0.00 | 4096 | 1.0000 | 1.0000 | 0.0476 | 1.0000 | 0.2062 |
| 0.10 | 64 | 0.9672 | 0.9666 | 0.6375 | 1.0000 | 0.6856 |
| 0.10 | 4096 | 0.0372 | 0.0375 | 0.0262 | 1.0000 | 0.1874 |
| 0.20 | 64 | 0.7957 | 0.7844 | 0.5075 | 1.0000 | 0.6282 |
| 0.20 | 4096 | 0.0126 | 0.0123 | 0.0160 | 1.0000 | 0.1621 |

Clean learned delta still decays because learned keys are not exactly orthogonal;
each hot-key correction modifies cold-key projections. The exact-key delta row
rules out an update-capacity explanation.

### Sample efficiency

Each encoder family was retrained from scratch at the frozen budget; no test
checkpoint selection was performed. Values are mean cosine over the two frozen
sample-efficiency cells.

| Steps per stage | Direct | Triality | Learned delta | Oracle delta | Fast weight |
|---:|---:|---:|---:|---:|---:|
| 25 | 0.992553 | 0.992584 | 0.828917 | 1.000000 | 0.384737 |
| 75 | 0.992533 | 0.992555 | 0.842469 | 1.000000 | 0.390277 |
| 150 | 0.992533 | 0.992555 | 0.860470 | 1.000000 | 0.396273 |
| 300 | 0.992533 | 0.992555 | 0.892878 | 1.000000 | 0.406895 |

Categorical slot routing solves this controlled alias geometry early. The
continuous key encoder improves steadily but has not reached orthogonal-key
quality by 300 steps per stage.

## Corrected local CUDA tier

The first timing artifact is superseded. Audit found that it constructed the
triality tensor on CPU and copied it to CUDA inside every timed call, fixed the
variant order, differentiated only values, never tuned delta chunks, omitted
address encoders, and undercounted backward peak memory when an old gradient
was already allocated.

The corrected Windows tier uses an RTX 2070 SUPER, PyTorch `2.12.0+cu130`,
float32, batch 2, and TF32 disabled. It reports both a pre-encoded core and an
encoder-inclusive tier. The latter gives every row a 192-parameter write and
192-parameter query encoder and differentiates values plus both encoders.
Backend selection was frozen from three disjoint tuning processes before three
fresh measurement processes. Each table entry is the median of process
medians; each length/variant has 150 forward and 75 forward+backward samples.

| Length | Direct forward | Triality forward | Delta forward | Fast forward | Direct fwd+bwd | Triality fwd+bwd | Delta fwd+bwd | Fast fwd+bwd |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1.671 ms | 2.304 ms | 5.426 ms | 4.950 ms | 5.163 ms | 6.195 ms | 14.305 ms | 9.511 ms |
| 256 | 1.910 ms | 2.536 ms | 6.434 ms | 5.970 ms | 6.023 ms | 7.073 ms | 17.597 ms | 11.652 ms |
| 1,024 | 2.519 ms | 2.929 ms | 8.220 ms | 7.746 ms | 8.302 ms | 9.203 ms | 21.576 ms | 14.479 ms |
| 2,048 | 3.115 ms | 3.538 ms | 8.823 ms | 8.332 ms | 11.094 ms | 11.963 ms | 23.611 ms | 15.786 ms |
| 4,096 | 4.474 ms | 4.998 ms | 9.362 ms | 8.884 ms | 19.506 ms | 20.318 ms | 24.925 ms | 16.676 ms |

Thus direct slots remain the fastest maintained **eager** end-to-end forward
row, while additive fast weights have the fastest long backward row but fail
the overwrite quality gate. Triality's bind/unbind overhead falls from about
38% at length 64 to 12% at 4,096. At length 4,096, incremental forward CUDA
allocation is 115.4 MiB direct, 115.7 MiB triality, 35.5 MiB delta, and 25.5
MiB fast weight. The eager delta tree is slower but substantially leaner.

The algebra was also tightened without changing the recurrence: both delta
composition and application now evaluate (LSR^T) as batched matrix products
instead of a three-operand `einsum`, and triality's invariant tensor is stored
on the problem device outside timing. Float64 recurrence, prefix, and gradient
tests remain exact to their maintained tolerances.

## Official fused FLA tier

The formerly open external-kernel gate is now run in an isolated WSL2 Linux
environment on the same RTX 2070 SUPER: `flash-linear-attention 0.5.2`,
`fla-core 0.5.2`, PyTorch `2.13.0+cu130`, Triton `3.7.1`, float16, batch 2.
The official `chunk_delta_rule` and `fused_recurrent_delta_rule` operators are
compared with direct slots, triality slots, and the local eager delta tree.
All rows have 64 recurrent scalars and receive identical hard keys, queries,
and values. Forward discrepancies are at most `0.001465`; local-versus-FLA
gradient discrepancies are at most `0.001307`.

As above, configurations are frozen from three tuning processes and measured
in three new processes with 150/75 samples per cell.

| Length | Direct | Triality | Local eager delta | FLA chunk | FLA recurrent | Fastest fwd+bwd |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1.351 ms | 1.722 ms | 4.903 ms | 0.773 ms | **0.252 ms** | FLA recurrent, 1.048 ms |
| 256 | 1.654 ms | 2.025 ms | 6.137 ms | 0.806 ms | **0.264 ms** | FLA recurrent, 1.061 ms |
| 1,024 | 2.465 ms | 2.733 ms | 8.466 ms | 0.832 ms | **0.520 ms** | FLA recurrent, 1.671 ms |
| 2,048 | 3.065 ms | 3.335 ms | 8.515 ms | 0.844 ms | **0.819 ms** | FLA recurrent, 2.243 ms |
| 4,096 | 4.110 ms | 4.374 ms | 9.040 ms | **0.988 ms** | 1.255 ms | FLA chunk, 2.577 ms |

The systems verdict therefore changes: the production-style delta kernels,
not the local slot prefix tree, are the fastest measured transport-free core.
FLA recurrent wins through length 2,048 and FLA chunk wins at 4,096. At the
longest length, FLA chunk is 4.16x faster than direct slots and uses 0.828 MiB
incremental forward allocation versus 57.4 MiB.

This does **not** reverse the quality result. The official standard delta op
has no per-token value-axis action argument, so the fused comparison freezes
noncommuting transport to identity and excludes address encoding. The quality
campaign includes supplied noncommuting actions and learned/corrupted
addresses. A transported fused kernel, an explicitly budgeted co-moving-frame
reduction, or a full Gated DeltaNet-2 layer is required before systems and
quality results can be combined. See the
[official FLA DeltaNet layer](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/delta_net.py)
and [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791).

## Task B: partial cross-view action and bilinear identification

The maintained blind-action artifact already supplies a ten-seed partial-action
cohort. At length 2048:

| Row | Mean cosine across seeds | Worst seed |
|---|---:|---:|
| Shared triality binding | 1.000000 | 1.000000 |
| Shared-family direct path | 1.000000 | 1.000000 |
| Independent binding | 1.000000 | 1.000000 |
| Independent direct path | 0.494682 | 0.407763 |
| Correct-negative direct oracle | 1.000000 | 1.000000 |

The shared family also recovers the held-out negative complement in 10/10
seeds. The independent family fits all supplied observations but averages
0.9040 complement cosine. Independent binding still retrieves nearly perfectly,
so it is correctly classified as a bypass: it does not consume the failed
negative action. The matched direct path exposes the completion failure.

The equivariant-identification gate supplies the generic bilinear controls:

- Spin(8) structured intertwiner: one fitted scalar, orbit MSE at most
  `4.53e-30`;
- restricted generic bilinear tensor: 512 fitted parameters, interpolates its
  rank-16 training support but has length-8 orbit MSE `0.604` to `0.722`;
- group-augmented generic tensor: full rank and orbit MSE at most `1.67e-29`,
  but receives four times as many endpoints;
- SO(3) cross-product control: reproduces the same structured-versus-restricted
  generic effect.

Thus the structured hypothesis class is sample-efficient on the frozen orbit
task, but neither generic equivariance nor an exceptional triality advantage is
ruled out.

One preregistered Task B row remains theorem-backed rather than independently
rerun: the delta memory using the independently fitted negative action. With
orthogonal one-hot keys, its overwrite, value-transport, and read equations are
exactly the direct-slot equations, so it should reproduce `independent_direct`.
Until that replay is materialized as an artifact, the strict Task B empirical
decision rule remains open.

> **Later resolution:** the strict attempt to reproduce the metric-only
> historical fit failed its frozen provenance gate. A separately preregistered
> prospective cohort retained all learned parameters and passed the Task B
> decision `10/10`, with direct/delta error exactly zero. The paragraph above is
> retained as the state of this original report, not the current programme
> status.

## Best next direction

1. Keep hard/discretized slots as the current quality leader, but use the FLA
   result—not the eager tree—as the systems baseline.
2. Develop discretized or orthogonalized learned routing; the dominant failure
   is address geometry, not delta capacity.
3. Retain the exact chunkwise delta core as the mandatory continuous-key and
   low-memory baseline.
4. Extend the fused tier to noncommuting value transport. A co-moving-frame
   reduction is algebraically plausible but must count its cumulative-action
   state and transforms; alternatively add the value action to a fused kernel.
5. Materialize the Task B independent-action delta replay to close the last
   empirical row.
6. Use triality where its cross-view action-sharing prior is actually exercised;
   do not advertise it as a generic overwrite mechanism.

## Canonical artifacts

- `artifacts/matched_learned_retrieval_task_a_seed0.json` through
  `seed9.json`: independently durable frozen raw rows;
- `artifacts/matched_learned_retrieval_task_a_seeds0_9.json`: validated
  aggregate;
- `artifacts/matched_memory_cores_cuda_rtx2070s_20260810.json`: superseded
  first-pass eager artifact, retained for audit history;
- `artifacts/matched_memory_cores_cuda_rtx2070s_frozen_aggregate_20260810.json`:
  corrected three-process local core/end-to-end aggregate;
- `artifacts/fla_delta_rule_cuda_rtx2070s_frozen_aggregate_20260810.json`:
  official FLA three-process transport-free core aggregate;
- `artifacts/matched_retrieval_campaign_synthesis_20260810.json`: claim-audited
  Task A/Task B synthesis;
- `artifacts/task_b_delta_action_replay_seeds0_9.json`: failed strict historical
  replay plus exact direct/delta control;
- `artifacts/task_b_paired_action_replication_seeds20_29.json`: prospective
  retained-parameter Task B close;
- `artifacts/matched_retrieval_campaign_synthesis_task_b_closed_20260810.json`:
  current claim-audited synthesis;
- `artifacts/matched_learned_retrieval_task_a_dev101_full.json`: excluded full
  development run.

SHA-256 digests are recorded in `ARTIFACTS.sha256`.

## Claim boundary

This is a controlled synthetic campaign. It establishes neither language-model
quality nor universal memory superiority. It does establish exact local scan
algebra, a reproducible address-geometry robustness result, a constrained
shared-action completion result, and a clear separation between memory update,
address inference, equivariant prior, and systems kernel.
