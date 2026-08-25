# G15B interleaved associative routing protocol

**Frozen:** 2026-08-25, after G15A-S and before any G15B outcome is inspected  
**Parent:** [G15 Spin-Dirac preregistration](G15_SPIN_DIRAC_PREREGISTRATION.md)  
**Entry evidence:** [G15A-S results](G15_SPIN_DIRAC_RESULTS.md#g15a-s-spanning-and-center-sensitive-chart-transfer) | **Executable runner:** [`g15b_interleaved_cohort.py`](g15b_interleaved_cohort.py) | **Task generator:** [`g15b_interleaved_tasks.py`](g15b_interleaved_tasks.py)

## Decision being tested

G15A-S established a learned composition chart under oracle keys, values,
erase/write timing, and query timing. It did not establish that the token model
can learn content address, last-write-wins editing, selective writing, or query
selection. G15B therefore asks:

> Can the current Spin-Dirac fast-weight law be explicitly commissioned to
> learn those controls on fresh, interleaved generic episodes, with full Spin
> transport noninferior to the frozen identity and commuting controls?

This is commissioned-controller evidence, not label-free discovery. A later
external-loss-only lane is allowed only after this identified controller gate
passes. Retrieval without the controller and intervention diagnostics below is
a model-task result, not a learned-control result.

## Frozen arms and shell

The primary arms remain exactly those in the parent preregistration:

| Arm | Transport | Second read sector |
|---|---|---|
| I | identity | identity copy |
| C | commuting `SO(2)^4` | fixed Clifford map |
| S | full factorized Spin(8) | fixed Clifford map |

The parent readout difference is preserved rather than rewritten after G15A.
An identity-transport/Clifford-read arm may be added only as a diagnostic if I
and C/S separate; it cannot replace a primary arm. A frozen broken-Spin arm is
conditional on an apparent S advantage and is not part of noninferiority.

Every primary arm uses the same:

- one `spin_dirac` block, no attention;
- width 64, four heads, and four `8 x 8` association matrices;
- causal depthwise convolution width 4 to parse local event triples;
- tied token/output embeddings, expansion 2, and dropout zero;
- tied normalized query/key projection and always-live query read;
- equivariant head-scalar erase/write/retention, bounded values, retention
  range `0.999` to `0.999999`, initialized at `0.9995`;
- trainable zero-initialized coordinate projection with no supplied coordinates;
- `HarmonicMuonAdamW`, weight decay `0.01`, global clip `1.0`, and one
  continuous optimizer state across phases.

No binary query-event/internal-query switch may be added. Prior Pure Spin
results showed that a hard-off event path can give the selected address exactly
zero temporal credit, while a continuously live query is already present in
[`spin_dirac_memory.py`](spin_dirac_memory.py).

## Shared-payload interleaved grammar

The old generic builders remain useful diagnostics, but their disjoint key,
value, and filler ranges plus front-loaded writes and trailing queries are too
shortcut-friendly for a controller-identification claim. G15B freezes a new
alphabet:

- five role markers: `PAD`, `WRITE`, `QUERY`, `SELECT`, `ITEM`;
- 64 shared payload symbols which may be a key, value, or unmarked distractor
  in different episodes.

The grammar is:

- `[WRITE, key, value]`: commit value under key;
- `[QUERY, key]`: predict the latest preceding committed value for key;
- `[SELECT, key, value]`: selected candidate, therefore commit;
- `[ITEM, key, value]`: unselected candidate, therefore do not commit;
- unmarked positions: distractors from the same payload alphabet.

Events occupy random nonoverlapping positions. Queries are interleaved with
writes and can occur before and after an overwrite of the same key. Every query
follows a valid write, and its target is determined only by the latest valid
write strictly before it. The four frozen strata are:

1. unique-key MQAR with several interleaved queries;
2. overwrite/collision with queries before and after changed overwrites;
3. keyed selective copy, where `ITEM` and `SELECT` share the same payload
   distribution but only selected triples are retrievable;
4. one exact-distance needle among shared-payload distractors.

The repository's older anonymous ordered selective-copy task is retained as a
separate diagnostic. It additionally asks for an ordinal pointer/pop mechanism
not present in a pure content-addressed `8 x 8` state and cannot veto this
controller gate without an oracle ceiling.

Leakage rejection is mandatory: the answer is absent from the query pair; the
target differs from the query key; the target does not occur in the query
token's four-position receptive field; each answer is replayed from the latest
preceding valid event; and future writes never affect an earlier query. Payload
and event schedules are byte-identical across arms.

## Frozen commissioned loss

The parent G15B preregistration permits identical commissioning labels across
arms. G15B binds those labels to:

```text
retrieval
+ 0.25 * causal reverse binding
+ 0.25 * unique-key address classification
+ 0.25 * balanced write timing
+ 0.10 * balanced overwrite erase timing
+ 0.01 * non-edit retention
```

- Retrieval is vocabulary cross-entropy at every query-key position.
- Reverse binding predicts the already observed key at every valid write value
  position. It is external and causal.
- Address classification groups all writes of the same key into one normalized
  prototype and asks each query vector to select the key identity at temperature
  `0.10`. It must not classify overwrite occurrences: the query knows the key,
  while erase/write timing determines which value is current.
- Write timing uses equally weighted positive and negative BCE terms and is
  positive only at `WRITE` or `SELECT` value positions.
- Erase timing is positive only at second-and-later valid writes of a key.
  First writes, `ITEM`, query, and distractor positions are negatives.
- Retention is normalized within its configured interval and targets the
  maximum only at non-edit positions.

No coordinate, action, carrier, state, or intermediate-geometry target is
allowed. Controller labels may not choose checkpoints, alter a seed schedule,
or rescue a failing quality run.

## Frozen optimizer and schedule

Quality seeds are `2309`, `2311`, and `2333`. A seed-23 infrastructure smoke
is non-evidentiary and cannot alter this protocol.

| Phase | Length | Live keys | Max writes | Queries | Batch | Updates | LR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 2 | 4 | 4 | 32 | 800 | 0.003 |
| 2 | 128 | 4 | 8 | 4 | 32 | 1,000 | 0.003 |
| 3 | 256 | 8 | 16 | 8 | 16 | 1,200 | 0.003 |
| 4 | 512 | 8 | 24 | 8 | 8 | 800 | 0.001 |
| 5 | 1,024 | 8 | 24 | 8 | 4 | 400 | 0.001 |

The task cycle is `MQAR, overwrite, overwrite, selective, needle`. Because the
needle contributes one exact-distance query while the other strata use the
phase query count, this yields exactly 375,360 scored retrieval decisions per
seed across 4,200 updates.
The explicit learning-rate decay is first-order and frozen, rather than a
post-failure continuation. There is no early stopping, competence pacing,
development recipe selection, or seed-specific extension.

## Preflight gates

Before quality training:

1. exact FP64 `forward_controls` replay with up to eight live orthogonal
   addresses must pass first write, unrelated-key preservation, same-key
   overwrite, and repeated-query reads below `1e-10`;
2. recurrent, parallel, arbitrary-chunk, and token-step paths must agree;
3. every generated episode must pass latest-preceding replay and all leakage
   checks;
4. oracle address/edit/query execution must reach at least `0.999` direct-read
   accuracy on every stratum;
5. finite, nonzero role-position activation gradients must reach query, key,
   value, write, overwrite erase, retention, and coordinates through their
   declared losses;
6. the unique-key address descent direction must be correct at scored queries;
7. every trainable tensor appears in exactly one optimizer group, and paired
   arms have identical trainable shapes, model seeds, and data hashes;
8. the clean commit, source/protocol hashes, useful label/event counts, state
   bytes, timings, peak allocation, and checkpoint hashes are recorded.

Failure stops the cohort and is classified as task/capacity, observability,
implementation, or optimizer-partition failure before any model-quality claim.

## Frozen evaluation and interventions

Each arm and seed receives held-out namespaces with at least 4,096 query
decisions per task/length at lengths 128, 512, and 1,024. Needle distances are
exactly 64, 448, and 960. Report:

- exact query and exact episode accuracy plus bits/query;
- unique-key query-address top-1 and correct-minus-best-wrong cosine margin;
- same-key address consistency across overwrites;
- write precision/recall/F1;
- overwrite-erase precision/recall/F1 and first-write/filler false positives;
- non-edit retention distribution and coordinate RMS/max;
- task, distance, live-key, collision, and query-location strata;
- useful event counts, synchronized time, peak allocation, and artifacts.

Run paired evaluation-only interventions through a fail-closed complete-control
override after learned controls are formed and before transitions:

1. no memory contribution;
2. write forced to zero;
3. erase forced to zero;
4. query addresses permuted among simultaneously live keys;
5. complete oracle address/write/erase replacement, diagnostic only.

Required signatures are: no-memory and no-write each reduce L512/L1024
accuracy by at least `0.50`; wrong query reduces multi-key accuracy by at least
`0.50`; no-erase reduces overwrite accuracy by at least `0.20` while changing
unique-key MQAR by no more than `0.05`; and oracle execution itself passes.

## Binding gates

Every arm and seed must satisfy:

- query accuracy at least `0.90` and exact-episode accuracy at least `0.50` on
  every task and evaluation length;
- unique-key address top-1 at least `0.98` with mean margin at least `0.20`;
- write-event F1 at least `0.98`;
- overwrite-erase recall at least `0.95` and non-overwrite erase false-positive
  rate at most `0.02`;
- non-edit retention fifth percentile at least `0.9995`;
- every causal intervention signature, finite check, provenance check, and
  train/evaluation namespace-separation check.

For the parent transport decision, S's three-seed mean may be no more than
`0.01` below the better of I and C in any task/length cell, and no individual S
seed may trail the better paired I/C seed by more than `0.03`. There is no
averaging rescue for an absolute controller or causal-use failure.

## Conditional external-loss-only lane

Only after the commissioned gate passes, a fresh-from-initialization lane may
remove address/write/erase/retention labels and keep retrieval plus reverse
binding. That would test externally commissioned causal association without
internal memory labels. A supervised checkpoint with labels later removed does
not establish autonomous discovery and must be named supervised commissioning
transfer instead.

## Explicit falsifiers and nonclaims

- Oracle failure means the task exceeds the state law or implementation; no
  learned result is admissible.
- Retrieval without address/gate metrics is not controller identification.
- Retrieval without the intervention drops is a bypass, not memory evidence.
- Learned failure with oracle success is a controller-learning failure.
- S noninferiority failure blocks G15C even if G15A-S remains valid.
- S beating I/C still does not establish Spin necessity on generic data.

A complete pass supports only an explicitly commissioned generic Spin-Dirac
controller with noninferior full-Spin transport. It does not establish
label-free learning, ordinary next-token learning, natural-language advantage,
triality capacity, fused efficiency, or parameter/compute scaling. All older
G15A/L/F/R/S results remain separate evidence and are never overwritten.
