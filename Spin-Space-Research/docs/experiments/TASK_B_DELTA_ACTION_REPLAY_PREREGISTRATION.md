# Task-B independent-action delta replay preregistration

- **Frozen:** 2026-08-10, before implementation results were inspected
- **Development seed:** `101`, excluded from every reliability count
- **Frozen reliability seeds:** `0` through `9`, one durable process per seed
- **Arithmetic:** CPU float64
- **Training budget:** the original independent-family budget, `500` Adam
  steps per curriculum stage followed by `150` LBFGS steps

## Question

Does a standard delta memory supplied with the independently fitted
negative-chiral action reproduce the matched direct-slot failure on held-out
Spin(8) action words, as predicted by the one-hot gauge equivalence?

This closes the only Task-B row in
[`MATCHED_LEARNED_RETRIEVAL_RESULTS.md`](MATCHED_LEARNED_RETRIEVAL_RESULTS.md)
that is currently theorem-backed but not independently materialized.

## Frozen inputs

- `artifacts/spin8_blind_alias_action_seeds0_9.json` is the authority for the
  original partial-action cohort and soft direct row.
- `src/spin8_blind_alias_action.py` supplies the unchanged teacher,
  calibration design, independent action-family training, alias world, random
  streams, and dense length ladder.
- `src/schurscan_delta_memory.py` supplies the standard factored delta update
  and work-efficient scan.
- The starting implementation commit is `c4b6310`.

The partial observations remain five vector/positive columns and a rank-two
negative calibration plane. Evaluation still uses the full negative
complement, fresh continuous aliases, full-dimensional values, noncommuting
action words, and lengths `16, 32, 64, 128, 256, 512, 1024, 2048`.

## Why soft and hard rows are separated

The historical direct row uses a low-temperature soft slot route. Standard
delta overwrite with key (k),

\[
S'=(I-kk^T)S+kv^T,
\]

is exactly a direct slot overwrite only when (k) is one-hot. Applying the
formula to a soft probability vector would change both the erase geometry and
the scientific question. The paired replay therefore reports:

1. the original soft independent-direct row, rerun only as a determinism audit;
2. independent-action direct memory using the learned route's hard `argmax`;
3. independent-action standard delta memory using the identical one-hot key;
4. correct-negative-action direct memory under the same hard route;
5. correct-negative-action standard delta memory under the same hard route.

No logical label is supplied to the hard rows. The label is used only after
the decision to measure routing agreement.

## Paired execution contract

Direct and delta states receive identical writes, rotations, queries, aliases,
values, and event masks. At a write with one-hot key (e_h),

\[
M'_h=v,\qquad
S'=(I-e_he_h^T)S+e_hv^T.
\]

At a value rotation (R), direct rows use (m_h'=Rm_h), while delta memory
uses (S'=SR^T). Querying uses the same one-hot row selector. The delta row
must call the maintained delta transition implementation; copying direct-slot
outputs is forbidden.

The replay also constructs complete direct and delta transition sequences and
compares their work-efficient parallel scans with recurrent execution.

## Metrics

For every seed and length, retain:

- query count;
- mean and minimum retrieval cosine;
- mean and maximum relative squared error;
- maximum direct/delta state and prediction discrepancy;
- learned hard-route stability relative to each label's learned center slot;
- original soft-row reproduction error;
- direct and delta parallel/recurrent errors;
- held-out negative-complement cosine;
- observed-column fit error;
- recurrent-state scalars.

Aggregate reports preserve every per-seed row and report means, minima,
maxima, and paired seed counts. An aggregate without exactly seeds `0` through
`9` is invalid.

## Frozen decision rules

The implementation gate passes only if:

- hard direct and hard delta states/predictions agree within `1e-10` in every
  seed and length;
- both parallel/recurrent errors are below `1e-9`;
- the soft independent-direct replay reproduces every stored mean cosine
  within `1e-12`;
- hard learned-route stability is at least `0.99` for writes and queries, with
  collision-free, cross-encoder-consistent center permutations;
- the correct-negative delta ceiling has mean cosine at least `0.995` and
  minimum cosine at least `0.98` at every length.

The preregistered Task-B representation-prior rule closes positively only if,
in at least `8/10` seeds:

- the independent family fits the supplied observations below `1e-6` MSE;
- the existing shared-family direct row has length-2048 mean cosine at least
  `0.995`;
- the independently transported hard delta row has length-2048 mean cosine at
  most `0.90`.

If hardening materially changes the independent failure relative to the soft
row, both outcomes remain visible. No result here can establish an exceptional
triality advantage over other correct equivariant priors, a generic memory-
capacity advantage, or a model-level quality result.

## Planned artifacts

- `artifacts/task_b_delta_action_replay_seed0.json` through `seed9.json`;
- `artifacts/task_b_delta_action_replay_seeds0_9.json`;
- a separate development artifact for seed `101`.

The preregistration is not itself a result.

## Pre-cohort implementation addendum: permutation-gauge correction

The excluded seed-`101` development smoke exposed a scoring error before any
frozen seed was run. A learned collision-free slot family is identifiable only
up to a global permutation. Comparing its hard slot index directly with the
numeric latent label therefore assigns an arbitrary gauge and can report low
"agreement" for a perfectly consistent router.

The frozen routing gate is corrected as follows:

- each label's reference slot is the `argmax` selected at that alias center;
- fresh write/query aliases are scored against their own encoder's reference
  slot for that label;
- write and query center maps must each be collision-free;
- the two center maps must agree label by label;
- write and query alias stability must each be at least `0.99`.

The label is still used only after routing, for diagnostics. No event stream,
training budget, memory equation, seed, action threshold, or retrieval decision
rule changes. The failed gauge-dependent development artifact is overwritten
after this correction and remains excluded from reliability counts.
