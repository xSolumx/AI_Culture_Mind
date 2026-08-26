# G15B-R4 ownership/background factorial protocol

**Frozen:** 2026-08-26, after sealing G15B-R3 and before inspecting any R4
intervention metric.

**Entry evidence:**
[`G15BR3_LOGICAL_COMPONENT_RESULTS.md`](G15BR3_LOGICAL_COMPONENT_RESULTS.md)

**Status:** prospective zero-update retained-checkpoint diagnostic

## Question

G15B-R3 gives each commissioned key an oracle replaceable component. It raises
ordinary overwrite by 12.2--12.8 points over learned symmetric erase and
reaches 1.0 on a constructed all-strata guard, but fails its frozen promotion
gate. The result leaves two specific ambiguities:

1. must the token immediately after a written value belong to the logical-key
   component, even when it is the next event marker; and
2. does background injection support or contaminate query-time decoding?

R4 asks whether a locally observable value-token-only slot is sufficient and
whether its read requires the shared background channel. It does not
retrospectively re-adjudicate R3.

## Bound evidence and runtime

Use the retained G15B identity checkpoints for seeds 2309, 2311, and 2333,
the exact G15B evaluation namespace, 4,096 query decisions per
task/length/seed cell, batch cap 16, and lengths 128, 512, and 1,024. Perform
zero optimizer updates.

Bind the sealed R3 artifact:

- file:
  [`artifacts/g15br3_logical_component_sm75_2026-08-26.json`](artifacts/g15br3_logical_component_sm75_2026-08-26.json);
- SHA-256:
  `0fe54b8ce38868d67a7ecb0cb888f2279d8809c2bbaf3ccbda678326ff808959`;
- required status: clean-start exact-SM75 `quality`, failed adjudication,
  `post_same_key_improved=true`, and decision to inspect write-tail ownership
  and background/component coupling.

The quality result must start from a clean commit on CUDA compute capability
`(7,5)`. Parent, checkpoint, protocol, and source hashes are recorded.

## Frozen factors

The erase-free per-key recurrence and reset timing remain exactly R3. At every
locally observable valid write of key `j`, reset component `C^(j)` before
adding the current value-token injection. Learned query, key, value, write,
retention, identity transport, output gate, and decoder remain frozen.
Collision history is never used.

### Injection ownership

- `V`: only the labelled write-value token belongs to the key component. Every
  `t+1` injection remains background.
- `VT`: the write-value token and its in-range `t+1` injection belong to the key
  component, exactly matching sealed R3. Final-token writes own only their
  value injection.

### Query-time background coupling

- `BG+`: read `C^(0) + sum_j C^(j)` at every token.
- `BG-`: at locally observable query-key positions, read `sum_j C^(j)` and
  exclude only `C^(0)`. Non-query positions use the full sum. `BG-` does not
  select the queried key component and does not use oracle query-slot routing.

The primary 2-by-2 LWW arms are:

1. `v_lww_bgplus`;
2. `v_lww_bgminus`;
3. `vt_lww_bgplus`, the sealed R3 mechanism reference;
4. `vt_lww_bgminus`.

Score the frozen controls:

- `learned`;
- `erase_free_no_reset_bgplus`, shared by `V` and `VT` because complete
  component summation reconstructs the same monolithic erase-free state;
- `v_no_reset_bgminus`;
- `vt_no_reset_bgminus`.

No arm changes the learned scalar controls except that every component arm
disables the already rejected symmetric erase. The two no-reset `BG-` controls
prevent background exclusion from being credited to replacement.

## Cohorts and query strata

Evaluate MQAR, ordinary overwrite, the deterministic R3 overwrite guard,
selective copy, and needle at every registered length. For ordinary overwrite
and the guard, report:

1. `before_any_overwrite`;
2. `after_unrelated_overwrite_only`;
3. `after_same_key_overwrite`.

The ordinary generator may leave the unrelated-only stratum empty; report zero
support and make no ordinary-cohort claim for it. The constructed guard must
populate all three strata at every length or adjudication fails closed.

## Integrity

- reproduce the sealed R3 `learned`, `erase_free_no_reset`, and
  `vt_lww_bgplus` query metrics within `1e-12`;
- ordinary model forward and the `learned` arm are bit-identical;
- ownership is exclusive and complete; `V` never owns `t+1`, while `VT` owns
  exactly each in-range `t+1` without overlap;
- reset affects exactly one logical component per valid write and occurs before
  current injection;
- LWW and its matching no-reset control are identical on tasks with no
  repeated key writes and at queries before the first same-key overwrite;
- both `BG+` no-reset ownership decompositions reconstruct the same
  erase-free monolithic state and query predictions;
- for each ownership rule, `BG+ read - BG- read` equals the background-
  component read at locally observable query positions;
- a separate FP64 algebraic contract compares monolithic, decomposed, and
  recurrent/parallel component states and reads at residual at most `1e-10`;
- FP32 scored paths must preserve query predictions for every registered replay
  proof, remain finite, and preserve all non-erase controls bitwise;
- local write and query-position decoders, reset masks, task fingerprints,
  parent/source hashes, clean SM75 provenance, and zero updates must pass.

The FP64 contract is a new prospective algebraic check, not a relaxation or
relabeling of R3's failed `5e-4` learned-logit gate. Continue to report tail
roles, final-token writes, state expansion, and wall time. The diagnostic is
not state-, parameter-, compute-, or time-matched.

## Frozen decision gates

An individual LWW arm passes only if every length satisfies all applicable
conditions below.

### Ordinary overwrite

- aggregate LWW accuracy improves at least `0.10` over `learned` and its
  ownership/background-matched no-reset control;
- `after_same_key_overwrite` improves at least `0.10` over both controls;
- within every seed, aggregate and post-same-key accuracy improve at least
  `0.05` over both controls;
- `before_any_overwrite` trails `learned` and its matching no-reset control by
  no more than `0.02`.

### Constructed guard

- aggregate and post-same-key accuracy are at least `0.995`;
- aggregate and post-same-key trail `learned` by no more than `0.005`;
- aggregate and post-same-key improve at least `0.10` over the matching
  no-reset control;
- before-any and after-unrelated-only accuracy are at least `0.98` and trail
  `learned` and the matching no-reset control by no more than `0.02`;
- all three guard strata have nonzero support.

These are prospectively ceiling-aware absolute guard gates. They do not alter
R3's frozen requirement for a 0.10 guard improvement over learned.

### Safety and integrity

- MQAR and selective copy trail `learned` by no more than `0.02`;
- needle accuracy is at least `0.999`;
- every integrity gate passes.

Fresh explicit-slot training is warranted only if at least one value-only arm
passes:

- if only `v_lww_bgplus` passes, screen explicit slots with a shared residual
  background channel;
- if only `v_lww_bgminus` passes, screen explicit slots with a protected slot
  read;
- if both pass, prefer `BG+` as the simpler state law unless `BG-` improves
  ordinary-overwrite accuracy or bits/query by a separately reported material
  margin;
- if only a `VT` arm passes, do not train: success remains dependent on
  ambiguous tail ownership;
- `vt_lww_bgplus` passing confirms the sealed R3 mechanism reference but can
  never by itself relabel R3 or authorize training;
- if no arm passes, reject only this frozen post-hoc ownership/background
  family.

Any authorized training requires its own frozen protocol, trainable
non-oracle ownership, fresh seeds/data, explicit occupancy state, and
parameter/state/compute controls. G15C and the present token-local controller
remain blocked.

## Exact reproduction target

The implementation must expose a smoke mode and an evidentiary quality mode:

```bash
PYTHONPATH=SSM-Models /home/local/.venvs/sm75-native-2026/bin/python \
  -m hybrid_memory_v1_4.g15br4_ownership_background \
  --mode quality --device cuda \
  --checkpoint-directory /home/local/g15b_bd5045a_quality_attempt2_checkpoints \
  --output /home/local/g15br4_<commit>_quality.json
```

Quality must fail closed on a dirty tree, non-CUDA execution, non-SM75
hardware, missing parent/checkpoint hashes, absent source hashes, or incomplete
integrity/support.

## Nonclaims

R4 remains zero-update, retained-checkpoint, commissioned-task mechanism
evidence with oracle component ownership. Reused G15B schedules are not fresh
generalization. A pass cannot establish autonomous slot learning, generic
association, ordinary-text recall, longer-context scaling, efficiency,
optimizer or tokenizer superiority, Spin benefit, G15C, or model-family
promotion. A failure does not falsify GDN2/KDA, dual-address edits, or an
explicit-slot architecture trained from scratch.
