# G15B-R2 collision-only erase protocol

**Frozen:** 2026-08-26, after G15B-R1 and before any R2 intervention metric is
inspected.

**Entry evidence:**
[`G15BR1_EVENT_ERASE_RESULTS.md`](G15BR1_EVENT_ERASE_RESULTS.md)

**Status:** prospective zero-update checkpoint diagnostic

## Question

G15B-R0 changed collision timing and truncated the learned write continuation.
G15B-R1 preserved the continuation but erased at first writes and overwrites
alike. R2 completes the missing factorial cell:

> Does symmetric erase help when it is applied only at true collisions while
> every learned write control, including the one-token continuation, remains
> bitwise unchanged?

This is oracle state-history timing. It is causal, but the existing token-local
controller cannot infer it. A pass can authorize an explicit occupancy-state
mechanism study; it cannot authorize retraining the same controller.

## Bound evidence and runtime

Use the same retained G15B identity checkpoints for seeds 2309, 2311, and 2333,
the exact G15B evaluation namespace, 4,096 decisions per task/length cell, and
batch cap 16 at lengths 128, 512, and 1,024. Perform zero optimizer updates.

Frozen artifact hashes:

- G15B: `f74d860e30ab40ec747521dfcecd74aac2bb75151206c25b7104d334727429eb`;
- G15B-R0: `4d92d6af2fb062cf2baaa035c4e4eff89d494dfcb56b9b666523bbbdbfe3cf9c`;
- G15B-R1: `c015b128846e4b5c63d927778815a87728a7d613369163b1027ed3dd9f0b2912`.

All artifacts must be clean-start evidentiary `quality` results from exact
SM75, use the frozen seeds, bind their direct parents, and have failed their
registered adjudications.

## Frozen interventions

All modes preserve learned query, key, value, write, retention, output gate,
decoder, identity transport, and the complete learned write continuation.

1. `learned`: untouched checkpoint controls.
2. `soft_collision_erase`: erase equals learned write amplitude only at true
   overwrite events and is zero elsewhere.
3. `exact_collision_erase`: erase is one only at true overwrite events and is
   zero elsewhere.

The collision labels come from commissioned task history. They are an oracle
causal intervention, not an autonomous controller implementation.

## Query strata

For overwrite cells, report each intervention separately on three mutually
exclusive causal strata:

1. `before_any_overwrite`: no overwrite of any key precedes the query;
2. `after_unrelated_overwrite_only`: at least one other key was overwritten,
   but the queried key was not;
3. `after_same_key_overwrite`: the queried key was overwritten before the
   query.

This separates stale-value correction from collateral damage to other keys.

## Integrity

- parent and checkpoint hashes must match exactly;
- baseline query accuracy, exact-episode accuracy, and bits/query must replay
  within `1e-12`;
- ordinary model-forward and reconstructed learned-control logits must be
  bit-identical;
- all non-erase controls must be bit-identical in every batch and arm;
- exact collision masks, the local-write decoder, and the R0 temporal-
  observability witness must pass;
- the result must start from a clean commit on CUDA compute capability `(7,5)`.

## Frozen decision

An intervention supports an explicit occupancy-state successor only if every
three-seed mean satisfies:

- overall overwrite improves by at least `0.10` at every length;
- `after_same_key_overwrite` improves by at least `0.10` at every length;
- `before_any_overwrite` and `after_unrelated_overwrite_only` trail learned by
  no more than `0.02` wherever the stratum is populated;
- MQAR and selective copy trail learned by no more than `0.02` at every length;
- needle accuracy is at least `0.999` at every length.

If both modes pass, select the smaller mean nonnegative MQAR/selective/guard-
stratum degradation, breaking ties in favor of `soft_collision_erase`.

If post-same-key recall improves but a guard stratum fails, select an
oblique/dual erase-address diagnostic. If post-same-key recall does not improve,
select exact logical-component replacement of the complete learned write
program. Do not train either route from this checkpoint result alone.

## Exact reproduction

From the repository root in the SM75-capable WSL environment:

```bash
PYTHONPATH=SSM-Models /home/local/.venvs/sm75-native-2026/bin/python \
  -m hybrid_memory_v1_4.g15br2_collision_erase \
  --mode quality --device cuda \
  --checkpoint-directory /home/local/g15b_bd5045a_quality_attempt2_checkpoints \
  --output /home/local/g15br2_<commit>_quality.json
```

## Nonclaims

No R2 result is fresh training or generalization evidence. It cannot promote
G15C, external-loss-only learning, the present token-local controller, an
optimizer, Spin transport, natural language, scaling, or a model family. All
earlier G15 results retain their original boundaries.
