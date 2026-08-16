# Exploratory frozen-checkpoint `2.A5` unseen-relation protocol

**Selection and interpretation frozen:** 2026-08-16T18:53:01+02:00
**Checkpoint cohort:**
[`pure_rotor_2a5_center_pilot300.json`](artifacts/pure_rotor_2a5_center_pilot300.json)
**Cohort SHA-256:**
`911815d9e104fa08e632161f97f41a966991a9102c70ca65e52a5f07d28d4476`

**Completed result:** evaluation finished at **2026-08-16T18:56:42+02:00**.
The Spin checkpoints pass the exploratory gate without retraining; see
[`PURE_ROTOR_2A5_UNSEEN_RELATION_RESULTS.md`](PURE_ROTOR_2A5_UNSEEN_RELATION_RESULTS.md).

This is an exploratory, no-retraining falsifier frozen before loading any model
checkpoint for the new words. It tests whether the pilot's success is confined
to the explicit `a,a=z` relation used in its paired evaluation.

## Deterministic word selection

Reconstruct the exact three realized training schedules. Enumerate token words
in increasing length and lexicographic token-ID order, with
`a=0`, `b=1`, `b_inverse=2`. Reject a word if:

1. it occurs as a contiguous substring in any realized seed-0/1/2 training
   input; or
2. it contains any adjacent local rewrite used by the pilot split:
   `a,a`, `b,b_inverse`, or `b_inverse,b`.

At each length, partition the survivors by exact `2.A5` product. Choose the
first length containing at least one identity word and one central word, then
choose the lexicographically first of each. Length 10 has one qualifying
central word but no identity word. Length 11 is the first joint length and
selects:

```text
identity e:
a b b b a b b b b b b
token IDs: 0 1 1 1 0 1 1 1 1 1 1

center z:
a b b a b b b a b_inverse b_inverse a
token IDs: 0 1 1 0 1 1 1 0 2 2 0
```

The exact multiplication table must verify products `e` and `z`. Both complete
words must have zero training occurrences in every seed. The selection rule is
allowed to inspect inputs and exact group labels, but not checkpoint logits or
predictions.

## Paired evaluation

Place the two length-11 blocks inside identical random contexts and score only
positions at or after block completion. Their projected A5 products then agree,
while their exact binary products differ by the center. Test early and late
placement at total lengths `16,64,128`, using two batches of 32 pairs and a
separately namespaced deterministic context schedule for each original seed.

Reuse the exact same 12 checkpoints without any optimizer update:

- Pure Rotor v2.1;
- identity-rotation ablation;
- Spin quaternion scan;
- Transformers Mamba-2.

Recompute and verify every checkpoint SHA-256 before evaluation. Re-run the
exact-table, projective A5, and float64 quaternion oracle controls. Store the
new schedule hashes and exact central-partner audits.

## Predeclared interpretation

1. A replicated Spin relation-generalization result requires central-margin
   accuracy above 75% for every seed and early length, not only for L16.
2. Exact and projective accuracy must be reported alongside the margin. A model
   can prefer the correct center partner while losing the underlying A5 state.
3. The Spin candidate must exceed every trained alternative in early L64 and
   L128 exact accuracy in every seed.
4. A failure would narrow the earlier result to the `a,a` split rather than
   invalidating the exact scan algebra.
5. A pass remains post-hoc exploratory evidence because the word-search rule
   was designed after the pilot result. It must not be relabelled as the
   preregistered pilot gate.
