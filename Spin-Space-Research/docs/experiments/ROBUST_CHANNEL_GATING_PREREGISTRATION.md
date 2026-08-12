# Robust Channel-Gating Preregistration

## Status

Written on 2026-08-03 after completing the pair-0/pair-1 subset lattice and
before evaluating any model on changed-generator equivalence-class index 2.
This is a prospective follow-up, not part of the earlier preregistrations.

## Question

Can a fixed, deployable channel gate recover the stable auxiliary ensemble
without searching subsets using test labels?

## Frozen model contract

For each of the ten deterministic GA checkpoints:

- select the anchor once as the maximum-commutator channel on the original
  token actions;
- freeze token actions, recurrent initial state, output-head weights and bias,
  and logit scale;
- fix `alpha_anchor = 1`;
- learn only the other three scalar gates with `alpha_c = sigmoid(beta_c)`, so
  every auxiliary coefficient lies in `[0, 1]`;
- use one gate vector for every length and generator alphabet.

This is a decoder intervention, not recurrence retraining.

## Data allocation fixed before index-2 evaluation

- Gate fitting: original alphabet and changed-generator class index 0, lengths
  `16, 32, 48, 64, 80, 128, 192, 256`, one deterministic batch of 256 per
  stratum.
- Model selection: changed-generator class index 1, every multiple of 16 from
  L16 through L256, one disjoint deterministic batch of 512 per length.
- Final test: changed-generator class index 2, every multiple of 16 from L16
  through L256, two new deterministic batches of 512 per length.
- Regression evaluation: original and indices 0 and 1 use the same final-test
  batch count and disjoint evaluation seeds after gate selection.

Generator classes quotient inverse re-labellings and are selected only by
repository order. No accuracy participates in choosing class index 2.

## Objective and selection

Initialize every auxiliary `beta` to zero (`alpha = 0.5`). Optimize 400 Adam
steps at learning rate `0.05`. For every fit stratum compute

`mean softplus(1 - (true_logit - largest_false_logit))`.

Aggregate strata with `0.1 * logsumexp(stratum_loss / 0.1)`, a smooth
worst-stratum objective. Do not add sparsity or an oracle-subset target.

At step 0 and every ten updates, score the fixed pair-1 selection cache. Choose
the checkpoint with the highest dense minimum accuracy, then highest dense mean
accuracy, then earliest step. Pair-1 labels may select the three gates; pair-2
labels may not.

## Baselines and reporting

On every final evaluation alphabet report dense per-length accuracy for:

1. anchor only;
2. all channels;
3. learned fixed gates.

The previously measured pair-0 oracle subset is an upper diagnostic only. It
is unavailable to the fitting algorithm and is not a fair deployable baseline.

## Gates

Per seed, the learned gate passes the primary test only if it is at least 90%
at every pair-2 dense length. Report pass counts and the full per-seed
distribution; do not promote only a mean.

The stronger reliability target is ten of ten pair-2 passes plus no learned
gate reducing the pair-1 or original dense floor by more than one percentage
point relative to the better of anchor-only and all-channels for that seed.
Failure of the stronger target is still informative: it means a three-scalar
fixed gate cannot identify the oracle-stable ensemble under this allocation.
