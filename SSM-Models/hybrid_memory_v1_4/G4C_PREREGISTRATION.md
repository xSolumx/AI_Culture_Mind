# Hybrid Memory v1.4.1 G4c Consolidation Preregistration

Status: frozen after G4b failed and before any G4b validation checkpoint is
continued.

Date: 2026-08-24

## Trigger

G4b passed on model seeds 1423 and 1427 but failed its all-seed gate because
seed 1429 reached only 84.814% at length 512. No G4b validation checkpoint has
received additional training.

The post-hoc hypothesis is that dense short-to-long training learned the rule
but left an initialization-sensitive final optimization basin. This test asks
whether one uniform, prospectively fixed length-512 consolidation phase repairs
all three checkpoints without seed-specific tuning or short-task forgetting.

G4b remains failed regardless of this result.

## Frozen continuation

Apply the identical phase independently to the existing seed 1423, 1427, and
1429 G4b checkpoints:

- 300 fresh updates;
- length 512;
- 16 stored pairs and 4 queries per episode;
- batch size 16;
- AdamW reinitialized at learning rate 0.001 and weight decay 0.01;
- association auxiliary coefficient 0.25;
- gradient norm clipped to 1.0;
- deterministic training namespaces derived from model seed plus 300,000;
- no architecture, initialization, checkpoint selection, or per-seed
  hyperparameter change.

This adds 19,200 scored query labels per seed. Association/write supervision
remains explicit and label-supervised.

## Frozen evaluation

After continuation, evaluate each seed on disjoint fresh namespaces:

- length 96, 16 pairs and 16 queries, 16 batches of 32 sequences;
- length 512, 16 pairs and 4 queries, 16 batches of 32 sequences;
- 8,192 L96 and 2,048 L512 query decisions per seed;
- exact query accuracy, exact-sequence accuracy, and bits/query;
- retained checkpoint SHA-256 and preregistration SHA-256.

## Gate

Pass only if every seed reaches both:

- at least 90% exact query accuracy at length 512; and
- at least 90% exact query accuracy at length 96.

An average cannot hide a weak seed. Any non-finite result, missing checkpoint,
or seed-specific deviation is a failure.

## Interpretation boundary

A pass establishes a robust label-supervised synthetic commissioning schedule:
dense short-to-long curriculum followed by a modest amount of training at the
target distance. It does not establish label-free learning, natural-language
quality, superiority over an upstream model, or a fused-kernel speed result.
