# Hybrid Memory v1.4.4 G8 Target-Distance Consolidation

Status: frozen after G7 failure analysis and before any G7 checkpoint receives
additional training.

Date: 2026-08-25

## Trigger

G7 competence pacing made every fresh seed strong at the trained L96, with a
94.312% minimum. It nevertheless failed at L512: mean 89.404%, minimum 85.889%.
It also showed that early-task competence is not monotonic. Seeds 1693 and 1699
missed their P2/P4 competence caps but later mastered P8/P16.

The remaining learning problem is target-distance retention, not merely
short-task acquisition. G8 directly trains that distribution. G7 remains
failed regardless of G8.

## Frozen continuation

Apply the identical continuation to G7 checkpoints 1693, 1697, and 1699:

- 600 fresh updates;
- 16 stored pairs and 4 queries;
- length 512;
- batch size 16;
- external query retrieval coefficient 1.0;
- external reverse-key reconstruction coefficient 0.25;
- restore the exact G7 AdamW optimizer state, then set learning rate to 0.001;
- retain weight decay 0.01 and gradient clip 1.0;
- training namespace: model seed plus 700,000;
- no seed-specific checkpoint, budget, or hyperparameter changes.

This adds 38,400 retrieval labels and 153,600 reverse-binding labels per seed.

## Frozen evaluation

- new L96 namespace: model seed plus 2,200,000;
- new L512 namespace: model seed plus 2,300,000;
- 16 batches of 32 sequences per seed and length;
- L96 has 8,192 decisions; L512 has 2,048;
- retain source and output checkpoint hashes, optimizer state, source commit,
  starting Git status, and preregistration hash.

## Gate

Pass only if every seed reaches at least 90% exact query accuracy at both L96
and L512. A mean cannot hide a weak seed. Non-finite or missing evidence fails
the gate.

## Claim boundary

A pass establishes a robust external-label synthetic training schedule that
includes direct target-distance consolidation. It does not rescue G7, establish
ordinary label-free next-token learning, validate natural-language quality,
prove upstream superiority, or establish fused speed.

