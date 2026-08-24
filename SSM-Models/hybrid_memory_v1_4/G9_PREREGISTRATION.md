# Hybrid Memory v1.4.4 G9 Fresh Combined-Schedule Validation

Status: frozen after G8 passed and before model seeds 1721, 1723, and 1733 are
run.

Date: 2026-08-25

## Question

Does the complete external causal learning schedule succeed from random
initialization on fresh seeds, without continuing exposed G7 checkpoints?

G8 passed as a prospectively frozen continuation, but it did not validate the
whole schedule from scratch. G9 is that missing gate.

## Frozen model and objective

- Hybrid Memory v1.4.4 tied-address configuration;
- query retrieval cross entropy coefficient 1.0;
- reverse-key reconstruction at observed value events coefficient 0.25;
- no internal address, memory-state, write-gate, or intermediate-logit labels;
- AdamW, weight decay 0.01, gradient clip 1.0;
- one optimizer state retained continuously across every phase.

## Frozen schedule

The first four fixed counts are the maximum updates actually consumed at each
phase across the exposed G7 seeds, not a competence gate. The final phase is
the successful frozen G8 distance continuation.

| Phase | Pairs | Queries | Length | Batch | Updates | LR |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 2 | 16 | 32 | 1,200 | 0.003 |
| 2 | 4 | 4 | 24 | 32 | 1,200 | 0.003 |
| 3 | 8 | 8 | 48 | 32 | 1,400 | 0.003 |
| 4 | 16 | 16 | 96 | 32 | 1,300 | 0.003 |
| 5 | 16 | 4 | 512 | 16 | 600 | 0.001 |

Per seed this supplies 1,292,800 retrieval labels and 1,408,000 reverse-
binding labels. Phase 5 uses a distinct training namespace and lowers the
existing optimizer's learning rate without resetting its state.

## Frozen seeds and evaluation

- model seeds: 1721, 1723, 1733;
- phases 1-4 training namespace: model seed plus 100,000;
- phase 5 training namespace: model seed plus 700,000;
- L96 evaluation namespace: model seed plus 2,400,000;
- L512 evaluation namespace: model seed plus 2,500,000;
- 16 batches of 32 sequences per seed and length;
- L96 has 8,192 decisions; L512 has 2,048;
- retain model and optimizer states, phase traces, source commit, starting Git
  status, preregistration hash, and checkpoint hashes.

## Gate

Pass only if every fresh seed reaches at least 90% exact query accuracy at both
L96 and L512. A mean cannot hide a weak seed. Missing or non-finite evidence
fails the gate.

## Claim boundary

A pass validates the complete external-label synthetic schedule from scratch
on three fresh seeds. It does not validate ordinary label-free next-token
learning, natural-language quality, upstream superiority, or fused speed.

