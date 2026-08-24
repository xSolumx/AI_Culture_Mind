# Hybrid Memory v1.4.4 G6 External-Learning Validation

Status: frozen after the G5b single-seed comparison and before model seeds
1643, 1657, and 1663 are run.

Date: 2026-08-24

## Trigger

G5b's corrected causal reverse-binding objective made every actual architecture
learn L96. On its single paired seed, v1.4.4 reached 97.375% L96 and 95.996%
L512, actual Transformers Mamba-2 reached 98.657% and 77.979%, and actual OLMo
Hybrid reached 100% and 58.105%.

This suggests that v1.4.4 can learn and retain the association using only
externally observable labels, but one seed is not robust evidence. G6 tests
only the local v1.4.4 claim across three fresh seeds.

## Frozen model and objective

- Hybrid Memory v1.4.4 tied-address configuration from G4f;
- final query retrieval cross entropy coefficient 1.0;
- reverse-key reconstruction at observed value events coefficient 0.25;
- no access to internal query/key projections, memory states, write gates, or
  intermediate block logits;
- AdamW 0.003, weight decay 0.01, gradient clip 1.0, batch size 32.

## Frozen curriculum

| Phase | Pairs | Queries | Length | Updates |
|---|---:|---:|---:|---:|
| 1 | 2 | 2 | 16 | 300 |
| 2 | 4 | 4 | 24 | 300 |
| 3 | 8 | 8 | 48 | 400 |
| 4 | 16 | 16 | 96 | 1,200 |

Each seed receives 774,400 retrieval labels and 774,400 reverse-binding labels.

## Frozen seeds and evaluation

- model seeds: 1643, 1657, 1663;
- training data seed: model seed plus 100,000;
- L96 evaluation seed: model seed plus 2,000,000;
- L512 evaluation seed: model seed plus 2,100,000;
- 16 batches of 32 sequences per length and seed;
- L96 has 8,192 query decisions; L512 has 2,048;
- retained checkpoints, hashes, source commit, clean/nonclean start status, and
  preregistration hash.

## Gate

Pass only if every fresh seed reaches at least 90% exact query accuracy at both
L96 and L512. A mean cannot hide a weak seed. Missing or non-finite artifacts
fail the gate.

## Claim boundary

A pass validates robust synthetic associative learning from external causal
retrieval and reverse-binding labels at the declared budget. It does not
validate ordinary label-free next-token learning, natural-language quality,
pretrained-model superiority, or fused speed.

