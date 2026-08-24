# Hybrid Memory v1.4.3 G4e Identity-Path Preregistration

Status: frozen after identity-path development seeds 1429 and 1459 completed
and before fresh model seeds 1481, 1483, and 1487 are run.

Date: 2026-08-24

## Trigger

v1.4.2 G4d failed because seed 1459 reached 86.328% at L512. The identity-
preserving v1.4.3 value/read path then reached 95.557% on prior weak seed 1429
and 90.381% on prior weak seed 1459 under the old budget. Both are exposed-seed
development results.

Seed 1459 cleared the threshold narrowly and its final-phase mean retrieval
loss remained 0.819. G4e therefore freezes a larger useful-label budget before
new seeds: the architecture and first three phases stay fixed, while the final
16-pair phase increases from 600 to 1,200 updates.

## Frozen model

- version 1.4.3;
- model dimension 64;
- `gated_delta -> attention`;
- four Gated Delta heads, key dimension 32, value dimension 16;
- normalized value injection;
- identity initialization for value projection and output projection;
- identity-centered output gate `1 + tanh(g)`;
- Gated Delta residual raw scale 0.0, giving sigmoid scale 0.5;
- attention residual raw scale -2.0;
- four attention heads, window 1024;
- convolution width 4, expansion 2, dropout 0, tied embeddings.

## Frozen training

- final retrieval coefficient 1.0;
- association/write/retention coefficient 0.25;
- intermediate post-memory retrieval coefficient 0.50;
- AdamW, learning rate 0.003, weight decay 0.01;
- gradient clip 1.0;
- batch size 32;
- fresh deterministic episode every update.

| Phase | Pairs | Queries | Length | Updates |
|---|---:|---:|---:|---:|
| 1 | 2 | 2 | 16 | 300 |
| 2 | 4 | 4 | 24 | 300 |
| 3 | 8 | 8 | 48 | 400 |
| 4 | 16 | 16 | 96 | 1,200 |

Useful query labels per seed: 774,400. The protocol remains explicitly label-
supervised.

## Frozen validation

- new model seeds: 1481, 1483, 1487;
- L96 evaluation namespace: model seed plus 1,000,000;
- L512 evaluation namespace: model seed plus 1,100,000;
- 16 batches of 32 sequences per length and seed;
- L96 has 8,192 query decisions; L512 has 2,048;
- exact query accuracy, exact-sequence accuracy, bits/query, checkpoint hash,
  preregistration hash, source commit, and start status retained.

## Gate

Pass only if every seed reaches at least 90% exact query accuracy at both L96
and L512. An average cannot hide a weak seed. Any non-finite value, missing
checkpoint, or protocol deviation fails the gate.

## Boundary

A pass validates robust label-supervised synthetic associative learning at the
declared 774,400-label budget. It does not establish label-free learning,
natural-language quality, upstream superiority, or fused speed.
