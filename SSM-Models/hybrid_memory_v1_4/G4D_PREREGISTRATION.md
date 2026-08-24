# Hybrid Memory v1.4.2 G4d Successor Preregistration

Status: frozen after v1.4.2 development seeds 1401 and 1429 completed and
before fresh model seeds 1451, 1453, and 1459 are run.

Date: 2026-08-24

## Trigger and diagnosis

G4b and G4c both failed their all-seed gates because seed 1429 remained weak.
The post-G4c diagnostic found:

- 100% key-address top-1 accuracy in all four heads for all seeds;
- near-chance retrieval when the Gated Delta mixer was disabled;
- almost unchanged retrieval when the attention mixer was disabled;
- worse key separation, larger filler/value norms, and larger fast-weight state
  norm in seed 1429.

This localized the remaining problem to interference/value conditioning inside
the memory core rather than route discovery or missing attention.

The v1.4.2 development successor doubled key dimension from 16 to 32,
RMS-normalized value injection to fixed per-head norm, and applied the same
retrieval loss immediately after the Gated Delta block. It reached 97.168% and
92.676% L512 accuracy on previously exposed seeds 1401 and 1429. Those are
development results only.

## Frozen model

- model dimension 64;
- layer plan `gated_delta -> attention`;
- four Gated Delta heads;
- key dimension 32 and value dimension 16 per head;
- value vectors L2-normalized then scaled to norm `sqrt(16)`;
- four attention heads with window 1024;
- causal depthwise convolution width 4;
- expansion 2, dropout 0, tied embeddings;
- semantic parallel affine Gated Delta scan.

## Frozen objective and optimizer

- final retrieval loss coefficient 1.0;
- explicit association/write/retention auxiliary coefficient 0.25;
- intermediate retrieval loss immediately after the Gated Delta block,
  coefficient 0.50;
- AdamW, learning rate 0.003, weight decay 0.01;
- gradient norm clipped to 1.0;
- batch size 32.

Both auxiliary objectives are synthetic-task label supervision. This is not a
label-free protocol.

## Frozen curriculum

| Phase | Pairs | Queries | Length | Updates |
|---|---:|---:|---:|---:|
| 1 | 2 | 2 | 16 | 300 |
| 2 | 4 | 4 | 24 | 300 |
| 3 | 8 | 8 | 48 | 400 |
| 4 | 16 | 16 | 96 | 600 |

Each seed receives 467,200 scored query labels on fresh deterministic episodes.

## Frozen seeds and evaluation

- fresh model seeds: 1451, 1453, 1459;
- these seeds were not used by G4a, G4b, G4c, or v1.4.2 development;
- disjoint evaluation namespaces derived from model seed plus 800,000 at L96
  and plus 900,000 at L512;
- 16 batches of 32 sequences per seed;
- L96: 16 pairs and 16 queries, 8,192 decisions per seed;
- L512: 16 pairs and 4 queries, 2,048 decisions per seed;
- exact query accuracy, exact-sequence accuracy, bits/query, source commit,
  start status, checkpoint SHA-256, and preregistration SHA-256 retained.

## Primary gate

Pass only if every fresh seed reaches at least 90% exact query accuracy at both
L96 and L512. An average cannot hide a weak seed. Any non-finite result,
missing checkpoint, or protocol deviation is a failure.

## Claim boundary

A pass validates robust label-supervised synthetic associative learning for
this configuration and schedule. It does not establish label-free learning,
natural-language quality, upstream-model superiority, or fused-kernel speed.
