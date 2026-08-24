# Hybrid Memory v1.4.4 G4f Tied-Address Preregistration

Status: frozen after failed seed 1481 was repaired in development and before
fresh model seeds 1511, 1523, and 1531 are run.

Date: 2026-08-24

## Trigger

G4e failed because seed 1481 remained at 78.223% L512 despite identity value
and readout paths and 774,400 useful labels. Geometry diagnostics localized the
failure to address alignment: seed 1481 had only 97.7-99.8% per-head address
top-1, query-key margin 0.22, and much higher association loss, while strong
seeds had 100% address accuracy and margin about 0.42.

v1.4.4 ties query and key projection weights and initializes the shared
projection orthogonally. On the already exposed failed seed 1481, this raised
L512 accuracy to 97.998% and reduced bits/query from 0.976 to 0.078. This is a
development result only.

## Frozen model

- model version 1.4.4;
- model dimension 64;
- layer plan `gated_delta -> attention`;
- four Gated Delta heads, key dimension 32, value dimension 16;
- one shared query/key projection, orthogonally initialized;
- normalized value injection;
- identity-initialized value and output projections;
- identity-centered output gate `1 + tanh(g)`;
- initial memory residual scale sigmoid(0) = 0.5;
- four attention heads, window 1024;
- convolution width 4, expansion 2, dropout 0, tied embeddings.

## Frozen objective, optimizer, and budget

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

Useful query labels per seed: 774,400. This remains an explicitly label-
supervised commissioning protocol.

## Frozen validation

- new model seeds: 1511, 1523, 1531;
- L96 namespaces: model seed plus 1,200,000;
- L512 namespaces: model seed plus 1,300,000;
- 16 batches of 32 sequences per length and seed;
- L96: 8,192 decisions; L512: 2,048 decisions;
- exact query accuracy, exact sequence accuracy, bits/query, source commit,
  empty/nonempty start status, preregistration hash, and checkpoint hashes.

## Gate

Pass only if every fresh seed reaches at least 90% exact query accuracy at both
L96 and L512. Any weak seed, non-finite result, missing artifact, or protocol
deviation fails the gate.

## Claim boundary

A pass validates robust label-supervised synthetic associative learning for
v1.4.4 at the declared budget. It does not validate label-free learning,
natural-language quality, upstream superiority, or fused speed.
