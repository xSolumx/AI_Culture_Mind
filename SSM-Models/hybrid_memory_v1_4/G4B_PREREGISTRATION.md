# Hybrid Memory v1.4.1 G4b Preregistration

Status: frozen before fresh model seeds 1423, 1427, and 1429 are run.

Date: 2026-08-24

## Question

After replacing value-only selected slots with a content-addressed Gated
DeltaNet state, can the v1.4 hybrid learn the 16-pair MQAR rule on fresh
episodes and exceed 90% exact query accuracy at length 512 across fresh model
seeds?

This is a label-supervised commissioning test. It is not a label-free model
promotion and it does not establish natural-language quality.

## Frozen diagnosis being tested

The failed G4a model had two independent structural problems:

1. each selected head stored values in 16 static addresses but did not store
   key signatures, so collisions could not be resolved by content; and
2. 600 updates at batch 8 exposed only 19,200 scored query labels, despite a
   much larger reported token count dominated by random filler.

The successor stores key-to-value associations in a matrix state and uses
explicit query/key alignment and write-event supervision during commissioning.

## Frozen model and optimizer

- model dimension: 64;
- layer plan: `gated_delta -> attention`;
- four Gated Delta heads, key dimension 16, value dimension 16;
- four attention heads, window 1024;
- causal depthwise convolution width 4;
- expansion 2, dropout 0;
- AdamW, learning rate 0.003, weight decay 0.01;
- gradient norm clipped to 1.0;
- batch size 32;
- association auxiliary coefficient 0.25;
- training scan: semantic parallel affine Gated Delta scan.

## Frozen curriculum

| Phase | Pairs | Queries per episode | Length | Updates |
|---|---:|---:|---:|---:|
| 1 | 2 | 2 | 16 | 300 |
| 2 | 4 | 4 | 24 | 300 |
| 3 | 8 | 8 | 48 | 400 |
| 4 | 16 | 16 | 96 | 600 |

Each update uses a fresh deterministic episode seed. Every model seed sees
467,200 scored query labels. The auxiliary uses task metadata to align read
queries with the matching write key, label value-token write events, and
penalize filler decay. These labels must not be described as label-free.

## Frozen validation cohorts

- model seeds: 1423, 1427, 1429;
- evaluation seeds are disjoint namespaces derived from model seed plus
  100,000 at length 96 and plus 200,000 at length 512;
- 16 evaluation batches of 32 sequences per seed;
- length 96 uses 16 queries per sequence;
- length 512 uses 4 queries per sequence;
- exact query accuracy, exact sequence accuracy, and bits/query are recorded;
- every checkpoint and the preregistration file are SHA-256 hashed.

## Primary gate

Pass only if every one of the three fresh model seeds reaches at least 90%
exact query accuracy over its 2,048 unseen length-512 queries. A mean above 90%
with any seed below 90% is a failure.

## Interpretation

A pass establishes that the content-addressed hybrid can be commissioned to
learn this synthetic association rule robustly under explicit task labels. It
does not establish:

- label-free MQAR learning;
- superiority over Mamba-2, Gated DeltaNet, attention, or any pretrained model;
- natural-text perplexity or downstream-task quality;
- FlashRT kernel compatibility or speed on the local SM75 GPU;
- that Spin(8), F4, or rotor transport should be active in the generic memory
  path.

A failure leaves the development seed results as development results and
blocks model-quality promotion.
