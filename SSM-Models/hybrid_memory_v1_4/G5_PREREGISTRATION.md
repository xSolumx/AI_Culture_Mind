# Hybrid Memory v1.4 G5 Actual-Upstream Learning Comparison

Status: frozen after the G4f pass and before model seed 1601 is run.

Date: 2026-08-24

## Question

Can the validated v1.4.4 architecture and two actual upstream architectures
learn the same MQAR task when every model receives only externally observable
causal labels?

This is a paired single-seed comparison, not a multi-seed validation gate.

## Frozen models

1. local Hybrid Memory v1.4.4 tied-address configuration from G4f;
2. actual Hugging Face Transformers 5.9 `Mamba2ForCausalLM`;
3. actual Hugging Face Transformers 5.9 `OlmoHybridForCausalLM` with one
   Gated DeltaNet layer and one full-attention layer.

The upstream configurations are the exact small models already exercised by
`upstream_probe.py`. All use vocabulary size 197, hidden size 64, two layers,
tied token embeddings, float32, and native PyTorch unfused fallbacks. Parameter
counts are recorded rather than assumed to be exactly matched.

## Frozen objective

Every architecture receives the same two causal cross-entropy terms:

- retrieval: logits at each `QUERY key` key position predict its associated
  value;
- write reconstruction: logits at each `WRITE key` key position predict the
  following value token.

The total loss is `retrieval + 0.25 * write_reconstruction`. There is no access
to v1.4.4 query/key projections, write gates, memory state, or intermediate
block logits. This is an externally supervised next-token commissioning
objective available to every causal language model.

## Frozen optimizer and data

- model seed 1601 for each architecture;
- shared data seed base 1661;
- AdamW, learning rate 0.003, weight decay 0.01;
- gradient clip 1.0;
- batch size 32;
- a fresh deterministic episode every update;
- phases: `(2,2,16,300)`, `(4,4,24,300)`, `(8,8,48,400)`, and
  `(16,16,96,1200)` as `(pairs,queries,length,updates)`.

Each model sees 774,400 retrieval labels and 774,400 write-reconstruction
labels. Batches are paired exactly across architectures.

## Frozen evaluation

- L96: 16 pairs, 16 queries, 16 batches of 32, seed namespace 1,601,601;
- L512: 16 pairs, 4 queries, 16 batches of 32, seed namespace 1,701,601;
- exact query accuracy, exact sequence accuracy, and bits/query;
- retained checkpoint and hash for every model;
- source commit and starting Git status.

## Interpretation

Results are descriptive. One seed cannot establish robustness or superiority.
The small upstream instances use the real library implementations but are
randomly initialized task models, not pretrained checkpoints. The official
pretrained `state-spaces/mamba2-130m` remains a separately pinned inference
probe because its vocabulary and scale make it a different adaptation
experiment.

