# Hybrid Memory v1.4 G5b Causal Reverse-Binding Comparison

Status: frozen after G5 diagnosis and before model seed 1621 is run.

Date: 2026-08-24

## Trigger

G5's proposed `WRITE key -> value` reconstruction target was statistically
unlearnable on this synthetic generator: each value is sampled independently
and had not yet appeared. All three models' reconstruction losses remained at
approximately `ln(64)`. The term supplied noise, not an event/binding signal.

Under the remaining retrieval signal, local v1.4.4 reached 5.69% L96 and 4.44%
L512, actual Transformers Mamba-2 reached 17.97% and 14.16%, and actual OLMo
Hybrid reached 97.52% at the trained L96 but only 25.44% at L512. This is a
negative single-seed result, not a validation verdict.

## Corrected externally observable target

At each observed `WRITE key value` value position, the model reconstructs the
already seen key. The total loss is:

`query retrieval cross entropy + 0.25 * reverse-key cross entropy`.

This target is causal and learnable: both key and value are in the prefix when
the auxiliary logit is produced. It directly rewards retaining the local
binding at the event without reading any model's internal memory state,
address projection, or write gate.

## Frozen protocol

- the same three architectures and exact configurations as G5;
- fresh model seed 1621 for every architecture;
- shared data seed base 1681;
- evaluation namespaces 1,621,621 (L96) and 1,721,621 (L512);
- AdamW 0.003, weight decay 0.01, gradient clip 1.0, batch 32;
- phases `(2,2,16,300)`, `(4,4,24,300)`, `(8,8,48,400)`, and
  `(16,16,96,1200)` as `(pairs,queries,length,updates)`;
- 774,400 retrieval labels and 774,400 reverse-binding labels per model;
- retained checkpoints, hashes, source commit, and starting Git status.

## Interpretation

G5b is a one-seed paired successor screen, not a robustness or superiority
claim. It asks whether a causally valid external reconstruction signal repairs
rule learning and length transfer. It does not erase G5 and does not replace
the multi-seed G4f commissioned-learning result.

