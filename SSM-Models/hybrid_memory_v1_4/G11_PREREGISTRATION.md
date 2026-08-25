# Hybrid Memory v1.4.5 G11 Natural Next-Token Screen

Status: frozen after G10 passed and after the TinyStories row snapshot was
retained, before the evidentiary G11 cohort is trained. Discarded engineering
forward/backward smokes used a separate `bench` namespace and did not produce
or select learning results.

Date: 2026-08-25

## Question

Can retention-safe v1.4.5 learn an ordinary causal next-byte objective on real
text, without the synthetic reverse-binding or internal memory labels, and how
does that learning curve compare with actual small Transformers Mamba-2 and
OLMo Hybrid implementations under the same data and update budget?

## Frozen data

- dataset: `roneneldan/TinyStories` via the Hugging Face Dataset Viewer;
- Hub revision at snapshot:
  `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`;
- license reported by the Hub: CDLA-Sharing-1.0;
- train rows 0--1999: 1,730,239 UTF-8 bytes, SHA-256
  `e25563b202a9f669b5d479d0cf94bd9a2c48a0050fcb569d63f1f6fc8580b4c1`;
- validation rows 0--255: 230,722 UTF-8 bytes, SHA-256
  `4b7aadf02f91e1dce13b699bd428bda3984ac4ee9ec17167db1813e0a34b1b53`;
- retained snapshot artifact SHA-256
  `ebe7c7c948f3e59781097ffa64e214da15364d61b38622d33dd076d40471adc6`;
- zero exact-story overlap in the selected official split prefixes;
- fixed ASCII story separator `\n\n<|story|>\n\n`;
- vocabulary: raw UTF-8 bytes 0--255, with no fitted tokenizer.

## Frozen models

All are two-layer, hidden-width-64 causal language models with tied token
embeddings and no dropout:

1. retention-safe Hybrid Memory v1.4.5 (`gated_delta -> attention`), using the
   exact G10 memory configuration and a 256-byte vocabulary;
2. actual `transformers.Mamba2ForCausalLM`, using the same small configuration
   as the retained G5 comparison except for vocabulary 256;
3. actual `transformers.OlmoHybridForCausalLM`, one linear-attention layer and
   one full-attention layer, likewise with vocabulary 256.

These are actual library implementations initialized from scratch, not names
attached to repository substitutes. Parameter counts and exact runtime class
names must be recorded. The separately retained pretrained
`state-spaces/mamba2-130m` probe remains an availability result, not a matched
member of this small-model training cohort.

## Frozen training and evaluation

- one paired model seed: 1811;
- data seed: 1817, independent of model RNG;
- sequence length: 256 input bytes and 256 next-byte targets;
- training batch size: 16;
- 2,000 updates per model (8,192,000 scored training bytes);
- AdamW, learning rate 0.001, weight decay 0.01, gradient clip 1.0;
- evaluation at updates 0, 500, 1000, and 2000;
- validation batch size 16, 32 fixed batches (131,072 scored bytes per
  evaluation);
- identical deterministic train and validation offsets for every model;
- retain final model/optimizer states and all learning-curve points.

The objective is only ordinary causal cross entropy: input byte at position
`t` predicts byte `t+1`. There are no association, write, address, routing, or
intermediate losses.

## Gate

The v1.4.5 ordinary-learning screen passes only if:

1. every reported loss and bits-per-byte value is finite;
2. held-out validation bits/byte improves by at least 2.0 from update 0 to
   update 2000; and
3. final held-out validation bits/byte is at most 4.0.

The upstream models are reported comparisons, not part of the v1.4.5 pass
condition. A single seed cannot establish superiority or cross-seed
robustness.

## Claim boundary

A pass establishes bounded ordinary next-token learning on a real-text corpus
for one paired small-model screen. It does not establish general language-model
quality, scaling, pretrained-model superiority, long-context natural-text
recall, or fused-kernel speed.
