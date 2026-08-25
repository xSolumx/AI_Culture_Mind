# G16 SM75 trained frontier shootout protocol

**Frozen:** 2026-08-25, after runtime qualification and before inspecting any
G16 training metric

**Hardware:** NVIDIA GeForce RTX 2070 SUPER, compute capability 7.5
**Purpose:** train stronger executable small models, rather than infer model
quality from layer probes or unmatched pretrained checkpoints

## Question

At about 125 thousand trainable parameters and the same lossless text stream,
which maintained architecture makes the best use of a fixed 4.096-million-token
training budget, and does any of them learn one-shot binding after ordinary
next-token training?

This is a development shootout. One seed can reject a recipe or select models
for confirmation, but cannot establish architecture superiority.

## Frozen development arms

| Arm | Implementation | Shape | Optimizer | Parameters |
|---|---|---|---|---:|
| `hybrid_v1_4_5` | exact retained G12C repository Gated Delta + window attention | width 48, key/value 24/12, expansion 5, two layers | `HarmonicMuonAdamW` | 124,534 |
| `hybrid_gdn2` | repository independent GDN2 edit law + window attention | width 48, key/value 28/12, expansion 4, two layers | `HarmonicMuonAdamW` | 124,414 |
| `mamba2` | repository wrapper around official fused `mamba_ssm.Mamba2` | width 56, four layers, state 32, head 16, expansion 2 | AdamW | 124,172 |
| `olmo_hybrid` | actual `transformers.OlmoHybridForCausalLM` Gated DeltaNet + full attention | width 64, intermediate 138, two layers | AdamW | 124,376 |

All counts must be within 1% of the 124,534 target. The Mamba-2 arm must bind
its imported distribution to the source checkout and revision already qualified
on SM75. The OLMo arm must record its exact Transformers runtime class and
installed Transformers/FLA versions. Random initialization is explicit; no
downloaded checkpoint is shrunk, distilled, or presented as one of these small
arms.

The local `hybrid_gdn2` arm implements the full decoupled retention/erase/write
law in this repository. It is not renamed as upstream FLA GDN2 or as KDA. Its
width, attention block, value dimension, retention settings, and identity paths
remain paired to v1.4.5; key width and MLP expansion are adjusted only to close
the parameter residual to 0.096%.

Upstream FLA GDN2 is ineligible because its actual SM75 chunk backward exceeded
the 1,200-second qualification limit twice. FlashRT is ineligible because its
published kernel requires SM80 and BF16. The initially proposed small Mamba-3
SISO shape is also ineligible: its head dimension 8 fails the actual backward
kernel's `K >= 16` contract. Mamba-3 MIMO selects an SM80-only TileLang MMA
path. No fallback or unqualified reshaping may replace any failed arm after the
freeze.

## Data, tokenizer, and target pairing

- frozen TinyStories snapshot and split hashes from G11--G13;
- frozen training-only Hugging Face ByteLevel BPE with 512 tokens, no unknown
  token, exact byte round trip, SHA-256
  `6b8031ebb2c899eaa780ee9216b2feddd00fac0a04265a6f3a5111a8dbda8ee1`;
- model seed 2333;
- identical macro-window starts and target token IDs across all arms;
- 4,096 target tokens per update;
- ordinary causal next-token cross entropy only;
- no retrieval, transport, routing, or memory commissioning loss.

The tokenizer increases raw-byte exposure relative to bytes. Every loss is
therefore reported in bits per raw byte (BPRB), with tokens and raw bytes also
recorded.

## Training curriculum

Train each model for exactly 1,000 updates:

| Updates | Context | Batch | Target tokens |
|---:|---:|---:|---:|
| 1--200 | 256 | 16 | 819,200 |
| 201--400 | 512 | 8 | 819,200 |
| 401--600 | 1,024 | 4 | 819,200 |
| 601--800 | 2,048 | 2 | 819,200 |
| 801--1,000 | 4,096 | 1 | 819,200 |

Learning rate is `1e-3`, weight decay `0.01`, and global gradient clipping is
1.0. There is no best-checkpoint selection or early stopping. Mamba-2 and OLMo
Hybrid use AdamW. The two repository hybrids use the previously selected
composite optimizer. This is a best-recipe quality table, not an optimizer-
isolated architecture theorem.

## Evaluation

At initialization and after every phase, record ordinary held-out BPRB. The
final checkpoint is evaluated at token contexts 256, 512, 1,024, 2,048, and
4,096 on identical validation macro-windows.

Each final checkpoint also receives the frozen paired counterfactual fact probe
at 512, 2,048, and 8,192 raw-byte distances. It compares the log probability of
the same continuation after a matching versus counterfactual earlier fact.
This is a one-shot binding diagnostic, not a natural-text perplexity measure.

Record trainable parameters, target-token digest, phase timing, synchronized
median update time, peak CUDA allocation, exact source/runtime provenance,
checkpoint hash, starting commit, and starting Git status.

## Frozen development decision

Integrity passes only if all metrics and gradients are finite, parameter-count
residuals are at most 1%, tokenizer/source identities match, and all four arms
have identical training target digests.

An arm is *development-qualified* only if:

1. final 256-token BPRB is at most 2.0;
2. final 4,096-token BPRB is at most 2.0;
3. final 4,096-token BPRB is no worse than 0.10 above `hybrid_v1_4_5`; and
4. its 8,192-byte mean counterfactual recall gain is finite.

These are elimination gates, not claims of superiority. A learned-recall pass
additionally requires mean gain of at least 0.02 nats at 8,192 bytes. Models
that development-qualify may enter a separately frozen three-fresh-seed
confirmation; no seed-specific tuning is allowed.

## Claim limits

A win here is bounded to one small TinyStories allocation on one SM75 GPU. The
arms are parameter-matched and token-target-matched, but not guaranteed
step-time matched. Model-specific optimizers confound a pure architecture
comparison by design. This experiment cannot establish a scaling law,
pretrained-model ranking, generic language superiority, or learned long-range
binding unless the corresponding frozen gate actually passes.
