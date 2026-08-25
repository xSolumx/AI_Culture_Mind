# G13 exact-target long-context curriculum preregistration

**Frozen:** 2026-08-25, before any G13 training or validation result  
**Parent evidence:** G12A--G12E  
**Status at freeze:** protocol and thresholds fixed; no G13 outcome inspected

> **Post-run protocol deviation (2026-08-25):** this frozen document specifies
> a 2,048-token attention window, but the frozen builder inherited
> `attention_window_size=1024` from `successor_screen.py`; the completed artifact
> also records 1,024. Both paired arms used the same 1,024-token window, so the
> paired curriculum comparison remains internally controlled. The run did not
> execute the written 2,048-window architecture, and later reports must use the
> actual 1,024 value. This notice preserves rather than rewrites the frozen spec.

## Question

Does ordinary next-token pretraining through
`256 -> 512 -> 1,024 -> 2,048 -> 4,096` tokens teach retention-safe Hybrid
Memory v1.4.5 to use longer natural-text histories, rather than merely execute
at a longer tensor shape?

G12 established robust ordinary-text learning with a train-only lossless BPE
tokenizer, but its factual counterfactual recall signal was tiny,
seed-inconsistent, and non-monotone after 256-token training. G13 therefore
separates ordinary held-out loss from a stricter recall-and-ablation decision.

## Frozen model and data

- Dataset: the exact G11/G12 pinned TinyStories snapshot, with unchanged train
  and validation byte hashes.
- Tokenizer: G12A's train-only lossless ByteLevel BPE vocabulary 512, verified
  by its serialized SHA-256 and exact round trip.
- Architecture: retention-safe Hybrid Memory v1.4.5, layer order
  `gated_delta -> attention`, width 48, FFN expansion 5, 124,534 trainable
  parameters, and a 2,048-token local-attention window. No parameter or
  architecture search occurs in G13.
- Optimizer: the G12B-selected `HarmonicMuonAdamW`, learning rate `1e-3`,
  decoupled weight decay `0.01`, global gradient clip `1.0`, no scheduler, no
  optimizer reset between phases.
- Fresh model seeds: 2011, 2017, and 2027. The paired arms start from identical
  parameters for a seed. The deterministic data-order seed remains 1817.

## Exact-target paired training

Each arm receives 1,000 updates and exactly 4,096 scored BPE targets per
update. For each seed and global update, one deterministic contiguous
4,097-token macro-window is selected. Both arms score the same target token IDs
and therefore the same represented raw bytes. Only the partition into
independent sequences differs.

| Phase | Updates | Curriculum length | Curriculum batch | Fixed control length | Fixed control batch |
|---:|---:|---:|---:|---:|---:|
| 1 | 200 | 256 | 16 | 256 | 16 |
| 2 | 200 | 512 | 8 | 256 | 16 |
| 3 | 200 | 1,024 | 4 | 256 | 16 |
| 4 | 200 | 2,048 | 2 | 256 | 16 |
| 5 | 200 | 4,096 | 1 | 256 | 16 |

Rows are overlapping only at the causal boundary: a row beginning at token
`k` predicts targets `k+1 ... k+L`. Consequently, every partition covers the
same 4,096 targets exactly once. This is a target-token and raw-byte exposure
match, not a FLOP or wall-time match.

Lengths above 1,024 run as live-state 1,024-token chunks. Recurrent states are
not detached, so gradients cross chunk boundaries. Chunked/full forward and
gradient parity must pass a unit test before CUDA training. If 4,096 still
cannot run on the available GPU, G13 stops and records the failure; it does not
silently shorten the phase or change precision.

## Frozen evaluation

- At updates 0, 200, 400, 600, 800, and 1,000, both arms receive ordinary
  held-out evaluation at 256 and 4,096 tokens.
- Final checkpoints also receive ordinary evaluation at 512, 1,024, and 2,048
  tokens. Each context length scores the same 32,768 held-out target IDs,
  partitioned from eight deterministic macro-windows exactly as in training.
- The paired counterfactual factual-recall probe retains G12D's construction
  and adds raw-byte distances 2,048, 4,096, and 8,192 to the existing 128,
  256, 512, and 1,024 distances. It records actual BPE prompt lengths so the
  8,192-byte row can be checked against the 2,048-token attention window.
- Final recall is measured for the full model, with the gated-delta mixer
  residual suppressed, and with the attention mixer residual suppressed.
  Suppression sets the selected mixer's residual scale to -30 without changing
  embeddings, normalization, or FFN paths. These are causal ablations, not
  retrained models.

## Decision gates

### Integrity

All losses, gradients, and scores must be finite; each paired seed must present
identical target IDs and cumulative raw bytes across arms; the phase-1 curves
must agree within `1e-6` BPRB; and every checkpoint and input hash is recorded.

### Ordinary long-context improvement

The curriculum passes only if:

1. its mean final 4,096-token BPRB is at least 0.02 below the fixed control;
2. it wins at 4,096 tokens in at least two of three paired seeds;
3. its mean 256-token BPRB regresses by no more than 0.02; and
4. no seed regresses at 256 tokens by more than 0.05.

### Learned long-range recall

The stronger learning claim passes only if, at 8,192 raw bytes:

1. all three curriculum seed-mean matching-minus-mismatched gains are positive;
2. their mean gain is at least 0.02 nats;
3. the curriculum mean exceeds the fixed control mean by at least 0.01 nats;
   and
4. suppressing gated-delta reduces the curriculum mean gain by at least 0.01
   nats.

The full G13 promotion requires integrity plus both gates. Passing only the
ordinary gate means context-length training improved held-out compression but
did not establish factual long-range memory. Failure rejects this curriculum
at the tested scale; it does not prove the architecture, optimizer, or
tokenizer globally incapable.

## Nonclaims

G13 is not compute-matched, not a scaling-law estimate, not an instruction or
question-answering benchmark, and not evidence that 4,096 tokens is an optimal
context. The fixed control deliberately spends less attention compute. Any
quality gain must be reported beside wall time and peak CUDA allocation.
