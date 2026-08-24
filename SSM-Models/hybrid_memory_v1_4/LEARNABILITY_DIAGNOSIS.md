# Hybrid Memory v1.4 Learnability Diagnosis and Pivot

Date: 2026-08-24

## Outcome first

G4a did not fail because the optimizer was broken or because 600 updates were
almost enough. Its primary selected memory did not represent the learning
problem it was being asked to solve.

The failed path stored values in statically selected slots. It did not store a
key signature beside each value, so it could not test content at read time or
resolve a collision. On the frozen 16-pair MQAR task, one head had only 16
addresses. Even a perfect deterministic key-mod-16 router therefore had a
roughly 68-74% last-write-wins ceiling on the actual cohorts. More training
could not move that mechanism through a greater-than-90% gate.

v1.4.1 pivots the primary memory to a content-addressed Gated DeltaNet matrix
state and moves selected blocks and Spin/F4/rotor structure to optional archive
or transport roles. A development curriculum with explicit association labels
then reached 94.34% at length 96 and 89.84% at length 512. On a larger unseen
2,048-query length-512 cohort the same checkpoint reached 92.68%; 300 fresh
length-512 continuation updates raised it to 94.19%.

This is a real synthetic rule-learning result under explicit task labels. It
is not yet a label-free or natural-language model promotion.

The strict three-seed G4b gate later failed because seed 1429 reached 84.814%
at L512, and uniform G4c consolidation raised it only to 87.598%. A post-G4c
diagnostic showed 100% address top-1 accuracy but seed-dependent key/value
interference. The v1.4.2 successor therefore doubled key dimension, normalized
value injection, and directly supervised the post-memory readout. It repaired
the previously weak seed in development, but fresh seed 1459 failed G4d at
86.328%.

v1.4.3 initialized the value projection, output projection, output gate, and
memory residual as an identity-preserving path. G4e nevertheless failed on
seed 1481 at 78.223% L512 while the other two seeds exceeded 97%. The weak seed
had an independently misaligned query/key address frame. v1.4.4 ties those
projections and initializes their shared map orthogonally. It repaired exposed
seed 1481 to 97.998% L512 in development, then passed prospectively frozen G4f
on every fresh seed with a 98.096% minimum L512 accuracy.

## What the present learning problem actually is

The task is not “remember a value for a long time.” It is four coupled
problems:

1. **event detection**: identify when a value token completes a `WRITE key
   value` triple and when a key token completes a `QUERY key` pair;
2. **binding**: write a value under a content representation of its key;
3. **retention**: avoid corrupting or decaying the binding over hundreds of
   filler tokens;
4. **content read**: reconstruct the value by comparing a query key with stored
   key content.

The G4a selected tier attempted event detection and static address selection,
but its state contained values only. Its router could learn a bin; it could not
represent or verify the association.

## Direct falsifiers and results

### 1. Static-address capacity ceiling

For every row, store the 16 values at `key_offset mod slots`, let later writes
overwrite earlier writes, and answer from the final slot. This is an oracle for
the selected mechanism's address rule because its router is perfect.

The development screen measured these exact last-write-wins ceilings:

| Static slots | Exact query ceiling |
|---:|---:|
| 4 | 21.88% |
| 8 | 49.22% |
| 16 | 66.41% |
| 32 | 89.06% |
| 64 | 100.00% |

Larger cohorts in the diagnosis placed the 16-slot ceiling between 67.97% and
74.22%, depending on the deterministic cohort. The conclusion is invariant:
16 value-only slots cannot support the frozen greater-than-90% claim.

Multiple heads do not automatically repair this. Without stored key content,
the readout has no causal fact telling it which head's colliding value is
correct.

### 2. Sparse useful supervision

G4a reported 2.46 million presented tokens, but only four query positions per
episode were scored. Its actual useful retrieval budget was:

`600 updates * batch 8 * 4 queries = 19,200 query labels`.

The successful development curriculum used 467,200 query labels and made the
query density explicit. Token count alone was a misleading budget measure
because most tokens were random filler.

### 3. Fixed-batch overfit

Two-layer attention, DeltaProduct, and selected-to-attention controls each
memorized one fixed batch at 100%. v1.4.1 also memorized its fixed 16-pair batch
at 100% without association supervision. The loss, optimizer, output head, and
gradient shell are therefore capable of fitting data.

The failure was fresh-episode rule learning, not a disconnected scalar loss.

### 4. Fresh-episode controls

At the old 600-update, batch-8 budget, selected memory, DeltaProduct, and a
two-layer attention control all remained near chance on fresh episodes. A
short-to-long curriculum copied mechanically from v1.2 also remained around
5-9% because v1.2's successful overwrite task used two repeatedly overwritten
keys, while G4a used 16 fresh keys. The curriculum idea transfers; the old task
difficulty and budget do not.

## What transferred from the earlier models

### Pure Spin v1.2

- Oracle address and event controls reached about 99.5%, locating the major
  bottleneck in routing rather than raw state arithmetic.
- Perfect routers could still produce weak retrieval seeds, demonstrating
  joint optimization and core-basin sensitivity.
- A corrected 2/3/5/8 short-to-long router curriculum was much more robust than
  fixed-depth training.
- Identity-preserving residuals, normalized value injection, direct reads, and
  explicit reconstruction signals were more reliable than assuming a
  structured controller would self-organize.

v1.4.1 keeps the staged curriculum, direct read, normalized key/query vectors,
and small identity-preserving residual. It does not copy v1.2's easier task and
claim the same evidence.

### Pure F4 v1.3

The F4 audit separated group action, address memory, routing, and hardware. On
generic data, identity transport was the correct default and exceptional
symmetry was not a free quality prior. Independent erase/write controls, full
cache accounting, and preserved read scale remain useful design rules.

### Pure Rotor and Pure Spin(8)

Rotor transport can be active and mathematically valid while associative
recall is worse than identity transport. Spin(8) structure is a transport or
feature-frame hypothesis, not a substitute for key-value memory. v1.4.1
therefore keeps Spin/F4/rotor tiers optional until generic content addressing
passes without them.

### Older SSM experiments

The earlier runs repeatedly showed that a nonzero gradient, a scan parity
test, or a faster kernel does not establish learnability. Temporal
observability, task-grammar alignment, useful-label counts, state capacity,
and fresh-seed controls must be checked before a large cohort.

## The v1.4.1 pivot

The new semantic Gated Delta state is a per-head matrix `S`:

`S_t = a_t S_(t-1) + beta_t k_t (v_t - k_t^T a_t S_(t-1))`

and the read is `q_t^T S_t`.

This stores the binding itself: normalized key content indexes a value vector.
The delta residual removes the value currently predicted at the key before the
new value is added. Learned `beta` controls writes and learned `a` controls
retention. The recurrence has both exact recurrent execution and an associative
parallel affine scan; full-sequence, arbitrary-chunk, and token-step states use
the same cache contract.

The default layer plan is now:

`gated_delta -> attention`

Attention supplies a strong local/global mixing path while Gated Delta supplies
bounded content-addressed state. Selected blocks remain available as a sparse
archive experiment, not the generic primary memory.

## Why explicit association supervision was used

Synthetic MQAR scores only the query tokens. A random filler-heavy episode
provides almost no local language-model signal that a value token should update
memory under the preceding key. During commissioning, v1.4.1 therefore uses
task metadata to:

- align each read query vector with the key vector emitted at its matching
  value event;
- label write strength at value events and suppress it elsewhere; and
- penalize filler decay.

This answers the capacity and optimization question: the architecture can
learn and retain the rule. It does not answer the label-free question. A
natural next-token corpus supplies dense local prediction signals, but that
must be tested rather than assumed.

## Upstream implementation boundary

- FlashRT Gated Delta Attention was pinned at repository revision
  `892f725c92033f8daf3de1329e1bba05b2747a39`, kernel version 6. Its published
  artifact is BF16, x86_64 Linux, and SM80+. The local RTX 2070 SUPER is SM75,
  so this row fails closed.
- Actual FLA 0.5.2 `chunk_gated_delta_rule` ran successfully in the WSL FLA
  environment on SM75.
- Actual Hugging Face Transformers Mamba-2 and OLMo Hybrid Gated DeltaNet
  implementations ran through their explicit unfused PyTorch fallbacks on
  native Windows.
- The actual pretrained `state-spaces/mamba2-130m` checkpoint at revision
  `3a5aea0c25d0fb43cc360e2c2aac82c26e3eed49` was downloaded and produced finite
  CUDA logits through official `mamba_ssm` in WSL.

These are real upstream implementation probes. They are not yet matched MQAR
quality comparisons and their timings cross different runtimes, so no speed
ranking is claimed.

G5 then performed a paired single-seed MQAR learning comparison with the actual
small Transformers Mamba-2 and OLMo Hybrid implementations. With internal
memory labels removed, v1.4.4 stayed at 5.69%/4.44% L96/L512, Mamba-2 reached
17.97%/14.16%, and OLMo Hybrid reached 97.52%/25.44%. OLMo's split isolates
training-length learning from long-filler retention.

The comparison also exposed that predicting a fresh random value from its
preceding key is an irreducible target, not useful next-token supervision. G5b
replaces it with reverse-key reconstruction after the value has been observed.
That retains an external, causal training signal while teaching the event to
carry both sides of the binding.

G5b made all three actual architectures learn L96. v1.4.4 reached 97.38% L96
and 96.00% L512; Mamba-2 reached 98.66% and 77.98%; OLMo Hybrid reached 100%
and 58.11%. The present solution is therefore not more structure or more
updates. It is a learnable causal event target plus a tied content-addressed
state that preserves the commissioned binding. G6 then tested that result on
three fresh local-model seeds.

G6 then failed that three-seed gate because seed 1643 reached only 89.21% L96
and 83.40% L512. The external association target was already learned; the weak
seed was still acquiring retrieval when each fixed phase ended. G7 therefore
replaces the fixed clock with capped competence pacing. This directly imports
the reliable lesson from Pure Spin v1.2: curriculum stages should represent
mastered capabilities, not merely equal update counts.

G7 disproved the stronger version of that hypothesis. Every seed learned L96,
but L512 still fell to an 85.89% minimum. Two seeds missed early competence
caps and later mastered harder phases, so phase accuracy was not monotonic.
The next learning problem was explicit distance coverage. G8 directly trained
all failed-G7 checkpoints at L512 with one uniform, frozen continuation. It
passed every seed: L512 mean/minimum rose to 94.61%/92.29% while L96 remained
95.63%/93.69%.

The resulting answer to the learning problem is layered. Content-addressed
state fixes representation capacity; tied address geometry fixes independent
frame alignment; causal reverse-binding reconstruction fixes the absence of a
learnable local association signal; and explicit target-distance training
fixes the mistaken assumption that short-task mastery implies retention.

## Current claim ledger

### Proved by code/tests

- the semantic recurrence is affine in state;
- recurrent and parallel scans agree numerically;
- arbitrary chunk replay agrees with full execution;
- invalid tokens freeze state and emit zero mixer output;
- gradients reach query, key, value, write, decay, gate, and output paths;
- cache bytes include both fast-weight and convolution state.

### Empirical development result

- 100% fixed-batch memorization;
- 94.34% fresh length-96 accuracy;
- 92.68% fresh length-512 accuracy on a larger pre-continuation cohort;
- 94.19% after length-512 continuation;
- finite real upstream FLA, Transformers, and pretrained Mamba-2 probes.

### Prospectively validated

- G4f fresh seeds 1511/1523/1531 all exceeded 98% exact query accuracy at L96
  and L512 under the frozen 774,400-label commissioning budget;
- the G4f artifact started from clean commit `ffc6efd`, retained all
  checkpoints, and records preregistration and checkpoint hashes.
- G8 uniformly continued all three G7 checkpoints and cleared 90% at both L96
  and L512 for every seed, with a 92.29% minimum L512 accuracy.

### Constrained

- the positive learning result uses synthetic task labels for association and
  write events;
- the local v1.4 recurrence is semantic PyTorch, not a fused kernel;
- extrapolation beyond the 1024 attention window has only small cohorts.

### Open

- robust tied-address commissioning passed G4f; learning the same rule without
  internal association/write labels remains open;
- fresh-from-scratch validation of the combined external reverse-binding plus
  target-distance schedule, now prospectively frozen as G9;
- label-free MQAR after a dense causal language-model pretraining phase;
- matched natural-text quality versus actual Mamba-2 and Gated DeltaNet;
- whether selected archive or Spin/F4/rotor transport adds value after the
  generic content-addressed core is stable;
- a fused SM75/next-hardware kernel path.
