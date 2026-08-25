# G14 preregistration: decoupled erase/write gate-law screen

Date frozen: 2026-08-25

## Question

Does the G13 failure require a different state-update law before another
optimizer, tokenizer, or long-context curriculum is justified?

G13 measured useful recurrent-state contribution to ordinary text but found
continuous writes and severe decay in written directions. Gated DeltaNet v1
uses one scalar `beta` both to erase an addressed row and to inject a value.
Gated DeltaNet-2 (GDN2) separates channel-wise key erasure `b` from
channel-wise value writing `w`. G14 tests only that representational and
optimization distinction. It is not a language-model promotion test.

## Frozen task

- One fixed address and eight value channels.
- Each example presents a random permutation of distinct one-hot features.
- The target is the multi-hot union of all presented features.
- Training lengths are sampled uniformly from 2 through 7.
- Evaluation uses held-out permutations at length 8.
- The recurrent state itself is scored; there is no learned decoder that can
  rescale or hide a deficient state.

The task is intentionally adversarial to a tied erase/write scalar. A GDN-v1
state is a nonnegative weighted average whose coefficients sum to at most one,
while the length-eight target has eight unit entries. GDN2 can represent the
target by learning zero erasure and unit write on the presented channel.

## Frozen arms

1. `gated_delta_v1`: one learned scalar sigmoid gate per event, used for both
   erasure and writing.
2. `gated_delta_v2`: one learned scalar erase gate plus an independently
   learned eight-channel write gate per event.

Both controllers see the same one-hot feature and normalized position. Both
are trained from scratch with AdamW, the same batch, seed, update count, and
learning-rate schedule. Parameter counts are reported, not hidden. This is a
mechanism screen, so parameter matching is not claimed.

## Frozen optimization and seeds

- seeds: `14001, 14002, 14003`;
- 600 updates;
- batch size 128;
- AdamW, learning rate `3e-2`, weight decay `0`;
- float32 on the available CUDA device, otherwise CPU;
- binary cross-entropy on the raw state after clamping it to `(1e-6, 1-1e-6)`.

## Frozen gates

G14 supports the update-law pivot only if all are true:

1. all runs have finite losses and gradients;
2. GDN2 held-out length-eight mean squared state error is at most `0.01` for
   every seed;
3. GDN2 exact bit accuracy at threshold 0.5 is at least `0.99` for every seed;
4. GDN2 beats GDN-v1 held-out MSE for every paired seed;
5. the implementation tests verify exact GDN-v1 reduction when GDN2 gates are
   tied and exact recurrent/parallel/arbitrary-chunk replay.

Failure of a gate rejects only this candidate mechanism screen. Passing does
not show natural-text learning, long-context recall, compute efficiency, or
superiority to an attention/SSM baseline.

## Decision after the screen

If G14 passes, the next costly cohort is a matched natural-text and delayed
binding comparison among v1.4.5 GDN, KDA-style channel decay with tied
erase/write, and GDN2. Optimizer families must be tuned jointly with learning
rate in that cohort. If G14 fails, do not spend a long-context cohort on this
implementation until the semantic or optimization failure is explained.
