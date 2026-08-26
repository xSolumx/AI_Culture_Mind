# G15B-T strict-history transactional-delta protocol

**Frozen:** 2026-08-26, after the failed R5-S quality adjudication and before
any G15B-T training batch, metric, or checkpoint.

**Status:** prospective fresh-model mechanism screen; retained G15B
checkpoints are out of scope

## Why this is a pivot rather than R5-R6

R5 identified strict causal convolution history at the write tail as the only
source whose protected LWW read passed every frozen performance and
bias-separation check. R5-S then failed all 135 scaled-logit checks while every
categorical, BPQ, state/read, transition, fingerprint, provenance, and FP64
contract passed.

The retained-checkpoint repair series is closed. G15B-T does not change an R5
threshold, rescore an R5 arm, or continue component-sum surgery. It trains a
fresh, parameter-matched memory whose control topology exposes the causal
history source directly.

## Hypothesis

The present learning problem is temporal controller placement:

- a full/current-token controller must infer a completed three-token write
  transaction while the final value is still entering the local convolution;
- strict history one position later already contains the complete
  `[write/select marker, key, value]` transaction;
- a content-addressed delta edit can implement overwrite without predicting a
  nonlocal collision bit;
- the ordinary residual/read path should be computed once, not reconstructed
  by summing expanded diagnostic components.

The primary G15B-T arm therefore uses:

```text
query and output gate <- full causal convolution view at t
commit/key/value/erase/write/retention <- strict-history view at t
state <- one monolithic content-addressed fast-weight matrix
```

The strict-history view includes the learned convolution bias and the first
`kernel_size - 1` taps. It is computed structurally from the cache/past taps,
with the bias added exactly once; it is never computed by subtracting a
rounded current contribution from the full convolution. Under a valid mask,
"history" means previous valid tokens. It is causal and constant-state in
streaming execution. The convolution cache is the pending transaction
register; this protocol does not claim an additional recurrent pending tuple.
It is not an oracle component label or future-token view.

## Frozen recurrence

For each head, let `q_t` be projected from the full view and let
`c_t, k_t, v_t, e_t, w_t, r_t` be projected from the selected edit view. The
occupancy/commit gate and erase gate are scalars per head,
`c_t = sigmoid(P_c(view_t))` and `e_t = sigmoid(P_e(view_t))`; `w_t` remains
channelwise. The semantic edit is

```text
A_t = (I - c_t e_t k_t k_t^T) diag(r_t)
B_t = k_t (c_t * w_t * v_t)^T
S_t = A_t S_{t-1} + B_t
y_t = q_t^T S_t
```

Commit, scalar erase, and channelwise write controls remain independent. With
unit `k_t` and `c_t,e_t in [0,1]`, the symmetric erase operator is a positive
contraction; its product with channelwise retention is nonexpansive. The
present GDN2 channelwise nonsymmetric erase form is not used for this claim.
This first screen does not add Spin transport, extra slots, a router, an
attention layer, or a new optimizer. The recurrence is affine and uses the
existing exact prefix scan; recurrent, parallel, chunked, and token-step views
must agree within the frozen numerical contracts below. Affine pairs compose
chronologically as `(A2 A1, A2 B1 + B2)`.

## Exact matched arms

### F — full-view GDN2 control

The new transactional-delta layer in `full` controller mode. Query, commit,
edit, retention, and output controls all consume the ordinary full
causal-convolution view. It has the same scalar commit projection as T.

### T — strict-history transactional delta

The transactional-delta layer in `history` controller mode. Query/output
consume the full view; all state-transition and injection controls consume
strict history. F and T both compute both full and history convolution views,
so they have exactly the same graph shape, learned parameter count,
fast-weight/convolution state bytes, and nominal convolution work. Record
active trainable parameter counts, measured step time, and peak memory.

### T-AUX — locally supervised timing diagnostic

The identical T architecture and initialization family, trained with an
additional balanced loss on the scalar `c_t` against the locally decodable
completed post-value tail mask. No channelwise write or erase label is added.
T-AUX is an unequal-objective diagnostic. It cannot promote the architecture,
but can distinguish architectural failure from LM-only credit-assignment
failure.

There is no Spin arm in this protocol. Identity must learn the edit law first.
The existing `gated_delta_v2` result may be reported as a contextual reference,
but it is not the matched F arm and cannot enter the frozen F/T promotion
margin because it lacks the scalar commit module and symmetric scalar erase.

## Phase 0: implementation qualification

Before training, require:

1. F and T have exactly equal total/active parameter counts, state scalars,
   streaming cache bytes, and nominal full-plus-history convolution work for
   every tested configuration.
2. Changing the current token input at position `t` changes the full view but
   leaves the strict-history preactivation and every edit control at `t`
   bit-identical; therefore T's affine transition, injection, and post-update
   memory state at `t` are bit-identical. Query/output may change.
3. Changing an in-window prior token produces a nonzero strict-history/edit
   control effect.
4. Strict-history cache execution is exactly causal under full, arbitrary
   chunk, token-step, and arbitrary valid-mask-hole paths. A transaction ending
   at a chunk boundary remains in the next chunk's cache. Invalid positions use
   affine identity/zero, emit zero mixer output, and retain both memory and
   convolution history.
5. In FP64, recurrent/parallel memory-state and read residuals are at most
   `1e-10`.
6. In FP32, model-logit recurrent/parallel, full/chunked, and full/token-step
   residuals are at most `5e-4`, with exact predictions.
7. State-transition and injection tensors are finite; measured transition
   spectral norms are at most one up to `1e-6`, consistent with the frozen
   symmetric scalar-erase/retention contraction.
8. A real loss sends nonzero finite gradients to strict-history convolution,
   key, value, erase, write, retention, output, embedding, and LM-head
   parameters.
9. Windows and WSL test suites pass; an exact SM75 CUDA smoke executes without
   fallback or architecture substitution.

Any failed Phase-0 contract blocks training.

## Phase 1: constructed mechanism screen

Use fresh seeds `2381`, `2383`, and `2389`; no G15B/R checkpoints or optimizer
states are loaded. Pair F, T, and T-AUX on identical synthetic schedules.

Training curriculum:

```text
128 -> 256 -> 512 -> 1,024 tokens
```

Evaluate on fresh, fingerprint-disjoint cohorts at lengths 128, 512, 1,024,
and 2,048 for:

- MQAR;
- overwrite with before/unrelated/after-same-key strata;
- selective copy;
- needle recall;
- the constructed overwrite guard;
- adversarial filler and event-marker roles at the post-value position.

The frozen primary F/T objective contains retrieval, reverse-binding, and
query-to-commit-address losses only. Address prototypes are gathered at valid
post-value commit/tail positions, never at the old value positions. Neither F
nor T receives write, erase, collision, occupancy, or timing labels. T-AUX adds
only the declared scalar-commit loss against the post-value tail target. It
does not reuse G15B's value-position write/erase targets. If an erase target is
ever added in a later protocol, it must fire on every valid commit, not only
collisions.

Optimizer, learning-rate schedule, weight decay, gradient clipping, batches,
target tokens, seeds, width, heads, state bytes, and update count are paired.
Record measured update time and peak CUDA memory.

### T promotion gates

T passes Phase 1 only if all are true:

1. every seed reaches at least `0.90` ordinary overwrite query accuracy at
   lengths 128, 512, and 1,024;
2. every three-seed overwrite mean exceeds matched F by at least `0.05`;
3. post-same-key-overwrite accuracy is at least `0.90` through L1024;
4. MQAR and selective-copy means are at least `0.95` through L1024 and no more
   than `0.02` below F in any seed/length cell;
5. needle accuracy is exactly `1.0` through L2048;
6. the constructed guard is at least `0.99`, including post-same-key strata;
7. query-address top-1 and completed-tail commit F1 are each at least `0.95`
   per seed through L1024;
8. current-token mutation leaves the trained edit controls invariant at the
   same position, while relevant history mutation changes them;
9. setting commit to zero or memory output to zero sharply degrades L512/L1024
   retrieval; permuting strict histories across the batch or shifting commit
   timing by `-1`/`+1` sharply degrades overwrite;
10. disabling erase hurts post-overwrite recall while preserving unique-key
    MQAR; a bias-only history control fails; and zero-gap
    write-to-query/write-to-write boundary cases pass;
11. all numerical, fingerprint, finite-output, checkpoint, and schedule-parity
    contracts pass.

If T scores well but the causal-use interventions do not move it, treat the
score as a residual-stream shortcut and fail the mechanism gate.

If T fails but T-AUX passes, conclude that the topology is sufficient but the
LM-only objective does not identify commit timing. Do not call T learned and
do not proceed to geometry. If both fail, reject this strict-history GDN2 law.
If T passes, freeze a separate natural-text/scaling protocol before further
training.

## Phase 2 boundary

Phase 2 is not authorized by this document. A later prospective protocol must
cover the user's requested:

- multi-seed natural-text robustness;
- ordinary pretraining followed by longer-context recall;
- parameter-, state-, token-, and measured-compute-matched scaling;
- byte-level versus train-only BPE tokenizer controls;
- the selected optimizer versus AdamW and a declared custom-optimizer
  ablation;
- contexts `256 -> 512 -> 1024 -> 2048 -> 4096`.

Only after an identity T arm passes both the constructed and natural-text
gates may fixed `SO(2)^4`, constrained `SU(3)` torus, or sparse Spin-frame
transport enter a matched factorial.

## Stop rules

- Never loosen a gate after its first qualifying run.
- Dirty-tree, non-SM75, missing-hash, incomplete-cell, schedule-mismatch, or
  fallback execution fails closed.
- Do not use current event-marker identity as a hidden oracle commit label.
- Do not treat T-AUX as evidence of LM-only learnability.
- Do not reuse value-position write/erase supervision for a post-value commit.
- Do not substitute a larger state, extra attention, or different optimizer
  into T while calling it a matched recurrence comparison.
- Stop after Phase 1 failure and document the negative result.

## Nonclaims

This protocol does not establish that strict-history control is universally
optimal, that R5 passed, that R5-S should be waived, or that a local
convolution is a complete transaction parser. It does not establish generic
association, natural-text quality, long-context scaling, tokenizer/optimizer
superiority, efficiency, a Spin advantage, or model-family promotion.
