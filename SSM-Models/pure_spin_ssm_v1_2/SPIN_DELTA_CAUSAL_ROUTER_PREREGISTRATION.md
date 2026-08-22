# Spin-Delta causal low-entropy router gate

**Protocol status:** specified before frozen outcome training.

## Question

The oracle-address intervention established that the two-slot Spin-Delta
recurrence can solve and length-extrapolate on the two-key overwrite task. It
intervened jointly on write timing, erase/write slot, erase strength, and query
slot. The failed autonomous controller, however, predicts each quantity from
the current block control alone even though a write is defined by the causal
three-token pattern `[WRITE,key,value]`.

This gate asks whether an explicit causal, low-entropy event/address router can
close that inference gap without changing the Spin recurrence or its CUDA
compiler.

## Candidate

The candidate prepends a small token router to the unchanged Spin-Delta core:

1. a learned token embedding;
2. one causal width-three convolution;
3. a pointwise nonlinear projection to write-event, write-slot, query-event,
   and query-slot logits;
4. binary/categorical straight-through decisions whose forward values are
   exactly zero/one and one-hot.

The router supplies differentiable controls to every Spin-Delta layer. A write
event gates both the affine drive and unit-strength erase. The predicted write
slot supplies the common erase/write key. The predicted query slot supplies
the read probe. The two-slot state, independent Spin transports, value drive,
raw-CUDA recurrence, readout, and language head are unchanged.

During training only, event and slot labels are derived from the synthetic
sequence grammar and contribute a balanced auxiliary classification loss.
The retrieval target remains the primary loss. Evaluation supplies token IDs
only: neither oracle controls nor router labels enter the model. This is
autonomous inference on the named grammar, but it is not self-supervised
language memory and not evidence that natural text provides such labels.

## Frozen task and budget

- sequence: `[WRITE,key,value] * writes + [QUERY,key]`;
- two keys and 32 possible values;
- train at 8 writes; evaluate at 8, 16, and 32 writes;
- `d_model=64`, two Spin(8) layers, two heads, two slots;
- router width 32, causal kernel width 3, temperature 1;
- retrieval loss plus auxiliary loss with coefficient 1;
- 800 AdamW steps, batch 128, 32 evaluation batches per length;
- learning rate `3e-3`, weight decay `0.01`, clip norm `1.0`;
- frozen seeds 449, 457, and 461;
- fixed order: existing learned continuous addresses, then causal discrete
  auxiliary router;
- identical initial Spin-Delta core tensors and identical batches within each
  seed.

Development may use seed 443 only. Frozen seeds must not be inspected until
the implementation, tests, summarizer, and this protocol are committed.

## Frozen decisions

**Autonomous retrieval capacity passes** only if every candidate seed reaches
at least 95% accuracy at both 8 and 16 writes and at least 93% at 32 writes.

**Router identification passes** only if every seed reaches all four of:

- write-event F1 at least 0.99;
- query-event F1 at least 0.99;
- write-slot accuracy at true write events at least 0.99;
- query-slot accuracy at true query events at least 0.99.

**Robust rescue passes** only if the candidate improves 16-write retrieval by
at least five percentage points in at least two seeds and on the three-seed
mean relative to the paired existing controller.

All three decisions are reported independently. Retrieval can pass while
router identification fails, because unused or compensating decisions may
exist. Router identification can pass while retrieval fails, which would
falsify the claim that the localized inference fix is sufficient.

## Non-claims

- The candidate has extra router parameters and train-time grammar labels; it
  is not a parameter-matched architectural superiority result.
- Straight-through gradients are an estimator, not exact differentiation of
  argmax.
- Success does not establish Shakespeare improvement, generic language-event
  discovery, or superiority to Mamba-2, Gated DeltaNet, or attention.
- Timing and memory are diagnostics only and cannot authorize a speed claim.
