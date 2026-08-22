# Spin-Delta query-event continuation gate

**Protocol status:** frozen before every outcome run.

## Question

The one-step topology audit proved that the initialized hard query event gives
the event head nonzero credit while multiplying query-slot credit by exactly
zero. Does a soft-to-hard query-event homotopy repair label-free joint learning
without changing the write path, slot estimator, core, objective, or exposure?

## Frozen arms

Both arms use the established 2/3/5/8-write curriculum: 100, 100, 200, and 400
updates respectively.

1. **Hard event:** the current straight-through hard query event at every
   update.
2. **Linear continuation:** at global update \(t\in\{1,\ldots,800\}\), use

   \[
   e_t=(1-\alpha_t)\sigma(z)+\alpha_t e_{\mathrm{hard}},\qquad
   \alpha_t=\frac{t-1}{799}.
   \]

The continuation therefore begins with the sigmoid probability and reaches
the exact hard forward event on the final update. The query-event temperature
remains 1. The hard straight-through categorical query slot is unchanged.
Evaluation always uses the ordinary hard router.

## Frozen cohort

- fresh initialization seeds: `719`, `727`, `733`;
- fresh minibatch-order seeds: `739`, `743`, `751`;
- one source model per initialization is constructed once and cloned into all
  six `(data order, arm)` descendants;
- identical batches are replayed between paired arms by resetting the data
  generator to the same seed;
- batch 128, 800 updates, AdamW `3e-3`, weight decay `0.01`, clip norm 1;
- `d_model=64`, two Spin(8) layers, width-three router, raw-CUDA recurrence;
- final retrieval cross-entropy is the only differentiated objective;
- no router labels, auxiliary router loss, oracle controls, or teacher
  controls;
- detached evaluation at 2, 3, 5, 8, 16, and 32 writes.

Both arms receive 102,400 retrieval labels and 2,009,600 training tokens.

## Gauge and measurements

One global binary slot permutation is selected per trained model over all
readiness lengths and applied jointly to write and query slots. Event metrics
are not gauge-adjusted. The report retains every paired cell, router metric,
factorial range, exposure count, state digest, implementation hash, and
environment record.

## Frozen decisions

**Mechanism repair passes** only if all of the following hold:

1. mean paired 16-write retrieval improves by at least 10 points;
2. the worst paired 16-write regression is no worse than 10 points;
3. the worst continuation 16-write cell exceeds the worst hard cell by at
   least five points;
4. the continuation's minimum query-event F1 over every cell and readiness
   length is at least 0.50 and exceeds the hard minimum by at least 0.25;
5. its gauge-correct query-slot accuracy minimum is at least 0.60 and exceeds
   the hard minimum by at least 0.10.

**Learning autonomy promotion** retains the stronger established bar. Every
continuation cell must reach 95% at 8 and 16 writes and 93% at 32 writes; both
16-write factorial-range maxima must be below five points; and all four
gauge-correct router metrics must be at least 0.99 at every readiness length.
The no-label/no-oracle contract must also pass.

The mechanism decision may pass while full promotion fails. All component
decisions are reported separately.

## Boundaries

This is a finite synthetic causal intervention. It is not a natural-language,
throughput, or general soft-routing result. An observed improvement would
support the diagnosed credit path; it would not prove that the Spin(8)
transport caused the improvement.
