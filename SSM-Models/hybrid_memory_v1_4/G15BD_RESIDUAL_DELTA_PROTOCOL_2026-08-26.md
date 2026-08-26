# G15B-D coupled residual-delta protocol

**Frozen:** 2026-08-26, after sealing the negative G15B-E quality artifact and
before any G15B-D training batch, checkpoint, or learning metric

**Predecessor:**
[`G15BE_PHASE1_RESULTS.md`](G15BE_PHASE1_RESULTS.md)

**Phase-0 result ledger:**
[`G15BD_PHASE0_QUALIFICATION_RESULTS.md`](G15BD_PHASE0_QUALIFICATION_RESULTS.md)

**Status:** implementation and exact-SM75 Phase-0 qualification are
prospective. No G15B-D learning result exists yet. Geometry and natural-text
promotion remain blocked.

## Diagnosed learning problem

G15B-E establishes that address learning is no longer the primary failure:
query-address top-1 is `1.0` throughout, MQAR/selective recall is high, the
memory-zero and binding-permutation interventions cause large losses, and all
numerical/provenance checks pass. The remaining failure is the learned edit
law. Two additive seeds require distributed non-event edits, additive state
norms grow strongly with context, and only seed `2483` improves materially over
the product arm. Independent erase and write amplitudes let the model fit the
commissioned objective with cancellation-dependent recurrent dynamics instead
of a stable local replacement algorithm.

G15B-D tests one change only: couple removal of the retrieved old content to
insertion of the new target with a single channelwise delta strength.

## Frozen arms

Both arms use the G15B-E full causal controller, model shell, optimizer
partition, dimensions, parameter tensors, initialization, state bytes, data,
and curriculum.

### P: historical product edit

\[
\beta_t=\sigma(\gamma_t)\sigma(\eta_t),\qquad
\alpha_{t,j}=\sigma(\gamma_t)\sigma(\omega_{t,j}),
\]

\[
S_t=(I-\beta_t k_tk_t^\top)\operatorname{diag}(r_t)S_{t-1}
    +k_t(\alpha_t\odot v_t)^\top.
\]

### D: coupled residual delta

The existing shared, erase, and channelwise write logits form one effective
edit strength

\[
\delta_{t,j}=\sigma(
  \gamma_t+\eta_t+\omega_{t,j}+b_\delta
),
\]

where fixed `b_delta` makes every initial `delta` equal to P's initial
effective erase and write strengths. For every value channel `j`,

\[
S_t^{(:,j)}=
(I-\delta_{t,j}k_tk_t^\top)
\operatorname{diag}(r_t)S_{t-1}^{(:,j)}
+\delta_{t,j}k_tv_{t,j}.
\]

Equivalently, after retention, D writes the addressed prediction residual:

\[
S_t^{(:,j)}=\widetilde S_{t-1}^{(:,j)}
+\delta_{t,j}k_t
\left(v_{t,j}-k_t^\top\widetilde S_{t-1}^{(:,j)}\right).
\]

Thus each channel's erase and write are one decision. The transition is
nonexpansive for unit `k`, `0 <= delta <= 1`, and bounded retention. Each value
channel remains an affine scan over `key_dim`; the recurrent state and cache do
not grow.

The write projection is not removed or replaced with a dummy parameter. Its
channelwise logits participate directly in `delta`, so P and D have identical
parameter names, shapes, counts, active tensors, and optimizer assignment.
The D semantic parallel scan expands the transition over value channels and is
therefore not compute- or temporary-memory-matched to P. Phase 1 must report
measured synchronized step time, tokens/second, and peak CUDA memory. Accuracy
promotion cannot be relabelled as an efficiency result; any later scaling claim
requires a separately compiled and measured compute-matched implementation.

## Phase 0: implementation qualification

Phase 0 must run from a clean committed WSL2 checkout on the exact local NVIDIA
SM75 device. It must establish conjunctively:

1. identical P/D parameter names, tensors, total/active counts, initial hash,
   optimizer partition, and FP32 state bytes;
2. initial effective erase/write residual at most `2e-8`, initial state/output
   residual at most `2e-6`, and exact initial predictions;
3. D effective erase and write strengths are exactly the same tensor;
4. finite strengths strictly inside `(0,1)` and maximum transition spectral
   norm at most `1 + 1e-6`;
5. an independent direct residual-delta formula matches the compiled affine
   transition and injection within `1e-10` in FP64;
6. FP64 recurrent/parallel, arbitrary-chunk, token-step, mask, and compact-valid
   parity within `1e-10`;
7. FP32 parity under the already frozen scale-aware bound, with every compared
   prediction exact;
8. finite nonzero gradients to query, key, value, shared edit, erase, write,
   retention, readout, convolution, embedding, and LM paths;
9. deterministic source hashes, Git status/commit, CUDA/runtime/device, and
   exact compute capability.

Failure stops G15B-D. Phase 0 is implementation evidence, not learning
evidence.

## Frozen Phase-1 cohort

Conditional on Phase 0, Phase 1 reuses the exact G15B-E quality budget:

- arms `P` and `D`;
- seeds `2481`, `2483`, `2489`;
- `128 -> 256 -> 512 -> 1024` training curriculum;
- `3,400` updates and `13,926,400` tokens per arm/seed;
- MQAR, overwrite, overwrite guard, selective copy, and needle evaluation at
  `128`, `512`, `1024`, and held-out `2048`;
- identical batches, optimizer groups, initialization tensors, parameters,
  state bytes, decision counts, checkpoint/source hashes, and numerical
  audits.

## Frozen Phase-1 gates

All gates are conjunctive.

- D overwrite at least `0.93` and post-same-key overwrite at least `0.92` for
  every seed through L1024;
- D MQAR/selective at least `0.98`, guard at least `0.99`, and needle exactly
  `1.0` under the same lengths as G15B-E;
- D mean overwrite minus P at least `0.02` at L128/L512/L1024;
- no D seed more than `0.03` below the D mean overwrite;
- memory-zero, valid-event-edit-zero, and coherent binding permutation each
  reduce the targeted D accuracy by at least `0.50` at L512/L1024;
- valid-event-only D execution stays within `0.02` of learned execution, while
  non-event-only accuracy is at most `0.50`;
- mean non-event `delta` is at most `0.05` and mean valid-event `delta` is at
  least `0.25` on overwrite at L512/L1024;
- D maximum state norm is at most `1.25` times matched P for every promoted
  seed/length;
- learned reconstruction, completeness, disjoint fingerprints, finite values,
  exact predictions, matched budgets, and provenance all pass.

If D passes constructed learning, the next experiment is a separately frozen
natural-text identity-memory screen. Spin/torus transport does not enter until
the identity edit law passes both constructed and natural-text gates.

## Explicit nonclaims

This protocol does not claim that residual delta is novel, that G15B-E failed
because of its optimizer or tokenizer, that the commissioned grammar is generic
language, or that a successful Phase 0/1 authorizes geometry, long-context
scaling, or model-family promotion.
