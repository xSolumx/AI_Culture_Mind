# G15B-R5-S numerical ratification protocol

**Frozen:** 2026-08-26, after sealing the formally failed R5 quality artifact
and before generating any R5-S stability batch or metric.

**Bound evidence:**
[`G15BR5_CAUSAL_TAIL_SOURCE_RESULTS.md`](G15BR5_CAUSAL_TAIL_SOURCE_RESULTS.md)

**Status:** prospective zero-update numerical ratification; R5 is not reopened

## Question

R5's background-free strict-history arm passed all 132 frozen performance and
bias-separation checks, but the overall artifact failed two numerical gates.
Every discrete replay metric and learned logit was exact, while:

- decomposed no-reset BPQ differed from R4 by at most `1.397359e-7` against an
  absolute `1e-12` threshold;
- no-reset component-state reconstruction reached `2.384186e-6` against
  `2e-6`;
- the additive background-read relation reached `3.576279e-6` against `2e-6`;
- the independently recomputed FP64 algebra remained exact to
  `3.996803e-15`.

R5-S asks whether fresh batches show any semantic divergence above fixed FP32
engineering bounds, with exact discrete behavior and independently exact FP64
algebra. A pass is consistent with reduction-order effects; it cannot prove
that reduction order caused the original R5 maxima because the sealed artifact
does not retain their per-batch locations.

R5-S does not change R5's frozen artifact, decision, thresholds, or pass list.
It is a new prospective stability cohort. A pass can ratify the numerics needed
to justify writing a separate fresh-training protocol; it cannot convert R5
itself into a formal pass or authorize training directly.

## Bound artifact and runtime

Bind:

- R5 artifact:
  [`artifacts/g15br5_causal_tail_source_sm75_2026-08-26.json`](artifacts/g15br5_causal_tail_source_sm75_2026-08-26.json);
- SHA-256:
  `ba627fe34e8dd29458fc1321b52c98242838c3b56e2abdc7e44c749f50aaa313`;
- exact R5 commit:
  `e039e499b44b8e9bbb1108eb456c051a4702ba4e`;
- required R5 status: evidentiary exact-SM75 quality, clean start, frozen overall
  fail, `h_lww_bgminus` performance pass with 132/132 checks, empty final pass
  list, exact batch fingerprints and discrete replay, failed replay/runtime
  integrity, and zero updates.

Use the same retained identity checkpoints for seeds 2309, 2311, and 2333 on
CUDA compute capability `(7,5)`. Quality must start from a clean commit and
record all parent, checkpoint, protocol, and source hashes.

## Fresh stability cohort

R5-S never re-scores R5 performance gates. It reconstructs every original
`g15b-eval` batch, reproduces each sealed aggregate cell digest, and retains
the individual fingerprint set. It then generates the new deterministic
`g15br5s-stability` cohort twice, requires both copies to be identical, and
requires every fresh individual fingerprint to be absent from the original
set.

Evaluate:

- checkpoint seeds 2309, 2311, and 2333;
- tasks MQAR, ordinary overwrite, constructed overwrite guard, selective
  copy, and needle;
- lengths 128, 512, and 1,024;
- 512 query decisions per task/length/checkpoint cell;
- batch cap 16;
- zero optimizer updates.

For each batch compare the direct erase-free monolithic recurrence with the
no-reset H, C, and B source/residual decompositions. LWW performance is not
recomputed.

## Frozen numerical bounds

Let `eps = torch.finfo(torch.float32).eps`.

R5-S freezes two complementary tolerances before fresh execution:

1. an absolute component-algebra ceiling of `5e-6`, reusing an already
   implemented R5 FP32 injection-integrity threshold as a prospective
   engineering ceiling. This is not a derived state/read forward-error bound;
2. a symmetric scaled-logit tolerance

   ```text
   abs(candidate - reference)
   <= 64 * eps * max(1, abs(candidate), abs(reference)).
   ```

The factor 64 is an engineering FP32 reduction allowance, not a theorem about
all scan lengths. Its scientific role is constrained by exact prediction
agreement, an independent FP64 contract at `1e-10`, and the separate `5e-6`
absolute ceiling. Report the maximum ratio to the scaled allowance.

For scalar BPQ, require absolute difference at most `1e-6` and report relative
difference diagnostically. The relative measure is not a second gate because,
with a denominator floored at one, it would be redundant with the stricter
absolute rule.

No tolerance applies to categorical behavior: query predictions, query
accuracy, and exact-episode accuracy must match exactly.

## Integrity gates

### Sealed R5 audit

- artifact hash and exact clean-SM75 provenance pass;
- R5 performance pass is exactly `h_lww_bgminus: true` with 132/132 checks;
- final R5 pass list remains empty and the formal decision remains unchanged;
- learned replay is bit-identical;
- all R4 query-accuracy and exact-episode replay residuals are zero;
- the only nonzero R4 replay residuals are no-reset BPQ and are at most `1e-6`;
- all R5 causal, source-assignment, shared-transition, finite-output,
  no-overwrite, and prefix-prediction invariants pass;
- R5 FP32 source-assignment residual is zero and injection-sum residual is at
  most `5e-6`;
- every R5 FP64 algebraic contract is at most `1e-10`.
- every current R5 helper/model/task source hash and retained checkpoint hash
  matches the sealed R5 artifact exactly.

### Fresh cohort audit

At every batch and for each source H, C, and B:

- local write, tail, and query masks match their audit labels;
- source plus exact signed residual reconstructs the full injection with
  assignment residual zero and sum residual at most `5e-6`;
- no-reset decomposed state reconstructs the direct monolithic state to
  absolute maximum residual `5e-6`;
- full read equals key read plus background read to absolute maximum residual
  `5e-6`;
- full injection is bit-identical to, and left/right transitions are computed
  bit-identically by two independent erase-free full-token transition calls;
- candidate logits meet the frozen 64-epsilon scaled tolerance;
- every query prediction, query accuracy, and exact-episode accuracy matches
  the direct monolithic path exactly;
- BPQ meets the frozen absolute tolerance and reports relative error;
- outputs are finite;
- every batch records its normalized state/read/logit/BPQ severity; the single
  worst normalized batch in each task/length/checkpoint cell is regenerated
  exactly and checked against a common FP64 scan/read reference;
- on that retained worst batch, report FP32 monolithic and decomposed distances
  separately from the common FP64 reference, state/read magnitudes, normalized
  errors, and the independently recomputed FP64 convolution/source/component/
  recurrent/parallel contract; require FP64 algebra at `1e-10`;
- reconstructed original aggregate digests equal R5, fresh generation is
  deterministic on a second replay, and individual fresh/original fingerprint
  sets are disjoint.

The fresh cohort is one eighth the decision count of R5 and cannot localize or
replay the original artifact's worst state/read batch. This limitation is
mandatory in the result interpretation.

Quality fails closed on any dirty start, non-CUDA device, non-SM75 hardware,
missing hash, incomplete cell, threshold miss, prediction mismatch, or
nonfinite output.

## Frozen decision

R5-S passes only if every sealed-artifact and fresh-cohort gate passes for all
three checkpoints.

- **Pass:** record that R5's formal fail is unchanged and that no semantic
  divergence was detected above the prospective engineering bounds on the
  smaller fresh cohort. The result is consistent with bounded FP32
  reduction-order effects and supports drafting, not executing, a separately
  frozen fresh-seed pending-write/commit training screen with learned
  addresses, explicit transaction occupancy, protected background-free read,
  and matched controls.
- **Fail:** stop retained-checkpoint tail repair and do not draft or execute
  that training screen until the failing numerical mechanism is understood.

No post-run threshold amendment is allowed. R5-S does not use performance
metrics to choose a tolerance, cannot prove the cause of R5's original maxima,
and never promotes an R5 arm by itself.

## Exact reproduction target

```bash
PYTHONPATH=SSM-Models /home/local/.venvs/sm75-native-2026/bin/python \
  -m hybrid_memory_v1_4.g15br5s_numerical_ratification \
  --mode quality --device cuda \
  --checkpoint-directory /home/local/g15b_bd5045a_quality_attempt2_checkpoints \
  --output /home/local/g15br5s_<commit>_quality.json
```

## Nonclaims

R5-S is a numerical stability cohort on commissioned synthetic tasks and
retained checkpoints. It does not establish autonomous transaction learning,
generic association, ordinary-text recall, longer-context scaling, efficiency,
optimizer/tokenizer superiority, Spin benefit, parameter/state/compute
matching, or model promotion. A pass supports only the next protocol-writing
step; training remains a separate experiment.
