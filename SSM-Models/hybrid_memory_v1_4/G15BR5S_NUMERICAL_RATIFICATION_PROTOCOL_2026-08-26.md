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

R5-S asks whether these failures reproduce as bounded FP32 reduction-order
effects on fresh batches, with exact discrete behavior and independently exact
FP64 algebra, or expose a real semantic divergence.

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

R5-S never re-scores R5 performance gates. It uses a new deterministic batch
namespace, `g15br5s-stability`, with no batch fingerprint equal to the
corresponding R5/R4 cell.

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

1. an absolute component-algebra ceiling of `5e-6`, reusing the already
   implemented R5 FP32 injection-integrity scale rather than choosing the
   observed `3.576279e-6` as the threshold;
2. a symmetric scaled-logit tolerance

   ```text
   abs(candidate - reference)
   <= 64 * eps * max(1, abs(candidate), abs(reference)).
   ```

The factor 64 is an engineering FP32 reduction allowance, not a theorem about
all scan lengths. Its scientific role is constrained by exact prediction
agreement, an independent FP64 contract at `1e-10`, and the separate `5e-6`
absolute ceiling. Report the maximum ratio to the scaled allowance.

For scalar BPQ, require both:

- absolute difference at most `1e-6`;
- relative difference at most `64 * eps` using a denominator floored at one.

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

### Fresh cohort audit

At every batch and for each source H, C, and B:

- local write, tail, and query masks match their audit labels;
- source plus exact signed residual reconstructs the full injection with
  assignment residual zero and sum residual at most `5e-6`;
- no-reset decomposed state reconstructs the direct monolithic state to
  absolute maximum residual `5e-6`;
- full read equals key read plus background read to absolute maximum residual
  `5e-6`;
- full transitions are the same shared tensor objects across sources;
- candidate logits meet the frozen 64-epsilon scaled tolerance;
- every query prediction, query accuracy, and exact-episode accuracy matches
  the direct monolithic path exactly;
- BPQ meets both frozen scalar tolerances;
- outputs are finite;
- the independent FP64 convolution/source/component/recurrent/parallel
  contract passes at `1e-10` for the first batch of every task/length cell;
- fresh fingerprints are deterministic, complete, and disjoint from the
  corresponding R5 fingerprints.

Quality fails closed on any dirty start, non-CUDA device, non-SM75 hardware,
missing hash, incomplete cell, threshold miss, prediction mismatch, or
nonfinite output.

## Frozen decision

R5-S passes only if every sealed-artifact and fresh-cohort gate passes for all
three checkpoints.

- **Pass:** record that R5's formal fail is numerically ratified but unchanged;
  support drafting, not executing, a separately frozen fresh-seed
  pending-write/commit training screen with learned addresses, explicit
  transaction occupancy, protected background-free read, and matched
  controls.
- **Fail:** stop retained-checkpoint tail repair and do not draft or execute
  that training screen until the failing numerical mechanism is understood.

No post-run threshold amendment is allowed. R5-S does not use performance
metrics to choose a tolerance and never promotes an R5 arm by itself.

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
