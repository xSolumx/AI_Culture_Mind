# G15B-E Effective-Edit Protocol

**Frozen:** 2026-08-26, after the sealed G15B-T Phase-1 result and before any
G15B-E training batch, metric, or checkpoint

**Predecessor result:**
[`G15BT_PHASE1_RESULTS.md`](G15BT_PHASE1_RESULTS.md)

**Status:** prospective identity-memory pivot; Phase 0 implementation
qualification pending. No G15B-E learning or model-quality result exists yet.

## Decision being tested

G15B-T learned useful addresses and used its recurrent memory, but its primary
strict-history arm failed the frozen overwrite, timing, robustness, and
numerical gates. Its recurrence exposed the loss only to the effective products

\[
\beta_t=c_te_t,
\qquad
\alpha_t=c_tw_t.
\]

Consequently, retrieval loss cannot identify `commit` \(c_t\) independently
from erase \(e_t\) and write \(w_t\). The observed per-head commit F1 is not a
well-posed target for the unsupervised language objective. The matched
full-view control also beat strict history on mean overwrite at every length,
so the next mainline test edits at the value position where the full causal
view already contains marker, key, and value.

G15B-E asks one narrow question:

> Does replacing multiplicatively factorized effective edit gates with a
> logit-additive parameterization materially improve fresh, autonomous learning
> when parameters, state, initialization, optimizer, tokens, and execution are
> held fixed?

This is an identity-memory experiment. It contains no torus, Spin, triality,
attention, timing BCE, grammar-labelled commit target, checkpoint reuse, or
optimizer change.

## Frozen arms

Both arms use one full-view transactional-delta block with the same query,
key, value, retention, shared event-intensity, erase, write, read, and output
projections. They have identical parameter shapes, parameter count, active
graph, FP32 state bytes, initialization seed, and training/evaluation data.

Let \(\gamma_t\) be a shared per-head continuous edit-intensity logit,
\(\eta_t\) a per-head erase logit, and \(\omega_t\) a channelwise write logit.
The shared logit is not assigned binary commit semantics and is never scored by
thresholded commit F1.

### P: multiplicative product control

\[
\beta_t=\sigma(\gamma_t)\sigma(\eta_t),
\qquad
\alpha_t=\sigma(\gamma_t)\sigma(\omega_t).
\]

### A: logit-additive effective edit

\[
\beta_t=\sigma(\gamma_t+\eta_t+\delta_\beta),
\qquad
\alpha_t=\sigma(\gamma_t+\omega_t+\delta_\alpha).
\]

The fixed offsets \(\delta_\beta,\delta_\alpha\) make the initial effective
erase and write strengths exactly equal to arm P. They are constants, not
learned parameters. Both arms use

\[
A_t=(I-\beta_t k_tk_t^\top)\operatorname{diag}(r_t),
\qquad
B_t=k_t(\alpha_t\odot v_t)^\top,
\]

\[
S_t=A_tS_{t-1}+B_t.
\]

Only the effective gates \(\beta_t\) and \(\alpha_t\) have edit semantics.

## Phase 0: implementation qualification

Phase 0 must run from a clean committed checkout on the exact local NVIDIA
SM75 device. It must establish:

1. exact P/A parameter-shape, active-parameter, initialization-hash, and state
   byte equality;
2. exact equality of initial effective erase/write gates;
3. finite effective gates strictly inside `[0,1]` and transition spectral norm
   at most `1 + 1e-6`;
4. FP64 recurrent/parallel, arbitrary-chunk, and token-step parity at absolute
   error at most `1e-10`;
5. FP32 chunk/step comparisons with exact predictions and prospectively scaled
   logit, state, read, and bits-per-query bounds recorded before Phase 1;
6. masked-token state preservation and compact-valid-token parity;
7. finite, nonzero gradients from the real LM loss to every declared path,
   including shared intensity, erase, and write projections;
8. deterministic artifact, source hashes, Git commit/status, WSL, CUDA, and
   exact compute-capability provenance.

Failure stops G15B-E. Phase-0 execution is not learning evidence.

## Phase 1: frozen constructed learning screen

Phase 1 uses fresh seeds `2481`, `2483`, and `2489`; a
`128 -> 256 -> 512 -> 1024` curriculum; the existing retrieval, reverse-binding,
and query/write-address objectives; and the same optimizer groups and schedule
for P and A. Each paired seed starts from the same parameter tensors and sees
hash-identical batches and evaluation schedules. No G15B-T checkpoint or
optimizer state may enter the cohort.

Evaluation uses MQAR, overwrite, selective copy, constructed guard, and needle
recall at lengths `128`, `512`, `1024`, and `2048`. Report per seed and length:

- overall and supported-stratum query accuracy;
- effective erase/write distributions at valid events and non-events;
- address top-1, while stating that address alignment is trained;
- state/read maxima, throughput, peak memory, and bits per evaluated query;
- complete data fingerprints and checkpoint/source hashes.

Ordinary-overwrite strata with zero support remain `null`; constructed guard
evidence must not be relabelled as ordinary-overwrite evidence.

## Frozen promotion gates

All gates are conjunctive.

### Quality and robustness

- overwrite accuracy at least `0.93` for every seed through L1024;
- post-same-key-overwrite accuracy at least `0.92` for every seed through
  L1024;
- MQAR and selective-copy accuracy at least `0.98` for every seed through
  L1024;
- needle accuracy exactly `1.0` for every seed through L2048;
- constructed guard accuracy at least `0.99` for every seed through L2048;
- A mean overwrite exceeds fresh P mean overwrite by at least `0.02` at each
  of L128, L512, and L1024;
- no seed is more than `0.03` below its arm's three-seed mean overwrite at any
  promoted length.

### Causal-use interventions

At L512 and L1024 for every A seed:

- memory-zero and valid-event-edit-zero each reduce the targeted accuracy by at
  least `0.50`;
- erase-zero reduces post-overwrite accuracy by at least `0.20` while reducing
  MQAR by at most `0.02`;
- permutation of valid-write key/value bindings reduces targeted accuracy by at
  least `0.50`;
- retaining edits only at valid WRITE/SELECT value positions stays within
  `0.02` of learned execution;
- retaining only ITEM/non-event edits cannot solve overwrite or selective copy.

Any failed provenance, schedule, fingerprint-disjointness, finite-output,
checkpoint, exact-prediction, or Phase-0 numerical bound stops promotion.

## Frozen decisions

- **A passes and P fails:** multiplicative latent-factor optimization was the
  tested learning problem; authorize a separate natural-text identity screen.
- **Both pass:** topology is sufficient; select only after a fresh measured
  efficiency comparison of the compiled effective gates.
- **P passes and A fails:** reject the additive parameterization.
- **Both fail while address top-1 remains 1.0:** the edit/value law remains the
  limiting mechanism; test a prospectively frozen residual-delta write law,
  not geometry.
- **Any causal-use intervention fails:** treat the apparent score as shortcut
  learning and stop.

No Phase-1 outcome directly authorizes Spin transport. Geometry may only enter
after an identity edit law passes its own constructed and natural-text gates.

## Explicit nonclaims

This protocol does not claim that G15B-T failed because of AdamW or Muon, that
logit addition is novel, that a supervised transaction parser would improve
ordinary language modelling, or that effective-edit gates solve long-context
natural text. It is a matched, falsifiable test of one diagnosed
parameterization problem.
