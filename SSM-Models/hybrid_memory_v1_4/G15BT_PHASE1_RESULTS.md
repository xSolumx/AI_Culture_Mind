# G15B-T Phase-1 transactional-controller results

**Protocol:**
[`G15BT_TRANSACTIONAL_DELTA_PROTOCOL_2026-08-26.md`](G15BT_TRANSACTIONAL_DELTA_PROTOCOL_2026-08-26.md)

**Phase-0 implementation result:**
[`G15BT_PHASE0_QUALIFICATION_RESULTS.md`](G15BT_PHASE0_QUALIFICATION_RESULTS.md)

**Exact quality artifact:**
[`artifacts/g15bt_phase1_quality_sm75_2026-08-26.json`](artifacts/g15bt_phase1_quality_sm75_2026-08-26.json)

**SHA-256:**
`6b6b991643ee6ddf50478f905dbaa53d9df9c8e52a10a8b52265dc8c12397fac`

**Execution commit:** `0c664f317d82789b5215da4658d2b59cbfe4c414`

**Status:** formal `T` fail; diagnostic-only `T-AUX` fail; stop G15B-T
before geometry

## Bottom line

The fresh strict-history transactional-delta architecture did not solve the
learning problem under its frozen Phase-1 screen. The primary `T` arm fails
the absolute overwrite, completed-tail commit, matched-margin, intervention,
and one numerical boundary gate. It is below the matched full-view `F` mean
on ordinary overwrite at every evaluated length.

The timing-supervised `T-AUX` diagnostic improves overwrite and commit timing,
but it also fails. It never reaches the required `+0.05` mean overwrite margin
over `F`, and seed 2381 remains below the `0.95` completed-tail commit-F1 gate.
The protocol's frozen decision is therefore:

> `stop G15B-T after the constructed Phase-1 failure`

This stops the G15B-T route before any torus or Spin transport comparison.
The passing Phase-0 implementation qualification remains valid, as do the
historical R5/R5-S results; none is retrospectively relabelled.

## Exact execution and integrity

The quality cohort ran from a clean tree on an NVIDIA GeForce RTX 2070 SUPER
with exact compute capability `(7,5)`, Python 3.11.16, PyTorch 2.9.0+cu128,
and CUDA 12.8 under WSL2. The artifact is evidentiary and contains all nine
seed/arm reports:

- seeds `2381`, `2383`, and `2389`;
- arms `F`, `T`, and diagnostic-only `T-AUX`;
- 3,400 updates and 13,926,400 training tokens per report;
- 30,600 updates and 125,337,600 training tokens in total;
- 67,033 total/active parameters and 4,864 FP32 state bytes per sequence for
  every arm;
- matched initial parameter hashes within every seed;
- passing optimizer partition, finite/nonzero gradient, learned-forward
  reconstruction, schedule, fingerprint-disjointness, checkpoint, and source
  provenance checks.

No G15B/R checkpoint or optimizer state entered this cohort. All declared
quality cells, interventions, checkpoints, and paired schedules completed. An
independent audit replayed the adjudication and verified all nine checkpoint
hashes.

## Ordinary overwrite

Three-seed mean query accuracy is:

| Length | `F` | `T` | `T-AUX` | `T - F` | `T-AUX - F` |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.895508 | 0.881022 | 0.930827 | -0.014486 | +0.035319 |
| 512 | 0.909668 | 0.890137 | 0.926432 | -0.019531 | +0.016764 |
| 1,024 | 0.909017 | 0.888835 | 0.930013 | -0.020182 | +0.020996 |
| 2,048 | 0.910319 | 0.879069 | 0.927897 | -0.031250 | +0.017578 |

`T` is inferior to `F` in every mean cell. `T-AUX` is above `F`, but its
largest mean gain is only `0.035319`, below the prospectively frozen `0.05`
gate at every length.

The primary arm is also seed-fragile:

| `T` seed | L128 | L512 | L1024 | L2048 |
|---:|---:|---:|---:|---:|
| 2381 | 0.813477 | 0.821289 | 0.811523 | 0.807129 |
| 2383 | 0.918945 | 0.928711 | 0.923828 | 0.916504 |
| 2389 | 0.910645 | 0.920410 | 0.931152 | 0.913574 |

Seed 2381 misses the `0.90` ordinary-overwrite gate at every length, whereas
2383 and 2389 are substantially stronger. This is a multi-seed failure, not a
claim that the recurrence cannot fit any seed.

The ordinary-overwrite `after_unrelated_overwrite_only` stratum has zero
support. Only the constructed guard supplies unrelated-only observations, at
768 decisions per cell. Ordinary-overwrite claims here therefore cover its
populated aggregate, before-overwrite, and after-same-key strata; they do not
establish ordinary unrelated-overwrite generalization.

## Addressing, commit timing, and causal use

Query-address top-1 is `1.0` throughout all recorded standard cells. Address
formation is therefore not the observed bottleneck. The primary objective
explicitly includes the frozen query-to-commit-address loss, so this is a
trained-address diagnostic, not independent evidence of unsupervised generic
association.

The completed-tail commit F1 gate is different:

- `T` fails the `0.95` threshold for every seed, task, and length;
- `T-AUX` seed 2381 remains approximately `0.8889` across cells;
- `T-AUX` seed 2383 is `0.9988--1.0`, and seed 2389 is
  `0.9840--0.9976`.

Thus explicit timing supervision largely identifies the post-value tail for
two seeds and improves overwrite means, but it does not produce a passing
three-seed mechanism. Because `T-AUX` has an unequal objective and fails its
own frozen gates, it cannot establish LM-only learnability or promote the
topology.

A post-hoc checkpoint diagnostic, not part of the preregistered adjudication,
explains the approximately `0.8889` F1 for `T-AUX` seed 2381. Head 2 fires at
every `WRITE_VALUE` and again at the correct tail: it has recall 1.0 and
precision 0.8, with exactly 6,144 false positives, all at role `WRITE_VALUE`
and offset zero. Unsupervised `T` heads also split prepare/value and tail
phases. This is exploratory evidence motivating a separately frozen staged
prepare/commit architecture, not a rescue of G15B-T or proof that the current
topology is sufficient.

The trained strict-history current-token interventions pass for all `T` and
`T-AUX` seeds: current-token mutation leaves edit controls, transition, and
injection invariant, while prior-history mutation has a nonzero effect. Most
causal-use interventions also pass, including commit-zero, memory-zero,
permuted-history, bias-only-history, erase-on-post-overwrite, and the majority
of timing shifts. They do not repair the failed learning gate. Notable
exceptions include the `+1` timing shift for `T` seed 2383 and some
erase-zero MQAR preservation checks.

`T` seed 2383 also misses the trained boundary numerical contract:

- chunk-boundary logit residual `8.9520215988e-4`;
- masked-step logit residual `8.8858604431e-4`;
- frozen maximum `5e-4`;
- predictions remain exact.

This is a numerical-parity failure only: its causal intervention passes, its
categorical predictions are unchanged, and it is not evidence of semantic or
causal leakage.

Finally, `T` seed 2389 reaches `0.99951171875` rather than exactly `1.0` on
needle recall at L2048, failing the frozen exact needle gate. Constructed
guards remain strong, but they cannot override these conjunctive failures.

## Interpretation boundary

The strongest supported conclusion is narrow: the tested one-block
strict-history symmetric-erase transactional GDN2 law, trained under the
frozen HarmonicMuonAdamW objectives and schedule, does not pass the
constructed three-seed mechanism gate. Timing supervision helps but does not
make the architecture pass.

This result does not establish that strict-history control is universally
wrong, that all transactional memories fail, or that a different explicit
slot/pending-transaction architecture could not learn. It provides no
natural-text quality, longer-context-pretraining, tokenizer, optimizer,
efficiency, scaling-law, torus, Spin, or model-family promotion evidence.
Those Phase-2 questions are not authorized after this Phase-1 failure.

The earlier clean execution smoke remains useful only as runner evidence:
[`artifacts/g15bt_phase1_smoke_sm75_2026-08-26.json`](artifacts/g15bt_phase1_smoke_sm75_2026-08-26.json).
