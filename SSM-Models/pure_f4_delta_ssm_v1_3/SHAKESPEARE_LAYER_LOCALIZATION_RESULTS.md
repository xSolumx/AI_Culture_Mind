# Shakespeare layer-localization fresh-seed result

**Research author:** Hayden Austin

**Date:** 2026-08-21

**Verdict:** the prospective gate failed. Early E6 transport is not promoted
as the v1.3 natural-text schedule.

## Frozen test

The protocol and decision rule were committed and pushed in
`SHAKESPEARE_LAYER_LOCALIZATION_PREREGISTRATION.md` before seeds 18--22 were
run. The candidate was `E6 -> identity`; the baseline was
`identity -> identity`. The gate required both at least 4/5 paired wins and a
mean improvement of at least 0.0100 validation bpb.

All inputs and validation targets came from the pinned, disjoint Tiny
Shakespeare train and validation slices documented in
`SHAKESPEARE_DEVELOPMENT_RESULTS.md`.

## Complete result

Positive improvement favors the candidate.

| Seed | Identity bpb | Early-E6 bpb | Improvement bpb |
|---:|---:|---:|---:|
| 18 | 3.3771 | 3.4079 | -0.0308 |
| 19 | 3.4462 | 3.4131 | +0.0331 |
| 20 | 3.4732 | 3.4838 | -0.0106 |
| 21 | 3.4177 | 3.4630 | -0.0453 |
| 22 | 3.5042 | 3.4971 | +0.0071 |
| **Mean** | **3.4437** | **3.4530** | **-0.0093** |

The candidate won 2/5 seeds and was worse on average. It failed both primary
criteria. Mean training throughput was also lower because the candidate still
constructs E6 actions in one layer.

## Consequence for v1.3

The seed-17 discovery was real but not stable. The correct conclusion is not
to search schedules until one looks favorable. On generic character-level
Shakespeare at this scale:

- identity transport is the supported natural-text benchmark reference;
- F4/E6 actions remain mathematically verified, executable research controls;
- dense exceptional transport has no demonstrated language-quality benefit;
- the next exceptional test needs task-aligned multi-view structure or a
  mechanism that activates transport only when an independently measured
  benefit pays for it.

This narrows the architectural horizon. The promising target is no longer
“exceptional action everywhere,” but a compiler-visible optional connection:
an identity fast path plus sparse, auditable exceptional frame changes. Such a
mechanism must beat identity under a new frozen gate before receiving a CUDA
kernel.

## Reproduction artifacts

- `artifacts/shakespeare_localization500_seed18_rtx2070s_20260821.json`
- `artifacts/shakespeare_localization500_seed19_rtx2070s_20260821.json`
- `artifacts/shakespeare_localization500_seed20_rtx2070s_20260821.json`
- `artifacts/shakespeare_localization500_seed21_rtx2070s_20260821.json`
- `artifacts/shakespeare_localization500_seed22_rtx2070s_20260821.json`
- `artifacts/shakespeare_localization_fresh_seeds18_22_summary_rtx2070s_20260821.json`

The summary artifact validates that dataset metadata, non-seed configuration,
and source hashes match across all five inputs before computing the paired
result. This remains a small-model, single-device development result, not a
comparison with Mamba or a promoted natural-language benchmark.
