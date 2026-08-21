# Tiny Shakespeare development results

**Date:** 2026-08-21  
**Status:** development evidence; no promoted language-quality claim

## Data contract

Every row in this report trains and validates on the pinned
[char-rnn Tiny Shakespeare corpus](https://github.com/karpathy/char-rnn/blob/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt).
There is no WikiText fallback or benchmark option in v1.3.

- revision: `6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e`
- full SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`
- tokenization: raw UTF-8 bytes
- split: chronological, disjoint 90% train / 5% validation / 5% test
- train SHA-256: `a9e24e23a1ec77744dad26844bfd5a09b6e041954e1eef0000e7f24cba6db735`
- validation SHA-256: `c64e90c160750d91a9d2d9b1ace5eab1a440f5a1701d32a8e97a179b7839f1c8`

The runner samples all model variants from the same CPU byte streams and the
same seeded validation batch list. The held-out test slice has not been used.

## Geometry screen: 50 updates

Configuration: seed 17, two layers, `d_model=32`, memory width 4, rank 2,
batch 2, length 64, four validation batches, AdamW at `3e-3`.

| Variant | Parameters | Validation bpb | Train tok/s |
|---|---:|---:|---:|
| identity delta | 35,578 | 4.4723 | 3,549.2 |
| F4 direct | 39,010 | 4.4889 | 2,554.3 |
| E6 direct | 40,726 | 4.4693 | 2,553.9 |
| E6 polar | 40,726 | 4.4694 | 2,207.6 |
| E6 Cartan/KAK | 42,574 | 4.4918 | 2,217.8 |

Direct and polar E6 differ by only `0.00004` bpb here. Direct executes about
16% more training tokens per second, so it is the development default. This is
an implementation choice conditional on this shape and GPU, not a theorem
about E6 parameterizations.

## Core and layer-localization screen: 500 updates

The same seed and model shape were trained for 500 updates and evaluated on 16
fixed validation batches.

| Variant | Layer schedule | Parameters | Validation bpb | Train tok/s |
|---|---|---:|---:|---:|
| identity delta | identity -> identity | 35,578 | 3.3473 | 3,519.1 |
| F4 direct | F4 -> F4 | 39,010 | 3.3738 | 2,512.7 |
| E6 direct | E6 -> E6 | 40,726 | 3.3600 | 2,589.0 |
| late E6 | identity -> E6 | 38,152 | 3.4007 | 2,893.3 |
| early E6 | E6 -> identity | 38,152 | **3.3268** | 2,943.4 |

Dense exceptional transport is not supported by this screen: identity-only
beats both all-F4 and all-E6 while being materially faster. The asymmetric
schedule is the interesting discovery. Early E6 followed by identity improves
on identity-only by `0.0205` bpb in seed 17, whereas reversing the schedule is
worse by `0.0533` bpb. Because this ordering hypothesis was selected after
seeing seed 17, it is explicitly discovery evidence.

## Interpretation and falsifiable conjectures

The current evidence suggests a more precise hypothesis than “more symmetry
is better”:

1. **Exceptional transport may be a feature-frame initializer, not a
   per-layer default.** The first layer can align newly formed local features;
   later identity transport can consolidate them without repeatedly moving the
   memory frame.
2. **The action is an inductive bias, not free capacity.** A general invertible
   right action can be moved into a co-moving representation. Its benefit must
   come from parameter sharing and task-aligned geometry, while its cost is
   immediate.
3. **Ordinary character prediction is an adversarially honest task for Spin
   structure.** Tiny Shakespeare supplies no declared Spin/F4 symmetry. A gain
   must therefore emerge from useful learned organization, not from matching a
   symmetry planted in the labels.
4. **If the early-E6 signal replicates, the next compiler target is sparse
   exceptional entry plus identity consolidation.** If it fails, v1.3 should
   retain E6 as an experimental mechanism and make identity the natural-text
   default.

The prospective fresh-seed decision rule is frozen in
[`SHAKESPEARE_LAYER_LOCALIZATION_PREREGISTRATION.md`](SHAKESPEARE_LAYER_LOCALIZATION_PREREGISTRATION.md).

## Artifacts and nonclaims

- `artifacts/shakespeare_architecture_screen50_seed17_rtx2070s_20260821.json`
- `artifacts/shakespeare_core_screen500_seed17_rtx2070s_20260821.json`
- `artifacts/exceptional_auto_profile_rtx2070s_20260821.json`

These artifacts are single-device development screens. They are not
parameter-matched against Mamba-2, not multi-seed quality evidence, and not a
fused-kernel result. No comparison to v1.2 or another model is inherited.
