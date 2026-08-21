# Prospective Shakespeare layer-localization gate

**Frozen:** 2026-08-21, after seed-17 discovery and before fresh-seed runs  
**Status:** prospective development gate

## Question

Does the discovered two-layer schedule `E6 -> identity` improve validation
bits per byte over `identity -> identity` on fresh Tiny Shakespeare runs?

## Frozen protocol

- corpus, revision, hashes, byte tokenization, and 90/5/5 split exactly as in
  `SHAKESPEARE_DEVELOPMENT_RESULTS.md`;
- fresh seeds 18, 19, 20, 21, and 22;
- only `identity_delta` and `early_e6_delta`;
- 500 AdamW updates, learning rate `3e-3`, weight decay `0.01`;
- batch 2, sequence length 64, 16 fixed validation batches per seed;
- two layers, `d_model=32`, memory width 4, update rank 2, depthwise-conv 4,
  Jordan mixer, Albert-invariant readout, and automatic parallel scan;
- RTX 2070 SUPER under the recorded WSL/Torch environment;
- no test-split evaluation and no hyperparameter changes after inspecting a
  fresh-seed result.

## Primary decision rule

The schedule passes this development gate only if both conditions hold:

1. `early_e6_delta` has lower validation bpb in at least four of five seeds;
2. its mean paired improvement is at least `0.0100` bpb.

Report every seed, the mean paired difference, throughput, and any nonfinite
gradient. Failure rejects this layer-localization result at the tested scale;
it does not prove that exceptional transport is useless in other regimes.

## Secondary observations

Parameter count, peak CUDA allocation, and training throughput are descriptive
only. They cannot rescue a failed primary quality gate. A passing result is
still not a Mamba comparison or a promoted model claim; it only justifies a
larger matched multi-model experiment.
