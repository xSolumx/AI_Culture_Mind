# G12 optimizer, tokenizer, robustness, and scaling preregistration

**Frozen before execution:** 2026-08-25  
**Parent evidence:** G10 and G11 artifacts, plus the repository-wide optimizer
and tokenizer audit  
**Status at freeze:** no G12 training or evaluation result had been inspected

## Questions

1. Does a geometry-aware composite optimizer improve v1.4.5 ordinary
   TinyStories learning relative to unchanged AdamW?
2. Can lossless training-only ByteLevel BPE reduce sequence length without
   hiding quality behind a tokenizer-dependent metric?
3. Does the selected intervention remain robust on three fresh model seeds?
4. After ordinary next-token pretraining only, does the model use information
   farther back than its training length in a counterfactual recall probe?
5. What happens at parameter-matched and measured-compute-matched scale points?

## Fixed data and metrics

- Dataset snapshot and train/validation hashes are exactly G11's pinned 2,000
  TinyStories training rows and 256 official validation rows.
- The fitted tokenizer sees the training byte stream only. Validation is never
  used to fit merges or choose vocabulary items.
- Raw UTF-8 bytes remain the control. Fitted candidates are lossless ByteLevel
  BPE with vocabularies 512 and 1,024, the complete byte alphabet, no unknown
  token, no special token, and exact train/validation round-trip checks.
- Every token target carries its represented count of original UTF-8 bytes.
  The cross-tokenizer primary metric is total target negative log likelihood
  divided by scored original bytes and `ln(2)`: bits per raw byte (BPRB).
- Accuracy is descriptive and is never compared across different vocabularies.

## Stage A: tokenizer audit and deterministic selection

The tokenizer audit reports vocabulary size, serialized tokenizer hash, train
and validation tokens, bytes per token, and exact round trips. The selected BPE
candidate is the smallest vocabulary reaching at least 2.0 training bytes per
token. If neither reaches the threshold, fitted tokenization is rejected. This
selection uses no model loss.

## Stage B: optimizer development screen

- Architecture/tokenizer: v1.4.5, width 64, raw bytes.
- Development seeds: 1823 and 1829.
- Paired deterministic data order seed: 1817.
- Updates: 500; batch 16; sequence length 256; learning rate `1e-3`; decoupled
  weight decay `0.01`; global gradient clip `1.0`.
- Arms:
  - unchanged PyTorch AdamW;
  - `HarmonicMuonAdamW`: PyTorch Muon with `match_rms_adamw` on eligible 2D
    hidden matrices, scalar-second-moment AdamW with no decay on write/decay
    controllers and residual scales, ordinary AdamW on all remaining tensors.
- The complete named parameter partition is recorded in the artifact.
- The composite advances only if its mean final BPRB improves by at least 0.02,
  neither seed regresses by more than 0.02, and all values are finite. Otherwise
  AdamW remains selected. This screen is developmental, not a final result.

## Stage C: fresh three-seed natural-text robustness

- Fresh model seeds: 1871, 1873, and 1877. They are disjoint from G11 and Stage
  B and are not replaceable after execution.
- Updates: 1,000; batch 16; sequence length 256 tokens; identical deterministic
  windows for arms sharing a tokenizer.
- Raw control: v1.4.5 width 64 with AdamW.
- Optimizer arm: same raw model with the Stage-B-selected optimizer. If AdamW is
  selected, this is explicitly a null intervention and is not duplicated.
- Tokenizer arm: the Stage-A-selected ByteLevel BPE using the selected optimizer.
  Its model shape is chosen before training as the closest trainable parameter
  count to the 119,962-parameter raw control over widths divisible by four and
  integer FFN expansions 1--6. Ties choose larger width, then smaller expansion.
- Robustness requires all three selected-arm runs finite, at least 2.0 BPRB
  improvement from initialization per seed, mean final BPRB no worse than the
  raw AdamW mean by more than 0.02, and worst-seed final BPRB no worse by more
  than 0.05. Passing is bounded evidence only.

## Stage D: longer-context recall after ordinary pretraining

No recall examples or auxiliary losses appear during training. Each Stage-C
checkpoint is evaluated at raw-byte distances 128, 256, 512, and 1,024 using
held-out TinyStories filler and paired prompts of the form `Name's secret word
was value ... Name's secret word was`. The supported value is changed only in
the counterfactual prefix. The score is the teacher-forced log-probability gain
for the complete correct value under the matching prefix versus the mismatched
prefix, reported in nats and per raw byte. Names, values, filler offsets, and
pairing are deterministic from seed 1901. This is a templated post-pretraining
probe, not natural-corpus perplexity and not evidence of instruction following.

The same checkpoints also receive ordinary held-out next-token evaluation at
token lengths 256, 512, and 1,024. This separates loss at longer execution
lengths from the counterfactual recall contrast.

## Stage E: parameter and measured-compute accounting

- Parameter counts are exact trainable counts. The selected BPE shape is
  parameter-matched as specified in Stage C.
- Training reports scored tokens, scored original bytes, wall time, CUDA peak
  allocated bytes, and an interleaved median update time after warmup.
- A post-hoc compute-matched comparison truncates each paired learning curve to
  the greatest recorded checkpoint not exceeding the smaller measured training
  wall budget. It is descriptive because wall time is hardware-specific.
- Actual Transformers Mamba2 and OLMo hybrid G11 classes are rebuilt at their
  recorded shapes and included in parameter/runtime tables. Their G11 result is
  historical context; it is not silently treated as a G12 three-seed result.

## Decision boundaries

- No optimizer is described as `10,000x` better without a measured ratio.
- A tokenizer win must use BPRB, exact byte accounting, and parameter reporting;
  token loss or token accuracy alone cannot establish it.
- Optimizer, tokenizer, robustness, long-context execution, recall, parameter
  matching, and measured compute are separate findings.
- A failed sufficient gate rejects that route; it does not prove AdamW or raw
  bytes are globally optimal.
