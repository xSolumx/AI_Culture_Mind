# G15B-E Phase-0 Qualification Result

**Protocol:**
[`G15BE_EFFECTIVE_EDIT_PROTOCOL_2026-08-26.md`](G15BE_EFFECTIVE_EDIT_PROTOCOL_2026-08-26.md)

**Artifact:**
[`artifacts/g15be_phase0_qualification_sm75_2026-08-26.json`](artifacts/g15be_phase0_qualification_sm75_2026-08-26.json)

**SHA-256:**
`41b49a6de9a74a563c4dc6f3c0571d8d9b0c7fd8fd95405232be22807d88936b`

**Execution commit:** `6c0b4aab75362432c6561fcb7d243c4a44f73a09`

**Status:** pass; authorize only the prospectively frozen G15B-E Phase-1
constructed P/A learning screen

## Bottom line

The product (`P`) and logit-additive (`A`) effective-edit laws are executable,
bounded, parameter/state matched, and numerically coherent on the exact local
SM75 GPU. This is implementation evidence, not evidence that either law
learns retrieval or improves a language model.

The run started from a clean checkout on the NVIDIA GeForce RTX 2070 SUPER,
compute capability `(7,5)`, under WSL2 with Python 3.11.16, PyTorch
2.9.0+cu128, and CUDA 12.8. No fallback path was used.

## Matching contract

Both arms have:

- 22,161 total and active parameters;
- 1,408 bytes of FP32 streaming state per sequence;
- identical state-dictionary keys and bit-identical initialized tensors;
- the same full causal controller view and affine fast-weight state law.

The initial effective erase and write residuals between P and A are each
`1.8626451492e-9`, below the frozen `2e-8` FP32 allowance. Their maximum FP64
transition spectral norms are `0.9995000000000025` and
`0.9995000000000027`; all tested transitions, injections, and effective gates
are finite and bounded.

## Execution parity

For both arms, FP64 recurrent/parallel, arbitrary-chunk, token-step,
masked-step, and compact-valid-token comparisons pass the fixed `1e-10`
logit/state/read bounds. The largest recorded FP64 logit residual is
`2.776e-16`.

FP32 uses the prospectively frozen scale-aware bound

```text
128 * eps(float32) * sequence_length * max(1, reference_absmax)
```

separately for logits, state, and read. At length 33 each bound is
`5.035400390625e-4`; observed maxima across both arms are:

- logits: `8.941e-8`;
- recurrent/convolution state: `4.470e-8`;
- memory read: `1.863e-9`.

All compared predictions are exactly equal. Every declared real-LM gradient
path is finite and nonzero in both arms, including the shared intensity,
erase, write, key/value, retention, read/output, embedding, convolution, and
LM-head projections.

## Interpretation boundary

The artifact authorizes only fresh-seed constructed Phase 1 under the frozen
protocol. It does not authorize natural-text training, attention or geometry,
does not repair or relabel the failed G15B-T result, and does not show that
logit-additive gates learn better than product gates. That comparison remains
prospective.
