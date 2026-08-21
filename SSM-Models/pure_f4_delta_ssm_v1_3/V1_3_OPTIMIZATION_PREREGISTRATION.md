# V1.3 eager optimization noninferiority gate

**Frozen:** 2026-08-21 after seed-101 development timing and before fresh
quality seeds 102--106

**Status:** prospective development gate

## Candidate and baseline

The candidate combines two algebraic reductions:

1. identity transport uses the one-sided affine scan and never materializes or
   prefix-multiplies a 27 by 27 identity action;
2. the Albert determinant uses the explicit `H_3(O)` cubic formula rather than
   two dense Jordan products.

The eager baseline uses the former generic two-sided scan, dense Jordan mixer,
and determinant-through-Jordan trace identity. Model parameters, initial
weights, optimizer, batches, and validation examples are paired by seed.

The one-sided scan already has bitwise float32 model-output and parameter-
gradient parity against the generic scan when both use the same determinant
backend. The fresh-seed gate therefore tests the only remaining numerical
change: evaluation order in the explicit cubic.

## Frozen protocol

- pinned Tiny Shakespeare corpus and the existing disjoint 90/5/5 byte split;
- seeds 102, 103, 104, 105, and 106;
- `identity_legacy` versus `identity_delta`, eager execution only;
- 300 AdamW updates, learning rate `3e-3`, weight decay `0.01`;
- batch 2, sequence length 64, 16 fixed validation batches per seed;
- two layers, `d_model=32`, memory width 4, update rank 2, depthwise-conv 4,
  Jordan mixer, and Albert-invariant readout;
- RTX 2070 SUPER under the recorded WSL/Torch environment;
- no test-split evaluation and no protocol change after inspecting a fresh
  seed.

## Quality decision rule

The eager candidate passes only if both conditions hold:

1. mean paired `candidate_bpb - legacy_bpb` is no greater than `+0.0100`;
2. no individual seed is worse by more than `+0.0500` bpb.

This is a noninferiority gate, not a claim that the optimization improves
language quality. All seeds and paired differences must be reported.

## Systems boundary

The seed-101 timing is discovery evidence only. Final speed claims require the
same source revision, interleaved CUDA-event samples, output/gradient parity,
and separate reporting of cold compilation time. `torch.compile` with
`reduce-overhead` is evaluated as an opt-in fixed-shape execution tier; it is
not covered by the eager quality gate and cannot become the default from a
single training trajectory.
