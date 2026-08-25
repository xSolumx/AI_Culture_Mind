# Program 03: Triality memory and intertwiner scans

**Research author:** Hayden Austin

**Status:** legacy combined route; use the
[canonical Programme 03 charter](../03-structured-memory-and-retrieval/README.md)
and route scan/identification/triality claims to Programmes 01, 02, and 04.

## Object

Associative recurrent lifts whose transition algebra contains equivariant
bilinear coupling between representation streams. `Spin(8)` triality supplies
one exceptional instance; the broader mathematical object is the triangular
intertwiner scan.

## Claims that currently survive

- A staged triangular bilinear lift can preserve associative prefix scanning
  and constant recurrent state.
- The maintained ordered scan now has a linear-work implementation with
  irregular-length, long-horizon, and full-gradient parity. On the local RTX
  2070 SUPER, its length-4,096 eager-PyTorch forward pass replicated a 3.59x
  speedup over the earlier Hillis--Steele tensor program. This is not a fused
  production-kernel comparison.
- The octonionic/triality binding map is an exact norm-preserving single-pair
  bind/unbind primitive under the stated normalization.
- Shared octonion machinery now also supplies an associative multiplication-
  operator scan: raw nonassociative products remain explicitly parenthesized,
  while `8 by 8` operators are scanned. The exact left-generator Lie closure
  is all `so(8)`. This is cross-listed machinery, not evidence for a
  triality-specific memory advantage.
- A separately maintained Pure Spin(8) v1.0 model now scans one shared Lie
  element in `8v`, `8s+`, and `8s-`. The 24-scalar tuple is faithful to all
  four central signatures, while a fixed octonion invariant couples the three
  streams only in the post-scan readout. Its frozen transport cohort passes,
  but supplying the true 28 Lie coordinates makes it an algebra-aligned
  mechanism result rather than evidence for generic triality memory.
- Under three hidden Haar changes of token and state basis, a 28-parameter
  `SO(8)` gauge learns the transported multiplication law and is identified
  only up to numerical `G2` automorphism residuals at most `2.17e-4`, exactly
  matching the classical stabilizer ambiguity. This is equivariant algebra
  identification, not a triality-specific sequence-model win.
- Terminal-only even-depth training sharpens that ambiguity to
  `G2 union -G2`: four curriculum checkpoints recover the positive coset and
  five the negative coset. A held-out odd length produces the predicted sign
  in all nine runs. This is a task-identifiability result, not a global
  optimization theorem.
- Multiplicity channels give exact orthogonal slot isolation for at most the
  multiplicity dimension; raw eight-dimensional superposition does not.
- In a controlled teacher-aligned identification gate, the known
  one-dimensional intertwiner family extrapolates exactly from a proper
  coordinate subspace. An unrestricted tensor fit to the same endpoints does
  not, while explicit group augmentation closes the gap. SO(3) reproduces the
  result, so it is not a triality-specific advantage.
- With supplied unit keys and correct shared actions, triality-bound addressed
  slots are exactly an orthogonal gauge of same-width direct slots. They cannot
  differ in capacity or retrieval error in that oracle regime.
- A block-preserving local homogeneous compiler now scans eight independent
  `9 x 9` affine slot matrices. It wins every tested RTX 2070 SUPER length in
  the optimized sweep and reversed long-length replication; the CPU requires a
  local/structured hybrid. This is a local eager-PyTorch result.
- A frozen 10-seed, 3,901,440-query campaign now matches direct slots,
  triality slots, learned-key chunkwise delta, oracle delta, and additive fast
  weights at 64 recurrent scalars. Hard/discretized slots beat learned delta
  in every corrupted overwrite cell for all 10 seeds; oracle delta is exact,
  so the result is about address geometry rather than delta-update capacity.
- Direct and triality slots tie exactly on every clean hard-route cell. Soft
  corruption produces small, sign-unstable differences and no
  triality-specific overwrite advantage.
- In the corrected matched eager CUDA tier, direct slots are fastest and
  chunkwise delta uses substantially less temporary memory. In a separate
  frozen three-process benchmark, the official FLA DeltaRule kernels are the
  clear transport-free systems winner: at length 4,096, compact-WY is 4.16x
  faster than direct slots and allocates 0.828 MiB incrementally versus
  57.422 MiB. That operator has no input for the programme's noncommuting
  per-token value action, so it does not settle the cross-view task.
- A new same-router hierarchy now separates routing from update law. Across 10
  seeds and 240 frozen cells, selecting one two-slot block improves mean cosine
  from `0.7927` to `0.9294` for direct memory and from `0.8945` to `0.9748` for
  DeltaRule while reducing ideal float32 value payload from 256 to 64 bytes.
  Hard routing makes direct and DeltaRule exactly equal on the separable
  synthetic world; a stronger-corruption frontier maps the expected sharp
  failure once argmax decisions flip.
- The full transported recurrence now has an exact invertible-action
  co-moving compiler. A stable float32 prefix/solve plus official fp16 FLA
  kernel includes all frame costs and, at length 4,096, is `5.20x` faster
  forward and `5.30x` faster forward+backward than direct slots on the local
  RTX 2070 SUPER. It uses 128 logical state scalars versus 64.
- Spin(9) supplies exact reversible `9 -> 16` Clifford binding and an
  equivariant nine-score Hopf coarse index. Dynamic slot tests still reduce to
  a direct-memory gauge, so no Spin(9)-specific capacity claim is made.

## Open empirical boundary

The local campaign establishes a hard-routing advantage over the maintained
learned-key delta and fast-weight rows, while the official fused FLA benchmark
establishes the opposite systems result for the transport-free recurrence.
The new compiler closes the standard-FLA transport gap but does not establish
superiority over fused Gated DeltaNet/DeltaProduct, p-BIM, or
Householder-product systems on learned cross-view tasks. The work still finds
no triality-specific memory-update advantage. The Task B delta row
using the independently fitted negative action is theorem-equivalent to the
direct row under hard keys but has not yet been rerun as its own artifact.

## Canonical evidence

Use the current
[`Programme 03 evidence ledger`](../03-structured-memory-and-retrieval/EVIDENCE_LEDGER.md).
The implementation protocol and raw benchmark hashes are in
[`INTERTWINER_SCHURSCAN_BENCHMARK_RESULTS.md`](../../Spin-Space-Research/docs/experiments/INTERTWINER_SCHURSCAN_BENCHMARK_RESULTS.md).
The first controlled structural generalization gate is in
[`INTERTWINER_SCHURSCAN_EQUIVARIANT_IDENTIFICATION_RESULTS.md`](../../Spin-Space-Research/docs/experiments/INTERTWINER_SCHURSCAN_EQUIVARIANT_IDENTIFICATION_RESULTS.md).
The current winner-by-regime systems verdict is in
[`SCHURSCAN_MEMORY_SCANNER_OPTIMIZATION_RESULTS.md`](../../Spin-Space-Research/docs/experiments/SCHURSCAN_MEMORY_SCANNER_OPTIMIZATION_RESULTS.md).
The frozen matched quality, sample-efficiency, failure-cohort, and CUDA verdict
is in
[`MATCHED_LEARNED_RETRIEVAL_RESULTS.md`](../../Spin-Space-Research/docs/experiments/MATCHED_LEARNED_RETRIEVAL_RESULTS.md).
The full Spin(8)/Spin(9) boundary, hierarchical routing campaign, and stable
transported FLA result are in
[`SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md`](../../Spin-Space-Research/docs/experiments/SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md).

## Next publishable question

Present Intertwiner SchurScans as the general theorem, with triality as an
example, and the matched retrieval campaign as a separate falsification-grade
empirical result. The supported-Linux compact-WY comparison is now complete;
the standard fused transported-value factorization is now complete. The next
systems step is a large-slot gathered hierarchical kernel and the next
scientific step is explicit Task B delta-action replay plus learned shared
Spin(8)/Spin(9) coarse routing on nontrivially overlapping aliases. Address
geometry—not scan capacity—remains the dominant local failure.

The unrestricted Dirac--Gram global proof is not a prerequisite for either
step. It is a separate sensing/design theorem and cannot substitute for memory
or throughput evidence.

The local operator-lift implementation and separate SSM claim boundary are in
[`OCTONION_FINAL_ONLY_RESULTS.md`](../../SSM-Models/experiments/OCTONION_FINAL_ONLY_RESULTS.md),
[`PURE_SPIN8_VS_MAMBA2_RESULTS.md`](../../SSM-Models/experiments/PURE_SPIN8_VS_MAMBA2_RESULTS.md),
[`OCTONION_OPERATOR_SCAN_RESULTS.md`](../../SSM-Models/experiments/OCTONION_OPERATOR_SCAN_RESULTS.md).
