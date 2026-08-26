# V1.3.1 research log — 2026-08-26

## Audit

- Replayed the untouched package: 39/39 tests passed.
- Confirmed the exact local device: RTX 2070 SUPER, compute capability 7.5,
  8 GB.
- Replayed the current source-bound WSL environment rather than installing an
  Ampere-only stack.
- Found that the historical retention default was `0.8808`, leaving about
  `0.00030` of an untouched state after 64 steps.
- Found that bounded retention did not bound the full noncompact E6 transition;
  the right action can have norm above one.
- Found that independent erase directions did not certify left-transition
  contraction.
- Found that the compact-action profiler differentiated a constant Frobenius-
  norm objective, producing a mathematically zero action gradient.
- Confirmed that `reduce-overhead` CUDA graphs fail for exceptional
  `matrix_exp`; the old optimization evidence applied only to identity.
- Confirmed that the Hillis-Steele semantic scan stores/composes full 27 by 27
  right actions at every level and is not a viable 2K/4K SM75 training path.

## Implementation decisions

- Completed the 27D compact ladder with named G2 and vector-stabilizer Spin(7)
  bases derived inside the existing Spin(8) alignment.
- Added rank, closure, containment, stabilizer, determinant and orthogonality
  audits for the new rungs.
- Raised default retention to `0.9995` and documented its half-life.
- Added independent write strength, tied Delta erase, normalized query, and
  E6 Frobenius/logarithmic-norm compensation.
- Added an exact direct rank-r recurrence that avoids dense H by H erase
  matrices and logarithmic right-action prefixes.
- Fixed the profiler VJP objective with a fixed random cotangent.
- Added an exact-SM75 hidden-coordinate composition harness and fail-closed
  summary.
- Upgraded the natural-text runner with the official fused Mamba-2, tight
  parameter matching, exact shared targets, common-parameter initialization,
  SM75 enforcement, git/runtime provenance, checkpoints, target hashes, and an
  optional HarmonicMuonAdamW arm.
- Added a fail-closed multi-seed quality summarizer.

## Evidence

- 46/46 package tests pass after the implementation.
- F4 versus Spin(9) and E6 versus F4 hidden-coordinate tasks pass all three
  seeds and extrapolate from length 4 to length 16.
- The matched natural-text cohort completes all 12 arm/seed cells. Official
  fused Mamba-2 wins all three seeds. Safe E6 is the best exceptional arm but
  remains worse by `0.15865` mean bpb.
- Fixed-shape `torch.compile(default)` executes identity and E6. The
  exceptional `reduce-overhead` combination is explicitly ineligible.
- The direct recurrence completes finite forward/backward execution at L2048
  and L4096 on exact SM75 for a small one-layer E6 configuration, peaking at
  about 308 MiB and 598 MiB. This is execution evidence only.

## Interpretation

The old conclusion "dense exceptional transport does not help generic text"
survives a much fairer memory law and a matched modern baseline. A new positive
conclusion is also supported: F4 and E6 coordinates genuinely learn when the
target contains directions unavailable to the predecessor subgroup.

The learning problem is therefore not inability to optimize the group action.
It is task alignment, autonomous event/address inference, and the cost of
moving a dense 27D frame every token. Further exceptional work should be sparse
and event-conditioned. E7 belongs in a separate 56D Freudenthal correctness
package; E8 remains algebra-only on SM75 until a sparse representation has a
specific falsifiable task.
