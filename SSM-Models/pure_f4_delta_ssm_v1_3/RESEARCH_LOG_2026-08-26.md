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

## V1.3.2 cheap-transport continuation

The cost of moving the 27D frame was reopened as the primary failure rather
than accepted as an architectural tax.

### Exact action redesign

- Derived disconnected blocks of size at most three for every maintained F4
  and E6 generator.
- Added canonical coordinates of the second kind: an ordered product of exact
  one-parameter subgroup actions.
- Kept the old direct exponential-of-a-sum chart as a scientifically distinct
  control; no equivalence between the two finite-coordinate charts is claimed.
- Added a float64 portable action oracle, a dense same-chart oracle, and an
  FP32 CUDA kernel compiled explicitly for `sm_75`.
- Added handwritten action backward that reconstructs intermediate primitive
  states with inverse group elements.

### Failed first integration

The first sparse model called four separate associative scans around four
native actions at length 128.  Although the isolated action was hundreds of
times faster than its dense oracle, the model took 3.11 seconds for 40 updates
versus 1.95 seconds for dense E6.  This falsified the idea that a fast side
kernel alone solved the systems problem.

### Fused recurrence

- Fused retention, independent erase, periodic primitive action, write, read,
  and final-state production into one SM75 kernel.
- Added handwritten reverse-time gradients for all seven recurrence inputs.
- Moved the exceptional coordinate controller to active event tokens only.
- Added a same-parameter dead-budget arm whose action head participates in
  backward and receives optimizer state while its state action is zero.
- Repaired benchmark target hashing so no CUDA-to-CPU synchronization occurs
  inside timed updates.
- Separated training peak allocation from final validation allocation.
- Added a fresh-process complete-step cost harness with raw timing samples,
  saved-tensor shape audit, source hashes, patch hash, parameter/optimizer/
  buffer bytes, and independent cheap-action and Mamba-competitive verdicts.

### Evidence before the quality cohort

The following counts and small-cell timings were the state at that historical
checkpoint; the representative continuation below supersedes its systems
boundary.

- Canonical WSL venv: Python 3.11.16, Torch 2.9.0+cu128, RTX 2070 SUPER,
  compute capability exactly 7.5.
- WSL native suite: 72/72 passing.
- Fused recurrence output/final error at most about `6.6e-9`; all seven input
  gradient families at most about `9e-8` from the portable recurrence.
- Isolated action: F4/E6 forward+backward speedups of `415.4x` and `628.5x`
  over the dense same-chart oracle.
- Complete sparse E6 step: `19.89 ms` median and `44,691,968 B` maximum peak,
  versus dense E6 at `47.68 ms` and `143,790,592 B`.
- Official fused Mamba-2 remains ahead at `13.01 ms` and `34,982,400 B`.

The clean-revision systems rerun gives `19.89 ms` sparse E6 versus `47.68 ms`
dense E6 and `13.01 ms` Mamba-2.  The cheap-action gate passes; the complete-
model Mamba systems gate fails.

The fresh seeds 2677/2683/2699 then completed 12/12 natural-text cells and
verified all external checkpoint hashes.  Sparse E6 beats the exact dead-
budget control in 2/3 seeds and improves mean bpb from `2.962925` to
`2.955639`, passing the frozen local quality gate.  Dense E6 remains slightly
better at `2.949676`; Mamba-2 remains decisively better at `2.790171` and wins
all three seeds.  Cheap exceptional transport is therefore promoted as a
learned component, while the complete model is not promoted over Mamba-2.
See [`SM75_PRIMITIVE_TRANSPORT_RESULTS.md`](SM75_PRIMITIVE_TRANSPORT_RESULTS.md).

## Representative SM75 continuation

The small regression harness hid two different questions.  Its active fused
arm and dead generic-scan arm did not share the same recurrent backend, so the
active/dead timing was not a valid marginal-action measurement.  The native
recurrence now accepts `transport_enabled`: active and dead share the kernel
launch, saved-state layout, event schedule, coordinate controller, backward
interface, parameter count, and optimizer state.  Dead skips only the 78
primitive factors and returns exact zero coordinate gradients.  Portable and
native parity tests cover both modes.

The representative pair is four candidate layers at width 126 versus four
official Mamba-2 layers at width 140: 679,866 versus 682,160 parameters.  The
first clean run exposed host activation memory rather than action memory.
Replacing autograd's Albert-product intermediates with an exact custom reverse
rule reduced candidate peak from 905.8 MB to 353.8 MB.  Non-reentrant block
checkpointing reduced it again to 222.7 MB while preserving outputs, states,
and gradients.

At batch 32 / length 128, the final clean candidate takes `67.48 ms` and
222.71 MB versus exact dead at `61.38 ms` / 222.45 MB, dense E6 at
`233.45 ms` / 665.52 MB, and Mamba-2 at `70.64 ms` / 232.30 MB.  The
representative cheap-action and Mamba systems gates both pass.  This is the
first complete matched cell where the exceptional candidate is within both
time and memory budgets of official Mamba-2 on exact SM75.

## Long-context execution diagnosis

A clean fixed-4,096-token shape ladder then falsified the idea that the
persistent recurrence scales across context shapes.  Candidate time rises
from `67.39 ms` at `(32,128)` to `563.57 ms` at `(1,4096)` while peak memory
stays flat.  Active/dead overhead remains within its gate everywhere.  The
failure is occupancy: one persistent CUDA block per stream leaves most of the
40-SM GPU idle at batch 1.

An exact repair now groups 32 token edits into a two-sided affine block map,
parallel-scans the block maps, and reconstructs within-block states.  Dense
27 by 27 actions occur only at the scheduled events.  Raw-scan and complete-
model output/state/gradient parity pass.  The hardest development cell falls
to `172.59 ms`, a `3.26x` improvement, but remains slower than Mamba-2 and has
not received a clean full-ladder promotion.  The open cost is block-map scan
composition and launch integration, not exceptional factor evaluation.

## Cached inference qualification

The inference harness now drives the official Mamba-2 `InferenceParams` path
and rejects widths that cannot execute its real cached convolution kernel.
The matched SM75-friendly pair is a two-layer candidate at width 204 with a
standard SwiGLU host and normalized vector read versus two-layer Mamba-2 at
width 224: 807,652 versus 808,176 parameters.

At batch 1 and prefix 4,096, sparse E6 bulk prefill is `10.107 ms` versus
Mamba-2 at `14.215 ms`; cache-building prefill is `10.409 ms` versus
`27.998 ms`.  Candidate recurrent cache is 6,624 bytes versus Mamba-2 at
481,280 bytes.  All three gates pass.  Cached one-token decode is `2.392 ms`
versus Mamba-2 at `1.408 ms`, so streaming promotion fails at `1.700x`.
Active E6 versus exact dead-action decode is `0.998x`: the group action is
below measurement noise, and the remaining decode cost is the unfused eager
host.  The lean two-layer arm is systems evidence only; no quality conclusion
is transferred from the earlier Jordan/invariant cohort.
