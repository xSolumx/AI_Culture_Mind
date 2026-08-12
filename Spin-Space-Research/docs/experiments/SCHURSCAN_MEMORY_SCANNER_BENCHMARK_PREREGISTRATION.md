# SchurScan memory-scanner benchmark preregistration

- **Date frozen:** 2026-08-10, before timing the new structured-slot backend
- **Scope:** maintained eager PyTorch implementations on the named local host;
  not a universal or fused-kernel ranking

## Question

For an exact 64-scalar addressed memory transition, which maintained local
prefix program gives the best forward latency and intermediate memory:

1. the original structured slot Hillis--Steele scan;
2. a new ordered work-efficient structured slot scan;
3. a generic dense direct-affine work-efficient scan; or
4. a generic dense homogeneous work-efficient scan?

All four rows execute the identical recurrence and emit every prefix. This
isolates scan representation and tree schedule rather than retrieval quality.

## Frozen transition

The state is eight slots of eight coordinates. One token is

\[
M_t[h]=r_t[h]A_tM_{t-1}[h]+b_t[h],
\]

where `A_t` is a shared contractive noncommuting orthogonal action,
`r_t[h]` lies in `[0.90, 0.999]`, and `b_t[h]` is a small random drive. Inputs
are generated canonically in float64 on CPU, cast to the requested dtype, and
moved to the device before scan-only timing.

The structured monoid applies equally to:

- direct slots transported in their value representation;
- triality-bound slots transported in the vector representation;
- slots whose supplied orthogonal action was compiled from Householder, rotor,
  or another parameterization.

Triality bind/unbind and action-compilation costs are outside the scan-only
timing and will be discussed separately. They do not change the prefix monoid.

## Frozen backends

- `slot_hillis`: existing structured (O(N\log N))-work tree;
- `slot_work_efficient`: ordered padded Blelloch tree with fewer than `3P`
  structured compositions;
- `dense_affine_work_efficient`: materialize the exact block-diagonal
  `64 x 64` action and use the direct affine tree;
- `dense_homogeneous_work_efficient`: pack that exact transition as a
  `65 x 65` homogeneous matrix and use the one-matmul tree.

For the two dense rows, scan-only timing uses prebuilt dense leaves. A separate
end-to-end timing includes structured-to-dense materialization. Neither dense
row is presented as a faithful fused DeltaNet/DeltaProduct implementation.
A dense delta update has the same 64-dimensional generic affine scan shape,
but modern chunkwise delta kernels exploit additional low-rank structure not
used here.

The general 24-scalar Intertwiner SchurScan benchmark remains a separate
quality/capacity regime and will only be cited for context; it is not called a
same-memory competitor.

## Protocol

- CPU: float64, batch 1, lengths 64, 256, 1,024, and 2,048;
- CUDA: float32, batch 2, the same lengths plus 4,096 if memory permits;
- five warmups and fifteen measured repeats;
- six PyTorch CPU threads;
- CUDA Events with explicit synchronization;
- median, minimum, p20, p80, mean, and population standard deviation;
- incremental CUDA output/intermediate allocation above prebuilt leaves;
- timing order rotated by length to expose simple order bias;
- TF32 disabled.

Input generation and recurrent-reference construction are excluded from every
timed region. Dense conversion is excluded only from the rows explicitly
labelled `scan_only`.

## Correctness gates

- all four prefix-state outputs agree with a sequential recurrence below
  `2e-10` in float64 and relative error below `5e-5` in float32;
- the work-efficient structured backend passes irregular non-power-of-two
  lengths and full-gradient parity in maintained unit tests;
- composition counts and dependency depths are reported separately from time;
- no timing is interpreted unless its correctness gate passes.

## Decision rule

The fastest median row is named per device and length. A global local winner is
reported only if one row wins every length at or above 256 on a device. If
latency and memory winners differ, both are reported. No threshold is frozen
for a required speedup: a crossover or a negative result is acceptable.

## Scientific boundary

This systems benchmark cannot establish an absolute best memory architecture.
It ranks only maintained eager implementations of one exact slot recurrence.
Retrieval quality is governed separately by address inference, overwrite law,
transport identification, and task distribution.

Existing controlled evidence already shows:

- triality and direct addressed slots tie exactly when keys/actions are
  supplied, now strengthened to an exact gauge equivalence;
- jointly balanced hard-slot routing beats the tested learned continuous delta
  keys on the frozen alias task, while an oracle delta rule ties exactly;
- additive fast weights fail overwrites on that task;
- none of those results covers a full modern fused Gated DeltaNet,
  Erase-then-Delta, DeltaProduct, or language-scale comparison.

## Optimization addendum: local homogeneous slot blocks

Frozen after the first benchmark exposed the structured-launch versus dense-
arithmetic crossover and after correctness tests for the new representation,
but before observing any timing for it.

The repository-wide algebra audit suggested combining two maintained ideas:

1. the Schur/isotypic scans avoid materializing a full Kronecker action by
   preserving independent representation blocks;
2. the Mamba-3 and generic SchurScan references show that a small homogeneous
   lift can replace several eager affine operations by one batched matmul.

For each slot, pack

\[
H_t[h]=
\begin{pmatrix}
1 & 0\\
b_t[h] & r_t[h]A_t
\end{pmatrix}
\in\mathbb R^{9\times9}.
\]

Scan the eight local blocks as an enlarged batch. This uses eight `9 x 9`
matrices per token rather than one dense `65 x 65` matrix. It preserves slot
independence, emits the identical recurrence, and expresses each composition
as one batched matmul.

Four rows are added without changing any original input or timing setting:

- local-homogeneous Hillis, prepacked scan-only;
- local-homogeneous work-efficient, prepacked scan-only;
- local-homogeneous Hillis, including packing;
- local-homogeneous work-efficient, including packing.

The original six backends remain frozen controls. The optimized winner is
again named per length/device with no required speedup threshold. Correctness,
gradient, memory, replication, and claim-boundary rules are unchanged.
