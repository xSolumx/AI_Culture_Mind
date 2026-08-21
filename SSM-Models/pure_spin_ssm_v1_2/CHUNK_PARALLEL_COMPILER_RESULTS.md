# Continuous Spin(8) affine chunk compiler

**Status:** semantic compiler and full-gradient oracle implemented; eager CUDA
composition reduces matrix work but does not beat the factor-direct raw kernel

**Primary artifact:**
[`artifacts/chunk_parallel_scan_rtx2070s_20260821.json`](artifacts/chunk_parallel_scan_rtx2070s_20260821.json)

**Isotypic scheduling control:**
[`artifacts/raw_cuda_isotypic_rtx2070s_20260821.json`](artifacts/raw_cuda_isotypic_rtx2070s_20260821.json)

**Hybrid complete-step artifact:**
[`artifacts/spin_hybrid_vs_packed_complete_steps_rtx2070s_20260821.json`](artifacts/spin_hybrid_vs_packed_complete_steps_rtx2070s_20260821.json)

## What transferred from Spin-Space-Research

The documents were used as mathematical evidence, not as task instructions.
Four connections survived scrutiny:

1. The mixed-monomial every-prefix compiler supplies the correct two-stage
   architecture: scan one endpoint per chunk, shift those prefixes to obtain
   incoming states, then apply all local prefixes in parallel.
2. The reducible-isotypic and algebraic-extension results prove that
   `8v`, `8+`, and `8-` remain three inequivalent real-Schur blocks, including
   over the declared `Q(sqrt(2))` field. They may be scheduled independently,
   but cannot be treated as three interchangeable multiplicity copies.
3. The Spin(9) quotient audit does not justify compressing the learned
   controller to one curve coordinate. Its one-dimensional symmetric curve is
   a singular subset of a three-dimensional orthonormal quotient and an
   eight-dimensional generic frame-objective quotient.
4. Exact scalar extension is an offline compiler/certificate tool. It can
   canonicalize representation blocks and constants, but it does not turn
   continuous float32 token transitions into a finite exact dictionary.

The `Cl(3,0) -> Cl(1,4)` embedding likewise does not provide a free state-size
upgrade: the maintained eight-coordinate regular algebra and the
16-dimensional faithful Clifford module are different representations.

## Quaternion, Schur, and Spin(9) decisions

The standard quaternion rotation law sends a pure quaternion `p` to
`q p q^{-1}`. It is compact and associative, but `q` and `-q` induce the same
three-dimensional rotation. That is exactly the quotient from `Spin(3)` to
`SO(3)`, not a faithful spinor action. The local Q8 certificates sharpen the
architecture consequence: conjugation exposes four actions, whereas
quaternion left multiplication exposes all eight and lets central `-1` act as
`-I`. A quaternionic left-action block is therefore a valid center-sensitive
`Spin(3)` control or subgroup ablation. It is not a drop-in replacement for
the current restricted `8v + 8+ + 8-` action, whose state and representation
contract are different.

The exact Spin(9) local theorem supplies a second reusable principle. Its
normal quotient is `V1 + 2V5`, and symmetry permits a coupled `2x2` Hessian on
the two equivalent `V5` copies. The branching theorem likewise factors an
information operator into small multiplicity-space matrices tensored with
irreducible identities. This supports multiplicity-space recurrence design
when equivalent copies actually occur. It does **not** authorize mixing the
three inequivalent Spin(8) triality representations as if they were copies of
one irrep, nor does strict local D-optimality imply language-model quality.

The benchmark audit reaches the same practical boundary from the ML side:
complete isotypic mixing and independent write/erase are justified candidates,
while triality superiority, static rotors, and synthetic mechanism accuracy
are not. Pure Spin v1.2 already keeps independent retention and drive. Its
remaining quality-oriented extension is a matched, opt-in Schur-legal mixer
over genuine channel multiplicities; it must earn promotion on natural data
and must preserve the affine composition law. It was not silently inserted
into this throughput-focused compiler change.

## Implemented affine compiler

For token transition

\[
h_t = a_tR_th_{t-1}+d_t,
\]

write its affine summary as `(a_t,R_t,d_t)`. Chronological composition is

\[
(a_2,R_2,d_2)\circ(a_1,R_1,d_1)
=
(a_2a_1,\ R_2R_1,\ d_2+a_2R_2d_1).
\]

This law is associative. `chunk_parallel_scan.py` therefore:

1. materializes the ordered factor action for any supported subgroup;
2. compiles every local affine prefix in parallel across chunks;
3. scans chunk endpoints with the ordered work-efficient tree;
4. shifts the endpoint prefixes to obtain every incoming chunk state; and
5. applies all local prefixes in parallel and removes identity padding.

The public `chunk_parallel` model backend supports non-power-of-two and
non-divisible lengths. Four float64 semantic fixtures cover lengths 1, 5, 8,
and 9, chunk sizes 1, 2, and 4, and gradients with respect to coordinates,
retention, drive, and initial state. Maximum accepted discrepancies are at the
`1e-11` scale. A full-model output and parameter-gradient parity test is also
maintained.

## CUDA result

At `(B=8,L=256,C=2,F=28)` on the RTX 2070 SUPER, the full token tree requires
766 batched `8x8` action compositions. The chunk compiler reduces this to:

| Chunk | Chunks | Compositions | Fraction of full tree | Forward vs full tree | F+B vs full tree |
|---:|---:|---:|---:|---:|---:|
| 4 | 64 | 382 | 0.499 | **1.161x faster** | **1.131x faster** |
| 8 | 32 | 318 | 0.415 | 0.972x | **1.099x faster** |
| 16 | 16 | 286 | 0.373 | **1.091x faster** | 0.967x |

The best final eager forward and complete forward/backward rows are both chunk
size 4. Reduced algebraic work does not map linearly to latency because longer
local-prefix chains increase launch and autograd depth. The timing ordering
between chunks 4 and 8 also moved across exploratory replays, so the artifact's
five-repeat medians are a local compiler gate rather than a universal tuning
rule.

The decisive negative result is comparison with the maintained factor-direct
raw kernel. Chunk size 4 is 6.33x slower forward and 7.99x slower
forward/backward. Materializing three trainable `8x8` actions and retaining
their autograd graph overwhelms the reduced scan-tree work. The eager compiler
is the semantic oracle, not the promoted training backend.

## Isotypic scheduling falsifier

A second CUDA backend scheduled `8v`, `8+`, and `8-` as separate forward
warps and reduced their private coordinate/retention gradients afterward. It
passed output and complete-gradient parity for 3, 6, 15, and 28 factors.
Forward improved for all four blocks, but its private-gradient reduction made
complete forward/backward slower:

| Factors | Split-forward speedup | Split forward+backward slowdown | Hybrid forward+backward speedup |
|---:|---:|---:|---:|
| 3 | 1.058x | 1.048x | **1.033x** |
| 6 | 1.017x | 1.032x | **1.032x** |
| 15 | 1.057x | 1.050x | **1.009x** |
| 28 | 1.068x | 1.035x | **1.026x** |

This led to the phase-specific `raw_cuda_hybrid` backend: isotypic-split
forward plus packed-warp backward. Across four alternating complete-step
cycles, the packed backend's median of cycle medians was 80,069 tokens/s and
the hybrid reached 81,301 tokens/s, a **1.015x** gain. Both reached the same
fixed-batch training loss. Hybrid is now the recommended raw training schedule;
the packed backend remains the semantic and performance control.

The result is a hardware schedule improvement, not a new mathematical model or
quality claim. Exact isotypic separation is mathematically valid, but the best
physical schedule on Turing differs between forward and reverse mode.

## Fused continuation

The next kernel must preserve the semantic oracle while removing its measured
costs:

1. fuse factor-to-local-affine compilation so PyTorch never stores the full
   per-factor action tape;
2. target chunk sizes 4 and 8, the only locally competitive rows;
3. fuse the endpoint scan, incoming-state shift, and local expansion;
4. implement controller/table gradients, not only initial-state backward;
5. keep packed triality execution unless a different physical layout wins a
   measured control; and
6. compare the fused result directly with `raw_cuda_factorized`, including
   complete optimizer steps and natural-data quality.

On Turing, FP16 WMMA may accelerate action summaries but changes the numerical
contract. It requires an explicit FP32-accumulation parity and training-quality
gate; Tensor Core use is not assumed merely because the matrices are small.

## Replay

```bash
PYTHONPATH=.. python -m pytest -q \
  test_chunk_parallel_scan.py test_model.py test_raw_cuda_training.py

PYTHONPATH=.. python benchmark_chunk_parallel_scan.py \
  --batch 8 --length 256 --channels 2 --factors 28 \
  --chunk-sizes 4 8 16 --warmups 2 --repetitions 5 \
  --output artifacts/chunk_parallel_scan_rtx2070s_20260821.json

PYTHONPATH=.. python benchmark_raw_cuda_isotypic.py \
  --output artifacts/raw_cuda_isotypic_rtx2070s_20260821.json

PYTHONPATH=.. python benchmark_spin_backend_steps.py \
  --cycles 4 --windows 5 --steps-per-window 10 --warmup-steps 10 \
  --output artifacts/spin_hybrid_vs_packed_complete_steps_rtx2070s_20260821.json
```
