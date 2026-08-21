# Isotypic-to-silicon compiler v2.1.1

## Result

Version 2.1.1 connects three previously separate layers:

1. an exact isotypic certificate supplies the irreducible real dimension,
   Schur type, and multiplicity;
2. a runtime contract states whether the same group action is shared across
   those copies; and
3. a hardware profile chooses an eager, scalar Triton, or Tensor-Core Triton
   realization for the exact runtime shape.

The compiler refuses to infer shared action from equal representation type.
Independently routed Spin(8) channels are not one isotypic matrix-multiply,
while repeated memories transported by one physical action are. Plan JSON
records the certificate and profile hashes and has a stable fingerprint.

The maintained Pure Spin(8) layer now has an opt-in `compiled_recurrent` mode.
The model version remains 1.1 because its checkpoint schema did not change;
2.1.1 is the compiler version.

## Constructive self-calibration

Let `P` be the `7 x 8` matrix whose rows are the images of seven ordered
orthonormal vector probes. The missing oriented row is the Hodge cofactor

\[
 q_j=(-1)^{7+j}\det P_{\widehat j},\qquad q\leftarrow q/\lVert q\rVert.
\]

Stacking `P` and `q` reconstructs the vector action in `SO(8)`. An adjacent-
Givens QR factorization gives a concrete product of plane rotations. The same
angles are evaluated in the maintained `8v`, `8s+`, and `8s-` generators. One
supplied lift bit selects the kernel of `Spin(8) -> SO(8)`: it leaves `8v`
fixed and changes the sign of both half-spin actions. Float64 reconstruction
residuals in the recorded audit are `1.23e-15` or smaller.

This is a charted constructive lift, not a global continuous canonicalization.
Rank-deficient probes, orientation boundaries, and the double-cover bit remain
explicit parts of the interface. That boundary is consistent with the known
topological obstruction to continuous canonicalization; robust systems should
use charts, frames, or weighted alternatives instead of silently assuming one
global selector.

## CUDA and Tensor-Core audit

Artifact:
[`spin8_compiler_v211_rtx2070s_20260821.json`](artifacts/spin8_compiler_v211_rtx2070s_20260821.json),
SHA-256 `b0a09d3aa58d3da3cb4adea73d6e5dd59466cc1e9965e8a698248def22fb7b4c`.

Hardware was an RTX 2070 SUPER (`sm_75`) under WSL2, PyTorch 2.4.0 and CUDA
12.1. The full-gradient FP32 continuous recurrence

\[
 h_t=s_t A_t h_{t-1}+d_t
\]

has gradients for `A`, `s`, `d`, and `h_0`. At `(B,L,m,R)=(4,128,4,3)`, the
fused kernel records `967.52 us` versus `108961.04 us` for the sequential
PyTorch oracle, a bounded `112.62x` speedup. Forward maximum error is
`1.08e-6`; maximum gradient error is `1.91e-5`. This compares to an eager
sequential oracle, not to a work-efficient tree, Mamba, or a complete trained
model.

The FP16 inference kernel packs the shared multiplicity axis in tiles of 16
and accumulates dot products in FP32. PTX contains `mma.sync`; the scalar
control does not. Tensor Cores win only the profiled `(B,L,m)=(32,128,16)`
cell, by `1.423x`. They lose the other seven cells. The compiler therefore
requires an exact matching profile and a 1.05 speed margin before selecting
that backend.

End-to-end self-calibration at `(4,16,4)` records only `1.059x` because Hodge
completion, Givens extraction, and three representation lifts dominate the
now-cheap recurrence. The next systems target is consequently a fused action
constructor, not a less conservative Tensor-Core heuristic.

## Relationship to external implementations

- [e3nn tensor products](https://docs.e3nn.org/en/stable/api/o3/o3_tp.html)
  expose irreducible paths and generated contraction code. Version 2.1.1
  adopts the separation between algebraic instructions and execution, but its
  recurrence and Spin(8) lift are different objects.
- [NVIDIA cuEquivariance](https://github.com/NVIDIA/cuEquivariance) separates
  equivariant descriptors from optimized CUDA implementations. This supports
  the descriptor/backend split; it does not certify this repo's exact
  decomposition or recurrence.
- [OpenEquivariance](https://github.com/PASSIONLab/OpenEquivariance) likewise
  separates tensor-product problem descriptions from JIT CUDA/HIP kernels.
  The relevant lesson is backend specialization, not inherited performance.
- [Triton matrix multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
  and [`tl.dot`](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)
  provide the mechanism used to obtain real `mma.sync` instructions. The
  compiler still verifies PTX and measures the local shape.
- [Mamba](https://arxiv.org/abs/2312.00752) demonstrates that scan algebra and
  hardware-aware implementation must be designed together. No Mamba quality
  or throughput comparison is implied by the local eager-oracle timing.
- [LieTorch](https://github.com/princeton-vl/lietorch) is a useful precedent
  for differentiating through Lie-group computations in tangent coordinates.
  This implementation instead exposes a charted Spin(8) lift and its center
  bit.
- Learned canonicalization is useful but cannot be assumed globally:
  [Kaba et al.](https://arxiv.org/abs/2211.06489) construct learned
  canonicalizers, while [Dym et al.](https://arxiv.org/abs/2402.16077) prove
  broad impossibility results for continuous canonicalization and motivate
  weighted frames.

Version 2.1.1 therefore adopts proven implementation patterns without
inheriting claims their abstractions do not justify.

## Reproduction

From `SSM-Models`:

```powershell
python -m pytest -q test_spin8_compiler_v211.py test_pure_spin8_ssm.py
wsl bash -lc 'cd /mnt/c/Users/HaydenLocal/Programming/AI_Culture_Mind/SSM-Models && PYTHONPATH=. python3 -m pytest -q test_spin8_compiler_v211.py test_pure_spin8_ssm.py'
wsl bash -lc 'cd /mnt/c/Users/HaydenLocal/Programming/AI_Culture_Mind/SSM-Models && PYTHONPATH=. python3 benchmark_spin8_compiler_v211.py'
```

## Claim boundary and design conclusion

Established here: typed certificate ingestion, refusal of false multiplicity,
profile-guarded dispatch, an actual Tensor-Core instruction path, full-gradient
continuous recurrence, and a differentiable charted seven-probe Spin(8)
application.

Not established: field novelty, global continuous lift selection, noisy-probe
conditioning, complex/quaternionic Tensor-Core lowering, cross-device policy,
action-constructor fusion, natural-task benefit, or superiority to modern SSMs.

The strongest design conclusion is conditional but hard to escape: an
equivariant compiler must carry algebraic type, action-sharing semantics,
precision, and measured hardware cost as separate proof obligations. Any one
of them erased too early permits an invalid or slower lowering.
