# GA-SSM foundational review, 2026-08-03

## Honest conclusion

The project contains a valid noncommutative affine SSM, exact recurrent state,
real scan/streaming parity, and an unusually rigorous finite-group compiler
program. It does not yet contain evidence of a superior language model, and a
single chiral Spin(8) stream has now been shown to have the same SO(8) action
capacity as a generic skew-matrix chart.

The most important new finding in this review is below the Spin(8) layer: the
maintained Cl(3) network implemented only half of its claimed symmetry-allowed
linear maps. That is now corrected constructively by the isotypic layer and the
representation-factored SchurScan.

## What is mathematically solid

- The affine transition `(d,R,u)` is closed under ordered composition. A prefix
  scan needs associativity, not commutativity.
- A strict uniform contraction and bounded write imply BIBO state stability.
- Rotor conjugation is norm preserving and grade preserving.
- Parallel floating-point trees and recurrent evaluation are mathematically the
  same recurrence, though not bitwise identical.
- Per-layer recurrent states are exposed end to end; token streaming uses fixed
  cache independent of context length.
- Spin-group center information is destroyed by sandwich actions and retained
  by spinor actions. The Q8 center falsifier is structurally valid.
- One positive half-spin chart and a generic SO(8) exponential chart are
  orthogonal coefficient-basis changes of the same 28-dimensional action
  family. Any single-stream difference under coordinatewise Adam is an
  optimizer-chart effect, not a capacity theorem.

## Foundational correction: grades are not irreducible types

For the claimed proper-rotation symmetry,

```text
scalar and pseudoscalar = two trivial copies
vector and Hodge-dual bivector = two standard 3D copies.
```

Consequently, grade preservation is stronger than Spin(3) equivariance. The
old `GradeLinear` set all cross-copy intertwiners to zero. The exact audit found
centralizer dimension 8 versus rank 4 for the old family. `Spin3IsotypicLinear`
now supplies the complete 8-dimensional one-channel commutant in both JAX and
PyTorch-compatible experimental code.

This also corrects the interpretation of the earlier grade-decay result.
Grade-specific damping did not destroy grade preservation. It imposed an
arbitrary diagonal basis on a multiplicity space whose equivalent copies may
need to exchange information. The likely failure is optimization and
anisotropy within that repeated-irrep space, not loss of the stated symmetry.

## New architecture: SchurScan

The principled state coordinate system is not “a bag of Clifford grades.” It is
an isotypic decomposition into representation space and multiplicity space.
For real-type irreps, use token actions

```text
M_t,lambda tensor rho_lambda(g_t).
```

`rho(g_t)` carries shared geometric phase; `M_t` carries content routing,
retention, and mixing between equivalent copies. Their products remain in the
same factored family, permitting an exact associative affine scan without a
full dense state matrix. For real representations of complex or quaternionic
type, `M_t` must be generalized to the corresponding Schur division algebra.

This construction unifies the useful part of geometric algebra (known group
representations and invariant tensor structure) with the useful part of a
selective dense SSM (noncommutative token-conditioned state routing). It also
explains why independently rotating isolated channels is too restrictive:
phase exists, but transition-time content mixing does not.

## Code review adjudication

### Real issues fixed

- Both group-action evaluators averaged batch means. They now accumulate summed
  cross entropy and divide by the exact prefix-label count. Historical cohorts
  used equal batches, so archived headline values are unchanged.
- `sqrt(1-decay^2)` used subtractive cancellation. It now evaluates
  `sqrt(-expm1(-2 Delta lambda))` in JAX and PyTorch.
- The maintained linear layer omitted half of the Spin(3) commutant. The
  isotypic implementation and tests now expose and repair that capacity gap.

### Reviewer claims rejected

- A fixed-key JAX permutation converted to NumPy is deterministic. JIT is not a
  requirement for PRNG correctness.
- The group-action target is the state after the same-position token, not the
  next token. Shifting it would create an off-by-one bug.
- Weight tying does not require an explicit inverse of every hidden channel
  mix. It is a modeling restriction worth ablating, not a mathematical
  inconsistency.
- `GeometricAttention` explicitly accepts `(B,L,8)`, not multichannel
  `(B,L,C,8)`. Its output shape does not silently discard a documented channel
  axis.
- Smoothing rotor normalization with `sqrt(norm^2+epsilon)` would make rotors
  non-unit and invalidate exact norm preservation. The exponential chart is
  the correct path around the identity; a zero raw quaternion has no continuous
  normalization.
- Associative scan correctness does not require power-of-two sequence length.

## Remaining optimization faults and hypotheses

1. **Retention/write coupling.** `u_t=sqrt(1-d_t^2)P(x_t)` is a stationary-
   variance convention, not a stability requirement. It weakens writes exactly
   when retention is long. Strict contraction plus an independently bounded
   write remains BIBO-stable. This deserves the next controlled ablation and is
   independently motivated by the 2026 Naju result.
2. **Artificial geometric gauge.** Natural language supplies no physical 3D
   frame. Equivariance can still regularize, but the frame must be treated as a
   learned gauge, not a discovered linguistic symmetry.
3. **Per-channel transition diagonalism.** Existing rotors act independently
   per channel. Cross-channel interaction occurs around the recurrence, not in
   the transported memory itself. SchurScan directly addresses this.
4. **Controller bottleneck.** Rotor direction is produced from a grade-linear
   projection and strength from four coarse invariants per channel. This may
   be too weak to select task-relevant operators even when the transition
   family has capacity.
5. **Adaptive-optimizer coordinates.** Adam is not invariant under a general
   orthogonal coefficient-basis change. Chart comparisons require SGD/natural-
   gradient controls or explicit optimizer equivariance audits.
6. **Kernel reality.** The PyTorch Cl(3) reference is a Python recurrence and
   Spin(8)'s Hillis-Steele implementation is an `O(N log N)`-work oracle. A
   production claim requires fused `O(N)`-work scan kernels and throughput
   measurements.
7. **Evidence scale.** The language result remains three seeds, 300 steps,
   context 64, on a tiny byte model. It is a mechanism hint only.

## Legacy `SpinorModel`

`SpinorModel/GALLM_1.py` is a conventional Transformer with dense layers and a
function named `geometric_product` that concatenates a dot product and a full
antisymmetric outer difference; it is not a closed Cl(3) implementation.
`geometric_layers.py` is a cleaner historical GA tensor implementation, but its
componentwise GELU and ordinary LayerNorm do not preserve the maintained
Spin(3) contract. These files should remain labeled historical prototypes. New
research belongs in the tested `SSM-Models` core.

## Literature calibration

- Clifford Group Equivariant Neural Networks establishes grade-respecting
  Clifford-group layers and geometric-product compatibility:
  https://arxiv.org/abs/2305.11141
- Selective Dense SSM demonstrates the need for dense token-conditioned
  transitions on regular languages: https://arxiv.org/abs/2412.19350
- DeltaProduct shows that products of Householder updates are a serious
  efficient expressivity baseline: https://arxiv.org/abs/2502.10297
- The ICLR 2026 diagonal-SSM result formalizes finite-precision non-Abelian
  state-tracking limits: https://openreview.net/forum?id=5bg5Ru5OML
- Naju separates retention and writing in a stable discrete recurrence:
  https://arxiv.org/abs/2607.21000

These sources cover the ingredients separately. They do not, from the search
performed for this review, establish the exact representation-factored
selective affine scan implemented here. That is a candidate novelty, not yet a
priority or publication claim.

## Triality law identification and memory boundary

The coupled triality branch has now crossed three distinct gates.

First, the implemented infinitesimal equivariance constraints on
Hom(S+ tensor S-, V) have nullity one and select the Clifford triality tensor
at cosine 1.0. Under a rank-16-of-64 observation design, the resulting
one-parameter invariant family generalizes perfectly in 3/3 seeds. Generic
bilinear and MLP families fit the same observations below 1.2e-6 MSE but fail
on continuously perturbed sources and unseen generator paths. This is
symmetry-driven identification of a supplied-action law, not a language result.

Second, the project explicitly rejects a high-capacity claim for one 8D
triality vector. Wrong-key bind--unbind terms retain full norm. Exact
multi-item retrieval comes from orthogonal multiplicity codes for K at most H.
Beyond that rank horizon, unit-norm tight frames attain the classical optimal
average interference bound (K-H)/H; they do not evade it.

Third, a shared-basis retention transition upgrades the static code to an
exact addressed dynamic memory. It overwrites one slot, transports every
retained slot by the shared Spin(8) action, and composes as an associative
affine scan. A length-128 stress test stays within 2.3e-15 of serial and
symbolic oracles.

Hiratani and Sompolinsky already established octonion quadratic binding,
multi-pair degradation, and expanded composition layers in 2022:
https://arxiv.org/abs/2204.07186. Tight frames are classical as well. Gated
DeltaNet-2 and Erase-then-Delta Attention already provide separated or
independently addressed erase/write operations with efficient recurrent
memory:

- https://arxiv.org/abs/2605.22791
- https://arxiv.org/abs/2606.26560

The candidate contribution is therefore narrower: integration of unique
cross-irrep identifiability, optimal multiplicity coding, shared Spin(8)
transport, exact overwrite, and parallel-recurrent SSM duality.

## Best next experimental order

1. Freeze a learned-address selective-copy or MQAR task. Compare the exact slot
   oracle against a same-width direct-slot memory, linear attention or
   bilinear fast weights, Householder transport, and diagonal complex SSMs.
2. Learn addresses and query keys without weakening the exact recurrent-state
   contract; distinguish address discovery from value transport.
3. Compare SchurScan against grade rotor, dense selective, Householder-product,
   and diagonal real/complex transitions under equal state, decoder, and
   compute budgets.
4. Orthogonally ablate complementary versus independent writes and nonlinear
   cleanup outside the scan.
5. Only after those gates, integrate the mechanism into the language model.
