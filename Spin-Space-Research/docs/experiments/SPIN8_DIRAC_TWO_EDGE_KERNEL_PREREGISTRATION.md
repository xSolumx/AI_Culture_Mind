# Two-Edge Equality-Kernel Preregistration

**Frozen:** 2026-08-06, after complete sector reconstruction and the principal
orthonormal transverse theorem, before nonvertex equality classification.

## Question

Does the new residual preserve positivity infinitesimally at every equality
stratum of the proved variable-Cayley one-edge theorem?

The complete two-edge orientation matrices have the exact form

\[
K_\pm(i)=K_0\pm iL_1+i^2L_2+O(i^3),
\]

where `K_0` is the proved one-edge group-circulant. If `v` is in the kernel of
`K_0`, positivity for both residual signs requires

\[
v^T L_1v=0.
\]

If this fails at any exact feasible point, one sign of sufficiently small `i`
has negative margin and the two-edge conjecture is false.

## Already known before this protocol

- all eight sector coefficient maps are exact and replayable;
- the even and odd sectors use the same exact order-four Hadamard table;
- the principal orthonormal equality line has strictly positive second-order
  margin transverse to `i=0`;
- all 32 coordinate vertices and 256 orientation margins have been audited;
  their 16 equality rows have zero odd first derivative.

Those facts are prior evidence, not prospective outcomes of this protocol.

## Frozen gates

### Gate 1: reconstruct `K_0`, `L_1`, and `L_2` exactly

- derive all entries from the published coefficient maps and certified forced
  character monomials;
- verify `K_0` against the independent published one-edge polynomials;
- verify the series against fresh direct determinants at disjoint rational
  points;
- store matrices or sparse coefficient maps, not only hashes or pass flags.

### Gate 2: classify equality components of `K_0`

- begin with the exact principal minors used in the one-edge proof;
- factor known coordinate and Cayley boundary components;
- use exact ideal decomposition, saturation, or resultants for residual
  components;
- use numerical optimization only to discover candidates, never to declare the
  equality set complete.

The classification must distinguish the full matrix kernel from zeros of an
individual redundant principal-minor factor.

### Gate 3: first-order kernel obstruction

For every certified feasible equality component:

- construct an exact basis for `ker(K_0)` on a generic point and all rank-drop
  subcomponents;
- compute the compression `V^T L_1 V`;
- require it to vanish identically for both signs.

Any nonzero exact entry is a stopping result. Rationalize an interior witness,
verify the direct determinant reversal exactly, and publish the counterexample.

### Gate 4: second-order Schur quotient

Only if Gate 3 passes, compute the effective second-order kernel form

\[
V^T\left(L_2-L_1K_0^+L_1\right)V,
\]

using a block inverse on the range of `K_0`, not a floating-point
pseudoinverse. Require exact positive semidefiniteness on each component.

### Gate 5: global positivity machinery

Only after the local gates pass may the project attempt paired principal-minor,
Schur-complement, Duffy/Bernstein, or SOS certificates on the complete cube.
Individual Walsh-sector positivity is explicitly forbidden as a target because
opposite-signed exact faces already disprove it.

## Promotion boundary

Passing this protocol proves local compatibility at every classified one-edge
equality stratum. It does not prove global two-edge positivity. The full theorem
still requires a certificate away from the equality set.

