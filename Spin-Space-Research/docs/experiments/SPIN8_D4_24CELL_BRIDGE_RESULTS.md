# D4, the 24-cell, and the 24 triality sensors

**Date:** 2026-08-06  
**Status:** exact bridge and exact non-equivalence certificate  
**Harness:** `src/spin8_d4_24cell_bridge.py`  
**Artifact:** `artifacts/spin8_d4_24cell_bridge_20260806.json`

## Three meanings of D4

This repository sits near three objects that are often given the same label:

1. the `D4` root system, whose 24 roots are the vertices of a regular
   24-cell;
2. Dynkin type `D4`, the type of `so(8)`, whose order-three outer symmetry is
   Spin(8) triality;
3. the legacy recurrence benchmark called `D4`, meaning the non-Abelian
   dihedral group of order eight.

Only the first two are directly related. The third is a small finite-state
benchmark and is not a root system.

## The exact classical bridge

Scale all weights by two. The three eight-point minuscule weight orbits are

- `8v`: the vectors `+-2e_i`;
- `8s+`: the eight half-cube words with even minus parity;
- `8s-`: the eight words with odd minus parity.

Their union has 24 unit vectors after undoing the factor of two. Its unordered
pair inner products are exactly

| inner product | pairs |
|---:|---:|
| `-1` | 12 |
| `-1/2` | 96 |
| `0` | 72 |
| `1/2` | 96 |

This is a standard 24-cell realization. The verifier also checks all 126
coordinate monomials through degree five and proves that their discrete
averages equal the corresponding spherical averages on `S^3`. Thus the set is
an exact spherical 5-design.

The matrix

\[
H=\begin{pmatrix}
1&1&1&1\\
1&1&-1&-1\\
1&-1&1&-1\\
-1&1&1&-1
\end{pmatrix},\qquad T=H/2,
\]

satisfies `TT^T=I` and `T^3=I` exactly. It cycles

\[
8_v\longrightarrow 8_{s^-}\longrightarrow 8_{s^+}\longrightarrow 8_v.
\]

This makes the `D4`--triality--24-cell connection explicit rather than
relying on the repeated number 24.

## The crucial non-equivalence

The repository's 24 **coordinate probes** are not these 24 points on `S^3`.
Each probe produces a rank-seven information projector in the
28-dimensional bivector parameter space. The sensor configuration therefore
lives in the Grassmannian `Gr(7,28)`, not in four-dimensional Euclidean space.

The exact projector geometry is

\[
\sum_{x\in\text{one coordinate basis}}P_r(x)=2I_{28},
\qquad
\sum_{\text{all 24 probes}}P(x)=6I_{28}.
\]

For two distinct probes in one view, `tr(PQ)=1`, and their seven-dimensional
subspaces intersect in one line. For probes in different triality views,

\[
PQP=\frac14P,
\qquad
\operatorname{tr}(PQ)=\frac74.
\]

Thus every cross-view pair is exactly isoclinic with squared cosine `1/4`.
The 24 sensors form a three-colour tight fusion-frame configuration.

### Non-vertex deformation audit

This tightness is not a coordinate-vertex rigidity result. Since `P_r(x)` is
quadratic in `x`, every orthonormal basis `{q_j}` in one view obeys

\[
\sum_jP_r(q_j)=2I_{28}.
\]

The exact artifact includes a non-coordinate rational witness obtained by the
`3/5`--`4/5` rotation of the first two basis vectors. Its projector sum is
still exactly `2I_28`. Thus the tight configuration belongs to a continuous
orthonormal-basis deformation family.

### Standard packing audit

Tight does not mean optimally packed. The complete 24-subspace family is not
equi-isoclinic: distinct same-view subspaces intersect in a line, giving
spectral coherence one. Its minimum squared chordal distance is

\[
7-\frac74=\frac{21}{4},
\]

while the fusion-frame simplex bound for `(d,r,n)=(28,7,24)` is

\[
\frac{nr(d-r)}{d(n-1)}=\frac{126}{23}>\frac{21}{4}.
\]

The coloured triality incidence law may still be interesting. Ordinary
uncoloured **spectral** packing optimality is ruled out: coherence one is
worse than the strictly subunit coherence of a generic pairwise-transverse
finite configuration. The chordal calculation proves only that this frame
does not attain the simplex upper bound. Without a separate attainability
result at `(d,r,n)=(28,7,24)`, it does not by itself rule out chordal
optimality.

## Plain-language version

The classical 24-cell is a set of 24 arrows in four dimensions. The sensor
construction is a set of 24 seven-dimensional sheets inside a
28-dimensional room. Triality organizes both collections into three groups
of eight, but arrows and sheets are not the same thing.

The surprising sensor law is that a sheet from one colour meets every sheet
from another colour at exactly the same angle. Eight sheets of any one colour
also balance perfectly: their projectors add to a multiple of the identity.

## Relation to non-universal optimality

Cohn, Conway, Elkies, and Kumar proved that the `D4` 24-cell is a spherical
5-design yet is not universally optimal: a deformation has lower energy for
the absolutely monotone potential `(1+t)^8`. That warns against turning
symmetry or finitely many exact moments into a global optimum claim.

This certificate does **not** say that either the 24-cell or the sensor fusion
frame is universally optimal. It says exactly what the common `D4` origin
explains, and exactly where the two geometries diverge.

## Sources and novelty boundary

- [The D4 root system is not universally optimal](https://arxiv.org/abs/math/0607447)
- [Exploring Triality Explicitly](https://arxiv.org/abs/2502.14016)
- [Binary encoding of spinors and Clifford multiplication](https://arxiv.org/abs/1905.10613)
- [Totally symmetric Grassmannian codes](https://arxiv.org/abs/2406.19542)

The minuscule-weight and 24-cell connection is classical. The result to
investigate for publication priority is the complete coloured `Gr(7,28)`
projector configuration and its role in triality sensor design. No novelty
claim for that formulation is made until a dedicated prior-art search is
complete.

## Replay

```powershell
$env:PYTHONPATH='src'
python -m spin8_d4_24cell_bridge `
  --output artifacts/spin8_d4_24cell_bridge_20260806.json
python -m unittest discover -s tests `
  -p "test_spin8_d4_24cell_bridge.py" -v
```
