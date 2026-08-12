# Fixed-plane orthonormality for the symmetric Spin(9) sensor

**Exact theorem note — 2026-08-07**
**Status:** exact global optimum on every fixed plane of the symmetric curve
**Dependencies:** [frame-operator reduction](SPIN9_FRAME_OPERATOR_REDUCTION.md)
and [curve stabilizer theorem](SPIN9_SYMMETRY_BRANCHING.md)

## Abstract

The symmetric Spin(9) three-spinor curve was originally optimized only among
orthonormal frames. This note removes every nonorthogonal deformation that
remains inside the same spinor three-plane. For each interior plane \(W(c)\)
on the curve, its orthogonal projector \(P_W\) uniquely maximizes the
information determinant over all positive-semidefinite trace-three frame
operators supported on \(W\). By Schur--Horn, this is exactly the set of frame
operators obtainable from three unit probes in that plane.

The proof uses the full \(\mathfrak{so}(3)\) plane stabilizer, invariance of the
information map, concavity of \(\log\det\), and the exact nine-dimensional
kernel of the frame-to-information map. It closes the fixed-plane
orthonormality gate, but not the global comparison against frames supported on
other three-planes.

## 1. Feasible frames on one plane

Fix an interior curve plane \(W\subset\mathbb R^{16}\) and let \(P_W\) be its
orthogonal projector. The complete supported frame domain is

\[
\mathcal D_W
=\{M\succeq0:\operatorname{im}M\subseteq W,\ \operatorname{tr}M=3\}.
\]

Every member has rank at most three. Conversely, the Schur--Horn argument in
the frame-operator theorem shows that every positive-semidefinite rank-at-most
three matrix of trace three is a sum of three unit rank-one projectors. Thus
\(\mathcal D_W\) is not a relaxation: it is exactly the three-unit-probe domain
with support constrained to \(W\).

## 2. Stationarity from the plane stabilizer

Let

\[
F(M)=\log\det I(M)
\]

where the information operator is positive definite. The map \(I\) is linear
in \(M\), and \(F\) is invariant under Spin(9) conjugation.

The curve-stabilizer theorem proves that the stabilizer of \(W\) contains the
full \(\operatorname{SO}(W)\cong\operatorname{SO}(3)\), acting irreducibly on
\(W\). Since \(P_W\) is fixed by this action, the restriction of
\(\nabla F(P_W)\) to \(W\) commutes with every element of
\(\operatorname{SO}(W)\). The real Schur lemma therefore gives

\[
P_W\nabla F(P_W)P_W=\alpha P_W
\]

for some scalar \(\alpha\). Every tangent direction
\(D\in\operatorname{Sym}(W)\) to \(\mathcal D_W\) has trace zero, so

\[
DF(P_W)[D]
=\operatorname{tr}(\nabla F(P_W)D)
=\alpha\operatorname{tr}D
=0.
\]

Thus the orthonormal projector is stationary against *all* supported
nonorthogonal frame deformations, not only an equicorrelation slice.

## 3. Concavity makes the optimum global

The function \(\log\det\) is concave on the positive-definite cone, and
\(M\mapsto I(M)\) is linear. Therefore, for every \(M\in\mathcal D_W\) with
positive-definite information,

\[
F(M)
\le F(P_W)+DF(P_W)[M-P_W]
=F(P_W),
\]

because \(\operatorname{tr}(M-P_W)=0\). If \(I(M)\) is singular, its
determinant is zero and the same determinant inequality follows by continuity.

## 4. Why equality is unique

Strictness could fail only if

\[
I(M-P_W)=0.
\]

The exact frame theorem identifies the kernel of the information map as

\[
\ker I=\operatorname{span}\{P_0,\ldots,P_8\}.
\]

Every nonzero element \(P(v)=\sum_i v_iP_i\) of this kernel satisfies

\[
P(v)^2=\lVert v\rVert^2I_{16}
\]

and hence has rank \(16\). In contrast, \(M-P_W\) is supported on the
three-dimensional space \(W\), so it has rank at most three. Their intersection
is therefore zero. The information map is injective on supported differences,
and strict concavity gives

\[
\boxed{
\det I(M)<\det I(P_W)
\quad\text{for every }M\in\mathcal D_W\setminus\{P_W\}.
}
\]

## 5. Exact deformation check at \(c=0\)

As an independent algebraic regression check, take the realizable supported
eigenvalue deformation

\[
M(r)=B\operatorname{diag}(1+2r,1-r,1-r)B^{\mathsf T}
\]

at \(c=0\). Exact FLINT-assisted interpolation and direct rational determinant
evaluation give

\[
\frac{\det I(M(r))}{\det I(P_W)}
=-\frac{(r-1)^{21}(r+2)^5(2r+1)^8(5r+4)^2}{512}.
\]

At the orthonormal point,

\[
\left.\frac{d}{dr}\log\det I(M(r))\right|_{r=0}=0,
\qquad
\left.\frac{d^2}{dr^2}\log\det I(M(r))\right|_{r=0}
=-\frac{459}{8}.
\]

This one-variable identity is not needed for the theorem; it is a concrete
falsifier that agrees with the invariant concavity proof.

## 6. What remains open

This theorem excludes all nonorthogonal reweightings supported on the
candidate plane. It does not compare the candidate against a nonorthogonal
frame on a different plane. The unrestricted global proof still needs either:

1. a global inequality that replaces every feasible frame by an orthonormal
   projector without decreasing the determinant; or
2. an invariant certificate over the full eight-dimensional frame-objective
   quotient.

The orthonormal Grassmannian subproblem is only three-dimensional modulo
Spin(9), but the action is nonpolar, so the known one-dimensional curve does
not exhaust it.
