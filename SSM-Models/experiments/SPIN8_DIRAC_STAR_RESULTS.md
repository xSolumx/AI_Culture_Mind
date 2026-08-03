# Exact Signed Star-Family Spin(8) Dirac--Gram Theorem

**Date:** 2026-08-04

**Preregistration:** `SPIN8_DIRAC_STAR_PREREGISTRATION.md`

**Harness:** `../spin8_dirac_star.py`

**Raw exact artifact:** `spin8_dirac_star_20260804.json`

**Artifact SHA-256:**
`c4fd00315ab83a16f2d0ed267b2534cd4fbe77e2dc445882bac857e0f8ae5dc2`

## Result

The strengthened Dirac--Gram inequality

\[
\det I(X)\leq \det(XX^T)^3\det I(Q)
\]

is proved over the complete signed four-parameter star family frozen in the
preregistration. This is the first exact multivariate result in this project
with three simultaneous frame correlations. It includes both orientations of
the Cayley coordinate. It is not the unrestricted seven-invariant theorem.

## Family and exact reduction

For

\[
\begin{aligned}
x_1&=q_1,\\
x_2&=a q_1+\sqrt{1-a^2}\,q_2,\\
x_3&=d q_1+\sqrt{1-d^2}\,q_3,\\
x_4&=g q_1+\sqrt{1-g^2}\,q_4,
\end{aligned}
\]

with `u=a^2`, `v=d^2`, `w=g^2`, `z=c^2`, exact determinant algebra gives

\[
\frac{1024\det I(X)}{\Delta^3}
=F(u,v,w,z)+(adg\Phi)H(u,v,w,z),
\qquad
\Delta=(1-u)(1-v)(1-w).
\]

The independently reconstructed exact polynomials have degrees and term counts

| polynomial | degree in `(u,v,w,z)` | nonzero terms |
|---|---:|---:|
| `F` | `(3,3,3,5)` | 360 |
| `H` | `(2,2,2,4)` | 86 |

The orientation dependence has therefore not been discarded by squaring the
Cayley coordinate. It is isolated in the single signed term `adg Phi H`.

## Positivity certificate

Let

\[
T(z)=(1-z)^3(9-z)^2,
\qquad A=T-F,
\]

and define

\[
Q=A^2-uvw(1-u)(1-v)(1-w)zH^2.
\]

The exact tensor-product Bernstein expansions on `[0,1]^4` have:

| certificate | Bernstein degree | negative coefficients | zero coefficients |
|---|---:|---:|---:|
| `A` | `(3,3,3,5)` | 0 | 195 |
| `Q` | `(6,6,6,10)` | 0 | 2078 |

Every Bernstein basis function is nonnegative on the box. Hence `A>=0` and
`Q>=0`. The second inequality implies

\[
A\geq
\sqrt{uvw(1-u)(1-v)(1-w)z}\,|H|
=|adg\Phi H|.
\]

Therefore `T >= F + adg Phi H` for either orientation, which is exactly the
strengthened determinant inequality on this family.

Zero Bernstein coefficients are recorded as boundary structure; they are not
silently upgraded into a complete equality classification.

## Why this is an exact certificate

The proof did not infer a polynomial from a floating-point fit.

1. Cauchy--Binet degree counting and the three rank-loss factors give the
   conservative bounds `deg(F)<=(4,4,4,7)` and
   `deg(H)<=(3,3,3,6)`.
2. The discovery grid used enough rational-circle nodes to determine every
   coefficient allowed by those bounds.
3. A disjoint rational-circle grid reconstructed the coefficient maps again.
4. The two canonical rational coefficient serializations were required to
   agree exactly.
5. Sixteen additional rational frames, each in both orientations, supplied 32
   exact off-grid determinant identities.
6. All Bernstein signs were evaluated as exact rational numbers.

Any mismatch, negative coefficient, or nonzero holdout residual was a frozen
failure condition.

## Structural correction to the global proof strategy

The coupled whitening flow preserves

\[
c=\Phi/\sqrt{\det G}
\]

exactly, but the strengthened ratio requires

\[
\frac{d}{dt}\log\det I\geq6\|G-I\|_F^2.
\]

The previously observed adversarial normalized derivative
`5.9214371088` is below six. Thus the flow cannot prove the strengthened cubic
bound, even though its plain determinant derivative remained positive in the
campaign. This correction is now explicit in `SPIN8_DIRAC_GRAM_RESULTS.md`.

The exact replacement is the Dirac--Schur reduction

\[
\det I=2^7 32^{-21}\det(8T-SS^T),
\]

derived from four graph-isometric seven-frames. It exposes the coupled
21-dimensional operator on which the residual proof should be organized.

## Honest boundary

A general four-frame has three residual Cholesky correlations absent from the
star family. Those variables can couple the four graph frames and are not
controlled by this certificate. Consequently:

- signed star-family theorem: **proved exactly**;
- strengthened global Dirac--Gram theorem: **open**;
- global five-query D-optimality of `81/1024`: **open**.

The next legitimate gate is a conditional-decorrelation lemma: at fixed star
coordinates and normalized Cayley value, show that introducing the three
residual Cholesky correlations cannot increase the normalized Dirac ratio (or
produce an exact counterexample). Proving that lemma in the Schur operator
would lift this theorem to general frames. More random frames would not
strengthen the present result.
