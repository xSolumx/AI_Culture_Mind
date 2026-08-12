# Exact conditioning along a symmetric three-spinor Spin(9) curve

**Theorem note — 2026-08-07**
**Status:** exact theorem on a complete one-parameter family
**Certificate:** [`spin9_three_spinor_conditioning.py`](../../src/spin9_three_spinor_conditioning.py)

## Abstract

Three generic real spinors are the sharp probe count for identifying a shared
\(\operatorname{Spin}(9)\) action. This note addresses the next question:
conditioning. On an explicit one-parameter curve of mutually orthonormal
spinor triples whose quadratic Hopf images are equiangular with common inner
product \(c\), the \(36\times36\) infinitesimal information determinant is

\[
\det I(c)
=\frac{(1-c)^{10}(c+2)^5(2c+1)^3}{2^{43}},
\qquad -\frac12<c<1.
\]

It has the unique maximizer

\[
c_\star=\frac{\sqrt{241}-17}{24}
\approx-0.0614927209891657.
\]

The result explains an independently observed GPU optimum to machine
precision. The complete characteristic polynomial factors into two quadratics
and one quartic with multiplicities \(7,5,3\). The result is exact on the
stated curve. It does not prove that the curve exhausts the equiangular locus,
nor global D-optimality over all spinor triples.

## 1. The symmetric family

Let \(P_0,\ldots,P_8\) be the symmetric Clifford involutions and define the
quadratic Hopf map

\[
H(s)_i=s^{\mathsf T}P_i s.
\]

For unit spinors, \(\lVert H(s)\rVert=1\). The constructed curve satisfies

\[
\langle s_i,s_j\rangle=\delta_{ij},
\qquad
\langle H(s_i),H(s_j)\rangle=c\quad(i\ne j).
\]

Positive definiteness of the Hopf Gram matrix and the explicit spinor lift give
the open parameter interval \(-1/2<c<1\). No orbit-classification claim is
made for all triples sharing these two Gram matrices.

The explicit representative is

\[
\begin{aligned}
s_1&=e_0,\\
s_2&=d e_1+b e_8,\\
s_3&=-d e_2+y e_{11}+z e_{12},
\end{aligned}
\]

where

\[
d^2=\frac{1+c}{2},
\qquad
b^2=\frac{1-c}{2},
\]

\[
y=\frac{c(1-c)}{2b(1+c)},
\qquad
z^2=\frac{(1-c)(1+2c)}{2(1+c)^2}.
\]

Direct substitution proves both Gram identities exactly.

## 2. Information operator

For the conventional generators

\[
G_{ij}=\frac12P_iP_j,
\qquad 0\le i<j\le8,
\]

let

\[
J(s)=\begin{bmatrix}G_1s&\cdots&G_{36}s\end{bmatrix},
\qquad
I(c)=\sum_{r=1}^{3}J(s_r)^{\mathsf T}J(s_r).
\]

The trace is constant:

\[
\operatorname{tr}I(c)=27.
\]

In the maintained bivector ordering, the matrix decomposes exactly into fixed
blocks of dimensions

\[
6+10+10+10=36.
\]

Their determinants are

\[
\frac{(1-c)(c+2)^2}{2^7}
\]

and three identical copies of

\[
\frac{(1-c)^3(c+2)(2c+1)}{2^{12}}.
\]

Multiplication gives

\[
\boxed{
\det I(c)
=\frac{(1-c)^{10}(c+2)^5(2c+1)^3}{2^{43}}
}.
\]

## 3. Complete characteristic polynomial

The determinant factorization is the constant term of a stronger spectral
identity. Define

\[
\begin{aligned}
Q_1(x,c)&=8x^2-8x+1-c,\\
Q_2(x,c)&=4x^2-7x+2+c,\\
Q_4(x,c)&=
16x^4-60x^3+(64+4c)x^2\\
&\hspace{3.5em}-(16+8c)x+1+c-2c^2.
\end{aligned}
\]

The six-dimensional block has characteristic polynomial

\[
\frac{Q_1Q_2^2}{2^7},
\]

and each ten-dimensional block has characteristic polynomial

\[
\frac{Q_1^2Q_2Q_4}{2^{12}}.
\]

Therefore

\[
\boxed{
\chi_{I(c)}(x)
=\frac{Q_1(x,c)^7Q_2(x,c)^5Q_4(x,c)^3}{2^{43}}.
}
\]

The leading powers multiply to \(2^{43}\), so the displayed polynomial is
monic. Its eight root branches have fixed multiplicities

\[
7,\ 7,\ 5,\ 5,\ 3,\ 3,\ 3,\ 3.
\]

Setting \(x=0\) recovers the determinant formula. The factorization explains
the complete observed eigenvalue multiplicity pattern and gives a compact
starting point for exact conditioning and endpoint-asymptotic calculations.

The powers should not be read as seven copies of a two-dimensional module,
five copies of another two-dimensional module, and three copies of a
four-dimensional module. At \(c=0\), the exact
[stabilizer certificate](SPIN9_SYMMETRY_BRANCHING.md) proves the opposite
tensor-factor interpretation:

\[
\mathfrak{spin}(9)\cong2V_7\oplus2V_5\oplus4V_3.
\]

Thus \(7,5,3\) are irreducible \(\mathfrak{so}(3)\)-module dimensions and
\(2,2,4\) are their multiplicity-space dimensions. That branching statement
is exact at the Cayley-null point. The factorization is exact on the whole
curve, but no fixed common stabilizer for the entire curve is asserted.

Because \(I(c)\) is real symmetric, positive definiteness means that all eight
root branches are positive real numbers. The weaker phrase "roots in the
right half-plane" is inappropriate here. A factor discriminant can vanish at
an eigenvalue collision inside the positive-definite region; it is not by
itself a boundary equation. On this family positivity follows directly from
the Gram construction together with the positive determinant on
\((-1/2,1)\).

## 4. Unique optimum

On the open feasible interval the determinant is positive and vanishes at both
boundary limits. Its logarithmic derivative is

\[
\frac{d}{dc}\log\det I(c)
=
\frac{3(12c^2+17c+1)}
{(c-1)(c+2)(2c+1)}.
\]

Of the two roots of the numerator, only

\[
c_\star=\frac{\sqrt{241}-17}{24}
\]

lies in \((-1/2,1)\). The derivative is positive before \(c_\star\) and
negative afterward. Hence this point is the unique global maximizer along the
complete open parameter interval of this curve.

The stationary polynomial is explicitly

\[
12c^2+17c+1.
\]

Its discriminant is \(241\), which is not a rational square. It is irreducible
over \(\mathbb Q\), splits over \(\mathbb Q(\sqrt{241})\), and is therefore
solvable by radicals. The second root \((-17-\sqrt{241})/24\) lies outside the
feasible interval.

## 5. How the result was found

A batched unconstrained search over spinor triples repeatedly converged to:

- an identity spinor Gram matrix;
- an equiangular Hopf Gram matrix;
- common correlation `-0.06149272098916542`;
- gradient norm below `1.6e-14` in a double-precision refinement.

Integer-relation recovery suggested \(12c^2+17c+1=0\). The exact block
calculation above was then derived independently. The numerical search is
discovery provenance, not part of the proof.

## 6. Remaining global gate

The theorem does not rule out a better point elsewhere on the equiangular
locus, much less a nonorthogonal or non-equiangular triple. The relevant domain
dimensions are now separated in the
[quotient audit](../SPIN9_QUOTIENT_DIMENSION_AUDIT_2026-08-07.md): the ordered
unit-triple quotient is generically nine-dimensional, the frame-objective
quotient is generically eight-dimensional, and the orthonormal-projector
subproblem has cohomogeneity three. The displayed curve occupies only one
dimension of that last quotient.

The next proof gate is therefore not a guessed six-dimensional orbit. It is
either an orthonormality theorem followed by a three-dimensional Grassmannian
certificate, or a direct certificate on the full rank-three frame domain.

One part of the orthonormality gate is now exact: the
[fixed-plane theorem](SPIN9_FIXED_PLANE_ORTHONORMALITY.md) proves that every
curve projector uniquely beats all nonorthogonal unit-probe frames supported
on the same three-plane. Moving the support plane remains open.

Until that domain is controlled, the phrase *globally D-optimal three-spinor
sensor* remains conjectural.

## 7. Artifact

The exact report is
[`spin9_three_spinor_conditioning_20260807.json`](../../artifacts/spin9_three_spinor_conditioning_20260807.json).
Its SHA-256 digest is
`4fe1435c15b18f7072dc2099d7528649cd88f65f42b548ca9fa392438eeaae0a`.
