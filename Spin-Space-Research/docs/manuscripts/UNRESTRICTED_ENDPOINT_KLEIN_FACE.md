# A Klein-Four Positivity Theorem on a Cayley-Endpoint Face

**Working computer-assisted theorem note — 2026-08-07**
**Status:** exact theorem on a complete four-variable boundary face
**Certificate:**
[`spin8_dirac_endpoint_klein_face.py`](../../src/spin8_dirac_endpoint_klein_face.py)

## Abstract

The unrestricted balanced \(\operatorname{Spin}(8)\) triality sensor has
sixteen orientation-dependent determinant margins. On the Cayley-endpoint
face

\[
u_a=u_h=0,
\qquad
z=c^2=1,
\]

exactly three nontrivial Walsh amplitudes survive. Their characters are the
three nonidentity elements of a Klein-four subgroup. Consequently the sixteen
margins reduce to four eigenvalues of a \(4\times4\) symmetric
group-circulant matrix, each with multiplicity four.

This note proves that matrix positive semidefinite for every
\((u_d,u_e,u_g,u_i)\in[0,1]^4\). The proof certifies all principal minors.
Three native Bernstein tensors are nonnegative directly. The remaining
boundary controls are resolved by an exact one-mode triangular chart and a
nested boundary-selector decomposition. The final determinant certificate
contains a factorization by the one-mode minor and a second polynomial with
42 nonnegative Bernstein controls, followed by a 24,336-control nonnegative
interior remainder.

This is a global theorem on the stated endpoint face. It does not establish
positivity on the other endpoint faces or in the unrestricted
seven-dimensional interior.

## 1. The endpoint reduction

Let \(A_0\) be the trivial Walsh amplitude and let the nontrivial amplitudes
be indexed by the seven-bit characters of the exact sign quotient. After
setting \(u_a=u_h=0\) and \(z=1\), every forced monomial vanishes except those
belonging to

\[
0,
\qquad
\alpha=(0,0,1,1,0,0,1),
\qquad
\beta=(0,1,0,1,0,1,0),
\qquad
\gamma=(0,1,1,0,0,1,1).
\]

The relation \(\gamma=\alpha+\beta\) holds in
\((\mathbb Z/2\mathbb Z)^7\). Thus

\[
\{0,\alpha,\beta,\gamma\}\cong
(\mathbb Z/2\mathbb Z)^2.
\]

Write the integer-scaled amplitudes as \(x,a,b,c\). The physical margins on
this face are the four eigenvalues of

\[
K=
\begin{pmatrix}
x&a&b&c\\
a&x&c&b\\
b&c&x&a\\
c&b&a&x
\end{pmatrix},
\]

each repeated four times in the original sixteen-orientation family. Hence
all physical margins are nonnegative exactly when \(K\succeq0\).

## 2. Finite principal-minor criterion

A real symmetric matrix is positive semidefinite if and only if all of its
principal minors are nonnegative. Klein-four symmetry leaves only six
distinct conditions:

\[
x\geq0,
\]

\[
x^2-a^2\geq0,
\qquad
x^2-b^2\geq0,
\qquad
x^2-c^2\geq0,
\]

\[
x^3-x(a^2+b^2+c^2)+2abc\geq0,
\]

and

\[
\begin{aligned}
\det K={}&x^4-2x^2(a^2+b^2+c^2)+8xabc\\
&+a^4+b^4+c^4
-2(a^2b^2+a^2c^2+b^2c^2)\geq0.
\end{aligned}
\]

Although \(a,b,c\) individually contain circle square roots, the squares and
the product \(abc\) are ordinary polynomials in
\((u_d,u_e,u_g,u_i)\). Every principal minor above is therefore an exact
rational polynomial on the unit four-cube.

## 3. The one-mode boundary certificate

At \(u_d=u_i=0\), only \(a\) survives. The matrix condition reduces to

\[
x^2-a^2\geq0.
\]

In the remaining variables \(U=u_e\) and \(G=u_g\), the polynomial is
symmetric. Split the square into the triangles \(U+G\leq1\) and
\(U+G\geq1\).

On the lower triangle use

\[
r=U+G,
\qquad
q=\frac{4UG}{r^2}.
\]

After removing the exact factor \(r^2\), the transformed polynomial has
multidegree \((5,3)\); all 24 Bernstein controls are nonnegative. On the
upper triangle use the same chart for \(1-U\) and \(1-G\). The transformed
polynomial has multidegree \((7,3)\), and all 32 controls are strictly
positive. Thus the one-mode boundary is certified on its complete square.

## 4. Linear, quadratic, and cubic minors

The exact Bernstein tensor of \(x\) has 256 controls, one zero, and no
negative coefficient.

Two of the three quadratic minors have 2,401-control nonnegative native
tensors. The remaining quadratic minor has one negative native control, but
that control lies on \(u_d=u_i=0\). Subtracting the one-mode face with the
degree-six Bernstein selector

\[
(1-u_d)^6(1-u_i)^6
\]

leaves a 2,401-control remainder with 51 zeros and no negative coefficient.

The cubic minor restricts on the same boundary to

\[
x(x^2-a^2).
\]

After extending this certified face with the degree-nine selector, the
remainder has 10,000 Bernstein controls, 109 zeros, and no negative
coefficient.

## 5. Nested determinant certificate

The determinant needs two nested boundary steps.

First set \(u_d=0\). The two remaining modes vanish and the complete face is
the exact square

\[
\det K\big|_{u_d=0}=(x^2-a^2)^2.
\]

Extend this face with \((1-u_d)^{12}\) and subtract it. The residual boundary
at \(u_i=0\) is exactly divisible by \(u_d\). Write the quotient as \(Q\).
Its boundary value at \(u_d=0\) factors exactly as

\[
Q_0(U,G)=(x^2-a^2)B(U,G),
\]

where \(B\) has multidegree \((5,6)\). All 42 Bernstein controls of \(B\)
are nonnegative. Hence \(Q_0\geq0\), because the first factor is the certified
one-mode minor.

The decomposition

\[
Q=Q_0(1-u_d)^{11}+Q_{\mathrm{int}}
\]

has a 1,716-control remainder with no negative coefficient. Finally, after
extending the complete \(u_i=0\) boundary with \((1-u_i)^{11}\), the
four-variable interior remainder has 24,336 Bernstein controls: 3,749 are
zero and none is negative.

Every term in the nested decomposition is nonnegative. Therefore
\(\det K\geq0\) on the complete four-cube.

## 6. Theorem

For every point satisfying

\[
u_a=u_h=0,
\qquad
z=1,
\qquad
(u_d,u_e,u_g,u_i)\in[0,1]^4,
\]

all sixteen unrestricted orientation margins are nonnegative. Equivalently,
the strengthened cubic Gram-volume inequality holds on this complete
Cayley-endpoint face.

## 7. Why this slice matters

Numerical maximization of the global Fourier-energy ratio approaches its
largest values near this endpoint geometry. There the margin distribution
does not become an arbitrary high-variance vector. It collapses toward a
small Fourier subgroup, and on the one-mode boundary the normalized margins
approach eight zeros and eight twos. The theorem explains why saturation of
the global energy inequality is harmless on this face.

The reusable proof principle is:

1. identify the subgroup of Walsh modes surviving on a boundary;
2. replace radical-bearing orientation inequalities by PSD of the associated
   group-circulant matrix;
3. certify its finite principal-minor hierarchy;
4. extend lower-dimensional certificates with exact Bernstein boundary
   selectors.

This is a stronger guide for the unrestricted problem than expanding
elementary-symmetric coefficients blindly.

## 8. Reproducibility and scope

The exact artifact is
[`spin8_dirac_endpoint_klein_face_20260807.json`](../../artifacts/spin8_dirac_endpoint_klein_face_20260807.json).
The focused publication test rebuilds the four amplitudes from the two-grid
unrestricted coefficient maps and replays every factorization and Bernstein
sign check.

The theorem is confined to one complete endpoint face. In particular, it
does not prove:

- the other Cayley-endpoint faces;
- positivity for \(0<z<1\) outside the already-certified families;
- the unrestricted seven-variable Dirac--Gram inequality;
- global five-query optimality across all allocations.
