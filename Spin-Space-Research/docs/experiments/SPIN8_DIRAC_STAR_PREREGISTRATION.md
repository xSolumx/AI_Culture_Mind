# Spin(8) Dirac Star-Family Theorem Preregistration

**Frozen:** 2026-08-04, after exploratory exact interpolation exposed the
star-family factorization, but before the independent rational-node replay.

## Scope

Fix an orthonormal Cayley representative

\[
q_1=e_0,\quad q_2=e_1,\quad q_3=e_2,\quad
q_4=c e_3+s e_4,\qquad c^2+s^2=1,
\]

and consider the signed one-factor four-frame

\[
\begin{aligned}
x_1&=q_1,\\
x_2&=a q_1+\sqrt{1-a^2}\,q_2,\\
x_3&=d q_1+\sqrt{1-d^2}\,q_3,\\
x_4&=g q_1+\sqrt{1-g^2}\,q_4.
\end{aligned}
\]

Write `u=a^2`, `v=d^2`, `w=g^2`, `z=c^2`,
`Delta=(1-u)(1-v)(1-w)`, and
`Phi=sqrt(Delta)c`.

This is a four-parameter subfamily of the full seven-invariant Gram--Cayley
domain. It contains three simultaneous correlations and both orientations of
the Cayley orbit. It is not the unrestricted theorem.

## Exact factorization target

The normalized determinant must decompose as

\[
\frac{1024\det I(X)}{\Delta^3}
=F(u,v,w,z)+(adg\Phi)H(u,v,w,z),
\]

where exact degree bounds derived from Cauchy--Binet and the three-dimensional
duplicate/calibrated rank losses are

\[
\deg F\le(4,4,4,7),\qquad \deg H\le(3,3,3,6).
\]

The exploratory interpolation predicts the stricter actual degrees
`(3,3,3,5)` and `(2,2,2,4)`.

Let

\[
T(z)=(1-z)^3(9-z)^2,\qquad A=T-F.
\]

The strengthened determinant inequality for both orientations follows if

\[
A\ge0,
\qquad
Q=A^2-uvw(1-u)(1-v)(1-w)zH^2\ge0
\]

on `[0,1]^4`.

## Frozen exact targets

The independent replay must recover:

- actual degree of `F`: `(3,3,3,5)`;
- actual degree of `H`: `(2,2,2,4)`;
- nonzero monomials: `360` for `F`, `86` for `H`;
- `A` Bernstein negatives: `0`; zeros: `195`;
- `Q` Bernstein negatives: `0`; zeros: `2078`;
- at least 16 exact off-grid rational holdouts: all identities exact.

Every coefficient and comparison is rational. Floating-point agreement is not
an acceptance criterion.

## Independent replay

The confirmation interpolation must use a different set of rational circle
nodes from the exploratory grid. Its recovered `F` and `H` coefficient maps
must be bit-for-bit identical after canonical rational serialization. Any
coefficient disagreement, negative Bernstein coefficient, or failed holdout
falsifies the theorem claim.

## Interpretation boundary

Passing proves the strengthened Gram-volume inequality over the complete
signed star family. It does not cover the three residual Cholesky correlations
that distinguish a general four-frame from the one-factor Gram family. The
global Dirac--Gram inequality remains open until those variables are included
or eliminated analytically.
