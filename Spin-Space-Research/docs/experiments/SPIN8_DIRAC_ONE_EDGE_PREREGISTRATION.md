# Variable-Cayley One-Edge Dirac--Gram Protocol

**Current protocol version:** 2. See
`SPIN8_DIRAC_ONE_EDGE_PROTOCOL_HISTORY.md` for the prospective ordering.

**Prospective version 1:** written 2026-08-04 before the numerical attack and
before any exact interpolation for this family. The ordering is recorded in
the execution transcript, not an independently timestamped commit.

## Target family

Let

\[
q_1=e_0,\quad q_2=e_1,\quad q_3=e_2,\quad
q_4=c e_3+s e_4,\qquad c^2+s^2=1,
\]

and

\[
\begin{aligned}
x_1&=q_1,\\
x_2&=a q_1+Aq_2,\\
x_3&=d q_1+D(e q_2+E q_3),\\
x_4&=g q_1+Cq_4,
\end{aligned}
\]

with four circle relations for `(a,A)`, `(d,D)`, `(e,E)`, `(g,C)` and
nonnegative complements. Write

\[
\Delta=A^2D^2E^2C^2,\qquad z=c^2.
\]

The target inequality is

\[
\frac{1024\det I(X)}{\Delta^3}
\le (1-z)^3(9-z)^2.
\]

This family has four active Gram correlations and variable normalized Cayley
coordinate. It contains both the exact Cayley-null edge theorem (`c=0`) and
the variable-Cayley signed-star theorem (`e=0`). It is not the unrestricted
seven-invariant theorem.

## Exact orientation gate

Before interpolation, a complete exact 32-sign Walsh transform at rational
interior points must have support exactly

\[
\{1,\;egc,\;adgc,\;ade\}.
\]

The sign characters form the four characters of a Klein-four quotient. If
`x=T-F` and the three signed sector amplitudes are `P,Q,R`, orientation-uniform
positivity is exactly equivalent to positive semidefiniteness of

\[
K=
\begin{pmatrix}
x&-P&-Q&-R\\
-P&x&-R&-Q\\
-Q&-R&x&-P\\
-R&-Q&-P&x
\end{pmatrix}.
\]

No scalar absolute-value relaxation may replace this exact four-sector gate.

## Numerical falsifier

Before expensive exact reconstruction:

- evaluate 200,000 deterministic float64 interior samples;
- optimize at least 32 adversarial starts inside `(-0.98,0.98)^5`;
- record the maximum log advantage
  `log(1024 det(I)/Delta^3)-log((1-z)^3(9-z)^2)`;
- if any advantage exceeds `1e-8`, locally improve it, rationalize it with
  rational-circle coordinates, and verify the reversal exactly before doing
  interpolation.

Absence of a numerical violation is only permission to attempt an exact
certificate. It is not evidence sufficient for theorem promotion.

## Exact-certificate boundary

If the falsifier survives, exact work must:

1. derive the four Walsh sectors from common triality symmetries;
2. derive conservative multidegrees before choosing interpolation grids;
3. reconstruct on disjoint rational grids with enough nodes for those bounds;
4. compare coefficient maps directly, not only by hashes;
5. check all 32 signs at exact off-grid points;
6. certify all tetrahedral principal-minor polynomials on the full cube.

Negative Bernstein coefficients reject that certificate basis, not the
inequality. The variable-Cayley edge theorem is promoted only if every exact
PSD condition is certified.

## Version-2 reconstruction contract

The exact one-dimensional factor audit, completed after the falsifier but
before either full tensor grid, gives the four-sector decomposition

\[
N=F+\chi_1P+\chi_2Q+\chi_3R
\]

with

\[
\begin{aligned}
P&=A^2D^2(eE)(gC)c(1-z)^3 H_1,\\
Q&=(aA^3)(dD)E(gC)c(1-z)^3 H_2,\\
R&=(aA)(dD)e(1-z)^3 H_3.
\end{aligned}
\]

The prospectively frozen full-grid degree bounds are

\[
\deg F=(3,3,3,3,5),\quad
\deg H_1=(2,2,2,2,1),\quad
\deg H_2=(1,2,2,2,1),\quad
\deg H_3=(2,2,2,3,1).
\]

Two disjoint grids must each use four nodes in every spatial squared variable
and six Cayley-squared nodes. Their directly compared coefficient maps must be
identical and must recover exactly these degrees. Eight disjoint rational
magnitude holdouts crossed with all 32 signs must agree exactly.

The squared sectors and product used in the tetrahedral minors are fixed as

\[
\begin{aligned}
P^2={}&(1-u)^2(1-v)^2r(1-r)w(1-w)z(1-z)^6H_1^2,\\
Q^2={}&u(1-u)^3v(1-v)(1-r)w(1-w)z(1-z)^6H_2^2,\\
R^2={}&u(1-u)v(1-v)r(1-z)^6H_3^2,\\
PQR={}&u(1-u)^3v(1-v)^2r(1-r)w(1-w)z(1-z)^9H_1H_2H_3.
\end{aligned}
\]

This amendment was informed by exact slice degrees and is not relabelled as
part of version 1.
