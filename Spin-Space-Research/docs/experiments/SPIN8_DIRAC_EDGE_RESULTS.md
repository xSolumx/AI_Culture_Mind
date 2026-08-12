# Exact Cayley-Null Four-Correlation Dirac--Gram Theorem

**Date:** 2026-08-04

**Preregistration:** `SPIN8_DIRAC_EDGE_PREREGISTRATION.md`

**Harness:** `../../src/spin8_dirac_edge.py`

**Raw artifact:** `../../artifacts/spin8_dirac_edge_20260804.json`

**Artifact SHA-256:**
`290f2536b4dfc4ddeb1763773361380e0f2791b628e5a732083a6418f41ef390`

## Result

The strengthened Dirac--Gram inequality

\[
\det I(X)\leq\det(XX^T)^3\det I(Q)
\]

is proved on the complete Cayley-null four-correlation edge family frozen in
the preregistration.  This advances the exact frontier from the three active
Gram correlations of the signed star family to four simultaneous correlations,
including one residual Cholesky edge.  It is not the variable-Cayley one-edge
theorem and not the unrestricted seven-invariant theorem.

## Exact family

With an orthonormal Cayley-null frame `(q1,q2,q3,q4)`, let

\[
\begin{aligned}
x_1&=q_1,\\
x_2&=a q_1+Aq_2,\\
x_3&=d q_1+D(e q_2+E q_3),\\
x_4&=g q_1+Cq_4,
\end{aligned}
\]

where all four rows are unit and `A,D,E,C` are the nonnegative Cholesky
diagonals.  The Gram determinant is

\[
\Delta=A^2D^2E^2C^2.
\]

## Formal degree and divisibility lemma

Let `J` be the stacked `40 x 28` shared-action observation Jacobian.  Exact
symbolic nullspaces over the rational-function field give rank 25 and nullity
three on both circle branches at each boundary

\[
A=0,\qquad D=0,\qquad E=0,\qquad C=0.
\]

The calculation lives in the circle-constrained coordinate ring

\[
\mathbb Q[a,A,d,D,e,E,g,C]/
(a^2+A^2-1,d^2+D^2-1,e^2+E^2-1,g^2+C^2-1).
\]

Near `A=0`, choose either analytic chart
`a=sigma sqrt(1-A^2)` with `sigma` in `{+1,-1}`.  Relative to the boundary
Jacobian, the full perturbation is `O(A)`; the change in `a` is only `O(A^2)`.
Because `rank J(0)<=25`, every 28-row minor is therefore `O(A^3)`.
Cauchy--Binet gives

\[
\det(J^TJ)=\sum_S\det(J_S)^2,
\]

so the restricted Gram determinant is `O(A^6)`.  It is polynomial/analytic in
the quotient coordinate.  To make the divisibility step formal, the four
circle relations form a Groebner basis with leading monomials
`a^2,d^2,e^2,g^2`.  The quotient is therefore a free rank-16 module over
`Q[A,D,E,C]` with basis

\[
\{a^\alpha d^\beta e^\gamma g^\delta:
\alpha,\beta,\gamma,\delta\in\{0,1\}\}.
\]

For the first pair, write each normal-form component as `p(A)+a q(A)`.  The
order-six bound on both branches `a=+sqrt(1-A^2)` and
`a=-sqrt(1-A^2)` implies, by their sum and difference, that `p` and `q` are
both divisible by `A^6`; the square root is a unit at `A=0`.  Repeating the
same argument for the other three circle pairs makes every one of the 16
ordinary polynomial coefficients divisible by `A^6,D^6,E^6,C^6`.  These are
distinct, pairwise-coprime variables in `Q[A,D,E,C]`, so their product divides
every coefficient.  Consequently

\[
A^6D^6E^6C^6=\Delta^3\quad\text{divides}\quad\det I(X).
\]

The maintained exact generator tables give rank seven for each varying
`8 x 28` query block.  Its information projector is quadratic in the query
coordinates, and a determinant term can select it at most seven times.  Thus
the raw degree is at most 14 in each circle-coordinate pair.  Dividing by the
corresponding sixth power leaves pair degree at most eight.  The even sector
therefore has squared degree at most four.  In the `adeAD` sector, removing
`aA`, `dD`, and `e` leaves squared degrees at most three in `u,v,r`, while the
unfactored `g` pair remains at most four.  This gives the conservative bounds

\[
\deg F\le(4,4,4,4),\qquad
\deg H\le(3,3,3,4).
\]

Both interpolation grids used five nodes in every squared variable, so they
cover these conservative bounds.  They independently recovered the stricter
actual degrees `(3,3,3,3)` and `(2,2,2,3)`.

## Exact Walsh reduction

The determinant sign dependence is not inferred from sampled sparsity.
Enumerating the maintained Cayley form's diagonal sign symmetries gives eight
elements fixing `e0`, with arbitrary projected signs `(t1,t2,t4)` on
`(e1,e2,e4)`.  After restoring positive Cholesky diagonals using
`P(x)=P(-x)`, they act on partial signs as

\[
(a,d,e,g)\mapsto(t_1a,t_2d,t_1t_2e,t_4g).
\]

The exact character annihilator of this group is `{1, ade}`.  Hence

\[
\frac{1024\det I(X)}{\Delta^3}
=F(u,v,r,w)+(adeAD)H(u,v,r,w),
\]

with `u=a^2`, `v=d^2`, `r=e^2`, and `w=g^2`.

The character statement also fixes the displayed radical amplitude.  Flipping
the signed diagonal representatives and then restoring positive Cholesky
diagonals with `P(x)=P(-x)` makes the nontrivial sector odd in `a,d,e,A,D`
and even in `E,C`; the sixth-order boundary factors already removed from the
determinant leave exactly `adeAD` times a polynomial in `u,v,r,w`.  Thus the
full ansatz, not merely its partial-sign character, is

\[
F(u,v,r,w)+adeAD\,H(u,v,r,w).
\]

The symmetry audit is performed across the actual triality generator tables,
not only the vector Cayley chart.  For each of the eight actions, exact
conjugation in `V`, `S+`, and `S-` produces the same 28 adjoint generator
signs.  The complete actions and signs are stored in the raw artifact.

## Positivity certificate

The two disjoint rational grids recovered byte-identical maps:

- `F`: 214 nonzero monomials, SHA-256
  `2cdc1496989243a0364ad23e768adbc1a4bedf37137c0817e19056bfcc2c7583`;
- `H`: 98 nonzero monomials, SHA-256
  `411d1d069f0186ead81fd73571f902e1836a88e8e0d645de1ecbdb818a4c8f59`.

Set

\[
M=81-F,
\qquad
Q=M^2-uvr(1-u)(1-v)H^2.
\]

At the native tensor-product Bernstein degrees:

- `M` has zero negative coefficients, one zero coefficient, and minimum
  positive coefficient 18;
- `Q` has zero negative coefficients, five zero coefficients, and minimum
  positive coefficient 162.

The zero indices are exactly

\[
(0,0,0,0)
\]

for `M`, and that index plus its four first coordinate neighbours for `Q`.
They are compatible with the orthonormal equality boundary; Bernstein control
zeros alone do not classify the polynomial's complete equality set.  These
indices and the minimum positive coefficients are stored
explicitly in both independently reconstructed certificate records in the raw
artifact, alongside the complete exact Bernstein coefficient arrays.

Therefore `M>=0` and `Q>=0` on the unit four-cube.  Since `M>=0`, the
discriminant inequality implies

\[
M\geq |adeADH|,
\]

which proves both surviving sign orientations.

## Independent checks

- two disjoint five-node rational grids: identical coefficient maps;
- full 16-sign Walsh anchor on each grid: no unexpected character;
- 16 off-grid magnitude frames times 16 signs: 256 exact determinants;
- maximum off-grid error: exactly zero;
- native Bernstein negative coefficients: zero in both certificates.

The foundational test runs a lightweight artifact verifier.  It reconstructs
both polynomials from stored monomial maps, recomputes their Bernstein arrays,
replays the exact symmetry and boundary-rank lemmas, and recomputes all 256
off-grid determinants.  It does **not** rerun either interpolation grid.  The
three-minute `python -m spin8_dirac_edge ...` command is the full reconstruction
replay and was run successfully for the published artifact.

## Boundary of the claim

Still open:

- nonzero normalized Cayley coordinate in this one-edge family;
- the other two residual Cholesky edges;
- the unrestricted Gram--Cayley inequality;
- global five-query D-optimality across all allocations.

The exact counterexample in
`SPIN8_CONDITIONAL_DECORRELATION_COUNTEREXAMPLE.md` also remains decisive:
this theorem is not a consequence of monotonically removing residual
correlations.  It succeeds because the full orientation polynomial is
certified directly.
