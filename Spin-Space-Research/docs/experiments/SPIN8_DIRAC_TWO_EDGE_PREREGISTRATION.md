# Variable-Cayley Two-Edge Dirac--Gram Preregistration

**Frozen:** 2026-08-06, before any determinant evaluation on this family

**Status:** prospective bridge beyond the proved one-edge theorem

## Why this residual comes next

The unrestricted Cholesky frame has residual partial correlations `(e,h,i)`.
The proved variable-Cayley one-edge theorem keeps `e` and sets `h=i=0`.

The next family activates `i` while retaining `h=0`. This choice is structural:
`i` changes only the final Gram edge `x3 dot x4`. Activating `h` would change
`x2 dot x4` and, because `e` is already active, also feed into `x3 dot x4`.
Thus the `i`-bridge is the smallest possible addition of exactly one new Gram
correlation.

No empirical result was inspected to choose between `h` and `i`.

## Frozen family

Let

\[
q_4=c e_3+s e_4,\qquad c^2+s^2=1,
\]

and

\[
\begin{aligned}
x_1&=e_0,\\
x_2&=a e_0+A e_1,\\
x_3&=d e_0+D(e e_1+E e_2),\\
x_4&=g e_0+G(i e_2+I q_4),
\end{aligned}
\]

with

\[
a^2+A^2=d^2+D^2=e^2+E^2=g^2+G^2=i^2+I^2=c^2+s^2=1
\]

and nonnegative complements. The Gram determinant is

\[
\Delta=A^2D^2E^2G^2I^2.
\]

The target is the same strengthened inequality:

\[
\frac{1024\det I(X)}{\Delta^3}
\le (1-c^2)^3(9-c^2)^2.
\]

This six-variable family contains the proved one-edge theorem at `i=0`. It is
not the unrestricted seven-invariant family because `h=0` remains fixed.

## Frozen sequence of gates

### Gate 1: common-triality sign symmetry

Derive the induced sign group from exact common adjoint conjugacies. The
expected parameter-sign action is

\[
(a,d,e,g,i,c)
\mapsto(t_1,t_2,t_1t_2,t_4,t_2t_4,t_3t_4)
\odot(a,d,e,g,i,c).
\]

With an eight-element sign group in six sign variables, the annihilator may
contain at most eight Walsh characters. This dimension count is a prediction,
not permission to assume that all eight sectors are nonzero. The exact
annihilator must be derived before anchor determinants are inspected.

### Gate 2: exact anchor support

At two disjoint rational-circle interior anchors:

- evaluate all 64 sign choices exactly;
- compute the full Walsh transform;
- require every observed character to lie in the symmetry annihilator;
- record which allowed sectors are identically absent at each anchor.

Agreement at anchors is reconnaissance, not a global vanishing proof.

### Gate 3: numerical falsification

Before interpolation:

- sample at least 200,000 deterministic float64 interior points;
- run at least 32 adversarial restarts inside `(-0.98,0.98)^6`;
- evaluate the log normalized determinant advantage;
- include boundary-biased samples near every coordinate face.

Any advantage above `1e-8` triggers local improvement, low-denominator
rational-circle approximation, and a direct exact determinant check. A proved
counterexample stops the positivity campaign and becomes the result.

### Gate 4: conservative degree audit

For every symmetry-allowed sector:

- derive radical/sign amplitude factors before division;
- bound raw determinant degree using Cauchy--Binet and rank seven per query;
- reconstruct several independent one-dimensional slices per variable;
- use disjoint nodes to detect hidden higher-degree coefficients;
- freeze conservative tensor degrees before any full grid.

A single slice is not a degree proof.

### Gate 5: staged exact reconstruction

Only after Gates 1--4 survive:

- reconstruct on two disjoint rational tensor grids;
- compare full coefficient maps, not only hashes;
- run fresh exact off-grid all-sign holdouts;
- cache sparse sector polynomials before forming principal minors.

### Gate 6: positivity or exact failure

Exploit the proved one-edge face `i=0` first. Attempt, in order:

1. the exact Cayley block basis;
2. a Schur complement around the proved face;
3. boundary-adapted Duffy charts in `i` and the existing difficult variables;
4. subdivided Bernstein or SOS certificates only if the smaller routes fail.

Negative Bernstein controls reject a certificate basis, not the inequality.

## Promotion rule

The two-edge theorem is promoted only if:

- the global sector restriction follows from exact common triality symmetry;
- two disjoint exact reconstructions agree;
- fresh direct exact holdouts pass;
- every orientation-sector eigenvalue is certified nonnegative over the full
  six-cube.

Numerical survival, optimizer convergence to equality, or a positive native
Bernstein screen alone cannot promote the theorem.

## Interpretation boundaries

Even a pass would prove only the `h=0` two-edge bridge. One residual Cholesky
correlation and the unrestricted Gram--Cayley theorem would remain open.
Global five-query D-optimality would additionally require the nonbalanced
allocation upper bounds.
