# Dirac--Gram Two-Edge Status: Structure, Reduction, and Positivity

> **Historical status note (2026-08-07).** This document accurately records
> the frontier on 2026-08-06. The finite `h=0` positivity gate was subsequently
> closed by the [certified triangular atlas](experiments/SPIN8_DIRAC_TWO_EDGE_ATLAS_RESULTS.md).
> The final `h` residual and the unrestricted theorem remain open.

**Date:** 2026-08-06

## The result ladder

### 1. The recurring Cayley spectrum has an exact block mechanism

For the balanced five-probe \(\operatorname{Spin}(8)\) triality sensor, the 28-dimensional
information operator is exactly block diagonal after one fixed bivector-basis
permutation:

\[
I(c)=I_8^{(0)}(c)\oplus I_8^{(1)}(c)
\oplus I_8^{(2)}(c)\oplus I_4(c).
\]

The two middle blocks are conjugate by a displayed, constant signed
permutation. Their determinants, together with those of the other blocks, are

\[
\frac{1-c^2}{4},\qquad
\frac{(1-c^2)(9-c^2)}{16},\qquad
\frac{(1-c^2)(9-c^2)}{16},\qquad 1.
\]

This explains at once:

- why the characteristic polynomial has repeated factors;
- why the determinant contains the factors
  \((1-c^2)^3(9-c^2)^2\);
- why the Cayley-null value is exactly \(81/1024\);
- why three information directions disappear at calibrated endpoints.

The important correction is that these are certified invariant *coordinate*
blocks. A representation-theoretic irreducibility classification under the
residual stabilizer is a promising next theorem, not something inferred from
the factorization.

### 2. The variable-Cayley one-edge inequality is proved

The strengthened Dirac--Gram inequality is now exact on the family with four
active Gram correlations and variable normalized Cayley coordinate. Common
triality symmetry reduces all orientations to a four-by-four tetrahedral
matrix. Exact positivity of its principal minors is proved over the entire
five-dimensional parameter cube.

The load-bearing final determinant has 203,978 monomials. Its certificate uses
two Duffy charts:

\[
u=t y,\quad v=t(1-y),
\qquad\text{and}\qquad
1-u=t y,\quad1-v=t(1-y).
\]

The upper chart has no negative Bernstein controls. In the lower chart, every
negative control is confined to radial layers zero and one. Those layers are
proved separately through

\[
B_0=G_0^2,
\qquad
B_1=G_0\left(G_0+\frac{G'_0}{12}\right),
\]

and exact positivity certificates for both factors. Thus all orientation
eigenvalues are nonnegative throughout the frozen family.

### 3. A second residual edge is locally stable

The next family activates a second Cholesky residual. Let

\[
z=c^2\in[0,1]
\]

denote the squared normalized Cayley coordinate, and let \(i\) be the new
physical edge coordinate. Exact Walsh reduction pairs the eight orientation
margins into four Hadamard eigenchannels. Near \(i=0\), each pair has the form

\[
m_{r,\pm}=\lambda_r\pm i\mu_r+i^2\nu_r+O(i^3),
\qquad r=1,\ldots,4.
\]

Here \(\lambda_r\) is the proved one-edge margin, \(\mu_r\) is the odd
first derivative, and \(\nu_r\) is the leading even curvature. This expansion
makes the immediate failure test explicit: if \(\lambda_r=0\) but
\(\mu_r\neq0\), one sign of a sufficiently small \(i\) would make the target
margin negative.

That obstruction does not occur on the orthonormal equality line. The only
nontrivial quadratic tangent block has determinant

\[
4(z-9)^3(z-1),
\]

which is positive for \(0\leq z<1\). At the sole degenerate endpoint \(z=1\),
the odd derivative vanishes on the tangent kernel and the exact surviving
margin is

\[
64s^2(2-s^2)>0,
\qquad 0<s\leq1.
\]

Thus the endpoint that appears flat to second order rises at fourth order.
This proves local two-edge stability around the complete orthonormal equality
line. It does not prove finite-edge positivity throughout the parameter cube.

An exact rational witness also rules out the tempting shortcut

\[
4\lambda_r\nu_r-\mu_r^2\geq0
\]

as a global certificate. This is a negative result about the quadratic
truncation, not a counterexample to the full Dirac--Gram inequality: higher
powers of \(i\) remain capable of restoring positivity.

### 4. The finite second edge reduces to ordinary polynomials

The full finite-edge expression initially contains nested square roots. Set

\[
x=i^2\in[0,1],\qquad y=\sqrt{1-x}\in[0,1].
\]

Every paired orientation margin is exactly expressible as

\[
m_\pm(y)=L(y)\pm\sqrt{1-y^2}\,R(y),
\]

where \(L\) and \(R\) are ordinary polynomials in \(y\), with coefficients
that depend polynomially on the five older squared coordinates. Both signs
are nonnegative if and only if

\[
L(y)\geq0
\quad\text{and}\quad
S(y):=L(y)^2-(1-y^2)R(y)^2\geq0.
\]

The equivalence retains the premise \(L\geq0\), so the single squaring creates
no spurious solutions. Exact reconstruction gives

\[
\deg_y L=6,
\qquad
\deg_y S=12
\]

in all four channels. The original radical problem has therefore become eight
finite polynomial-positivity problems: four degree-six center conditions and
four degree-twelve separation conditions.

An exact endpoint-jet audit further shows that every reconstructed sector core
is independent of (c^2) at (i^2=1). Its first (i^2)-derivative there is
divisible by

\[
(1-d^2)(1-e^2)(1-g^2),
\]

in addition to the endpoint factors already forced in that sector.
Equivalently, after (i^2=1-y^2), core variation starts at order (y^2);
order-(y) terms can arise only from explicit Walsh-character complement
factors. This narrows the endpoint proof geometry but does not establish
positivity.

A float64 CUDA campaign checked 851,968 points across the interior and all
twelve coordinate faces without finding a violation. That is strong
falsification evidence, but it is not an exact positivity proof between the
sampled points.

## Evidence standard

The promoted one-edge theorem rests on exact arithmetic, not floating-point
survival:

- two disjoint tensor grids reconstructed identical rational polynomials;
- 256 disjoint direct rational determinants match the reconstruction exactly;
- 1,901,250 chart controls were evaluated with integer arithmetic;
- the exceptional boundary layers have exact algebraic identities;
- the final artifact cryptographically links the reconstruction, determinant
  cache, holdouts, lower-order minors, and assembled acceptance predicate;
- a replay verifier recomputes the stored holdout predictions and all
  lightweight acceptance checks.

The two-edge statements have a deliberately split evidence status. The local
kernel theorem and the finite radical-to-polynomial reduction are exact. The
dense CUDA campaign is a counterexample search only; it does not promote the
finite two-edge inequality to a theorem.

The audit also found and corrected an earlier unsupported sentence claiming
the 256 holdouts had already run. The source artifact had honestly retained
`holdouts_pending: true`; the missing experiment was run before theorem
promotion.

## Why this forms a coherent paper

The exact block theorem reveals the spectral anatomy of the balanced sensor.
The one-edge theorem then proves a nonorthogonal deformation family using
symmetry reduction and boundary-adapted positivity. Finally, the two-edge work
shows both what survives locally and how the remaining global question reduces
to a finite polynomial certificate. These are successive layers of one story:
triality symmetry converts a large signed determinant problem into invariant
blocks, then into scalar margins, and finally into ordinary polynomial signs.

A defensible paper claim is:

> This work derives an exact invariant-block factorization for the balanced
> \(\operatorname{Spin}(8)\) triality information family, proves the strengthened
> Dirac--Gram bound on a complete variable-Cayley four-correlation family, and
> reduces the next finite residual edge to four degree-six and four
> degree-twelve polynomial positivity gates.

This is narrower—and stronger—than claiming global five-query optimality.

## The honest frontier

The work does **not** yet prove the unrestricted seven-invariant Dirac--Gram
inequality. Two residual Cholesky edges remain. It also does not prove the
nonbalanced allocation upper bounds or global D-optimality among all
five-query designs.

The second residual edge is now active and algebraically reduced, but its
finite positivity remains open. Exact Bernstein-support analysis has closed a
previous prerequisite: the one-edge equality set is precisely

\[
z=1\quad\text{or}\quad(u,v,r,w)=(0,0,0,0).
\]

The distinction between physical and reduced equality is essential here. The
complete two-edge family carries the same common factor \((1-z)^3\), so the
\(z=1\) component disappears when the proved factor is cancelled. The reduced
one-edge determinant therefore has only the orthonormal zero set, and that
component already passes the full local kernel test. There is no additional
hidden equality kernel left to audit.

The next decisive exact gate is consequently to factor the finite endpoint
layers \(y=1\) and \(y=0\), together with their first inward derivatives. The
\(y=1\) face must recover the proved reduced one-edge theorem and its local
jet; \(y=0\) is an independent five-dimensional face. Only after both faces
are understood should the interior Bernstein/Duffy calculation begin.

One further residual Cholesky edge then separates this bridge from the
unrestricted seven-invariant inequality. Global five-query D-optimality and
nonbalanced allocation bounds remain separate open problems.

## Primary evidence

- [Cayley block theorem](experiments/SPIN8_CAYLEY_BLOCK_THEOREM.md)
- [Variable-Cayley one-edge theorem](experiments/SPIN8_DIRAC_ONE_EDGE_RESULTS.md)
- [Two-edge boundary-kernel theorem](experiments/SPIN8_TWO_EDGE_BOUNDARY_KERNEL_RESULTS.md)
- [Finite two-edge polynomial reduction](experiments/SPIN8_TWO_EDGE_FINITE_REDUCTION_RESULTS.md)
- [Cayley block artifact](../artifacts/spin8_cayley_blocks_20260806.json)
- [Final Duffy certificate](../artifacts/spin8_dirac_one_edge_duffy_20260806.json)
- [Exact holdouts](../artifacts/spin8_dirac_one_edge_holdouts_20260806.json)
- [Reproducibility protocol](REPRODUCIBILITY.md)
