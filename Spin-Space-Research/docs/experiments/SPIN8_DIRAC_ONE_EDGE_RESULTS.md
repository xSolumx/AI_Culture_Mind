# Variable-Cayley One-Edge Theorem Results

**Date:** 2026-08-04

**Protocol:** `SPIN8_DIRAC_ONE_EDGE_PREREGISTRATION.md`

**Protocol history:** `SPIN8_DIRAC_ONE_EDGE_PROTOCOL_HISTORY.md`

**Theorem-completion date:** 2026-08-06

## Result in one sentence

The strengthened Dirac--Gram determinant inequality is now proved exactly on
the complete variable-Cayley four-correlation one-edge family: two exact
Duffy charts cover the full five-cube, boundary identities close their only
exceptional layers, all tetrahedral principal minors are nonnegative, and all
256 preregistered off-grid orientation checks agree with direct determinants.

## What passed

- The exact common-triality sign group has Walsh annihilator
  `{1,egc,adgc,ade}`.
- A frozen 200,000-sample, 32-restart float64 CUDA attack found no violation.
  Optimization converged to the known orthonormal equality line; this is a
  falsifier pass, not a theorem.
- Two disjoint exact `4 x 4 x 4 x 4 x 6` grids recovered directly identical
  coefficient maps.
- The recovered multidegrees are exactly
  `F=(3,3,3,3,5)`, `H1=(2,2,2,2,1)`, `H2=(1,2,2,2,1)`, and
  `H3=(2,2,2,3,1)`.
- Term counts are respectively `1220`, `101`, `78`, and `192`.
- Eight disjoint rational magnitude points crossed with all 32 sign patterns
  now give **256/256 exact matches** between direct `28 x 28` determinants and
  the reconstructed sector formula.

The reconstruction artifact is
`../../artifacts/spin8_dirac_one_edge_exact_20260804.json`, SHA-256
`2f69dac6c572a7f001bfe29b16dedc04fe91e51dfa18ed8397c54ec3f2ad73e4`.

## Evidence correction: the holdout gate was initially missing

An earlier version of this document said that the 256 exact holdouts had
passed, while the stored reconstruction artifact correctly retained
`holdouts_pending: true`. That sentence was unsupported. The missing gate has
now been implemented separately in `spin8_dirac_one_edge_holdouts.py` and run
over every preregistered case. The new holdout artifact stores all 256 exact
rational direct and predicted values; its mismatch count is zero.

This correction matters: two matching interpolation grids are strong internal
evidence, but they are not a substitute for disjoint direct determinant
checks.

## Positivity gate

Writing `x=T-F`, native exact Bernstein positivity passes for `x`,
`x^2-Q^2`, and `x^2-R^2`. It fails as a certificate for `x^2-P^2` with only
two negative controls, the worst `-16/27`. This does not falsify the
inequality.

Exact subdivision first certified the complete half-cube `z<=1/2` and three of
four `(r,w)` half-boxes for `z>=1/2`. The unresolved region was

\[
r,w\in[0,1/2],\qquad z\in[1/2,1],
\]

and every original negative control lay in the `u=v=0` layer. Recursive boxes
tracked toward `r=w=0` without terminating. This was a basis problem, not a
counterexample.

On that face, the polynomial is symmetric in `r,w` and depends on them only
through `s=r+w` and `p=rw`. Two exact cube charts cover the complete feasible
`(s,p)` region. Their Bernstein tensors have zero negative coefficients:

- `r+w<=1`: 51 boundary zeros, minimum positive `64/7`;
- `r+w>=1`: no zeros, minimum `2809/4`.

Together with the nonnegative off-face controls, this proves
`x^2-P^2>=0` on the full five-cube.

After removing the exact `(1-z)^9` factor, the cubic principal minor has only
three negative native controls, all on `u=v=0`. There it factors as

\[
C_3=X(X^2-P^2),
\]

so `X>=0` and the proved first minor prove the complete cubic minor.

The final determinant, after removing `(1-z)^12`, has 203,978 monomials and
only 21 negative native controls. The exact decomposition

\[
D_4=(X^2-P^2)^2+uvK
\]

does not close the proof: `K` has 116,846 negative native Bernstein controls.
That certificate basis is therefore rejected.

A better Duffy chart writes `u=t y` and `v=t(1-y)`. A float64 discovery screen
found negative controls only in radial layers zero and one; all layers 2--24
were positive. Those first two layers have exact identities

\[
B_0=G_0^2,
\qquad
B_1=G_0\left(G_0+\frac{G'_0}{12}\right).
\]

The first factor is the proved symmetric face. The second factor has an exact
Bernstein tensor with zero negatives, ten boundary zeros, and minimum positive
coefficient `256/9`. The complementary triangle
`1-u=t y, 1-v=t(1-y)` also had no negative controls in the discovery screen.

The crash-resilient staged implementation has now completed the full exact
integer replay:

- reduced determinant: `203,978` exact monomials;
- lower Duffy chart: `950,625` exact Bernstein controls;
- lower-chart negatives: `393`, all confined to radial layers zero and one;
- radial layers 2--24: zero negative controls;
- upper Duffy chart: `950,625` exact controls and zero negatives;
- boundary layer zero: `G0^2`, with two exact symmetric charts and zero
  negatives;
- boundary layer one: `G0(G0+G0_prime/12)`, with zero negatives;
- compact remainder terms `Q`, `R`, and `PQR`: mechanically verified to
  vanish to first order on the boundary face.

The staged assembly also links and rechecks the prior certificates for `x`,
all three quadratic principal minors, and the cubic principal minor. The final
determinant is therefore not promoted in isolation: every principal-minor
condition for the tetrahedral orientation matrix is certified.

## Complete equality classification

An exact zero-support audit now sharpens nonnegativity to a complete equality
theorem. In the lower Duffy chart, all 4,500 zero controls obey

\[
k_t+k_r+k_w\le3,
\]

with the angular and Cayley indices unrestricted. The negative controls remain
confined to the already-certified radial layers zero and one, whereas radial
layers four and twenty-four are strictly positive at every remaining control.
The upper Duffy chart is strictly positive.

At the sole possible radial boundary \(t=0\), the determinant is \(G_0^2\).
Its exact symmetric lower chart vanishes only at \(r=w=0\), and its upper chart
is strictly positive. Therefore the reduced determinant vanishes exactly at

\[
u=v=r=w=0.
\]

Restoring the previously proved common factor gives the complete equality set

\[
\boxed{z=1\quad\text{or}\quad(u,v,r,w)=(0,0,0,0)}.
\]

This closes the equality-stratum ambiguity that remained after the positivity
proof. The result is replayed from the exact determinant cache by
`spin8_dirac_one_edge_equality.py`.

## Exact artifacts

- reconstructed sector polynomials:
  `../../artifacts/spin8_dirac_one_edge_exact_20260804.json`;
- determinant cache:
  `../../artifacts/spin8_dirac_one_edge_determinant_cache_20260806.json`,
  SHA-256 `dea6cb98...482a6f`;
- 256 direct holdouts:
  `../../artifacts/spin8_dirac_one_edge_holdouts_20260806.json`, SHA-256
  `695c4b78...856ee`;
- assembled theorem certificate:
  `../../artifacts/spin8_dirac_one_edge_duffy_20260806.json`, SHA-256
  `edac47a3...c790b`.
- equality-set certificate:
  `../../artifacts/spin8_dirac_one_edge_equality_20260806.json`, SHA-256
  `543f9fcc...a4e7`.

## Scientific status

The variable-Cayley one-edge inequality is now an **exact theorem on the
frozen target family**. This advances beyond both earlier slices:

- setting `c=0` recovers the Cayley-null edge theorem;
- setting `e=0` recovers the variable-Cayley signed-star theorem.

It is not the unrestricted seven-invariant theorem. Two residual Cholesky
edges remain absent from this family. The unrestricted Gram--Cayley inequality,
the nonbalanced allocation upper bounds, and global five-query D-optimality
therefore remain open.
