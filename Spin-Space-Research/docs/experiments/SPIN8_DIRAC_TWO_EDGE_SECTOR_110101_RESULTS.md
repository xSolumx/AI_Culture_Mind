# First Exact Two-Edge Sector Reconstruction

**Date:** 2026-08-06  
**Preregistration:** `SPIN8_DIRAC_TWO_EDGE_RECONSTRUCTION_PREREGISTRATION.md`  
**Target Walsh sector:** `110101`

## Result

The first complete six-variable residual sector of the two-edge Dirac--Gram
family has been reconstructed exactly. Two disjoint rational tensor grids
produced the identical coefficient map, 32 new off-grid points matched direct
determinants exactly, and the reconstructed polynomial contains the global
factor

\[
1-a^2=A^2.
\]

This is a rigorous sector theorem. It is not yet a positivity theorem for the
full two-edge family.

## Frozen reconstruction

The proof-safe residual degree bound was `(3,3,3,3,3,3)` in

```text
(a2,d2,e2,g2,i2,c2).
```

Each exact grid therefore contained `4^6=4096` points. At every point, the
target Walsh coefficient was recovered from eight quotient-representative
direct `28 x 28` information determinants. The computation used four saved
1024-point slabs per grid so that no process held the entire exact campaign.

| Check | Exact result |
|---|---:|
| Alpha-grid points | 4,096 |
| Beta-grid points | 4,096 |
| Direct determinants for both grids | 65,536 |
| Fresh holdout points | 32 |
| Fresh holdout determinants | 256 |
| Coefficient maps equal | yes |
| Holdout equalities | 32/32 |

The two grids returned the same coefficient-map digest:

```text
74cd02a0bff60c96653ede136fd733399e05d38f6692f9cb4f6e01a2f9b4f7c8
```

Because the structural degree ceiling was proved before the grids ran, the
first grid uniquely determines the sector polynomial. The second grid and the
fresh holdouts are independent safeguards against implementation and assembly
errors.

## Unexpected exact compression

The reconstructed residual `H_110101` has:

- `243` nonzero rational coefficients;
- observed multidegree `(2,2,2,2,1,1)`, strictly below the frozen ceiling;
- the exact factorization

  \[
  H_{110101}=(1-a^2)Q_{110101};
  \]

- a quotient `Q_110101` with `162` nonzero coefficients and multidegree
  `(1,2,2,2,1,1)`.

The global amplitude theorem had already forced one factor `aA` into this
sector's signed monomial. The newly proved factor `1-a^2=A^2` therefore raises
the total complement power from `A` to `A^3`. This is exactly the extra
endpoint multiplicity previously seen on one-dimensional slices, but it is
now a full six-variable identity rather than a pattern inferred from slices.

The factor was certified by exact polynomial division over `QQ`, zero
remainder, and exact recomposition. The maintained test rebuilds that division
from the published coefficient map.

## Nested boundary law

The 162-term quotient is itself far less generic than its degree permits. Put

\[
Q_0=Q_{110101}(a^2,d^2,e^2,g^2,0,0).
\]

Exact division and recomposition prove

\[
Q_{110101}=Q_0+D^2E^2G^2R,
\]

where `Q0` has 42 terms and `R` has only 28 terms. Moreover, `R` is multilinear
in all six squared coordinates. In the abbreviated variables
`(a,d,e,g,i,c)=(a2,d2,e2,g2,i2,c2)`, it has the compact form

\[
R=-\frac12\left[
c(a-1)(1-i)(de+d-e-g)+iN(a,d,e,g)
\right],
\]

with

\[
\begin{aligned}
N={}&6adeg+25ade+9ad-6aeg-25ae-9ag\\
&+60deg-47de-13d-60eg+47e+13g.
\end{aligned}
\]

Thus the new residual and Cayley coordinates can influence this sector only
when all three intervening complement directions `D,E,G` remain nonzero. That
is an exact algebraic dependency chain. It strongly suggests that the useful
global coordinates are a base polynomial plus boundary-ideal corrections,
not a flat six-dimensional monomial tensor.

Two transverse faces collapse even further:

\[
Q|_{d^2=1}=3(a^2-1)(g^2-1)^2,
\]

and

\[
Q|_{g^2=1}
=3(a^2-1)(d^2-1)^2(e^2-1)(3e^2+1).
\]

Both identities are independent of `i^2,c^2`, and the first also loses `e^2`.
They were verified by exact zero remainders against the published coefficient
map. On the unit cube the first face is nonpositive and the second is
nonnegative. This explicitly demonstrates why proving every Walsh sector
nonnegative would be a category error: positivity belongs to the eight signed
orientation margins assembled from the sectors.

## What a non-specialist should take from this

Imagine a complicated formula depending on six sliders, with four settings
sampled for each slider. The conservative plan allowed as many as 4,096
independent coefficient positions. Exact evaluation showed that only 243 are
used. Then those 243 terms turned out to share a hidden boundary factor, so
the genuinely new core needs only 162 terms.

Most importantly, this was not guessed from decimals. Two unrelated exact
grids produced the same rational formula, and 32 points that neither grid had
seen were recomputed from the original determinants and matched exactly.

## Artifacts and replay

- coefficient map:
  `../../artifacts/spin8_dirac_two_edge_sector_110101_coefficients_20260806.json`;
- disjoint-grid comparison:
  `../../artifacts/spin8_dirac_two_edge_sector_110101_20260806.json`;
- fresh holdouts:
  `../../artifacts/spin8_dirac_two_edge_sector_110101_holdouts_20260806.json`;
- exact factor certificate:
  `../../artifacts/spin8_dirac_two_edge_sector_110101_factor_20260806.json`.
- exact transverse-face certificate:
  `../../artifacts/spin8_dirac_two_edge_sector_110101_faces_20260806.json`.

The artifact verifier recomputes row digests, coefficient uniqueness and
degrees, all stored rational holdout equalities, and the complete exact
division/recomposition. It does not accept a stored `passed` flag by itself.

## Scientific boundary and next step

Only sector `110101` has been reconstructed globally. Individual Walsh sectors
need not be nonnegative; the final theorem concerns the eight orientation
margins obtained by their signed sums. Consequently this result does **not**
license a positivity claim for `Q_110101`, the other seven sectors, or the full
two-edge inequality.

The immediate mathematical opportunity is to derive this nested boundary-ideal
form from the observation Jacobian and test its analogues in the other seven
sectors. If that succeeds, the apparent 61,321-point problem will collapse to
small boundary polynomials plus multilinear corrections before any Bernstein
or sum-of-squares certificate is attempted.
