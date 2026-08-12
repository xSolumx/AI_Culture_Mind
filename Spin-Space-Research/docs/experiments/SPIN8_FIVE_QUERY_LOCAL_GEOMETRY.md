# Exact local geometry of the balanced five-query sensor

**Date:** 2026-08-06  
**Status:** exact local theorem and exact finite one-probe atlas  
**Harness:** `src/spin8_five_query_local_geometry.py`  
**Artifact:** `artifacts/spin8_five_query_local_geometry_20260806.json`

## Why this audit was necessary

The coordinate sensor repeatedly reaches

\[
\det I=\frac{81}{1024}.
\]

That fact alone does not exclude nearby probes whose coordinates are not
vertices of the maintained basis. The `D4` 24-cell literature gives a sharp
warning: a highly symmetric configuration can satisfy many exact moment laws
and still admit a better continuous deformation for another objective.

The correct local domain is the product of five unit seven-spheres. Its
tangent space has dimension `5 x 7 = 35`.

## Exact Riemannian Hessian

For a tangent vector `u` at a unit probe `x`, use the spherical expansion

\[
x(u)=x+u-\frac12\lVert u\rVert^2x+O(\lVert u\rVert^3).
\]

The query projector is quadratic in its probe. Applying the exact differential
identity

\[
d^2\log\det I
=\operatorname{tr}(I^{-1}d^2I)
-\operatorname{tr}(I^{-1}dI\,I^{-1}dI)
\]

produces a rational `35 x 35` Hessian. Its complete spectrum is

| eigenvalue | multiplicity |
|---:|---:|
| `0` | 28 |
| `-22` | 4 |
| `-158/9` | 2 |
| `-232/9` | 1 |

All first derivatives vanish exactly.

The shared infinitesimal `Spin(8)` action supplies a `35 x 28` orbit-tangent
matrix. It has rank 28, and the Hessian annihilates it exactly. Since the
Hessian nullity is also 28, its kernel is precisely the shared symmetry orbit.

Therefore the seven-dimensional quotient Hessian is negative definite.

> The balanced sensor is a strict local maximum of `log det I`, modulo the
> unavoidable shared `Spin(8)` symmetry.

This is stronger than numerical convergence and weaker than global
five-query optimality.

## All 35 coordinate great circles

Move one query from a coordinate vector `x` toward an orthogonal coordinate
`u`:

\[
x(c,s)=cx+su,\qquad c^2+s^2=1.
\]

All 35 choices reduce exactly to four determinant laws:

| curves | determinant |
|---:|---|
| 15 | `81/1024` |
| 12 | `3 c^6(c^2+2)(c^2+5)^2/4096` |
| 4 | `c^6(c^2+1)(4c^2+5)^2/2048` |
| 4 | `c^6(c^2+8)^2/1024` |

The 15 flat curves have tangent directions in the exact symmetry kernel. For
each of the other 20 curves,

\[
\det I(c)-\frac{81}{1024}=(c^2-1)R(c^2),
\]

where `R(z)` has strictly positive coefficients. Hence the determinant is
strictly smaller for `|c|<1`. At the orthogonal replacement boundary `c=0`,
all 20 nonflat curves have rank 25 and their determinants vanish to order six.

## Boundary sensitivity

This supplies two different boundary checks:

1. **Projective endpoints.** `c=+1` and `c=-1` represent the same information
   projector and attain equality.
2. **Rank boundary.** `c=0` either remains on a flat determinant family or
   loses exactly three information directions. No hidden determinant increase
   occurs on any single-query coordinate boundary arc.

The broader Gram--Cholesky boundary remains separately audited by the Dirac
program. In particular, earlier numerical whitening-flow work found a
near-rank-three boundary that falsified that proposed proof route; it was not
converted into evidence for the global theorem.

## Plain-language version

Imagine five movable cameras aimed at a 28-dimensional object. The best-known
arrangement has 35 small ways to wiggle the cameras. Twenty-eight wiggles just
rotate the whole experiment together, so they change nothing. In every one of
the seven genuinely new directions, the information gets worse. That proves
the arrangement is a real local peak, not a coordinate-grid illusion.

Following one camera all the way around any coordinate circle gives the same
answer: the path is either flat for symmetry reasons, or it gets worse and can
eventually lose three dimensions of information.

## Exact claim boundary

Proved:

- stationarity under every continuous tangent perturbation;
- strict local maximality after quotienting by shared `Spin(8)`;
- exact identification of all 28 flat infinitesimal directions;
- all 35 finite one-query coordinate great-circle determinant laws;
- exact ranks and sixth-order determinant collapse at their nonflat boundary.

Still open:

- distant coupled deformations of two or more probes;
- exceptional non-coordinate equality components;
- global optimality among exactly five equal-cost pure queries;
- the unrestricted Gram--Cayley inequality.

## Replay

```powershell
$env:PYTHONPATH='src'
$env:OMP_NUM_THREADS='2'
python -m spin8_five_query_local_geometry `
  --output artifacts/spin8_five_query_local_geometry_20260806.json
python -m unittest discover -s tests `
  -p "test_spin8_five_query_local_geometry.py" -v
```
