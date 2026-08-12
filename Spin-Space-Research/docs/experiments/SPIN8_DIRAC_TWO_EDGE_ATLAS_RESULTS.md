# A Certified Two-Edge Dirac--Gram Inequality

**Date:** 2026-08-07
**Status:** computer-assisted exact-domain theorem on the complete frozen
`h=0` two-edge family
**Certificate:**
[`spin8_dirac_two_edge_atlas.py`](../../src/spin8_dirac_two_edge_atlas.py)
**Raw artifact:**
[`spin8_dirac_two_edge_atlas_20260807.json`](../../artifacts/spin8_dirac_two_edge_atlas_20260807.json)

## The theorem

Let the four moving unit probes use the lower-triangular chart

\[
\begin{aligned}
x_1&=e_0,\\
x_2&=a e_0+A e_1,\\
x_3&=d e_0+D(e e_1+E e_2),\\
x_4&=g e_0+G\bigl(i e_2+I(c e_3+s e_4)\bigr),
\end{aligned}
\]

where every displayed pair lies on a unit circle,

\[
a^2+A^2=d^2+D^2=e^2+E^2=g^2+G^2=i^2+I^2=c^2+s^2=1,
\]

and the omitted Cholesky coordinate is fixed to (h=0). Let (I(X)) be the
balanced ((1,2,2)) Spin(8) triality information operator, let
(Delta=det(XX^{\mathsf T})), and let
(Q=(XX^{\mathsf T})^{-1/2}X) be the row-orthonormal completion. Then

\[
\boxed{
\det I(X)\leq \Delta^3\det I(Q)
}
\]

for every feasible point of this six-parameter family and every allowed sign
orientation. Equivalently,

\[
1024\,\Delta^2\det I(X)
\leq
(\Delta-\Phi^2)^3(9\Delta-\Phi^2)^2
\]

on the same domain.

This closes the finite two-edge gate that had previously been known only
through exact reduction, local analysis, and numerical falsification. It does
not include the final residual coordinate (h); the unrestricted
seven-invariant Dirac--Gram inequality remains open.

## Why a new chart was necessary

The exact Walsh reconstruction gives eight physical orientation margins. In
the original circle coordinates they contain nested square roots, so a native
tensor-product Bernstein expansion is neither small nor sign-definite.

For every nonnegative circle pair use the rational half-angle chart

\[
q(t)=\frac{2t}{1+t^2},
\qquad
Q(t)=\frac{1-t^2}{1+t^2},
\qquad 0\leq t\leq1.
\]

After multiplication by the common denominator

\[
\prod_{j=1}^{5}(1+t_j^2)^6(1+t_6^2)^4>0,
\]

each orientation margin becomes an integer polynomial of multidegree

\[
(12,12,12,12,12,8).
\]

The denominator is strictly positive, so this transformation preserves every
sign. The integer construction is separable and is replayed directly from the
maintained exact sector artifact; it does not require a six-variable symbolic
expansion.

## A complete triangular cover

One Bernstein box is still insufficient. The proof therefore partitions
selected coordinate squares into the two triangles

\[
(x_i,x_j)=(u,uv),
\qquad
(x_i,x_j)=(uv,v),
\qquad (u,v)\in[0,1]^2.
\]

Both children are retained at every split. This point is essential: a single
favourable triangle would cover only half of its coordinate square.

The frozen atlas contains 34 leaves. Three margins require depth one, four
require depth two, and one exceptional margin requires depth five. The latter
is not accepted by tolerance; its final cancellation controls are resolved in
exact integer arithmetic.

| Hadamard channel | Odd sign | Leaves | Maximum depth | Smallest certified positive lower bound | Exact fallbacks |
|---:|---:|---:|---:|---:|---:|
| 0 | (+) | 2 | 1 | (9.7222\times10^{-2}) | 0 |
| 0 | (-) | 4 | 2 | (2.3148\times10^{-3}) | 0 |
| 1 | (+) | 4 | 2 | (6.6551\times10^{-3}) | 1,248 |
| 1 | (-) | 12 | 5 | (2.2494\times10^{-6}) | 4,608 |
| 2 | (+) | 4 | 2 | (3.4722\times10^{-2}) | 1,248 |
| 2 | (-) | 4 | 2 | (2.8935\times10^{-3}) | 1,248 |
| 3 | (+) | 2 | 1 | (8.3333\times10^{-2}) | 0 |
| 3 | (-) | 2 | 1 | (6.9444\times10^{-2}) | 0 |

Every one of the 8,352 fallback controls is exactly zero after positive
integer row scaling. No negative fallback occurs.

## The four certificate layers

### 1. Exact numerator construction

The sector coefficients are rational and have common denominator four. The
half-angle substitutions are assembled using integer convolution and exact
integer tensor products. All resulting power coefficients are exactly
representable as binary64 integers, but their authoritative form remains the
integer tensor.

### 2. Exact domain cover

The two triangular maps above cover a complete square and meet on its
diagonal. Recursing on both children therefore gives a complete cover, not a
sample. The artifact records every leaf path, and its verifier compares the
stored paths with the frozen tree.

### 3. Outward Bernstein enclosure

For a degree-(n) power polynomial, the Bernstein transform uses the positive
weights

\[
b_k=\sum_{j\leq k}a_j\frac{\binom{k}{j}}{\binom{n}{j}}.
\]

The verifier propagates the standard binary64 dot-product error bound

\[
\gamma_n=\frac{nu}{1-nu},
\qquad u=2^{-53},
\]

and rounds every nonnegative error operation outward. A control whose lower
endpoint is positive is therefore certified positive under the declared
IEEE-754 execution model. Controls unreachable from any nonzero power
coefficient are exactly zero by the lower-triangular support relation.

### 4. Exact cancellation replay

The remaining controls are not accepted because their floating-point centres
look small. For each selected univariate row,

\[
n!\,b_k
=
\sum_{j\leq k}
a_j\binom{k}{j}j!(n-j)!,
\]

so multiplication by a positive row denominator converts the control to an
integer without changing its sign. The implementation computes only the
Cartesian rows containing unresolved controls. On the largest nominal leaf,
this reduces an 88,049,169-control tensor to 1,888 exact integer questions;
all 1,888 are zero.

## Independent checks and failure-resistant design

The focused theorem tests verify:

1. exact evaluation of both triangle substitutions at rational points;
2. containment of exact rational Bernstein controls in the outward intervals;
3. sign agreement between direct rational controls and scaled integer rows;
4. byte identity of the coefficient input;
5. equality of stored and frozen cover paths;
6. exact control accounting at every leaf.

The lightweight artifact verifier checks integrity and bookkeeping. A complete
proof replay is the explicit certificate command; it is intentionally not
misdescribed as a cheap unit test.

```powershell
$env:PYTHONPATH = "src"
python -m spin8_dirac_two_edge_atlas --workers 6
python -m unittest discover -s tests -p "test_spin8_publication_theorems.py" -v
```

## Scientific boundary

Established here:

- global nonnegativity of all eight physical margins on the complete frozen
  `h=0` two-edge domain;
- non-vertex interiors and every boundary represented by the atlas;
- a replayable separation between interval-positive, structural-zero, and
  exact-cancellation controls.

Not established here:

- the final (h) residual;
- the unrestricted seven-invariant Gram--Cayley inequality;
- the equality set of the two-edge theorem;
- global optimality among all exact five-query allocations;
- any machine-learning advantage of a triality recurrence.

The result is therefore a substantial new theorem slice and the first complete
domain-wide certificate beyond the variable-Cayley one-edge family. It is not
the unrestricted theorem.

## Artifact identity

SHA-256 of the regenerated strict-enclosure artifact:

```text
c6c221a5fb129c512c7858844356f87f2842d6dceb7b7bb8277edb9c31c086d0
```
