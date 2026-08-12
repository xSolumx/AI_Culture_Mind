# Exact weighted-design correction for the balanced Spin(8) sensor

**Date:** 2026-08-06  
**Status:** exact negative result plus exact global approximate-design theorem  
**Harness:** `src/spin8_approximate_design_audit.py`  
**Artifact:** `artifacts/spin8_approximate_design_audit_20260806.json`

## The domain distinction

The existing open problem asks for the best design made from exactly five
unit-cost pure queries. Classical approximate experimental design asks a
different question: how should an arbitrary total observation budget be
distributed over any number of query locations?

Those problems were previously both described informally as “global
D-optimality.” They must now be kept separate.

## The exact sensitivity test

Let

\[
I=\sum_{k=1}^5P_{r_k}(x_k)
\]

be the information matrix of the balanced Cayley-null sensor. For a candidate
unit query `x` in view `r`, define

\[
s_r(x)=\operatorname{tr}\!\left(I^{-1}P_r(x)\right).
\]

Because `P_r(x)` is quadratic in `x`, this is an ordinary quadratic form
`x^T S_r x`. The three exact spectra are:

| view | eigenvalues of `S_r` |
|---|---|
| `V` | `17/3`, `10` x4, `38/3` x2, `15` |
| `S+` | `67/12` x2, `119/12` x2, `155/12` x2, `175/12` x2 |
| `S-` | the same multiset as `S+` |

For the normalized approximate design `M=I/5`, the maximum sensitivity is

\[
5\max_{r,x}s_r(x)=5(15)=75.
\]

The parameter dimension is 28. The directional derivative of `log det M`
toward a point mass at `(r,x)` is

\[
\operatorname{tr}(M^{-1}P_r(x))-28.
\]

Concavity therefore gives the exact equivalence condition

\[
\operatorname{tr}(M^{-1}P_r(x))\leq28
\quad\text{for every unit query}.
\]

Since `75>28`, the balanced equal-weight design is **not** D-optimal in the
broader approximate-design domain.

## An exact rational counterexample

Keep the same five support points. Give the vector query weight `alpha` and
each of the other four queries weight `beta=(5-alpha)/4`, so total cost remains
five. The determinant is exactly

\[
D(\alpha)=
-\frac{\alpha^3(\alpha-5)^{21}(7\alpha+5)^4}{2^{60}}.
\]

Its derivative is

\[
D'(\alpha)=
-\frac{\alpha^2(\alpha-5)^{20}(7\alpha+5)^3
(196\alpha^2-125\alpha-75)}{2^{60}}.
\]

The unique interior maximum of this reweighting family is

\[
\alpha_*=\frac{125+5\sqrt{2977}}{392}
\approx1.014820044,
\qquad
\beta_*\approx0.996294989.
\]

No irrational arithmetic is needed for the counterexample. The rational
choice

\[
\alpha=101/100,\qquad\beta=399/400
\]

increases the determinant by an exact positive rational number. Its relative
gain is approximately `0.055095%`. The gain is small, but the reversal is
exact and not a floating-point sign decision.

The weight-segment boundaries are also exact. At `alpha=0`, the vector query
is removed and the information rank is 25; the determinant vanishes to order
three. At `alpha=5`, all four other weights vanish, leaving one rank-seven
projector; the determinant vanishes to order 21. Thus the displayed optimum is
genuinely interior to this weight segment, not an artifact of ignoring a
singular endpoint.

## The global approximate-design optimum

Take the eight coordinate probes of any one triality representation and give
them uniform weight. Their exact tight-frame identity gives

\[
M=\frac18\sum_{j=1}^8P_r(e_j)=\frac14I_{28}.
\]

For every unit query in every view,

\[
\operatorname{tr}(M^{-1}P_s(x))
=4\operatorname{tr}P_s(x)=4(7)=28.
\]

The sensitivity condition is saturated everywhere. Therefore this design is
globally D-optimal over the complete union of the three unit spheres in the
approximate-design domain.

## Plain-language version

Five equally priced measurements are like being required to buy exactly five
whole instruments. Approximate design lets you buy fractional amounts of as
many instruments as you want. A plan can be best under the first rule and lose
under the second.

The old balanced plan is still a serious candidate when exactly five whole
queries are required. But once fractional measurement time is allowed, a tiny
shift in time already improves it, and spreading time evenly over eight
orthogonal probes produces perfect isotropy.

## Corrected claim boundary

Proved exactly here:

- the equal balanced sensor fails approximate D-optimality;
- an explicit rational reweighting improves its determinant;
- the best point in the displayed one-parameter weight family is exact;
- the uniform eight-point one-view design is globally D-optimal among all
  approximate designs over all three views.

Still open:

- global D-optimality among exactly five equal-cost pure queries;
- the unrestricted equal-five-query Gram--Cayley inequality;
- whether another five-point support with optimized unequal weights is the
  smallest-support approximate optimum.

The equivalence argument is the matrix-valued version of the classical
[Kiefer--Wolfowitz D/G-optimality theorem](https://projecteuclid.org/ebooks/berkeley-symposium-on-mathematical-statistics-and-probability/Proceedings-of-the-Fourth-Berkeley-Symposium-on-Mathematical-Statistics-and/chapter/Optimum-Experimental-Designs-V-with-Applications-to-Systematic-and-Rotatable/bsmsp/1200512174).

## Replay

```powershell
$env:PYTHONPATH='src'
python -m spin8_approximate_design_audit `
  --output artifacts/spin8_approximate_design_audit_20260806.json
python -m unittest discover -s tests `
  -p "test_spin8_approximate_design_audit.py" -v
```
