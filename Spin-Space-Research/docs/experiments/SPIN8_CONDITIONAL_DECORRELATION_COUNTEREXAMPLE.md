# Exact Counterexample to Coordinatewise Cholesky Decorrelation

**Date:** 2026-08-04

**Harness:** `../../src/spin8_conditional_counterexample.py`

**Raw artifact:** `../../artifacts/spin8_conditional_counterexample_20260804.json`

**Artifact SHA-256:**
`fb972be8812f5bda34862ea49ba9e80b62ad5137a4a049615e87d397ab1e102d`

## Result

The proposed reduction from a general four-frame to the signed star family by
setting the three residual Cholesky partial correlations to zero is false.
The failure is not numerical: a small rational-circle witness reverses the
claimed monotonicity in exact arithmetic by a factor greater than three.

This does **not** falsify the global Dirac--Gram conjecture.  It falsifies one
specific proof route.

## How the witness was found

A deterministic 200,000-frame float64 attack found a normalized log gain of
`0.9315319804` over the corresponding star frame.  Adam optimization inside
the strict partial-correlation box `(-0.97,0.97)^7` increased the gain to
`1.1684945053`.  The optimized point was then replaced by the low-denominator
rational-circle coordinates

\[
(-14/19,\ 39/50,\ 5/31,\ -39/50,\ -11/64,\ 46/59,\ -7/9).
\]

Every coordinate `t` is mapped exactly to

\[
p=\frac{2t}{1+t^2},\qquad
q=\frac{1-t^2}{1+t^2},\qquad p^2+q^2=1.
\]

The resulting exact partial correlations are

\[
\begin{aligned}
a&=-532/557,&d&=3900/4021,&e&=155/493,\\
g&=-3900/4021,&h&=-1408/4217,&i&=5428/5597,\\
c&=-63/65.
\end{aligned}
\]

## Exact reversal

Let `X` be the complete Cholesky frame and `X_star` be obtained by retaining
`(a,d,g,c)` while setting `(e,h,i)=(0,0,0)`.  Exact symbolic determinant
evaluation gives

\[
\frac{\det I(X)/\det(G_X)^3}
     {\det I(X_{\rm star})/\det(G_{X_{\rm star}})^3}
=3.216818855745476\ldots,
\]

with normalized log gain

\[
1.1683929382305633\ldots.
\]

The ratio is stored as one exact rational integer quotient in the raw artifact;
the positive normalized difference is stored exactly as well, and there is no
tolerance in the acceptance test.  Exact positivity of all four leading Gram
principal minors supplies a Sylvester certificate before decimal eigenvalues
are reported.  The general Gram determinant is
approximately `1.4684920285e-5`, and its eigenvalues are all positive.  The
normalized Cayley coordinate remains exactly `-63/65` because the Cholesky
deformation preserves the oriented whitened four-plane.

## What is falsified

At fixed root partial correlations `(a,d,g)` and fixed normalized Cayley
coordinate `c`, removing the residual partial correlations `(e,h,i)` need not
increase `det(I)/det(G)^3`.

## What remains open

- another invariant-preserving deformation may still prove the conjecture;
- the invariant polynomial inequality remains unfalsified;
- the signed star theorem remains exact on its stated family;
- global five-query D-optimality remains open.

This counterexample is therefore a proof-strategy correction, not a theorem
counterexample.
