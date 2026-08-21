# Spin(9) explicit candidate collars

**Status:** four exact finite-radius local theorems; compact complement open

**Code:** [`spin9_candidate_explicit_collar.py`](../../src/spin9_candidate_explicit_collar.py)

**Artifact:** [`spin9_candidate_explicit_collar_20260821.json`](../../artifacts/spin9_candidate_explicit_collar_20260821.json), SHA-256
`2efdce2f181850dddf01273aa95f450ec5848c1f88eaece995a48eb167acf3d7`

The candidate normal form is made quantitative by the Cartan substitution

\[
p=\frac{3+9z^2}{2}r^2,
\qquad y=\frac{9z^2-1}{2}r^3,
\qquad 0\le z\le1.
\]

After exact scaling over `Q(sqrt(241))`, the candidate gap is

\[
G=Q(x)^2H(x)+r^2K(x,r,z).
\]

On each rational candidate-root cell, univariate Bernstein controls give an
exact positive lower bound `A_min` for the coefficient of `p`. A global
coefficient `L1` bound gives

\[
K\ge\frac32A_{\min}-rM.
\]

The chosen radii make `rM <= 3 A_min/4`, uniformly over every shape
`0 <= z <= 1`. Together with `Q^2 H >= 0`, this proves candidate maximality on
four explicit semialgebraic collars. Their radial radii are

\[
10^{-47},\qquad10^{-52},\qquad10^{-34},\qquad10^{-6}.
\]

These are conservative existence radii, not estimates of the true basin. The
tiny first three values reflect the exact near-interlacing exposed by the
normal-form certificate and a deliberately crude absolute coefficient bound.

This closes an explicit finite-radius neighborhood around every equality
preimage on the first `V1+V5` graph. It does not certify the compact complement
at the exact algebraic target, the second `V5`, or the unrestricted quotient.
