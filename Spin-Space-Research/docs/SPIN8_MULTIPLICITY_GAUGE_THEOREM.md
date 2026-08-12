# The Repeated-View Multiplicity Gauge

## Result

When several Spin(8) probes use the same triality representation, their total
information contribution depends only on one positive-semidefinite covariance
matrix. The individual probe list is not identifiable and should be quotiented
by an exact orthogonal gauge acting across the repeated probes.

This is a representation-independent fact about the maintained sensor model,
not a numerical pattern.

## The theorem

For one representation `r`, let its 28 infinitesimal generators be `G_p` and
define

\[
J_r(x)_{kp}=(G_p x)_k,
\qquad
P_r(x)=J_r(x)^T J_r(x).
\]

Stack `m` probes as the rows of `X`. Then

\[
\sum_{j=1}^m P_r(x_j)_{pq}
=\sum_j x_j^T G_p^T G_q x_j
=\operatorname{tr}(G_p^T G_q X^T X).
\]

Thus the complete repeated-view contribution is a linear function of

\[
C=X^T X,
\]

and nothing else. For any `U` in `O(m)`,

\[
(UX)^T(UX)=X^T X,
\]

so

\[
\sum_j P_r((UX)_j)=\sum_jP_r(x_j).
\]

This proves an exact `O(m)` multiplicity-space gauge.

## Two probes: correlation becomes energy imbalance

For unit probes `x,y` with correlation `r=x dot y`, use the 45-degree gauge

\[
u=\frac{x+y}{\sqrt2},\qquad v=\frac{x-y}{\sqrt2}.
\]

Then

\[
u\cdot v=0,
\qquad
\|u\|^2=1+r,
\qquad
\|v\|^2=1-r,
\]

while `P_r(u)+P_r(v)=P_r(x)+P_r(y)` exactly. A correlation between repeated
views is therefore gauge-equivalent to unequal energy assigned to two
orthogonal modes.

## Why the Dirac--Gram target also survives

Mixing two rows of the four-frame by an orthogonal matrix preserves its Gram
determinant. The Cayley four-form changes by the determinant of that row action,
so its square is also preserved. Therefore both sides of the invariant
Dirac--Gram inequality survive the same gauge transformation.

The checked certificate uses a genuinely correlated rational pair with
`r=3/13`. It verifies, in all three maintained triality representations, that
the projector sum is unchanged. The canonical gauge produces orthogonal modes
with exact squared norms `16/13` and `10/13`.

## What changed scientifically

The second residual in the current two-edge bridge joins two probes in the same
negative-spinor representation. It should not be treated as merely a sixth
unstructured scalar. Its invariant content is a change in the spectrum and
orientation of a rank-two covariance operator.

That gives the next proof a smaller and more natural domain:

1. quotient repeated probes by the multiplicity `O(2)` gauge;
2. parameterize their rank-two covariance by its two eigenvalues and its
   two-plane;
3. reuse concavity of `log det` in the covariance where the target invariants
   permit it;
4. apply exact sector positivity only to the irreducible residual variables.

This does **not** prove the two-edge inequality. It removes a redundant choice
of probe basis and explains why brute-force interpolation in all six Cholesky
coordinates is mathematically wasteful.

## Plain-language version

Imagine two identical kinds of sensors. Rotating the two sensor readouts into
new mixtures does not create or destroy information; it only renames the two
channels. Two correlated unit sensors can always be renamed as two perpendicular
sensors, one carrying energy `1+r` and the other `1-r`.

The old coordinates made that renaming freedom look like extra complexity. The
covariance description removes it before the difficult proof begins.

## Scope

- Proved: exact repeated-view covariance dependence and orthogonal gauge.
- Proved: exact two-probe correlation-to-energy conversion.
- Proved: invariance of the information determinant, Gram determinant, and
  squared Cayley invariant under the permitted row gauge.
- Proved subsequently: global positivity of the frozen `h=0`
  variable-Cayley two-edge family; see
  [the atlas theorem](experiments/SPIN8_DIRAC_TWO_EDGE_ATLAS_RESULTS.md).
- Open: the final `h` residual and unrestricted Dirac--Gram inequality.
