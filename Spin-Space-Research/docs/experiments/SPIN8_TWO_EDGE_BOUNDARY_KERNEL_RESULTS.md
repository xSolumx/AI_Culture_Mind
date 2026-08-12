# Spin(8) Two-Edge Boundary-Kernel Results

> **Subsequent result (2026-08-07).** This local theorem remains valid and is
> now complemented by a complete-domain `h=0` proof in the
> [two-edge triangular atlas](SPIN8_DIRAC_TWO_EDGE_ATLAS_RESULTS.md).

**Date:** 2026-08-06

**Status:** exact local theorem, exact negative certificate result, and CUDA
falsification evidence. The global two-edge Dirac--Gram inequality remains
open.

## Result in one sentence

The new Cholesky edge is exactly stable to second order around the complete
orthonormal equality line—including its only degenerate Cayley endpoint—but a
separate exact rational witness proves that the most obvious global quadratic
Schur certificate cannot work away from that line.

## Why this was the correct gate

The all-sector reconstruction had already reduced eight orientation margins
to four paired eigenchannels:

\[
K_\pm=K_0\pm iL_{\rm odd}+i^2L_{\rm even}+\cdots.
\]

If a one-edge eigenvalue `lambda` is zero while the matching odd derivative
`mu` is nonzero, one choice of the sign of a sufficiently small `i` makes the
margin negative. That would immediately disprove the proposed two-edge
theorem. Thus the load-bearing implication is

\[
\lambda_r=0\quad\Longrightarrow\quad\mu_r=0.
\]

Because the even and odd blocks use the same exact Hadamard table, this is a
four-scalar question rather than a new `8 x 8` matrix problem.

## Exact quadratic jet

Let `z=c^2` and let `a,d,e,g` be the physical transverse coordinates at the
orthonormal equality line. In every channel, the quadratic part of the old
margin has diagonal coefficients

\[
\frac52(z-9)(z-5),\qquad
2(z-9)(z-3),\qquad
2(z-9)(z-3),\qquad
2(z-9)(z-3).
\]

The only quadratic cross term is between `e` and `g`. Its signed amplitude is

\[
8\sqrt z\,(z-9)eg.
\]

The determinant of that `2 x 2` block factors exactly as

\[
\boxed{4(z-9)^3(z-1)}.
\]

It is positive for `0 <= z < 1` and zero only at `z=1`. Therefore the complete
quadratic margin is positive-definite everywhere along the equality line
except at that single endpoint.

The odd derivative begins at quadratic, not linear, order:

\[
\mu_r^{(2)}=(z-23)(z-9)dg
\]

in all four channels. At the endpoint's only tangent null direction, `d=0`,
so this odd term vanishes exactly.

There is a second symmetry: the coefficient `nu` of the new physical
coordinate `i^2` on the equality line is

\[
\boxed{\nu_r(z)=\frac52(z-9)(z-5)},
\]

for all four channels. This is exactly the stiffness of the original
`a`-direction. Locally, the new residual edge is not arbitrary; it enters as
an isometric copy of an already protected tangent direction.

## Closing the degenerate endpoint to all orders

For the two channels whose quadratic form becomes singular at `z=1`, set the
equal squared magnitudes `e^2=g^2=s`. Direct exact evaluation gives

\[
\lambda_{1,2}=64s^2(2-s^2),\qquad \mu_{1,2}=0.
\]

Thus the apparent tangent zero is lifted at fourth order and is strictly
positive for every `0 < s <= 1`. It is not a hidden non-vertex equality curve.

## An exact negative result: the naive global Schur strategy fails

It was tempting to try proving the whole cube from the quadratic residual

\[
4\lambda_r\nu_r-\mu_r^2\ge0.
\]

That statement is false. The rational-circle point

```text
(a2,d2,e2,g2,c2) = (25/169, 16/25, 25/169, 16/25, 1600/1681)
```

has positive `lambda` in all four channels but negative `nu` and negative
`4 lambda nu - mu^2` in all four. The exact residuals are stored as rational
numbers in the artifact; their decimal values are approximately

```text
-437.3238, -188.8744, -78.4434, -314.4931.
```

This does **not** falsify the full determinant inequality. Higher powers of
`i^2` are present and can restore positivity. It does decisively reject the
unmodified quadratic-truncation certificate, preventing a large exact proof
campaign from being spent on a false auxiliary claim.

## GPU falsifier

The float64 CUDA campaign evaluated:

- the full five-dimensional one-edge cube;
- all ten coordinate faces;
- a 50/50 mixture of uniform and boundary-biased samples;
- 131,072 points per region, 1,441,792 points total;
- all four Hadamard eigenchannels at every point.

It found zero points with `abs(lambda) < 1e-10` and `abs(mu) > 1e-7`. This is
falsification evidence only. It is not used as an exact global equality-set
proof.

## Independent arithmetic replay

Python-FLINT independently rebuilt the univariate coefficient slices directly
from the stored reconstruction map and confirmed:

- all four diagonal jet coefficients;
- the `eg` cross-amplitude core;
- the `dg` odd-amplitude core;
- vanishing of the other possible quadratic odd terms;
- the factorization `4(z-9)^3(z-1)`.

The FLINT run used six threads. The complete CUDA run peaked near 1.01 GiB of
process-tree RAM; the FLINT replay peaked near 0.552 GiB. Both ran under the
15 GiB watchdog with six-core affinity.

## Plain-language version

Picture the old theorem as a bowl whose bottom is a line. Adding the new edge
could have tilted the bowl at its bottom; if so, moving a tiny distance in one
sign would immediately go downhill and disprove the theorem.

It does not tilt. The first new force is exactly zero all along the bottom.
Almost everywhere the bowl curves upward in every direction. At one endpoint
there is a direction that looks flat under a quadratic microscope, but a more
exact calculation shows it rises like the fourth power. So that apparent flat
direction is safe.

However, far from the bottom the bowl is too complicated to describe by only
its first two curvature terms. An exact example shows that shortcut gives the
wrong sign. The full higher-order shape must be retained.

## Scientific boundary

Proved exactly here:

- the complete quadratic jet at the orthonormal equality line;
- positive-definiteness for `z<1`;
- annihilation of the odd derivative on the sole quadratic endpoint kernel;
- all-order quartic lifting on that endpoint path;
- failure of the naive global quadratic Schur certificate.

Not proved here:

- that the orthonormal line is the complete equality set of the one-edge
  theorem;
- boundary-kernel compatibility at any as-yet-unclassified equality stratum;
- positivity for finite values of the second edge;
- the global two-edge or unrestricted Gram--Cayley theorem.

## Next best strategy

Do not discard higher powers of `i^2`. This next reduction is now complete:
each paired margin is equivalent to degree-six and degree-twelve polynomial
gates after one reversible squaring. See
`SPIN8_TWO_EDGE_FINITE_REDUCTION_RESULTS.md`. The next exact task is to factor
their `y=0` and `y=1` endpoint layers before attempting staged Bernstein/Duffy
positivity in the interior.

## Artifacts and code

- `../../src/spin8_dirac_two_edge_kernel.py`
- `../../src/spin8_dirac_two_edge_kernel_flint.py`
- `../../artifacts/spin8_two_edge_kernel_falsifier_20260806.json`
- `../../artifacts/spin8_two_edge_kernel_flint_20260806.json`
- `../../artifacts/spin8_two_edge_kernel_resource_20260806.json`
- `../../artifacts/spin8_two_edge_kernel_flint_resource_20260806.json`
- `../../artifacts/spin8_two_edge_kernel_test_resource_20260806.json`

SHA-256:

- falsifier artifact:
  `3312b6f26b22c64378ede04f5ef7745acda627d3721510fd5205b72df018a5b0`;
- FLINT replay artifact:
  `7a0c65f5684fc9ac71db22be0d559206d47336cca86c63c56b6446b7e298408d`.

Validation: all 171 maintained tests passed in 300.426 seconds. The bounded
test process peaked at 3.953 GiB; its resource artifact SHA-256 is
`848e35e20f817492f6a0eaa5fe379254137d0b0d7a54155b1bbeb830bb357f5b`.
