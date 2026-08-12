# Spin(8) Finite Two-Edge Polynomial Reduction

> **Subsequent result (2026-08-07).** The reduction recorded here remains the
> exact algebraic input to a now-complete positivity proof on the frozen `h=0`
> domain. See the [certified triangular atlas](SPIN8_DIRAC_TWO_EDGE_ATLAS_RESULTS.md).

**Date:** 2026-08-06

**Status:** exact algebraic reduction and CUDA falsifier pass. Global
nonnegativity of the resulting polynomials remains open.

## Result in one sentence

Every pair of finite second-edge orientation margins is exactly equivalent to
one degree-six and one degree-twelve polynomial inequality in a single edge
coordinate; all radicals disappear after one reversible squaring, and an
851,968-point float64 CUDA campaign found no violation on the full six-cube or
any of its twelve coordinate faces.

## The reduction

Let `x=i^2`. The exact Walsh sectors split into four types according to whether
they carry the lower coordinate `sqrt(x)`, the complement coordinate
`sqrt(1-x)`, both, or neither. Every Hadamard eigenchannel therefore has the
form

\[
m_\pm(x)=A(x)+\sqrt{1-x}B(x)
 \pm \sqrt{x}\left(C(x)+\sqrt{1-x}D(x)\right).
\]

The exact reconstructed `i^2` degrees are

```text
even sectors: 3, 2, 1, 2
odd sectors:  2, 1, 1, 2
```

Now set

\[
y=\sqrt{1-x},\qquad x=1-y^2,\qquad 0\le y\le1.
\]

Define

\[
L(y)=A(1-y^2)+yB(1-y^2),
\]

and

\[
R(y)=C(1-y^2)+yD(1-y^2).
\]

Then

\[
m_\pm=L(y)\pm\sqrt{1-y^2}\,R(y).
\]

Both signs are nonnegative if and only if

\[
\boxed{L(y)\ge0}
\]

and

\[
\boxed{S(y)=L(y)^2-(1-y^2)R(y)^2\ge0}.
\]

This equivalence is exact: the first condition establishes the sign of the
center, and the second says its square is at least the square of the odd
displacement. There is only one squaring, and its sign premise is retained, so
no spurious solutions are introduced.

The exact degrees are

\[
\deg_y L=6,\qquad \deg_y S=12
\]

in every one of the four channels at the generic exact audit point.

## Exact endpoint-jet flag law

The first endpoint audit is now complete at the sector level. For every one of
the eight reconstructed core polynomials (H_m), exact substitution gives

\[
\deg_{c^2}\!\left(H_m\big|_{i^2=1}\right)=0.
\]

Thus the core value at the fully active endpoint is independent of the Cayley
coordinate. The first transverse derivative satisfies the stronger ideal
membership

\[
\left.\frac{\partial H_m}{\partial i^2}\right|_{i^2=1}
\in (1-d^2)(1-e^2)(1-g^2)\,\mathcal J_m,
\]

where \(\mathcal J_m\) also contains every endpoint factor already forced by
that sector's Walsh character. Since (i^2=1-y^2), the polynomial-core
variation begins at order (y^2). Any order-(y) endpoint term in a complete
orientation margin must therefore come from a character amplitude carrying
the explicit complement factor (I=y), not from hidden linear variation in
the reconstructed core.

This is an exact flag-geometry reduction, not a positivity proof. In
particular, the assembled endpoint margins can still depend on (c^2) through
their forced character monomials.

## Why this matters

The apparent finite-edge problem contained several nested square roots and
eight signed sectors. A direct exact determinant or repeated-squaring attack
would produce a very large certificate with difficult sign bookkeeping.

The new formulation replaces it with eight ordinary polynomial families:
four `L` polynomials and four `S` polynomials. Their dependence on the other
five squared coordinates remains, but the new edge itself is now an ordinary
degree-6/12 interval variable. This makes the following tools legitimate:

- exact univariate Bernstein conversion in `y`;
- interval subdivision only where a coefficient function changes sign;
- Duffy charts inherited from the proved one-edge boundary;
- exact endpoint factoring at `y=1` (the one-edge base) and `y=0` (the fully
  active second edge).

## CUDA falsifier

The numerical campaign sampled:

- the full six-cube `(a2,d2,e2,g2,c2,y)`;
- all twelve coordinate faces;
- 65,536 genuinely advancing samples per region;
- a 50/50 uniform and Beta(1/4,1/4) boundary-biased mixture;
- all four channels and both exact-equivalent gates.

Totals:

```text
regions:                         13
base points:                 851,968
negative L below -1e-9:            0
negative S below -1e-8:            0
```

The closest sampled values were:

```text
minimum L:  0.0067173  on a2=0
minimum S:  0.00004510 on a2=0
```

This is falsification evidence only. The GPU does not certify the sign between
sampled points.

## Plain-language version

The old expression looked like a machine containing four different square
roots. We found the right dial to turn: use the complementary length `y`
instead of the squared edge `x`. After that change, checking both possible
signs is the same as checking two normal polynomials.

One polynomial says the midpoint of the two answers is above zero. The other
says the distance from that midpoint to either answer is not large enough to
cross zero. Nothing approximate happened in this conversion.

## Scientific boundary

Proved exactly:

- the four-part radical form for every channel;
- the one-squaring equivalence;
- the global degree bounds and generic exact degrees `6` and `12`;
- exact agreement with the original eight-sector expression at the audit
  point in all four channels.

Not proved:

- `L>=0` or `S>=0` on the entire six-cube;
- the finite two-edge Dirac--Gram inequality;
- the third residual edge or unrestricted Gram--Cayley theorem.

## Next exact gate

Convert `L` and `S` to Bernstein form in `y` while leaving their base-coordinate
coefficients symbolic. First audit the endpoint layers `y=1` and `y=0` and the
first inward derivatives. The `y=1` layer must reduce to the proved one-edge
margin and its boundary-kernel jet; the `y=0` layer is an independent five-cube
face. Only after those endpoint layers are factored should a staged
Bernstein/Duffy campaign be launched for the interior `y` controls.

## Code and artifacts

- `../../src/spin8_dirac_two_edge_finite.py`
- `../../src/spin8_dirac_two_edge_endpoints.py`
- `../../tests/test_spin8_dirac_two_edge_finite.py`
- `../../artifacts/spin8_dirac_two_edge_endpoints_20260806.json`
- `../../artifacts/spin8_two_edge_finite_falsifier_20260806.json`
- `../../artifacts/spin8_two_edge_finite_resource_20260806.json`
- `../../artifacts/spin8_two_edge_finite_test_resource_20260806.json`

SHA-256:

- result artifact:
  `7b619ebd7f7728f4b56d7a4168381a1e00f2d22ec852ed296daa0187ca0a29f7`;
- resource artifact:
  `f59b4f90cc72ed23a64c92f61222f7056180ea220f60ef13251d117d40b6e7b9`.

Validation: all 174 maintained tests passed in 312.698 seconds. The bounded
test process peaked at 3.964 GiB; its resource artifact SHA-256 is
`ab25a428165ef3799c239e305f0232614461ff3d03c7d81a57a2408a019c4ca2`.
