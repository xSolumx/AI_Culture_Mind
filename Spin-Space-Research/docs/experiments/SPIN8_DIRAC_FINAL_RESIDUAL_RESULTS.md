# The Final Dirac--Gram Residual: Exact Slice and Global Reduction

**Date:** 2026-08-07
**Status:** exact theorem on the full final-residual extension of the former
equality slice; exact 16-sector reduction of the unrestricted chart; global
seven-invariant positivity remains open
**Preregistration:**
[`SPIN8_DIRAC_FINAL_RESIDUAL_PREREGISTRATION.md`](SPIN8_DIRAC_FINAL_RESIDUAL_PREREGISTRATION.md)
**Harness:**
[`spin8_dirac_final_residual.py`](../../src/spin8_dirac_final_residual.py)
**Artifact:**
[`spin8_dirac_final_residual_20260807.json`](../../artifacts/spin8_dirac_final_residual_20260807.json)

## 1. What changed

The complete `h=0` two-edge atlas leaves one lower-triangular coordinate
uncontrolled.  In the full chart,

\[
\begin{aligned}
x_1&=e_0,\\
x_2&=a e_0+A e_1,\\
x_3&=d e_0+D(e e_1+E e_2),\\
x_4&=g e_0+G\bigl(h e_1+H(i e_2+I(c e_3+s e_4))\bigr).
\end{aligned}
\]

The exact chart audit gives

\[
\Delta=det(XX^{\mathsf T})=A^2D^2E^2G^2H^2I^2,
\qquad
\Phi(X)=ADEGHIc.
\]

Hence (c=\Phi/\sqrt\Delta) throughout the full-rank chart.  The target
orthonormal determinant is therefore still

\[
\det I(Q)=\frac{(1-c^2)^3(9-c^2)^2}{1024}.
\]

The QR and symmetric-polar completions need not be the same labelled frame.
The balanced-flag theorem supplies the missing bridge: their internal
(2+2) splits differ by an (SO(4)) action inside the same four-plane, and
the complete information spectrum is invariant under that split action.  A
new exact cross-split regression mixes positive- and negative-chiral rows by a
rational (SO(4)) matrix and verifies equality of the full characteristic
polynomials.

## 2. Exact theorem on the complete former equality slice

Set

\[
a=d=e=g=i=0,
\qquad r=h^2,
\qquad z=c^2.
\]

No small-(h) approximation is made.  The complete determinant factors as

\[
\det I=rac{H^6s^6}{16384},F_1(r,z)F_2(r,z),
\]

where, after (H^2=1-r) and (s^2=1-z),

\[
F_1=r^2+4rz-12r-4z+36,
\]

and

\[
F_2=4(rz-3r-z+9).
\]

Consequently the normalized target margin is exactly

\[
\boxed{
(1-z)^3(9-z)^2-rac{1024\det I}{H^6}
=r(1-z)^3q(r,z)
},
\]

with

\[
q(r,z)=\frac14\bigl(
-r^2z+3r^2-4rz^2+25rz-45r+8z^2-96z+216
\bigr).
\]

In the tensor-product Bernstein basis of bidegree ((2,2)) on
([0,1]^2), the coefficient matrix is

\[
\begin{pmatrix}
54&42&32\\
387/8&607/16&29\\
87/2&69/2&53/2
\end{pmatrix}.
\]

Every coefficient is positive; the minimum is (53/2).  Therefore the
strengthened Dirac--Gram inequality holds for every physical (h) and every
Cayley coordinate on this slice, with equality exactly when

\[
h=0\quad\text{or}\quad c^2=1.
\]

This is a global two-variable theorem.  It is strictly stronger than the
previous transverse-Hessian check, but it is not the unrestricted theorem.

## 3. Exact reduction of the seventh invariant

The common signed-diagonal triality action has eight elements on the complete
chart.  Its annihilator in the seven sign variables has exactly 16
characters.  Thus the unrestricted determinant has 16 physical Walsh
sectors—not 128 unrelated sign cases.

If a sector mask is

\[
m=(m_a,m_d,m_e,m_g,m_h,m_i,m_c),
\]

then its forced final-coordinate factor is

\[
h^{m_h}H^{m_i\oplus m_c}.
\]

The (h) factor follows from the `h=0` sign symmetry.  The (H) exponent
records which normalized approach sectors remain distinguishable when the
nested (i,c) coordinates collapse at (H=0).

There is also a conservative global degree ceiling.  The last query is a
rank-seven update, so Cauchy--Binet bounds the determinant degree in its vector
coordinates by 14.  The established boundary-rank lemma supplies the factor
(H^6) removed by (Delta^3).  After the forced factor above is removed,
the residual is a polynomial in (r=h^2) of degree at most

\[
\boxed{
\deg_r\le
\left\lfloor\frac{8-m_h-(m_i\oplus m_c)}{2}\right\rfloor
\le4.
}
\]

An exact generic anchor reconstructed all 16 sectors from the 16 sign cosets,
then checked independent rational (h)-nodes.  Every holdout matched.  The
observed degrees were at most three, below the conservative ceiling of four.

This is the important computational reduction: the final residual does not
create an unbounded or opaque analytic object.  It adds one squared variable
of degree at most four and doubles the Walsh sector count from 8 to 16.

## 4. Numerical falsification and exact adjudication

The complete seven-coordinate campaign used the local RTX 2070 SUPER and
included:

- 1,000,000 mixed uniform and boundary-biased samples;
- 64 gradient restarts for 1,500 steps;
- exact rational-circle replay of every floating-point candidate above
  (10^{-9}).

The unscaled float64 determinant produced 92 apparent positives, with the
largest reported log ratio (30.17).  All 92 occurred in severely
near-singular regimes.  Every one reversed under exact rational replay;
the exact violation count is zero.  The largest gradient run converged to the
known orthonormal equality manifold, with log ratio zero, rather than to a
challenger.

Peak allocated CUDA memory was only 103,784,448 bytes.  These computations are
counterexample searches.  Their failure to find a witness does not establish
global nonnegativity.

## 5. Scientific conclusion

The final coordinate is not a hidden immediate counterexample, and it is not
an uncontrolled high-degree obstruction.  Two exact statements now survive:

1. the entire final-residual extension of the former equality slice is
   positive, with an explicit strict Bernstein residual;
2. the unrestricted sign problem reduces to 16 sectors whose final squared
   coordinate has degree at most four after known radical factors are removed.

The remaining gate is finite and explicit: reconstruct the 16 seven-variable
sector maps on two disjoint exact grids, verify independent holdouts, then
construct a complete positivity atlas or exact counterexample.  Until that
last sign certificate exists, the unrestricted Dirac--Gram inequality remains
open.

## 6. Replay

```powershell
$env:PYTHONPATH='src'
python -m spin8_dirac_final_residual `
  --random-samples 1000000 `
  --batch-size 4096 `
  --restarts 64 `
  --steps 1500 `
  --workers 6 `
  --output artifacts/spin8_dirac_final_residual_20260807.json
```

The published artifact SHA-256 is

```text
6fd4aa07aca24230b576492dc4453aa264e672c7495b3d3a69ee90898aa17865
```
