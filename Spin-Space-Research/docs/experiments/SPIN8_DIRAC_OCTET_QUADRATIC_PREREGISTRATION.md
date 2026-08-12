# Adjacent-Endpoint Octet Quadratic Gate

**Frozen before the native quadratic audit — 2026-08-08**
**Target:** the three quadratic principal minors of the second Schur block

## Question

On the adjacent Cayley-endpoint face

\[
u_a=0,\qquad c^2=1,\qquad u_h=1-y^2,
\]

the eight surviving orientation margins form a
\((\mathbb Z/2\mathbb Z)^3\) group-circulant.  The exact block reduction in
`spin8_dirac_endpoint_octet.py` proves that their nonnegativity is equivalent
to

\[
X\succeq0,
\qquad
Z=X^2-(1-y^2)R^2\succeq0.
\]

The first condition and the scalar principal minor of (Z) are already
proved.  This gate asks whether the three quadratic principal minors

\[
q_j=Z_0^2-Z_j^2,
\qquad j=1,2,3,
\]

are nonnegative on the complete five-cube
\((u_d,u_e,u_g,u_i,y)\in[0,1]^5\).

## Frozen evidence hierarchy

1. A native tensor-product Bernstein representation with no negative exact
   coefficient proves the corresponding (q_j\ge0) on the five-cube.
2. A negative native Bernstein coefficient rejects only that certificate
   basis.  It is not evidence that (q_j<0).
3. If native positivity fails, the exact locations of all negative controls
   will be classified by boundary support.  The already-proved common square
   on (u_d=u_g=0) is the first permitted boundary selector.
4. Floating-point optimization is a falsifier only.  A negative value changes
   theorem status only after an interior rational point is reconstructed and
   the sign is verified by exact arithmetic.
5. The quadratic gate does not prove the adjacent endpoint face.  The cubic
   and determinant minors of (Z) remain separate obligations.

## Acceptance and rejection

- **Quadratic pass:** all three (q_j) receive domain-wide exact sign
  certificates.
- **Exact disproof:** one (q_j) is negative at an exactly verified feasible
  point.
- **Inconclusive:** native Bernstein positivity fails and no exact
  counterexample or complete boundary-adapted certificate is obtained.

## Resource contract

- FLINT uses at most six worker threads, leaving two logical cores available.
- Each minor is audited in a fresh process so temporary exact tensors are not
  accumulated across families.
- Symbolic peak memory must remain below 16 GiB.
- GPU falsification, if needed, runs only after checking for active compute
  jobs and uses bounded batches.

No result from this gate may be promoted to the unrestricted seven-variable
Dirac--Gram theorem.

## Prospective amendment after the depth-two atlas

**Added before computing any blow-up quotient coefficients — 2026-08-08.**

For the first quadratic family, a complete dyadic half-cube audit certified
30 of 32 coarse boxes.  Refinement certified all 32 children of `00010` and
31 of 32 children of `00001`; the sole failure was the self-similar equality
corner `00001/00001`.  Repeating dyadic subdivision at that corner would not
constitute a finite proof.

The frozen local replacement is the max-coordinate blow-up of the five
nonnegative deviations

\[
(u_d,u_e,u_g,u_i,1-y).
\]

Five charts choose one deviation as the maximum radius (r), write every
other deviation as (r x_j), and use (0\le r\le1/4).  In every chart the
minor must be exactly divisible by (r^4), matching the independently
extracted order-four equality germ.  A chart passes only if the quotient has
an exact domain-wide positivity certificate on its five-cube.  All five
charts are required; positivity of selected directions or the leading
homogeneous form alone is insufficient.

**Second prospective amendment, before the radial-face atlas.**  In the
(u_d)-pivot chart, the exceptional divisor reduced to an exact four-factor
signed-square product plus a two-variable sign factorization.  The remaining
radial remainder has an exact first-order radius divisor; after division, its
only unresolved object is the (u_i=0) face.  That four-variable quotient
will be split into its 16 dyadic half-boxes.  The face passes only if every
box has a nonnegative exact Bernstein tensor.  A partial box count is not a
certificate.

## Prospective amendment for the second quadratic mode

**Added after both depth-two atlases completed and before computing any
second-mode blow-up coefficient — 2026-08-08.**

For mode `0101010`, the exact dyadic audit independently reproduced the same
finite localization pattern: 30 of 32 coarse boxes passed, every child of
`00010` passed, and only `00001/00001` failed among the children of `00001`.
No equality chart from mode `0011001` is assumed to transfer.

The sole residual box will be covered by the five max-coordinate blow-up
charts of

\[
(u_d,u_e,u_g,u_i,1-y).
\]

Every chart must again have exact radial order four. Each chart requires its
own domain-wide exact quotient certificate. Coincident negative-control
counts, numerical similarity, or reuse of a first-mode pass flag is
insufficient.

## Prospective amendment for the third quadratic mode

**Added after both depth-two atlases completed and before computing any
third-mode blow-up coefficient — 2026-08-08.**

For mode `0110011`, the independent exact dyadic audit again localized every
native Bernstein failure to the same self-similar equality corner: 30 of 32
coarse boxes passed, all 32 children of `00010` passed, and 31 of 32 children
of `00001` passed.  The only residual box is `00001/00001`.  This agreement is
evidence of a common boundary geometry, but no algebraic identity or
certificate from either earlier mode is assumed to transfer.

The residual box will therefore be covered by all five max-coordinate
blow-up charts of

\[
(u_d,u_e,u_g,u_i,1-y).
\]

Each chart must have exact radial order four.  Each radial quotient must then
receive its own exact domain-wide nonnegativity certificate.  If a native
Bernstein tensor contains negative controls, any replacement factorization or
subdivision must be recorded before it is used and must cover the complete
residual chart; numerical agreement, a positive tangent form, or reuse of an
earlier mode's pass flag is not sufficient.
