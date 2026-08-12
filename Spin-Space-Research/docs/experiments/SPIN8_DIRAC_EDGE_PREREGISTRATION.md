# Spin(8) Cayley-Null Edge-Family Preregistration

**Protocol version:** 2, amended 2026-08-04 after the successful certificate
to harden artifact metadata and verification.  See
`SPIN8_DIRAC_EDGE_PROTOCOL_HISTORY.md`.

**Prospective version 1:** written 2026-08-04 after a numerical attack falsified coordinatewise
conditional decorrelation and after an exploratory parity audit exposed a
two-character determinant, but before the independent exact coefficient
reconstruction specified below.

This ordering is recorded by the execution transcript, not by an independently
timestamped Git commit.  The document is therefore a prospective protocol for
the version-1 run, but not a cryptographically independent preregistration.

## Why this gate exists

The signed star theorem covers three of the six Gram correlations of a general
four-frame.  It does not follow that the other correlations can be removed
monotonically: a fresh 200,000-frame attack found that residual Cholesky
correlations can increase the Gram-normalized information determinant by a
factor greater than 2.5 relative to the corresponding star frame.

The next exact target therefore adds one residual Cholesky edge while retaining
the Cayley-null plane, which is the information-maximizing orthonormal orbit.
This tests four simultaneous Gram correlations without assuming a false
coordinatewise monotonicity lemma.

## Frozen family

Let `q_1=e_0`, `q_2=e_1`, `q_3=e_2`, and `q_4=e_4`, so the normalized Cayley
coordinate is zero.  Define

\[
\begin{aligned}
x_1 &= q_1,\\
x_2 &= a q_1 + A q_2,\\
x_3 &= d q_1 + D(e q_2 + E q_3),\\
x_4 &= g q_1 + C q_4,
\end{aligned}
\]

where `A^2=1-a^2`, `D^2=1-d^2`, `E^2=1-e^2`, and
`C^2=1-g^2`, with nonnegative diagonal factors.  Write

\[
u=a^2,\quad v=d^2,\quad r=e^2,\quad w=g^2,
\qquad
\Delta=(1-u)(1-v)(1-r)(1-w).
\]

This is not the unrestricted theorem.  It covers one of the three residual
Cholesky correlations and fixes the normalized Cayley coordinate to zero.

## Exact parity target

The exact symmetry certificate below reduces reconstruction to two sign
choices at every interpolation node.  A complete 16-sign Walsh anchor on each
grid and every 16-sign off-grid holdout must contain exactly two characters:

\[
\frac{1024\det I(X)}{\Delta^3}
=F(u,v,r,w)+(adeAD)H(u,v,r,w).
\]

Every other Walsh character must vanish exactly on both complete-grid anchors.
The conservative interpolation multidegrees are

\[
\deg F\le(4,4,4,4),\qquad
\deg H\le(3,3,3,4).
\]

Each disjoint grid therefore contains five nodes per squared variable.  The
exploratory target is the stricter recovered degree `(3,3,3,3)` for `F` and
`(2,2,2,3)` for `H`; a higher recovered degree, an unexpected character, or an
off-grid disagreement fails the gate.

The parity restriction must also be derived independently of interpolation.
The harness enumerates every diagonal sign action preserving the maintained
Cayley form and fixing `e_0`.  Their projections onto `(e_1,e_2,e_4)` must
cover all eight sign triples.  After restoring positive Cholesky diagonals
using `P(x)=P(-x)`, the induced action on `(a,d,e,g)` is
`(t_1,t_2,t_1 t_2,t_4)`.  Its exact Walsh annihilator must be precisely the
trivial and `ade` characters.

Version 2 additionally requires a common exact adjoint-conjugacy check in all
three maintained triality representations.  For every diagonal vector action,
the harness must exhibit coordinate actions on `V`, `S+`, and `S-` whose
conjugation induces the same 28 generator signs.  This is a verifier hardening
condition, not a retrospectively added positivity gate.

## Positivity certificate

At `c=0`, the orthonormal target is `1024 det I(Q)=81`.  Put

\[
A_0=81-F,
\qquad
Q_0=A_0^2-uvr(1-u)(1-v)H^2.
\]

The theorem follows for both surviving orientations if `A_0>=0` and `Q_0>=0`
on the unit four-cube.  The preregistered certificate is native tensor-product
Bernstein positivity: every exact Bernstein coefficient of both polynomials
must be nonnegative.  Degree elevation, subdivision, floating-point SOS, and
post-hoc domain restriction are not part of this gate.

A negative native Bernstein coefficient would fail this certificate gate.  It
would not, by itself, falsify the polynomial inequality; Bernstein degree
elevation, subdivision, or a separately preregistered exact SOS certificate
would remain logically possible.

## Independent reconstruction and holdouts

- Discovery and confirmation use disjoint five-node rational-circle sets.
- Canonically serialized coefficient maps for `F` and `H` must match exactly.
- Sixteen off-grid rational magnitude frames are crossed with all 16 sign
  orientations, for 256 exact determinant comparisons.
- The maximum exact comparison error must be zero.

## Promotion boundary

A pass proves the strengthened Dirac--Gram inequality on this complete
Cayley-null four-correlation edge family.  It does not prove the result for
nonzero Cayley coordinate, for either remaining residual correlation, or on
the unrestricted seven-invariant domain.
