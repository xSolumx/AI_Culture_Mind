# Exceptional transport ladder and carrier boundary

**Status:** implemented through E6(-26); E7/E8 are separate-carrier research
programmes, not aliases for more 27D generators

## Precise ladder

At the compatible Lie-algebra level the intended real-form chain is

\[
\mathfrak g_{2(-14)}
\subset\mathfrak{spin}(7)
\subset\mathfrak{spin}(8)
\subset\mathfrak{spin}(9)
\subset\mathfrak f_{4(-52)}
\subset\mathfrak e_{6(-26)}
\subset\mathfrak e_{7(-25)}
\subset\mathfrak e_{8(-24)}.
\]

At the global group level covers, centers, connected components and the chosen
embedding must also be specified. The displayed symbols are therefore a
research map, not a claim that one unnamed matrix representation contains
every group.

## Three-carrier tower

| Carrier | Dimension | Largest implemented/natural linear action | Invariants |
|---|---:|---|---|
| Albert algebra `J = H_3(O)` | 27 | `E6(-26)` | cubic determinant; positive trace metric only for compact F4 |
| Freudenthal system `F(J) = R + R + J + J*` | 56 | `E7(-25)` | symplectic form and quartic norm |
| `e8(-24)` adjoint | 248 | `E8(-24)` | Lie bracket and indefinite Killing form |

The current model is the first row. The executable 27D ladder is now

\[
G_2\subset Spin(7)\subset Spin(8)\subset Spin(9)\subset F_4\subset E_{6(-26)}.
\]

`Spin(7)` is the named vector-stabilizer embedding inside the existing
triality-aligned `Spin(8)`: it fixes the scalar unit in the first off-diagonal
octonion carrier. `G2` fixes the corresponding units in two carriers; the
third is then fixed automatically. Generator ranks, closure, containment and
stabilizers are tested and recorded by `audit_algebra.py`.

The 27D restriction branches as

\[
J\downarrow Spin(9)=2\cdot1\oplus9\oplus16,
\]

\[
J\downarrow Spin(8)=3\cdot1\oplus8_v\oplus8_{s+}\oplus8_{s-},
\]

\[
J\downarrow Spin(7)=4\cdot1\oplus7\oplus8\oplus8,
\qquad
J\downarrow G_2=6\cdot1\oplus3\cdot7.
\]

These decompositions explain what the smaller actions can move inside the
same state. They do not imply a language-quality ordering.

## Why E7 needs a new model

The honest linear E7 carrier is

\[
\mathfrak F(J)=\{(\alpha,\beta,X,Y):
\alpha,\beta\in\mathbb R,\ X\in J,\ Y\in J^*\},
\]

with

\[
56=1_{-3}\oplus27_{-1}\oplus27^*_{+1}\oplus1_{+3}
\]

under `E6(-26) x SO(1,1)`. A valid implementation must construct all 133
generators and verify

\[
G^T\Omega+\Omega G=0,
\qquad Dq(z)[Gz]=0,
\]

for the symplectic form `Omega` and Freudenthal quartic `q`. Applying the same
27D E6 matrix independently to both halves would be wrong: the dual half uses
the contragredient action.

The affine scan itself accepts a custom 56D linear generator bank. What does
not yet exist is the Freudenthal algebra, invariant readout, exact generator
construction, or SM75 kernel. That work belongs in a separate correctness
package before it enters a language model.

## Why E8 stops before the SM75 language model

The compatible grading is

\[
248=(133,1)\oplus(56,2)\oplus(1,3)
\]

under `E7(-25) x SL(2,R)`. The familiar 57D quasiconformal realization is
nonlinear and therefore does not preserve the present two-sided affine scan
law. Linear transport requires the 248D adjoint action

\[
x\mapsto \exp(\operatorname{ad}_a)x.
\]

A dense per-token `248 x 248` exponential and backward pass is not credible on
an 8 GB SM75 device. E8 is therefore algebra-only unless a future sparse
root-subgroup/event-only construction first passes a task-specific mechanism
gate. No fallback or reduced matrix is labelled E8.

## Stability across compact and noncompact rungs

Compact G2 through F4 are orthogonal in the trace metric. E6(-26) is
noncompact, so retention below one does not by itself bound

\[
S' = LSR^T+B.
\]

The direct v1.3.1 chart uses tied Delta erase by default and compensates
retention by a certified Frobenius/logarithmic-norm upper bound on the E6
symmetric component:

\[
\rho_{\mathrm{effective}}
=\rho\exp(-\lVert P\rVert_F).
\]

This is conservative but makes the stability policy explicit. Historical
uncompensated and independent-erase settings remain named controls.

The canonical-product v1.3.2 path instead uses the product bound

\[
\log\lVert R\rVert_2
\leq\sum_{a\in\mathrm{noncompact}}
|\theta_a|\lVert G_a\rVert_2.
\]

The generator norms and compact/noncompact mask are precomputed once, so this
bound is evaluated only for real sparse events.  The native SM75 recurrence
applies the 52/78 exact small-block primitives directly and never constructs a
dense 27 by 27 action.  This demonstrates the systems strategy that any future
56D or 248D rung would require: sparse exact root/subgroup products first,
followed by a fused carrier-specific recurrence.  It does not remove the need
to construct and verify the correct Freudenthal or adjoint carrier.

## Decision

A full **multi-carrier mathematical ladder** is possible. A single 27D
"ever-more-exceptional" transport is not. The next upper rung, if pursued, is
a separate 56D E7 correctness package. It is not promoted into natural-text
training merely because the algebra exists.

Primary orientation: [Helenius on Freudenthal systems](https://arxiv.org/abs/1005.1275),
[Dray, Manogue and Wilson on the E7/E8 division-algebra construction](https://arxiv.org/abs/2401.10534),
and [Gunaydin on the E8(-24) quasiconformal realization](https://arxiv.org/abs/hep-th/0409263).
