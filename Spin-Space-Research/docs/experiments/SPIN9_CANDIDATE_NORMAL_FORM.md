# Spin(9) candidate invariant normal form

**Status:** exact local algebraic bridge; explicit finite collar remains open

**Code:** [`spin9_candidate_normal_form.py`](../../src/spin9_candidate_normal_form.py)

**Artifact:** [`spin9_candidate_normal_form_20260821.json`](../../artifacts/spin9_candidate_normal_form_20260821.json)

Artifact SHA-256:
`f61d3de551610698f0f343186ee027c250d43484391fa71916bd86e3a0bf6c15`.

## Result

Let

\[
G(x,p,y)=R_*\,\delta(x,p,y)^{14}-N(x,p,y)
\]

be the reconstructed determinant gap at the exact algebraic candidate ratio
`R_*`. Over the positive real embedding of `Q(sqrt(241))`, define the quartic

\[
Q(x)=x^4-(17+\sqrt{241})x^3
 +\frac{19+\sqrt{241}}2x^2
 +\frac{17+\sqrt{241}}2x+\frac14.
\]

Its four real roots are exactly the four pure-`V1` graph preimages of the
symmetric candidate. Exact polynomial division gives

\[
G(x,0,0)=Q(x)^2 H(x),
\]

with zero remainder and `deg(H)=76`. Exact quadratic-field Sturm sequences
then prove that `H` is positive at all four roots of `Q`.

More decisively, if

\[
A(x)=\left.\frac{\partial G}{\partial p}\right|_{p=y=0},
\]

then `A` is positive at all four candidate preimages. Thus the invariant
leading form is

\[
G=Q^2H+pA+yB+O(p^2,py,y^2),
\qquad H>0,\quad A>0
\]

on the equality fiber. Since the orbit domain satisfies
`27 y^2 <= 2 p^3`, the `y` term is `O(p^(3/2))`. Consequently the exact
reconstructed candidate gap is strictly positive in some neighborhood of each
preimage, off the equality fiber.

## The nearly invisible separation

At three preimages, a root of `A` lies extraordinarily close to the
corresponding root of `Q`: the separations are approximately `10^-122`,
`10^-128`, and `10^-108`. The certificate therefore uses rational cells of
width `10^-140`. Each cell contains exactly one `Q` root, no `A` root, and has
positive exact endpoint signs for both `A` and `H`. The fourth cell needs only
width `10^-20`.

These decimal widths describe rational interval endpoints; no floating-point
sign is trusted. This extreme conditioning explains why the global cube sees
an apparently singular collar even though the exact transverse coefficient is
positive.

## What this closes—and what it does not

Closed: the exact bridge from the pure-line quartic equality fiber to a
positive first mixed radial coefficient in the reconstructed invariant chart.
This supplies the algebraic normal form needed by candidate-centered local
charts and independently matches the qualitative strict-local Hessian theorem.

Open: an explicit rational finite radius with quantitative remainder bounds,
the compact complement at the exact algebraic target, the second supported
`V5`, and the unrestricted quotient. The next certificate should turn the
qualitative domination above into four explicit semialgebraic collars.
