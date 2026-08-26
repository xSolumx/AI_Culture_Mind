# Mathematical design: Albert memory and exceptional transport

## One carrier, six nested actions

Let

\[
J=H_3(\mathbb O),\qquad \dim_{\mathbb R}J=27,
\]

with Jordan product (x\circ y=(xy+yx)/2). Its automorphism algebra is

\[
\mathfrak f_4=\operatorname{Der}(J),\qquad \dim\mathfrak f_4=52.
\]

The reduced structure algebra is

\[
\mathfrak e_{6(-26)}
=\operatorname{Der}(J)\oplus L(J_0)
=\mathfrak f_4\oplus 26,
\]

where (J_0) is trace-free and (L_x(y)=x\circ y). The code constructs the
52 derivations as commutators of Jordan left multiplications and appends the 26
traceless self-adjoint directions. Both (F_4) and (E_{6(-26)}) preserve the
Albert cubic determinant; only (F_4) preserves the positive trace metric.

Fixing one primitive diagonal idempotent gives a 36-dimensional Spin(9)
stabilizer. Fixing the full ordered diagonal Jordan frame gives a
28-dimensional Spin(8) stabilizer. Inside that alignment, fixing one scalar
octonion unit gives the 21-dimensional vector-stabilizer Spin(7); fixing the
corresponding units across the triality carriers gives the 14-dimensional G2
automorphism algebra. Thus the former separate local programmes become
restrictions of one action carrier:

\[
G_2\subset\mathrm{Spin}(7)\subset\mathrm{Spin}(8)\subset
\mathrm{Spin}(9)\subset F_4\subset E_{6(-26)}.
\]

The numerical stabilizer bases are runtime alignments inside the exact-data
(F_4) span; they are not relabelled as new exact subgroup theorems.

## Exponential action without a false global-chart claim

For a generator bank (G_a), one factor is

\[
R(\theta)=\exp\!\left(\sum_a\theta_aG_a\right).
\]

The exponential map keeps every learned tangent step inside the connected
matrix group. Compact connected (F_4) is covered by exponentials. A general
noncompact connected group need not be the image of one exponential, so the
implementation also supports

\[
R_t=\exp(A_{t,m})\cdots\exp(A_{t,1}).
\]

The factor count is a model choice, not a theorem baked into the scan.

For the Cartan decomposition

\[
\mathfrak e_{6(-26)}=\mathfrak f_4\oplus\mathfrak p,
\]

v1.3 also provides two structured global-action hypotheses. The polar chart is

\[
R=K\exp(P),\qquad K\in F_4,\quad P\in\mathfrak p,
\]

and the Cartan/KAK chart is

\[
R=K_1 A K_2,
\]

where the implemented two-dimensional (A) is the diagonal determinant-one
radial subgroup. These charts expose compact frame transport separately from
noncompact content scaling. They do not assert uniqueness of the factors.

The direct chart remains the default. This is not a retreat to a global-chart
claim: successive tokens compose their exponentials in the recurrent action,
so temporal depth already generates products of Lie exponentials. On the
local GPU, the direct chart matched polar E6 at the 50-update Shakespeare gate
and reduced complete training cost. Polar and KAK remain executable whenever
explicit within-token factor separation is the hypothesis under test.

## Rank-r independent delta memory

Let (S_t\in\mathbb R^{H\times V}), with (V=27) for the built-in
exceptional actions. Define

\[
A_t=\left(I-\sum_{j=1}^r k_{t,j}e_{t,j}^{\mathsf T}\right)D_t,
\qquad
B_t=\sum_{j=1}^r k_{t,j}z_{t,j}^{\mathsf T}.
\]

Then

\[
S_t=A_tS_{t-1}R_t^{\mathsf T}+B_t,
\qquad
o_t=S_t^{\mathsf T}q_t.
\]

Tying (e_{t,j}=\beta_{t,j}k_{t,j}) recovers block DeltaRule. Version 1.3.1
uses this contractive tied law by default because unconstrained independent
erase does not certify stability. Independent and unconstrained directions
remain explicit falsifiers. A separate sigmoid write-strength gate controls
the additive term, and queries are normalized by default.

For noncompact E6(-26), bounded left retention alone is insufficient because
the right action can expand. The direct and polar charts therefore compensate
retention using a conservative bound on the symmetric tangent:

\[
\rho_{\mathrm{effective}}
=\rho\exp(-\lVert P\rVert_F).
\]

Compact G2 through F4 have zero compensation in the invariant trace metric.

## Readout keeps both direction and exceptional scale

Plain RMS normalization quotients away the magnitude of a memory read. That is
an accidental restriction for overwrite memory, not an exceptional-algebra
principle. The default Albert readout therefore concatenates

\[
\operatorname{RMSNorm}(x),\quad
\frac{\operatorname{tr}(x)}{\sqrt 3},\quad
\log(1+\|x\|_2^2/27),\quad
\frac{\det(x)}{\sqrt{1+\det(x)^2}}.
\]

The first term supplies a stable 27D direction; the three scalar channels
restore linear scale, energy, and the bounded cubic exceptional invariant.
This is an architectural hypothesis with gradient tests, not a claim that
these are sufficient statistics for language.

## The actual associative object

A transition is the triple ((A,R,B)) acting by

\[
T(S)=ASR^{\mathsf T}+B.
\]

Chronological composition is

\[
(A_2,R_2,B_2)\circ(A_1,R_1,B_1)
=\left(A_2A_1,\ R_2R_1,\ B_2+A_2B_1R_2^{\mathsf T}\right).
\]

This operation is associative, so it supports a prefix scan. Neither raw
octonion multiplication nor the Albert Jordan product is claimed associative;
they are used pointwise to construct linear operators and nonlinear channel
features.

### Identity specialization and numerical algebra

When (R_t=I), the value-axis action is not merely cheap; it is absent from the
mathematics. The transition reduces to

\[
(A_2,B_2)\circ(A_1,B_1)
=\left(A_2A_1,\ B_2+A_2B_1\right).
\]

The dedicated one-sided compiler implements this smaller monoid directly. It
does not construct identity matrices or multiply value-space prefixes, and it
is bitwise float32 equal to the generic identity specialization in the tested
model path.

The Albert determinant has two implemented formulas. The explicit cubic

\[
abc-a\lVert z\rVert^2-b\lVert y\rVert^2-c\lVert x\rVert^2
+2\operatorname{Re}((xz)\bar y)
\]

equals the determinant recovered from Jordan traces in exact algebra and in
float64 value/gradient tests. It is not interchangeable during finite-
precision optimization: the changed contraction order perturbs float32
gradients, and a prospective five-seed Shakespeare gate detected a mean
quality regression. The default therefore deliberately retains the denser
Jordan-trace evaluation. This is a numerical-analysis decision, not a change
to the Albert algebra.

## Why the Riemann sphere and PGL are controls, not the core

The Riemann sphere is (\mathbb{CP}^1), and (PGL(2,\mathbb C)) acts on it by
Möbius transformations. This gives a clean two-coordinate projective-router
control, implemented in `projective.py`.

Projectivizing the recurrent memory itself would identify (x\) and
\(\lambda x\). That deletes amplitude, while delta overwrite needs amplitude
to distinguish what was written and how strongly it should be erased.
Furthermore (PGL(2,\mathbb C)) is noncompact and is not an (F_4) chart.
The exceptional analogue of projective geometry is instead the octonionic
projective plane (\mathbb OP^2=F_4/\mathrm{Spin}(9)), realized by rank-one
Albert elements. This may become a router/address manifold without replacing
the 27D memory.

## Carrier horizon

The 27D Albert carrier ends honestly at E6(-26). Continuing the compatible
real-form chain to E7(-25) requires the 56D Freudenthal system with its
symplectic and quartic invariants. A linear E8(-24) continuation requires the
248D adjoint carrier; the 57D quasiconformal realization is nonlinear and does
not satisfy this affine scan law. The exact boundary and branching data are in
[`EXCEPTIONAL_LADDER.md`](EXCEPTIONAL_LADDER.md).

## Sources and horizon

- [Exponential map in Lie theory](https://en.wikipedia.org/wiki/Exponential_map_in_Lie_theory),
  [Riemann sphere](https://en.wikipedia.org/wiki/Riemann_sphere), and
  [projective linear group](https://en.wikipedia.org/wiki/Projective_linear_group)
  are the three user-supplied orientation links.
- [Kollross, octonions, triality and (F_4)](https://arxiv.org/abs/1802.08075)
  gives an explicit octonionic (F_4) construction.
- [Bernardoni et al., geometry of (F_4)](https://arxiv.org/abs/0705.3978)
  realizes (F_4) as Albert automorphisms and (\mathbb OP^2=F_4/Spin(9)).
- [Corradetti et al., octonionic planes and real forms](https://arxiv.org/abs/2203.02671)
  connects Albert rank-one elements, projective/hyperbolic planes, and real
  forms of (F_4) and (E_6).
- [Dray, Manogue and Wilson, a division-algebra representation of (E_7)](https://arxiv.org/abs/2401.10534)
  motivates the honest next extension: a 56D Freudenthal carrier with its
  symplectic and quartic invariants. That is a future theorem and kernel gate,
  not a feature name attached to the present model.
