# Exact Reconstruction of the Unrestricted Spin(8) Dirac--Gram Margin

**Date:** 2026-08-07
**Status:** exact global polynomial identity and exact tangent-cone theorem;
global positivity remains open
**Structural certificate:**
[`spin8_dirac_unrestricted_structure.py`](../../src/spin8_dirac_unrestricted_structure.py)
**Grid evaluator:**
[`spin8_dirac_unrestricted_grid.py`](../../src/spin8_dirac_unrestricted_grid.py)
**Reconstruction:**
[`spin8_dirac_unrestricted_reconstruct.py`](../../src/spin8_dirac_unrestricted_reconstruct.py)
**Independent comparison:**
[`spin8_dirac_unrestricted_compare.py`](../../src/spin8_dirac_unrestricted_compare.py)
**Tangent certificate:**
[`spin8_dirac_unrestricted_tangent.py`](../../src/spin8_dirac_unrestricted_tangent.py)
**Full-sector energy certificate:**
[`spin8_dirac_unrestricted_energy.py`](../../src/spin8_dirac_unrestricted_energy.py)

## 1. Scope

Write the complete lower-triangular chart as

\[
x_1=e_0,
\qquad
x_2=a e_0+A e_1,
\]

\[
x_3=d e_0+D(e e_1+E e_2),
\]

\[
x_4=g e_0+G\left(
h e_1+H\left(i e_2+I(c e_3+s e_4)\right)
\right),
\]

with seven circle relations

\[
a^2+A^2=\cdots=c^2+s^2=1.
\]

The exact chart invariants are

\[
\Delta=\det(XX^{\mathsf T})=A^2D^2E^2G^2H^2I^2,
\qquad
\Phi=ADEGHIc.
\]

Consequently the normalized Cayley coordinate is exactly \(c\) at every
full-rank chart point.

The target remains the strengthened Dirac--Gram inequality

\[
\det I(X)
\leq
\Delta^3\frac{(1-c^2)^3(9-c^2)^2}{1024}.
\]

This report establishes the complete finite polynomial representation of its
margin. It does **not** yet establish the sign of that representation over the
entire seven-dimensional cube.

## 2. Exact sign quotient

The common signed-diagonal triality action induces a sign group of order
\(1024\) on the fourteen physical circle coordinates

\[
(a,A,d,D,e,E,g,G,h,H,i,I,c,s).
\]

Its character annihilator contains exactly sixteen sectors. Each sector has a
unique lower-coordinate character and a unique complementary-coordinate
character. Hence every sector has the exact form

\[
M_\mu
=
a^{\mu_a}A^{\nu_a}\cdots c^{\mu_c}s^{\nu_c}
P_\mu(a^2,d^2,e^2,g^2,h^2,i^2,c^2),
\]

where \(\mu,\nu\in\{0,1\}^7\) are fixed by triality and \(P_\mu\) is an
ordinary rational polynomial.

This is a symmetry theorem, not an interpolation observation: the complete
triality conjugacy action derives the sixteen characters before any determinant
grid is evaluated.

## 3. Exact boundary divisor and multidegree

All fourteen boundary branches were checked symbolically. On both branches of
each of

\[
A=0,\ D=0,\ E=0,\ G=0,\ H=0,\ I=0,\ s=0,
\]

the \(40\times28\) observation Jacobian has rank \(25\) and nullity \(3\).
The nullspace residual is identically zero in exact arithmetic. The two-branch
normal form in the seven-circle quotient ring therefore gives the common
divisor

\[
A^6D^6E^6G^6H^6I^6s^6
=
\Delta^3(1-c^2)^3.
\]

Every query projector has rank seven. Cauchy--Binet bounds each raw coordinate
pair degree by fourteen; removing the common sixth-power boundary factor and
the forced parity monomial leaves degree at most four in each squared
coordinate. Thus the unrestricted problem reduces exactly to sixteen bounded
polynomials in seven variables.

## 4. Two disjoint exact reconstructions

Two disjoint five-node rational half-angle grids were used. Each grid contains

\[
5^7=78{,}125
\]

chart points. At every point, sixteen orientation representatives were
evaluated with native FLINT rational matrices, giving

\[
16\cdot 5^7=1{,}250{,}000
\]

exact \(28\times28\) determinants per grid and \(2{,}500{,}000\) determinants
in total. The campaign was split into 25 atomically written, independently
hashed tiles per grid.

Tensor-product interpolation was performed independently on the two grids.
Every coefficient outside the predeclared degree box vanished exactly. More
strongly, all sixteen complete coefficient maps agree between the two grids,
coefficient for coefficient.

The observed multidegree is lower than the conservative bound:

- at most three in each of the six Gram residual coordinates;
- at most two in \(c^2\);
- several nontrivial sectors have still lower coordinate degrees.

The trivial sector contains 10,649 nonzero coefficients. The sparsest top
character contains only 182.

Finally, 32 rational points disjoint from both interpolation grids were
evaluated directly. All sixteen reconstructed sectors matched at every point:
512 additional exact determinant identities.

These facts establish the polynomial identities globally under the proved
multidegree ceiling. They are not a positivity argument.

## 5. Exact tangent cone along the equality line

Let

\[
u_a=a^2,\quad u_d=d^2,\quad u_e=e^2,\quad
u_g=g^2,\quad u_h=h^2,\quad u_i=i^2,
\qquad z=c^2.
\]

Along the orthonormal equality line \(u_a=\cdots=u_i=0\), the trivial Fourier
sector begins with

\[
\frac52(9-z)(5-z)(u_a+u_i)
+2(9-z)(3-z)(u_d+u_e+u_g+u_h).
\]

Only two nontrivial sectors occur at the same radial order. They couple
\((u_e,u_g)\) and \((u_d,u_h)\), with physical cross-term magnitude

\[
8(9-z)\sqrt{z}.
\]

Each coupled \(2\times2\) block therefore has discriminant

\[
16(9-z)^3(1-z)\geq0,
\qquad 0\leq z\leq1.
\]

Equivalently, its smaller eigenvalue is

\[
2(9-z)(3-z-2\sqrt z)
=
2(9-z)(1-\sqrt z)(3+\sqrt z)\geq0.
\]

All other thirteen nontrivial amplitudes vanish at radial order
\(3/2\) or higher. Hence the complete tangent cone is positive semidefinite
for the entire Cayley interval. Its only additional tangent degeneracy occurs
at the calibrated endpoint \(z=1\).

That endpoint degeneracy is not flat at the next order. For each of the
sixteen physical orientations, the quadratic form at \(z=1\) is the sum of
two signed squares,

\[
32(d\pm h)^2+32(e\pm g)^2.
\]

On its two-dimensional null cone, write the remaining free amplitudes as
\(q=d\) and \(p=e\), with the signs of \(h\) and \(g\) chosen according to
the orientation. Direct extraction from the exact seven-variable coefficient
maps gives the same quartic in every orientation:

\[
128(p^2+q^2)^2.
\]

Thus every nonzero tangent-null direction at the calibrated endpoint lifts
strictly positively at fourth order. This closes the exceptional endpoint of
the local analysis; it does not replace the still-required domain-wide
positivity certificate.

The full weighted endpoint blow-up is stronger. Put

\[
\begin{aligned}
a&=\varepsilon^2\alpha,& i&=\varepsilon^2\iota,&
d&=\varepsilon q,& e&=\varepsilon p,\\
h&=\sigma_d\varepsilon q+\varepsilon^2y,&
g&=\sigma_e\varepsilon p+\varepsilon^2x,&
s&=\varepsilon w,& c&=\sqrt{1-\varepsilon^2w^2},
\end{aligned}
\]

where \(\sigma_d,\sigma_e\in\{\pm1\}\) select the four tangent-null sign
types. Exact truncated arithmetic gives the same weighted degree-four form in
all four types:

\[
16\left[
5\alpha^2+5\iota^2
+8(p^2+q^2)^2
+4w^2(p^2+q^2)
+2x^2+2y^2
\right].
\]

This form is manifestly nonnegative and vanishes only at the blow-up origin.
It simultaneously controls motion away from the endpoint, transverse motion
away from each tangent-null relation, and activation of the two residual
directions that occur at higher weight.

## 6. A global coupled-core theorem on two-thirds of the Cayley interval

Let \(A_0\) be the trivial Fourier amplitude and let \(A_{eg}\) and
\(A_{dh}\) be the two amplitudes that occur at first radial order. Their
circle factors disappear after squaring, so

\[
C=A_0^2-A_{eg}^2-A_{dh}^2
\]

is an ordinary rational polynomial in the seven squared coordinates. Native
FLINT multiplication gives multidegree

\[
(6,6,6,6,6,6,4)
\]

and 522,747 nonzero power coefficients. After restricting
\(0\le z=c^2\le2/3\), the exact tensor-product Bernstein tensor has shape
\(7^6\times5\): all 588,245 scaled coefficients are nonnegative, 35 are
exactly zero, and none is negative. Therefore

\[
A_0^2\ge A_{eg}^2+A_{dh}^2,
\qquad
(u_a,u_d,u_e,u_g,u_h,u_i)\in[0,1]^6,\qquad 0\le z\le\frac23.
\]

The first coefficient obstructing this *native certificate basis* beyond that
interval is

\[
\frac{32}{9}(\tau-9)^2(\tau^2-14\tau+9),
\]

whose first positive root is \(7-2\sqrt{10}\). A negative Bernstein
coefficient beyond that point is not a counterexample to \(C\ge0\); it only
marks the reach of this unsplit basis.

This theorem controls the two first-order coupled modes globally on the stated
Cayley interval. It does not yet absorb the other thirteen Fourier amplitudes.

### 6.1 The complete Fourier-energy bound

The same exact construction includes all fifteen nontrivial amplitudes. After
inserting every proved circle-factor square, define

\[
E=A_0^2-\sum_{\mu\ne0}A_\mu^2.
\]

The integer-scaled polynomial has 525,665 nonzero power coefficients and the
same multidegree \((6,6,6,6,6,6,4)\). On
\([0,1]^6\times[0,2/3]\), its exact Bernstein tensor has 588,245
coefficients: 35 are zero and none is negative.

On the full cube, the unsplit tensor has only four negative controls. Their
indices show that every obstruction lies on one of the two coupled faces
\((u_e,u_g)\) or \((u_d,u_h)\), with all other residual coordinates zero.
The two face polynomials are identical after relabelling. For
\(2/3\le z\le1\), each square is covered by two symmetric triangular charts:

\[
r=x+y,\qquad q=\frac{4xy}{r^2}
\]

on \(x+y\le1\), and the same construction applied to \(1-x,1-y\) on the
complementary triangle. After removing the exact (r^2) boundary factor, the
lower chart has 175 nonnegative Bernstein coefficients; the upper chart has
225 strictly positive coefficients.

Extend the two faces back into the seven-cube with their degree-six Bernstein
boundary selectors. The exact identity is

\[
\begin{aligned}
E={}&E_{eg}(1-u_a)^6(1-u_d)^6(1-u_h)^6(1-u_i)^6\\
&+E_{dh}(1-u_a)^6(1-u_e)^6(1-u_g)^6(1-u_i)^6+R.
\end{aligned}
\]

The remainder \(R\) has a full \(7^6\times5\) Bernstein tensor with 495 zeros
and no negative coefficient. The face charts and remainder therefore prove
globally that

\[
A_0^2\geq\sum_{\mu\ne0}A_\mu^2
\qquad\text{on }[0,1]^7.
\]

By Hadamard orthogonality, the right-hand side is the mean-square deviation
of the sixteen physical orientation margins from their common mean \(A_0\).
The exact native Bernstein tensor of \(A_0\) itself has 12,288 controls, three
zeros, and no negative coefficient, so the mean is globally nonnegative.
Thus their RMS deviation is at most their mean globally. This absorbs every
Fourier sector into one exact aggregate inequality. It still does not imply that each
orientation margin is nonnegative: an RMS bound permits an isolated negative
outlier. The remaining gate is therefore an orientation-wise or stronger
one-sided certificate, not another reconstruction problem.

The theorem also settles the first four coefficients in an exact
orientation-invariant hierarchy. If \(e_k\) is the \(k\)-th elementary
symmetric polynomial of the sixteen margins, then

\[
e_1=16A_0\geq0,
\qquad
e_2=112A_0^2+8E\geq0.
\]

For the cubic coefficient, let \(T\) be the sum over the 35 unordered triples
of distinct nontrivial Walsh characters whose product is trivial. Exact
Newton expansion gives

\[
e_3=112A_0(4A_0^2+E)+32T.
\]

Writing \(a\) for the fifteen nontrivial amplitudes extended by zero at the
trivial character, one has \(6T=\langle a*a,a\rangle\). Young's convolution
inequality, Cauchy--Schwarz, the global energy theorem, and
\(\sqrt{15}\le4\) give

\[
|T|\le\frac23A_0^3,
\qquad
e_3\ge\frac{1280}{3}A_0^3\ge0.
\]

This analytic certificate avoids expanding the 4,411,890-term triple
polynomial.

The fourth coefficient follows from moments alone. Normalize
\(M_i=A_0(1+y_i)\), so that \(\sum y_i=0\) and
\(r=\sum y_i^2\le16\). Exact expansion gives

\[
\frac{e_4}{A_0^4}
=1820-\frac{91}{2}r+\frac{13}{3}\sum y_i^3
+\frac18r^2-\frac14\sum y_i^4.
\]

Using \(\sum y_i^3\ge-r^{3/2}\) and \(\sum y_i^4\le r^2\), the
lower envelope decreases to \(2348/3\) at \(r=16\). Therefore

\[
e_4\ge\frac{2348}{3}A_0^4\ge0.
\]

All sixteen margins are nonnegative exactly when every coefficient of

\[
\prod_\omega(t+M_\omega)
=t^{16}+e_1t^{15}+\cdots+e_{16}
\]

is nonnegative. A negative margin would otherwise produce a positive root.
Thus the remaining global gate can be attacked through the twelve
orientation-invariant, radical-free coefficients \(e_5,\ldots,e_{16}\),
rather than through sixteen separate radical-bearing margins.

### 6.2 Exact Cayley-endpoint Klein-four face

The high-energy numerical regime localizes near a boundary on which the
Fourier spectrum becomes sparse. This boundary can be settled exactly. On

\[
u_a=u_h=0,
\qquad
z=1,
\]

only three nontrivial sectors survive. Their masks are

\[
0011001,
\qquad
0101010,
\qquad
0110011,
\]

and the third is the binary sum of the first two. Together with the trivial
sector they form a Klein-four subgroup. The sixteen physical margins reduce
to the four eigenvalues of

\[
K=
\begin{pmatrix}
x&a&b&c\\
a&x&c&b\\
b&c&x&a\\
c&b&a&x
\end{pmatrix},
\]

each repeated four times.

The exact certificate proves all principal minors nonnegative on
\((u_d,u_e,u_g,u_i)\in[0,1]^4\). The quadratic and cubic obstructions reduce
to the already-certified one-mode triangular face plus nonnegative Bernstein
remainders. The determinant requires two nested boundary selectors. Its last
corner quotient factors as the one-mode minor times a \((5,6)\)-degree
polynomial with 42 nonnegative Bernstein controls; the final four-variable
remainder has 24,336 controls and no negative coefficient.

Therefore every physical margin is nonnegative on this complete endpoint
face. This is the first exact matrix-valued boundary continuation of the
global energy theorem. It does not cover other endpoint faces or the
unrestricted interior.

## 7. Numerical proof discovery and rejected shortcuts

A 100,000-point mixed interior/boundary GPU screen found the stronger
diagonal-dominance pattern

\[
A_0-\sum_{\mu\ne0}|A_\mu|>0
\]

at every sampled non-equality point. This is presently a proof-discovery
hypothesis, not a theorem.

Two tempting simplifications were tested and rejected:

1. the uniform Cauchy certificate
   \(A_0^2\geq15\sum_{\mu\ne0}A_\mu^2\) is false;
2. fixed independent weights \(|A_\mu|\leq\lambda_\mu A_0\) are too loose,
   because the sampled sector-wise suprema sum to about \(2.96\).

These negative results show that any successful dominance proof must exploit
the mutually exclusive boundary geometry of the forced monomials, rather than
treating the fifteen sectors independently.

## 8. Claim boundary and next exact gate

The following statements are established exactly:

- the complete seven-circle sign quotient;
- the global divisor \(\Delta^3(1-c^2)^3\);
- conservative multidegree at most four in every squared coordinate;
- identical unrestricted coefficient maps from two disjoint grids;
- 32 complete off-grid exact holdouts;
- the positive-semidefinite tangent cone along the full orthonormal equality
  line;
- the strictly positive quartic lift of all sixteen endpoint tangent-null
  cones;
- the positive-definite weighted leading form for the complete singular
  endpoint blow-up;
- the global coupled-core inequality
  \(A_0^2\ge A_{eg}^2+A_{dh}^2\) for \(0\le c^2\le2/3\);
- the stronger full-sector Fourier-energy inequality
  \(A_0^2\ge\sum_{\mu\ne0}A_\mu^2\) on the complete seven-cube.

The following statement remains open:

\[
M_\chi\geq0
\quad\text{for all sixteen physical orientations and every point of }[0,1]^7.
\]

The next exact campaign should combine two routes. The invariant route should
construct \(e_5,\ldots,e_{16}\) through the finite Walsh group algebra and
test them in increasing order. The boundary route should classify the Walsh
subgroups surviving on the remaining Cayley-endpoint strata and apply
group-circulant principal-minor certificates before returning to the
interior. Both routes remove orientation radicals. Boundary-supported
Bernstein decompositions should be attempted before any large generic SOS
relaxation. Neither a finite GPU screen nor the reconstructed identities alone
may be promoted to the unrestricted inequality.
