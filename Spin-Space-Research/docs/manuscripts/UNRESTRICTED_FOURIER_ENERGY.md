# A Global Fourier-Energy Inequality for Unrestricted Spin(8) Triality Sensor Margins

**Working computer-assisted theorem note — 2026-08-07**
**Status:** exact theorem on the complete seven-circle chart
**Certificate:**
[spin8_dirac_unrestricted_energy.py](../../src/spin8_dirac_unrestricted_energy.py)

## Abstract

The unrestricted balanced \(\operatorname{Spin}(8)\) triality sensor gives
sixteen orientation-dependent determinant margins after quotienting the exact
signed-diagonal symmetry. Their Walsh expansion has one trivial amplitude
\(A_0\) and fifteen nontrivial amplitudes \(A_\mu\). This note proves

\[
A_0^2\geq\sum_{\mu\ne0}A_\mu^2
\]

on the complete seven-dimensional Gram--Cayley cube. Equivalently, the
root-mean-square deviation of the sixteen physical margins from their common
mean is no larger than that mean.

The proof is exact. A native \(7^6\times5\) Bernstein tensor certifies the
low-Cayley region and isolates exactly four high-Cayley control obstructions.
All four lie on two isomorphic coupled boundary faces. Symmetric triangular
blow-ups certify those faces, while degree-six Bernstein boundary selectors
split the full polynomial into two nonnegative face extensions and a
588,245-control nonnegative remainder. The theorem is a global second-moment
law. It does not imply that every physical margin is nonnegative, so the
unrestricted cubic Gram-volume inequality remains open.

## 1. The sixteen orientation margins

Let \(M_\omega\), with \(\omega\in(\mathbb Z/2\mathbb Z)^4\), denote the
sixteen physical margins reconstructed on the unrestricted seven-circle
chart. Exact triality conjugacy gives the Walsh expansion

\[
M_\omega=A_0+\sum_{\mu\ne0}\chi_\mu(\omega)A_\mu,
\]

where every character value is \(\pm1\). After extracting the proved parity
monomial from each sector, the residual parts are rational polynomials in

\[
(u_a,u_d,u_e,u_g,u_h,u_i,z)
=(a^2,d^2,e^2,g^2,h^2,i^2,c^2)\in[0,1]^7.
\]

Squaring an amplitude removes every remaining circle radical. Hence

\[
E=A_0^2-\sum_{\mu\ne0}A_\mu^2
\]

is an ordinary rational polynomial on the seven-cube.

## 2. Global theorem and Parseval consequence

### Theorem 2.1

For every point of the unrestricted seven-circle chart,

\[
E\geq0.
\]

The integer-scaled polynomial has 525,665 nonzero power coefficients and
multidegree \((6,6,6,6,6,6,4)\).

The mean has the required sign: the exact Bernstein tensor of \(A_0\), at
multidegree \((3,3,3,3,3,3,2)\), has 12,288 controls, three zeros, and no
negative coefficient. Thus \(A_0\geq0\) on the complete cube.

### Corollary 2.2

Writing

\[
\overline M=\frac1{16}\sum_\omega M_\omega=A_0,
\]

one has

\[
\frac1{16}\sum_\omega(M_\omega-\overline M)^2
=\sum_{\mu\ne0}A_\mu^2
\leq \overline M^2.
\]

The equality is Parseval's identity for the sixteen Walsh characters. The
inequality is Theorem 2.1 together with the certified sign \(A_0\geq0\).

### Corollary 2.3 (first four orientation invariants)

Let \(e_k\) be the elementary symmetric polynomials of the sixteen margins.
Then

\[
e_1=16A_0\geq0,
\qquad
e_2=112A_0^2+8E\geq0,
\qquad
e_3\geq\frac{1280}{3}A_0^3\geq0,
\qquad
e_4\geq\frac{2348}{3}A_0^4\geq0.
\]

Indeed, Walsh orthogonality gives

\[
\sum_\omega M_\omega^2
=16\left(A_0^2+\sum_{\mu\ne0}A_\mu^2\right),
\]

and the displayed identity follows from
\(2e_2=(\sum M_\omega)^2-\sum M_\omega^2\).

For \(e_3\), let \(a\) be the function on
\((\mathbb Z/2\mathbb Z)^4\) whose value at the trivial character is zero and
whose other fifteen values are the nontrivial Walsh amplitudes. If \(T\) is
the sum over the 35 unordered triples of distinct nontrivial characters whose
product is trivial, then

\[
e_3=112A_0(4A_0^2+E)+32T.
\]

Every unordered triple occurs six times in the convolution inner product, so

\[
6T=\langle a*a,a\rangle.
\]

Young's convolution inequality and Cauchy--Schwarz give

\[
|T|
\leq\frac{\sqrt{15}}6
\left(\sum_{\mu\ne0}A_\mu^2\right)^{3/2}
\leq\frac23A_0^3,
\]

where the final step uses Theorem 2.1 and \(\sqrt{15}\leq4\). Since
\(E\geq0\),

\[
e_3
\geq448A_0^3-\frac{64}{3}A_0^3
=\frac{1280}{3}A_0^3.
\]

This proof avoids expanding the 4,411,890-term triple polynomial.

For \(e_4\), assume first that \(A_0>0\) and write
\(M_i=A_0(1+y_i)\). The energy theorem gives

\[
\sum_i y_i=0,\qquad r=\sum_i y_i^2\leq16.
\]

Expanding the fourth elementary symmetric polynomial around the all-ones
vector and applying Newton's identities to the centered variables gives

\[
\frac{e_4}{A_0^4}
=1820-\frac{91}{2}r+\frac{13}{3}\sum_i y_i^3
+\frac18r^2-\frac14\sum_i y_i^4.
\]

The norm inequalities

\[
\sum_i y_i^3\geq-r^{3/2},
\qquad
\sum_i y_i^4\leq r^2
\]

give the lower envelope

\[
1820-\frac{91}{2}r-\frac{13}{3}r^{3/2}-\frac18r^2.
\]

It decreases on \(0\leq r\leq16\), and its endpoint value is
\(2348/3\). If \(A_0=0\), the energy theorem forces every nontrivial
amplitude to vanish, so \(e_4=0\).

## 3. The four native obstructions

On \([0,1]^6\times[0,2/3]\), the exact Bernstein transform of \(E\) has
588,245 coefficients: 35 vanish exactly and none is negative.

On the complete cube, the same native basis has only four negative controls:

\[
\begin{aligned}
&(0,0,1,1,0,0,3),\qquad(0,0,1,1,0,0,4),\\
&(0,1,0,0,1,0,3),\qquad(0,1,0,0,1,0,4).
\end{aligned}
\]

Thus every native obstruction lies on one of the coupled coordinate faces
\((u_e,u_g)\) or \((u_d,u_h)\), and only in the two highest Cayley layers. A
negative Bernstein control is not a negative value of the polynomial. Here it
identifies exactly where the basis must be adapted.

## 4. Exact coupled-face charts

Set all residual coordinates except \(u_e=x\) and \(u_g=y\) to zero, and call
the resulting polynomial \(E_{eg}(x,y,z)\). Exact coefficient comparison
shows that the \((u_d,u_h)\) face is the same polynomial after relabelling,
and

\[
E_{eg}(x,y,z)=E_{eg}(y,x,z).
\]

For \(2/3\leq z\leq1\), split the unit square into two triangles. On
\(x+y\leq1\), use

\[
r=x+y,\qquad q=\frac{4xy}{r^2},\qquad z=\frac23+\frac t3.
\]

The arithmetic-geometric mean inequality gives \(0\leq q\leq1\). The exact
factor \(r^2\) divides the transformed face. After division, the polynomial
has multidegree \((6,4,4)\), and all 175 Bernstein coefficients are
nonnegative.

On \(x+y\geq1\), apply the same construction to \(1-x\) and \(1-y\). The
transformed polynomial has multidegree \((8,4,4)\), and all 225 Bernstein
coefficients are strictly positive. Together with the low-Cayley tensor,
these charts prove both coupled faces nonnegative on their complete domains.

## 5. Boundary-supported decomposition

Define the two face extensions

\[
\begin{aligned}
\widehat E_{eg}
&=E_{eg}(1-u_a)^6(1-u_d)^6(1-u_h)^6(1-u_i)^6,\\
\widehat E_{dh}
&=E_{dh}(1-u_a)^6(1-u_e)^6(1-u_g)^6(1-u_i)^6.
\end{aligned}
\]

The sixth powers are not arbitrary multipliers. In degree-six Bernstein form,
\((1-u)^6\) has a single nonzero control, at boundary index zero. These
factors therefore extend each face certificate without leaking into unrelated
control layers.

Exact subtraction gives

\[
E=\widehat E_{eg}+\widehat E_{dh}+R.
\]

The complete Bernstein tensor of \(R\) again has 588,245 controls: 495 are
zero and none is negative. Since both face extensions and \(R\) are
nonnegative, Theorem 2.1 follows.

## 6. Interpretation and limitation

The theorem shows that the total nontrivial Fourier energy of the orientation
margins can never exceed the square of their mean. It also shows that the
apparent seven-variable certificate obstruction was concentrated entirely on
two low-dimensional boundary geometries.

The conclusion is deliberately one step short of the unrestricted
Dirac--Gram theorem. A mean can dominate the root-mean-square fluctuation while
one member of a sixteen-point family is still negative. The remaining proof
obligation is one-sided:

\[
M_\omega\geq0
\qquad\text{for every \(\omega\) and every point of \([0,1]^7\).}
\]

The global energy inequality materially narrows that obligation, but does not
silently discharge it.

One saturation stratum is now understood exactly. On
\(u_a=u_h=0,\ z=1\), the Fourier support collapses to a Klein-four subgroup.
The companion
[endpoint theorem](UNRESTRICTED_ENDPOINT_KLEIN_FACE.md) proves the associated
\(4\times4\) group-circulant matrix positive semidefinite on the complete
remaining four-cube. Thus near-saturation caused by a sparse boundary spectrum
is harmless on that face; other boundary strata and the unrestricted interior
remain open.

There is an exact orientation-free continuation. Form

\[
q(t)=\prod_\omega(t+M_\omega)
=t^{16}+e_1t^{15}+\cdots+e_{16}.
\]

The margins are real. Therefore they are all nonnegative if and only if all
coefficients \(e_1,\ldots,e_{16}\) are nonnegative. The forward implication is
immediate. Conversely, a polynomial with nonnegative coefficients has no
positive real root, whereas a negative margin would give the positive root
\(t=-M_\omega\). Corollary 2.3 settles the first four coefficients globally.
The remaining unrestricted gate is equivalently the positivity of twelve
orientation-invariant polynomials \(e_5,\ldots,e_{16}\).

## 7. Reproducibility

The proof artifact is
[spin8_dirac_unrestricted_energy_20260807.json](../../artifacts/spin8_dirac_unrestricted_energy_20260807.json).
The focused theorem test reconstructs the energy polynomial from the exact
sector maps, checks the four native obstruction indices, replays both
triangular face charts, and verifies the global remainder tensor. The artifact
is covered by [ARTIFACTS.sha256](../../ARTIFACTS.sha256).
