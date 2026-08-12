# An Exact Cubic Gram-Volume Inequality for Signed-Star Spin(8) Triality Sensors

**Working computer-assisted theorem manuscript — 2026-08-06**
**Status:** exact theorem on the full four-parameter signed-star subfamily
**Certificate:** [`spin8_dirac_star.py`](../../src/spin8_dirac_star.py)

## Abstract

The balanced \(\operatorname{Spin}(8)\) triality sensor associates a
\(28\times28\) information Gram operator \(I(X)\) with four moving
unit probes. A cubic Gram-volume conjecture predicts that correlations cannot
improve the determinant after the natural volume penalty is removed. This
work proves that inequality on the full signed-star subfamily, in which three
probes correlate independently with a common orthonormal-frame direction and
the polar-orthonormalized frame has arbitrary signed Cayley calibration.

The determinant reduces exactly to an even polynomial plus one orientation
character. Two independent rational interpolation grids recover identical
coefficient maps, 32 off-grid signed determinants agree exactly, and two
tensor-product Bernstein certificates prove the required inequalities on the
four-cube. A structural compression yields an additional result: the
orientation polynomial factors by

\[
(1-u)(v-w)(1-z)^3,
\]

and the strengthened inequality is strict throughout
\(0<u,v,w,z<1\). Exact Bernstein-support analysis further shows that the
normalized equality set is precisely \(z=1\) or \(u=v=w=0\). The theorem is
exact for the signed-star subfamily; three residual Cholesky correlations
remain outside its domain, so the unrestricted Dirac--Gram conjecture remains
open.

## 1. Information geometry and target inequality

Use the three triality representations and query projectors
\(P_\alpha(x)\) defined in the companion Cayley-spectrum manuscript. Fix
one vector probe and consider a balanced \((2,2,1)\) allocation. Let

\[
X=\begin{bmatrix}x_1^{\mathsf T}\\x_2^{\mathsf T}\\x_3^{\mathsf T}\\x_4^{\mathsf T}\end{bmatrix}
\in\mathbb R^{4\times8}
\]

have unit rows and positive-definite Gram matrix

\[
\Gamma=XX^{\mathsf T},
\qquad \Delta=\det\Gamma>0.
\]

Its symmetric row whitening, or polar orthonormalization, is

\[
Q=\Gamma^{-1/2}X,
\qquad QQ^{\mathsf T}=I_4.
\]

The proposed cubic Gram-volume inequality is

\[
\boxed{
\det I(X)\le \Delta^3\det I(Q)
}.
\]

Earlier project notes call this the *strengthened Dirac--Gram inequality*.
That is repository terminology, not an established use of “Dirac inequality”
in the mathematical literature. The descriptive name above is used here to
avoid confusion. Section 3 proves that the exponent three is the exact generic
loss order on each of the three star boundary faces.

## 2. The full signed-star subfamily

Let \(q_1,q_2,q_3,q_4\) be an orthonormal four-frame. Define

\[
\begin{aligned}
x_1&=q_1,\\
x_2&=a q_1+Aq_2,\\
x_3&=d q_1+Dq_3,\\
x_4&=g q_1+Gq_4,
\end{aligned}
\]

where

\[
A^2=1-a^2,\qquad D^2=1-d^2,\qquad G^2=1-g^2.
\]

Write the orthonormal row frame as

\[
\widehat Q=
\begin{bmatrix}
q_1^{\mathsf T}\\q_2^{\mathsf T}\\q_3^{\mathsf T}\\q_4^{\mathsf T}
\end{bmatrix},
\qquad
L=
\begin{bmatrix}
1&0&0&0\\
a&A&0&0\\
d&0&D&0\\
g&0&0&G
\end{bmatrix}.
\]

Then \(X=L\widehat Q\), and the Gram matrix is explicitly

\[
\Gamma=LL^{\mathsf T}
=\begin{bmatrix}
1&a&d&g\\
a&1&ad&ag\\
d&ad&1&dg\\
g&ag&dg&1
\end{bmatrix}.
\]

This display is the meaning of *signed star*: the three correlations with the
first row are free, while the other three are their star products. A general
four-row Cholesky chart has three additional residual correlations and lies
outside this theorem.

Set

\[
u=a^2,\qquad v=d^2,\qquad w=g^2,
\]

and let

\[
c=\Phi(q_1,q_2,q_3,q_4),
\qquad z=c^2.
\]

The two orthonormal frames must not be conflated. Since

\[
Q=(LL^{\mathsf T})^{-1/2}L\widehat Q=O\widehat Q,
\qquad O\in O(4),
\]

alternation of the Cayley form gives

\[
\Phi(Q)=\det(O)\Phi(\widehat Q),
\qquad \Phi(Q)^2=z.
\]

The polar matrix \(O\) need not respect the positive/negative \(2+2\) row
split. Proposition 2.2 of the companion
[*Balanced Cayley Information Spectra*](CAYLEY_INFORMATION_SPECTRUM.md)
handles precisely this point: the classical four-plane orbit theorem and the
exact internal-split isotropy audit together show that every balanced
orthonormal information flag with Cayley square \(z\) is equivalent, not merely under
within-pair \(O(2)\times O(2)\) changes. Its exact isotropy certificate proves
that the four-plane stabilizer acts as all of \(SO(4)\). Therefore

\[
\det I(Q)=\frac{T(z)}{1024},
\qquad T(z)=(1-z)^3(9-z)^2.
\]

The Gram determinant is

\[
\Delta=(1-u)(1-v)(1-w)=A^2D^2G^2.
\]

The Cayley value of the nonorthogonal frame is \(ADGc\), so the signed
orientation monomial entering the normalized determinant is

\[
\omega=adg\,ADGc,
\qquad
\omega^2=uvw(1-u)(1-v)(1-w)z.
\]

## 3. Why a finite polynomial reconstruction is a proof

Work in the circle quotient

\[
\mathcal R=
\frac{\mathbb Q[a,A,d,D,g,G,c,s]}
{(a^2+A^2-1,\ d^2+D^2-1,\ g^2+G^2-1,\ c^2+s^2-1)}.
\]

### Lemma 3.1 — exact boundary rank loss

With the row/view order displayed explicitly, the observation Jacobian is

\[
J=
\begin{bmatrix}
J_V(e_0)\\
J_+(e_0)\\
J_+(a e_0+A e_1)\\
J_-(d e_0+D e_2)\\
J_-(g e_0+G(c e_3+s e_4))
\end{bmatrix}
\in\mathbb R^{40\times28}.
\]

On every circle branch of each star boundary, its symbolic rank over the
corresponding rational-function field is 25:

| Boundary | Circle branches | Generic rank | Universal rank upper bound |
|---|---:|---:|---:|
| \(A=0,\ a=\pm1\) | 2 | 25 | 25 |
| \(D=0,\ d=\pm1\) | 2 | 25 | 25 |
| \(G=0,\ g=\pm1\) | 2 | 25 | 25 |

The certificate stores three exact nullspace vectors for each of the six
branches and verifies every residual entry is zero. Symbolic rank is the
maximum rank of a polynomial matrix, so special parameter values may reduce
the rank further but cannot raise it above 25.

### Lemma 3.2 — rank loss forces minor divisibility

Let \(J(t)\) be a polynomial \(m\times n\) matrix with
\(\operatorname{rank}J(0)\le n-r\). Every \(n\times n\) minor of \(J(t)\)
is divisible by \(t^r\).

Indeed, expand a selected determinant multilinearly in the columns of
\(J(t)=J(0)+tJ_1+t^2J_2+\cdots\). A term of order below \(r\) contains more
than \(n-r\) columns from \(J(0)\), so those columns are dependent and the term
vanishes. The same proof applies coefficientwise to an analytic power-series
matrix, which is the form obtained after selecting either circle branch.
Cauchy--Binet then gives

\[
\det(J(t)^{\mathsf T}J(t))
=\sum_S\det(J_S(t))^2,
\]

which is divisible by \(t^{2r}\). Applying the lemma with \(n=28\) and
\(r=3\) yields sixth-order divisibility in \(A,D,G\).

### Lemma 3.3 — the three divisibilities combine in the circle quotient

For the stated lexicographic order, the four circle relations have leading
monomials \(a^2,d^2,g^2,c^2\). Hence \(\mathcal R\) is a free rank-16 module
over \(\mathbb Q[A,D,G,s]\), with basis

\[
\{a^\epsilon d^\eta g^\theta c^\kappa:
\epsilon,\eta,\theta,\kappa\in\{0,1\}\}.
\]

To see divisibility without treating \(a\) and \(A\) as independent, write
each normal-form coefficient as \(p(A)+a q(A)\). On the two analytic branches
\(a=\pm\sqrt{1-A^2}\), Lemmas 3.1–3.2 give sixth-order vanishing in \(A\).
Taking the branch sum and difference shows that \(A^6\) divides both \(p\)
and \(q\), since \(\sqrt{1-A^2}\) is a unit at \(A=0\). Repeating the argument
for \(D\) and \(G\), and using coprimality of these three diagonal variables,
proves

\[
A^6D^6G^6=\Delta^3\mid\det I(X)
\quad\text{in }\mathcal R.
\]

The power is generically exact rather than merely a lower bound. The
reconstructed normalized determinant has the following exact nonzero face
values:

\[
\begin{array}{c|c}
\text{face evaluation}&1024\det I(X)/\Delta^3\\ \hline
u=1,\ v=w=z=0&25/2\\
v=1,\ u=w=z=0&75/2\\
w=1,\ u=v=z=0&75/2
\end{array}
\]

Thus no fourth generic power of \(1-u\), \(1-v\), or \(1-w\) divides the
normalized determinant.

### Lemma 3.4 — exact parity module

The maintained signed-diagonal \(\operatorname{Spin}(7)\) actions fixing
\(e_0\), the projective identities \(P_\alpha(x)=P_\alpha(-x)\), and the
parameter redundancy

\[
(G,c,s)\mapsto(-G,-c,-s)
\]

generate a 128-element sign group on
\((a,A,d,D,g,G,c,s)\). Exact common-adjoint conjugacy is checked in all three
triality representations. The annihilator of this sign group contains exactly
two parity characters:

\[
(0,0,0,0,0,0,0,0),
\qquad
(1,1,1,1,1,1,1,0).
\]

Consequently the invariant submodule is precisely

\[
\mathbb Q[u,v,w,z]
\oplus
aAdDgGc\,\mathbb Q[u,v,w,z].
\]

There are no independent terms odd only in \(a\), \(d\), \(g\), or any other
proper subset. This symmetry lemma—not interpolation—is what makes the
two-sector ansatz exhaustive.

### Lemma 3.5 — finite degree space

After division by \(\Delta^3\), the normalized determinant has the form

\[
\frac{1024\det I(X)}{\Delta^3}
=F(u,v,w,z)+\omega H(u,v,w,z),
\]

with conservative multidegree bounds

\[
\deg F\le(4,4,4,7),
\qquad
\deg H\le(3,3,3,6).
\]

Stack the five query Jacobians into a \(40\times28\) matrix \(J\), so
\(I=J^{\mathsf T}J\). By Cauchy--Binet, \(\det I\) is a sum of squares
of \(28\times28\) minors of \(J\). Every maintained generator is exactly
skew-symmetric, so \(G_rx\in x^\perp\) and each query block has rank at most
seven. A nonzero minor therefore uses at most seven rows from any varying
query block. Since a selected row is linear in a circle pair, a squared minor
has pair degree at most 14.

Division by \(A^6,D^6,G^6\) leaves pair degree at most eight. The even sector
therefore has degrees at most four in \(u,v,w\); extracting \(aA,dD,gG\)
from the orientation sector lowers those bounds to three. No Gram factor is
removed from the Cayley pair, giving bounds seven for \(F\) and six after
extracting \(c\) for \(H\). This proves the displayed conservative
multidegrees. ∎

### Exact reconstruction protocol

Put

\[
\rho(t)=\left(\frac{2t}{1+t^2}\right)^2.
\]

The exact circle-parameter nodes are:

| Grid | Spatial parameters \(t\) for \(u,v,w\) | Cayley parameters \(t\) for \(z\) |
|---|---|---|
| Discovery | \(0,1/8,1/4,1/2,3/4\) | \(0,1/10,1/6,1/4,1/3,1/2,2/3,3/4\) |
| Confirmation | \(0,1/10,1/5,2/5,3/5\) | \(0,1/12,1/7,1/5,2/7,2/5,3/5,4/5\) |

The interpolation nodes are the corresponding exact rational numbers
\(\rho(t)\). They are distinct within each coordinate list. Every complementary
circle coordinate is nonzero, so \(\Delta\ne0\) throughout both grids; the
odd-sector division also omits the zero anchor, making every factor of
\(\omega\) used there nonzero.

The finite-dimensional ansatz is reconstructed twice:

1. a discovery grid with five rational-circle nodes for each of
   \(u,v,w\) and eight for \(z\);
2. an independently chosen, nonidentical confirmation grid of the same
   cardinalities—the grids share only the all-zero tensor anchor;
3. exact even/odd separation using both signs of \(c\);
4. canonical rational coefficient serialization and byte-level hash
   comparison;
5. 16 additional rational frames, each checked in both orientations.

Both grids recover

| Polynomial | Exact multidegree | Nonzero monomials |
|---|---:|---:|
| \(F\) | \((3,3,3,5)\) | 360 |
| \(H\) | \((2,2,2,4)\) | 86 |

The even interpolation basis is
\(u^iv^jw^kz^\ell\), with \(0\le i,j,k\le4\) and \(0\le\ell\le7\), and has
dimension \(5^3\cdot8=1000\). The odd basis has
\(0\le i,j,k\le3\), \(0\le\ell\le6\), and dimension
\(4^3\cdot7=448\). Distinct nodes make each univariate Vandermonde matrix
invertible, so their tensor products determine unique polynomials in these
spaces.

All 32 off-grid determinant residuals are exactly zero. These holdouts protect
against implementation and serialization errors; after the degree-space and
interpolation identities above are established, they are not a separate
mathematical proof.

## 4. Positivity theorem

Define the orientation-independent margin

\[
M=T-F
\]

and its discriminant

\[
D=M^2-uvw(1-u)(1-v)(1-w)zH^2.
\]

For a multidegree \(\mathbf n=(n_1,\ldots,n_4)\), this manuscript uses the
standard tensor-product Bernstein convention

\[
B_{\boldsymbol\alpha}^{\mathbf n}(x)
=\prod_{i=1}^4
\binom{n_i}{\alpha_i}
x_i^{\alpha_i}(1-x_i)^{n_i-\alpha_i}.
\]

The reported coefficient magnitudes, including the smallest positive
coefficients below, refer to this normalization.
All Bernstein transforms in the certificate are performed over
\(\mathbb Q\) at the stated native multidegrees. No floating-point sign test,
degree elevation, or box subdivision is used to establish positivity.

### Theorem 4.1 — signed-star cubic Gram-volume inequality

For every \(u,v,w,z\in[0,1]\) and both orientation signs,

\[
T(z)\ge F(u,v,w,z)+\omega H(u,v,w,z).
\]

Equivalently, every full-rank signed-star frame satisfies

\[
\det I(X)\le\det(XX^{\mathsf T})^3\det I(Q).
\]

#### Proof

The native tensor-product Bernstein expansions on \([0,1]^4\) have no
negative coefficients:

| Polynomial | Bernstein degree | Negative coefficients | Zero coefficients |
|---|---:|---:|---:|
| \(M\) | \((3,3,3,5)\) | 0 | 195 |
| \(D\) | \((6,6,6,10)\) | 0 | 2,078 |

Every tensor-product Bernstein basis function is nonnegative on the cube.
Therefore \(M\ge0\) and \(D\ge0\). The latter gives

\[
M^2\ge\omega^2H^2.
\]

Since \(M\ge0\),

\[
M\ge|\omega H|,
\]

which proves the claim for both orientations. ∎

## 5. Structural compression and strictness

The original certificate contains many zeros because it includes forced
Cayley-boundary factors. Exact division reveals

### Theorem 5.1 — exact structural factorization

The three reconstructed certificate polynomials factor in
\(\mathbb Q[u,v,w,z]\) as

\[
M=(1-z)^3\widetilde M,
\]

\[
H=(1-u)(v-w)(1-z)^3\widetilde H,
\]

and

\[
D=(1-z)^6\widetilde D.
\]

These identities are exact polynomial divisions. In particular, the
orientation-sensitive sector is antisymmetric under \(v\leftrightarrow w\)
and vanishes to third order at the calibrated Cayley boundary \(z=1\).

#### Proof

Substitution of the two exact reconstructed coefficient maps into the
definitions of \(M\) and \(D\), followed by division in
\(\mathbb Q[u,v,w,z]\), gives quotient polynomials with zero remainder for
all three displayed factors. The independent FLINT replay repeats those
divisions over \(\mathbb Q\). The reduced polynomial \(\widetilde H\) is
invariant under \(v\leftrightarrow w\), so the factor \(v-w\) makes \(H\)
antisymmetric. ∎

The reduced orientation polynomial is

\[
\begin{aligned}
\widetilde H=-\frac12\bigl(&uvwz+7uvw-uvz-uv-uwz-uw+uz-5u\\
&-vwz-3vw+vz-3v+wz-3w-z+9\bigr).
\end{aligned}
\]

The reduced Bernstein certificates are dramatically smaller:

| Polynomial | Degree | Coefficients | Negative | Zero | Smallest positive |
|---|---:|---:|---:|---:|---:|
| \(\widetilde M\) | \((3,3,3,2)\) | 192 | 0 | 3 | \(32/3\) |
| \(\widetilde D\) | \((6,6,6,4)\) | 1,715 | 0 | 20 | \(512/9\) |

### Corollary 5.2 — strict interior inequality

For \(0<u,v,w,z<1\),

\[
\det I(X)<\Delta^3\det I(Q).
\]

#### Proof

Every Bernstein basis function is strictly positive in the open cube. Each
reduced certificate has nonnegative coefficients and at least one strictly
positive coefficient; therefore
\(\widetilde M>0\) and \(\widetilde D>0\) there. Hence
\(M>|\omega H|\). ∎

### Corollary 5.3 — exact orientation-blind subfamilies

The normalized determinant's orientation-sensitive amplitude contains the
factor

\[
\omega H
\propto
\sqrt{uvw(1-u)(1-v)(1-w)z}\,(1-u)(v-w)(1-z)^3.
\]

In particular, orientation dependence disappears when \(v=w\), even away
from the orthonormal and calibrated boundaries. This is a genuine symmetry of
the star determinant, not an artifact of squaring the Cayley coordinate.

### Theorem 5.4 — complete equality classification

On the normalized closed parameter cube, equality holds exactly when

\[
z=1
\qquad\text{or}\qquad
(u,v,w)=(0,0,0).
\]

Consequently, on the full-rank signed-star family the same two loci are the
complete equality set: the calibrated Cayley endpoint and the orthonormal
star.

#### Proof

At native multidegree \((3,3,3,2)\), the only zero Bernstein controls of
\(\widetilde M\) are

\[
(0,0,0,k),\qquad 0\le k\le2.
\]

At native multidegree \((6,6,6,4)\), the only zero controls of
\(\widetilde D\) are

\[
(i,j,k,\ell),
\qquad i+j+k\le1,
\qquad 0\le\ell\le4.
\]

All other controls are strictly positive. If at least one of \(u,v,w\) is
positive, an active spatial Bernstein basis function has index at least one
for \(\widetilde M\) and at least two for \(\widetilde D\); hence both reduced
polynomials are strictly positive. At \(u=v=w=0\), only the displayed zero
controls remain active, so both vanish. Restoring the exact factors
\((1-z)^3\) and \((1-z)^6\) proves the stated alternatives. ∎

There is a separate, deliberately distinguished statement on the singular
Gram faces \(u=1\), \(v=1\), or \(w=1\). The *unnormalized* continuously
extended inequality is trivially an equality there because of its common
factor \(\Delta^3\). The normalized quotient is generally nonzero on those
faces and its inequality remains strict away from the two loci classified
above.

## 6. Exactness, replay, and artifact contract

The proof has four independent obligations:

1. **structural bound:** Cauchy--Binet, rank loss, parity, and quotient-ring
   reduction place the determinant in a finite polynomial space;
2. **reconstruction:** two independently chosen exact grids, sharing only the
   all-zero anchor, recover identical maps;
3. **identity:** exact off-grid signed determinants agree with those maps;
4. **positivity:** exact rational Bernstein coefficients are nonnegative.

The lightweight artifact verifier checks stored-map integrity and signs. A full
proof replay recomputes both interpolation grids and the determinant holdouts:

```powershell
$env:PYTHONPATH = "src"
python -m spin8_dirac_star --output reproduced_star.json
python -m spin8_dirac_star_foundations --output reproduced_star_foundations.json
python -m spin8_dirac_star_structure
python -m spin8_publication_flint_crosscheck `
  --output reproduced_flint_crosscheck.json
python -m unittest discover -s tests -p "test_spin8_publication_theorems.py"
```

The original, foundational, and reduced-structural artifacts are
[`spin8_dirac_star_20260804.json`](../../artifacts/spin8_dirac_star_20260804.json),
[`spin8_dirac_star_foundations_20260806.json`](../../artifacts/spin8_dirac_star_foundations_20260806.json),
and
[`spin8_dirac_star_structure_20260806.json`](../../artifacts/spin8_dirac_star_structure_20260806.json).
The independent arithmetic-backend report is
[`spin8_publication_flint_crosscheck_20260806.json`](../../artifacts/spin8_publication_flint_crosscheck_20260806.json).
It imports the stored rational coefficient maps into FLINT, repeats the exact
factor divisions, and independently recomputes all 1,907 reduced Bernstein
coefficients. It also evaluates the three exact-order face witnesses
\(25/2,75/2,75/2\) in the second arithmetic backend. It does not regenerate
the interpolation samples or derive the
Spin(8) projectors, so the full replay remains a separate obligation.

## 7. What the theorem does not prove

A general lower-triangular Cholesky chart for four unit vectors contains three
additional residual correlations. They can couple the four Jacobian graph
frames and are absent here. Therefore:

- the signed-star inequality is proved exactly;
- strictness in its open parameter box is proved exactly;
- the unrestricted seven-invariant Dirac--Gram inequality remains open;
- global five-query D-optimality remains open.

An exact rational counterexample elsewhere in the archive already falsifies
the tempting claim that the missing residual correlations can always be
removed monotonically at fixed star coordinates. The global proof must control
the residual variables jointly rather than delete them one at a time.

Katz and Shnider supply background for the Cayley form, its comass, triality,
and its \(\operatorname{Spin}(7)\) stabilizer. They do not contain or imply the
new information determinant, signed-star reduction, finite parity module, or
Bernstein certificate proved here. Farouki supplies the Bernstein-basis
normalization and positivity background; de Klerk and Laurent is broader
optimization context rather than a load-bearing step in this proof.

## References

1. M. G. Katz and S. Shnider,
   [*Cayley form, comass, and triality isomorphisms*](https://arxiv.org/abs/0801.0283),
   Israel Journal of Mathematics 169 (2009), 117–135.
2. R. T. Farouki,
   [*The Bernstein polynomial basis: a centennial retrospective*](https://doi.org/10.1016/j.cagd.2012.03.001),
   Computer Aided Geometric Design 29 (2012), 379–419.
3. E. de Klerk and M. Laurent,
   [*Error bounds for some semidefinite programming approaches to polynomial minimization on the hypercube*](https://optimization-online.org/2010/04/2591/),
   SIAM Journal on Optimization 20 (2010), 3104–3120.
