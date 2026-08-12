# Balanced Cayley Information Spectra for Spin(8) Triality Sensors

**Working theorem manuscript — 2026-08-06**
**Status:** exact symbolic theorem on the orthonormal balanced allocation
**Certificate:** [`spin8_cayley_criteria.py`](../../src/spin8_cayley_criteria.py)

## Abstract

Five infinitesimal observations of a shared \(\operatorname{Spin}(8)\)
action define a \(28\times28\) information Gram operator. For the balanced
allocation consisting of one vector, two positive-chiral, and two
negative-chiral probes, the orthonormal design space reduces to a single
Cayley parameter \(c\in[-1,1]\). This work gives an exact invariant-block
decomposition of the complete family into fixed coordinate subspaces of
dimensions \(8+8+8+4\). The determinant is

\[
\det I_c=\frac{(1-c^2)^3(9-c^2)^2}{1024}.
\]

The factorization yields more than D-optimality. Across the entire nonsingular
orbit, \(\operatorname{tr}I_c=35\) and
\(\operatorname{tr}(I_c^2)=67\) are constant, whereas
\(\operatorname{tr}(I_c^{-1})\) and
\(\operatorname{tr}(I_c^{-2})\) are strictly increasing functions of
\(c^2\). Consequently the Cayley-null orbit \(c=0\) uniquely maximizes
the determinant and simultaneously minimizes both inverse spectral moments.
The calibrated endpoints \(c=\pm1\) have rank 25, with the three lost
eigenvalues vanishing at the same rate \((1-c^2)/8\). These conclusions are
exact on the orthonormal balanced family; they do not establish global
five-query optimality among nonorthogonal designs or other allocations.

## 1. Setting

Let \(V,S^+,S^-\cong\mathbb R^8\) be the vector and two real chiral-spinor
representations of \(\operatorname{Spin}(8)\). Fix a bivector basis
\(E_1,\ldots,E_{28}\) of \(\mathfrak{so}(8)\), and write
\(G_r^{(\alpha)}\) for its skew-symmetric generator matrices in view
\(\alpha\in\{V,+,-\}\).

For a unit probe \(x\in\mathbb R^8\), define

\[
J_\alpha(x)
=\begin{bmatrix}
G_1^{(\alpha)}x&\cdots&G_{28}^{(\alpha)}x
\end{bmatrix}
\in\mathbb R^{8\times28},
\]

and its information contribution

\[
P_\alpha(x)=J_\alpha(x)^{\mathsf T}J_\alpha(x).
\]

Each \(P_\alpha(x)\) is positive semidefinite of rank seven: the stabilizer
of a unit vector is \(\operatorname{Spin}(7)\), so seven infinitesimal
directions move the probe and 21 do not.

The balanced information operator is

\[
I=P_V(v)+P_+(p_1)+P_+(p_2)+P_-(n_1)+P_-(n_2).
\]

The analysis concerns orthonormal balanced designs after the standard triality
identification described below.

## 2. Why one Cayley coordinate is sufficient

### Lemma 2.1 — gauge reduction

The \(\operatorname{Spin}(8)\) action is transitive on the unit sphere in
\(V\). Hence \(v\) may be fixed to \(e_0\). Its stabilizer is
\(\operatorname{Spin}(7)\). Clifford multiplication by \(e_0\)
identifies the restrictions of \(S^+\) and \(S^-\) to this stabilizer,
so the four remaining probes become an orthonormal four-frame in one copy of
\(\mathbb R^8\).

The four-frame retains its \(2+2\) view split. Before quotienting the
same-view basis freedom, it determines an oriented flag
\(P_+\oplus P_-\), where \(P_+\) and \(P_-\) are orthogonal oriented
two-planes. The information operator itself permits all of \(O(2)\) in each
pair, so it ultimately forgets both individual plane orientations.

### Proposition 2.2 — balanced-flag normal form

For every orthonormal balanced design there are \(c,s\in\mathbb R\), with
\(c^2+s^2=1\), such that

\[
(v;p_1,p_2;n_1,n_2)
\sim_{\mathrm{spec}}
(e_0;e_0,e_1;e_2,c e_3+s e_4).
\]

Here \(\sim_{\mathrm{spec}}\) means that the two information operators are
orthogonally conjugate in the 28-dimensional action-parameter space and hence
have the same spectrum. The allowed reductions are a common
\(\operatorname{Spin}(8)\) action and independent orthogonal changes of basis
inside each same-view probe pair.

#### Proof

Lemma 2.1 fixes \(v=e_0\) and turns the four spinor probes into an
orthonormal four-frame in a common \(\operatorname{Spin}(7)\)-module. For
either chiral view and any orthonormal pair \((x,y)\), linearity of the query
Jacobian gives

\[
P_\alpha(r x+t y)+P_\alpha(-t x+r y)
=(r^2+t^2)\bigl(P_\alpha(x)+P_\alpha(y)\bigr).
\]

Thus the information operator depends on each probe pair only through its
two-plane, not through the chosen orthonormal basis of that plane. This
identity is also checked entry by entry in the exact block certificate for
both chiral views.

Berndt and Tamaru show that the \(\operatorname{Spin}(7)\) action on the
oriented Grassmannian \(\widetilde{\operatorname{Gr}}_4(\mathbb R^8)\) has
cohomogeneity one, with the Cayley and oppositely oriented Cayley planes as
its two singular orbits [3, p. 3436]. Along the standard normal geodesic the
invariant Cayley value is \(c\in[-1,1]\); it distinguishes the principal
orbits when \(|c|<1\).

The additional assertion needed here concerns the internal \(2+2\) split and
is certified rather than inferred from the abstract name of the isotropy
group. At the exact rational principal representative \((c,s)=(3/5,4/5)\),
solving

\[
(I-P_W)X|_W=0,
\qquad X\in\mathfrak{spin}(7),
\]

gives a six-dimensional four-plane stabilizer. Its restricted matrices span
all six dimensions of \(\mathfrak{so}(W)\). The connected stabilizer image is
therefore \(\operatorname{SO}(W)\), not merely an unspecified quotient of an
abstract \(\operatorname{Spin}(4)\). Principal isotropy groups are conjugate
throughout the principal stratum, so the same conclusion holds for every
\(|c|<1\). Since \(\operatorname{SO}(4)\) is transitive on oriented
orthogonal \(2+2\) splittings, with stabilizer
\(\operatorname{SO}(2)\times\operatorname{SO}(2)\), the split carries no
further continuous invariant.

The dimension count makes the local quotient explicit. The oriented
four-plane Grassmannian has dimension \(4(8-4)=16\); its oriented
\(2+2\)-splitting fiber has dimension \(2(4-2)=4\). A principal split flag has a
two-dimensional stabilizer, hence a (21-2=19)-dimensional
\(\operatorname{Spin}(7)\) orbit inside the 20-dimensional flag space. Thus

\[
\frac{\{\text{oriented orthogonal }2+2\text{ plane flags}\}}
{\operatorname{Spin}(7)}
\cong [-1,1],
\]

with coordinate \(c\), by the cited global four-plane orbit classification
and the full-\(SO(4)\) split action just established. Passing from oriented
flags to the information-equivalence relation permits a reflection in either
same-view pair, identifies \(c\) with \(-c\), and gives the interval
\(z=c^2\in[0,1]\). Exact endpoint calculations give nine-dimensional plane
stabilizers at \(c=\pm1\), but their restricted images
still span all of \(\mathfrak{so}(4)\); hence their (2+2) splits are also
equivalent. The displayed normal form therefore covers the principal and
singular strata. The exact isotropy calculation supporting the split step is
[`spin8_cayley_flag.py`](../../src/spin8_cayley_flag.py). ∎

The calibrated and anti-calibrated planes are the two singular oriented
orbits. Reversing one same-view pair basis changes the sign of \(c\) without
changing its information contribution. The information spectrum therefore
depends on the unoriented coordinate \(z=c^2\in[0,1]\).

The classical ingredients are the cohomogeneity-one action and the fact that
the Cayley form has unit comass and stabilizer \(\operatorname{Spin}(7)\);
see Berndt and Tamaru [3, p. 3436] and
[Katz–Shnider](https://arxiv.org/abs/0801.0283). The repository independently
checks the maintained sign convention against all 21 infinitesimal stabilizer
generators. The division between classical and mechanically checked proof
layers is recorded in the
[flag-quotient audit](../CAYLEY_FLAG_QUOTIENT_AUDIT_2026-08-06.md).

After fixing \(v=e_0\), the displayed \(p_i,n_i\) coordinates below live in
the common \(\operatorname{Spin}(7)\)-identified spinor copy; the first
\(e_0\) is in \(V\), whereas the second is a coordinate in that spinor copy.
An exact spectral representative is

\[
(v;p_1,p_2;n_1,n_2)
=(e_0;e_0,e_1;e_2,c e_3+s e_4),
\qquad c^2+s^2=1.
\]

## 3. Constant invariant blocks

Let \(I(c,s)\) be the information operator of the representative above.
In the maintained bivector basis, the off-diagonal support graph has four
connected components independent of \(c\) and \(s\). A fixed coordinate
permutation therefore gives

\[
I(c,s)=I_8^{(0)}\oplus I_8^{(1)}\oplus I_8^{(2)}\oplus I_4.
\]

All identities below hold in the quotient ring

\[
\mathbb Q[c,s]/(c^2+s^2-1).
\]

### Theorem 3.1 — exact block characteristic laws

Use the monic convention

\[
\chi_j(\lambda)=\det(\lambda I-I_j).
\]

With spectral variable \(\lambda\), the four block polynomials are

\[
\begin{aligned}
\chi_0(\lambda)=-\frac14(\lambda-1)^2
&\bigl(2c\lambda-c-2\lambda^3+8\lambda^2-6\lambda+1\bigr)\\
&\cdot\bigl(2c\lambda-c+2\lambda^3-8\lambda^2+6\lambda-1\bigr),
\end{aligned}
\]

\[
\begin{aligned}
\chi_1(\lambda)=\chi_2(\lambda)=\frac1{16}
&(c-2\lambda^2+4\lambda-1)
(c-2\lambda^2+6\lambda-3)\\
&\cdot(c+2\lambda^2-6\lambda+3)
(c+2\lambda^2-4\lambda+1),
\end{aligned}
\]

and

\[
\chi_3(\lambda)=(\lambda-1)^2(\lambda^2-3\lambda+1).
\]

Although the first two displays are written as rationally scaled products,
their expansions are monic. For example, if

\[
P(\lambda)=2\lambda^3-8\lambda^2+6\lambda-1,
\qquad L(\lambda)=2\lambda-1,
\]

then

\[
\chi_0(\lambda)
=\frac14(\lambda-1)^2
\bigl(P(\lambda)+cL(\lambda)\bigr)
\bigl(P(\lambda)-cL(\lambda)\bigr),
\]

which also makes its dependence on (c^2) manifest.

Moreover, \(I_8^{(1)}\) and \(I_8^{(2)}\) are conjugate by a fixed
signed-permutation matrix. The repeated factors are therefore structural, not
an accidental factorization of the full degree-28 polynomial.

#### Proof

The support decomposition is a direct identity in the bivector basis. Each
block characteristic polynomial is computed over \(\mathbb Q[c,s]\) and
reduced by \(s^2=1-c^2\). Subtraction from the displayed polynomial gives
zero in the quotient. A stored signed-permutation matrix \(U\) satisfies

\[
U I_8^{(1)}=I_8^{(2)}U,
\qquad U^{\mathsf T}U=I_8,
\]

entry by entry in the same quotient. The certificate is implemented in
[`spin8_cayley_blocks.py`](../../src/spin8_cayley_blocks.py). ∎

## 4. Determinant and rank

Evaluating the four block polynomials at \(\lambda=0\) gives

\[
\det I_8^{(0)}=\frac{1-c^2}{4},
\]

\[
\det I_8^{(1)}=\det I_8^{(2)}
=\frac{(1-c^2)(9-c^2)}{16},
\]

and \(\det I_4=1\). Hence

\[
\boxed{
\det I_c=\frac{(1-c^2)^3(9-c^2)^2}{1024}
}.
\]

Writing \(z=c^2\),

\[
\frac{d}{dz}\det I_c
=-\frac{(1-z)^2(9-z)(29-5z)}{1024}<0,
\qquad 0\le z<1.
\]

Thus \(c=0\) is the unique unoriented D-optimum on this orbit and

\[
\det I_0=\frac14\left(\frac9{16}\right)^2=\frac{81}{1024}.
\]

On the closed interval (0\le z\le1), the derivative vanishes only at the
singular endpoint (z=1); it is strictly negative throughout (0\le z<1).

At \(c=\pm1\), exactly one determinant factor from each eight-dimensional
block vanishes; direct exact rank evaluation gives
\(\operatorname{rank}I_{\pm1}=25\).

## 5. Simultaneous spectral optimality

The characteristic law contains two invariants that were previously hidden by
the determinant calculation.

### Theorem 5.1 — fixed direct moments

For every \(c\in[-1,1]\),

\[
\operatorname{tr}I_c=35,
\qquad
\operatorname{tr}(I_c^2)=67.
\]

The first identity also follows because each of the five unit probes contributes
a rank-seven orthogonal projector. The second is a genuinely coupled invariant
of the balanced Cayley family. Consequently the spectral mean, variance, and
second-moment participation ratio

\[
r_{\mathrm{eff},2}(I_c)
=\frac{(\operatorname{tr}I_c)^2}{\operatorname{tr}(I_c^2)}
=\frac{1225}{67}
\]

remain constant even while the determinant changes. This quantity is not the
matrix stable rank \(\lVert I_c\rVert_F^2/\lVert I_c\rVert_2^2\), which varies
because the largest eigenvalue varies with \(c\).

Equivalently, the first two power sums are fixed. Newton's identities then
fix the first two elementary coefficients of the degree-28 characteristic
polynomial, while its constant coefficient—the determinant—varies with (z).

### Theorem 5.2 — exact inverse moments

For the nonsingular family \(0\le z<1\),

\[
\operatorname{tr}(I_c^{-1})
=\frac{11z^2-206z+387}{(1-z)(9-z)},
\]

and

\[
\operatorname{tr}(I_c^{-2})
=\frac{19z^4-76z^3+786z^2+2676z+8883}
{(1-z)^2(9-z)^2}.
\]

Both functions are strictly increasing on \([0,1)\). Their unique minima
occur at the Cayley-null orbit:

\[
\operatorname{tr}(I_0^{-1})=43,
\qquad
\operatorname{tr}(I_0^{-2})=\frac{329}{3}.
\]

#### Proof

If \(p(\lambda)=\det(\lambda I-I_c)\), logarithmic differentiation gives

\[
\operatorname{tr}(I_c^{-1})=-\frac{p'(0)}{p(0)},
\]

and

\[
\operatorname{tr}(I_c^{-2})
=-\left.\frac{d^2}{d\lambda^2}\log p(\lambda)\right|_{\lambda=0}.
\]

Substitution of the exact block polynomial yields the displayed rational
functions. The first derivative is

\[
\frac{d}{dz}\operatorname{tr}(I_c^{-1})
=\frac{96(z^2-6z+21)}{(9-z)^2(1-z)^2}>0,
\]

because \(z^2-6z+21=(z-3)^2+12\).

For the second inverse moment,

\[
\frac{d}{dz}\operatorname{tr}(I_c^{-2})
=\frac{16N(z)}{(9-z)^3(1-z)^3},
\]

where

\[
N(z)=12609+336z-630z^2-8z^3-19z^4.
\]

For \(0\le z\le1\), the elementary bound

\[
N(z)
\ge12609-630-8-19
=11952>0
\]

already proves the required sign; the omitted term \(336z\) is nonnegative.
As an independent exact certificate, the implementation also records the
degree-four Bernstein coefficients
\(12609,12693,12672,12544,12288\), all positive. ∎

### Corollary 5.3 — one orbit, three aligned criteria

Within the orthonormal balanced family, the Cayley-null orbit uniquely:

1. maximizes \(\det I\) (D-optimality);
2. minimizes \(\operatorname{tr}(I^{-1})\) (A-optimality);
3. minimizes \(\|I^{-1}\|_F^2=\operatorname{tr}(I^{-2})\).

This alignment is not a generic consequence of fixed trace. It follows from
the exceptional exact spectrum.

### Theorem 5.4 — equal-rate first-order rank loss

Let \(z=c^2\), put \(\varepsilon=1-z\), and approach either calibrated
endpoint from the nonsingular orbit. Exactly three eigenvalue branches vanish, one from each
eight-dimensional block, and all three have the same leading law:

\[
\lambda_j(z)=\frac{1-z}{8}+O\bigl((1-z)^2\bigr),
\qquad j=1,2,3.
\]

At the endpoint, the largest surviving eigenvalue is \(2+\sqrt2\). Hence

\[
\kappa_2(I_c)
=\frac{8(2+\sqrt2)}{1-z}+O(1).
\]

The inverse moments consequently diverge as

\[
\operatorname{tr}(I_c^{-1})
=\frac{24}{1-z}+O(1),
\]

and

\[
\operatorname{tr}(I_c^{-2})
=\frac{192}{(1-z)^2}
+O\!\left(\frac1{1-z}\right).
\]

#### Proof

At \(c=1\), the three singular block polynomials factor as

\[
\chi_0(\lambda)
=\lambda(\lambda-1)^3(\lambda^2-4\lambda+2)
(\lambda^2-3\lambda+1),
\]

and

\[
\chi_1(\lambda)=\chi_2(\lambda)
=\lambda(\lambda-2)^2(\lambda-1)^3
(\lambda^2-3\lambda+1).
\]

Each zero root is simple. Implicit differentiation of
\(\chi_j(c,\lambda_j(c))=0\) at \((c,\lambda)=(1,0)\), followed by
\(1-c^2=-2(c-1)+O((c-1)^2)\), gives

\[
\begin{array}{c|cc|c}
\text{block}&\partial_\lambda\chi_j(1,0)&
\partial_c\chi_j(1,0)&\lambda_j'(1)\\ \hline
0&-2&-\tfrac12&-\tfrac14\\
1&-4&-1&-\tfrac14\\
2&-4&-1&-\tfrac14
\end{array}
\]

and therefore

\[
\lim_{z\uparrow1}\frac{\lambda_j(z)}{1-z}=\frac18
\]

for all three blocks. The competing largest root from the other factors is
\((3+\sqrt5)/2<2+\sqrt2\), so the endpoint factorizations identify
\(2+\sqrt2\) as the largest surviving root. The remaining asymptotics follow
by inversion and summation. The spectrum depends on the unoriented coordinate
\(z\), so the result is the same at \(c=-1\). ∎

## 6. Computational proof object

The proof is exact at six distinct layers:

1. the three triality generator families are rational and satisfy the common
   Lie-algebra and equivariance contracts;
2. principal and endpoint four-plane stabilizers, their effective
   \(\mathfrak{so}(4)\) images, and the local one-dimensional flag quotient
   are checked over \(\mathbb Q\); the global interval classification remains
   the cited classical theorem input;
3. the \(28\times28\) information family is constructed exactly;
4. block support, characteristic identities, and conjugacy are checked in the
   circle quotient;
5. the design-criterion formulas and derivative signs are reconstructed from
   the block polynomial with exact rational arithmetic;
6. the three endpoint eigenvalue slopes are obtained by exact implicit
   differentiation of the singular block factors.

Reproduce the theorem with:

```powershell
$env:PYTHONPATH = "src"
python -m spin8_cayley_blocks
python -m spin8_cayley_flag
python -m spin8_cayley_criteria
python -m unittest discover -s tests -p "test_spin8_publication_theorems.py"
```

The publication artifacts are
[`spin8_cayley_blocks_20260806.json`](../../artifacts/spin8_cayley_blocks_20260806.json)
,
[`spin8_cayley_flag_20260806.json`](../../artifacts/spin8_cayley_flag_20260806.json),
and
[`spin8_cayley_criteria_20260806.json`](../../artifacts/spin8_cayley_criteria_20260806.json).
The derivative identities are also recomputed with python-flint in
[`spin8_publication_flint_crosscheck_20260806.json`](../../artifacts/spin8_publication_flint_crosscheck_20260806.json).
That cross-check imports the maintained rational coefficient maps; it does not
reconstruct those maps from the triality representation matrices. It changes
the arithmetic backend for polynomial division, differentiation, and
Bernstein conversion. By contrast, the SymPy block certificate constructs the
28-by-28 information family directly from the maintained rational generator
matrices before deriving the block laws.

## 7. Scope and open problems

This theorem covers the complete orthonormal balanced information family. It
does not prove:

- that orthonormal completion improves every nonorthogonal balanced frame;
- the allocation-wise upper bounds for \((4,1,0)\), \((3,2,0)\), or
  \((3,1,1)\);
- global five-query D- or A-optimality across all allocations;
- a sequence-model advantage for triality.

The exact signed-star and one-edge Dirac--Gram theorems extend parts of the
first question beyond orthonormal frames. The unrestricted problem remains
open.

## References

1. R. Harvey and H. B. Lawson Jr.,
   [*Calibrated geometries*](https://doi.org/10.1007/BF02392726),
   Acta Mathematica 148 (1982), 47–157.
2. M. G. Katz and S. Shnider,
   [*Cayley form, comass, and triality isomorphisms*](https://arxiv.org/abs/0801.0283),
   Israel Journal of Mathematics 178 (2010), 187–208.
3. J. Berndt and H. Tamaru,
   [*Cohomogeneity one actions on noncompact symmetric spaces of rank one*](https://arxiv.org/abs/math/0505490),
   Transactions of the American Mathematical Society 359 (2007), 3425–3438.
4. C. McRae,
   [*Exploring Triality Explicitly*](https://arxiv.org/abs/2502.14016), 2025.
