# Exact determinant bound on the full Spin(9) spin-two Cartan slice

**Computer-assisted theorem note — 2026-08-11**  
**Status:** exact finite-radius theorem for every pure \(V_5\) graph over the
Cayley-null plane  
**Reconstruction:**
[`spin9_v5_cartan_reconstruction.py`](../../src/spin9_v5_cartan_reconstruction.py)  
**Certificate:**
[`spin9_v5_cartan_certificate.py`](../../src/spin9_v5_cartan_certificate.py)

## Abstract

At the Cayley-null three-plane \(P_0\), the Grassmann normal slice for the
Spin(9) action is \(V_1\oplus V_5\), where
\(V_5\cong\operatorname{Sym}_0(3)\) is the spin-two module. The earlier local
Hessian theorem controls this module only infinitesimally, and the preceding
finite-radius result controls only its zero-cubic and axisymmetric rays.

This note treats the complete five-dimensional \(V_5\) family. The stabilizer
reduces every \(V_5\) vector to a two-dimensional Cartan section. On that
section the normalized information determinant is

\[
R(p,y)=\frac{N(p,y)}
{(1+8p+20p^2+16p^3-16y^2)^{14}},
\]

where \(N\in\mathbb Z[p,y]\) has weighted degree \(84\), with
\(\deg p=2\) and \(\deg y=3\). A deterministic two-embedding modular lift
proves this formula in characteristic zero. A six-cell exact Bernstein atlas
then proves

\[
R(p,y)<\frac{101}{100}
\qquad
(p\geq0,\ 27y^2\leq2p^3).
\]

The known algebraic symmetric candidate has determinant ratio greater than
\(101/100\), so it strictly beats every pure \(V_5\) graph over \(P_0\).

This is not a theorem on the coupled \(V_1\oplus V_5\) slice, the complete
Grassmann quotient, or the unrestricted rank-three frame space.

## 1. Cartan geometry of \(V_5\)

Use the Cayley-null orthonormal frame

\[
B=\begin{bmatrix}
e_0 & \frac{e_1+e_8}{\sqrt2} &
\frac{-e_2+e_{12}}{\sqrt2}
\end{bmatrix}.
\]

In the exact normal-slice basis, choose the Cartan variations \(Z_a,Z_t\)
with nonzero entries

\[
\begin{aligned}
Z_a:;&(1,1,2),(2,2,1),(5,2,-1),(6,1,-1),(8,1,-2),\\
     &(9,0,-2\sqrt2),(11,2,-1),(12,2,1),(14,0,\sqrt2),(15,1,1),\\
Z_t:;&(4,2,-\sqrt2),(6,0,-1),(10,2,\sqrt2),
       (14,1,\sqrt2),(15,0,1).
\end{aligned}
\]

Rows and columns are zero-indexed in this display to match the executable
certificate. They obey

\[
Z_a^{\mathsf T}Z_a=\operatorname{diag}(10,10,4),\qquad
Z_t^{\mathsf T}Z_t=\operatorname{diag}(2,2,4),
\]

and

\[
Z_a^{\mathsf T}Z_t=
\begin{pmatrix}0&2&0\\2&0&0\\0&0&0\end{pmatrix}.
\]

For

\[
X(u,v)=B+uZ_a+vZ_t,
\]

the Gram matrix and determinant factor exactly as

\[
X^{\mathsf T}X=
\begin{pmatrix}
1+10u^2+2v^2&4uv&0\\
4uv&1+10u^2+2v^2&0\\
0&0&1+4u^2+4v^2
\end{pmatrix},
\]

\[
\delta(u,v)=
(1+4u^2+4v^2)
(1+10u^2-4uv+2v^2)
(1+10u^2+4uv+2v^2),
\]

In invariant coordinates

\[
p=3u^2+v^2,
\qquad
y=\sqrt2\,u(v^2-u^2),
\]

this becomes

\[
\delta=1+8p+20p^2+16p^3-16y^2.
\]

The spin-two orbit space is

\[
p\geq0,
\qquad
27y^2\leq2p^3.
\]

This is the familiar eigenvalue description of
\(\operatorname{Sym}_0(3)\): radius and one normalized cubic shape determine
an orbit. Therefore a theorem on one closed Cartan chamber is a theorem for
every vector in \(V_5\).

## 2. Exact determinant reconstruction

Let \(G_1,\ldots,G_{36}\) be the maintained doubled integer Spin(9)
generators. For the graph frame \(X\), form the three observation matrices

\[
O_c=(G_iX_{:c})_{i=1}^{36}
\]

and the common-denominator information matrix

\[
J(u,v)=\sum_{c,d=1}^3
\operatorname{adj}(X^{\mathsf T}X)_{cd}\,O_cO_d^{\mathsf T}.
\]

Thus \(J=\delta I\). Each entry of \(J\) has total degree at most six, so
\(\det J\) has total degree at most \(216\).

The reconstructed identity is

\[
\det J(u,v)=\det J(0,0)\,\delta(u,v)^{22}N(p,y),
\]

with

\[
\det J(0,0)=17179869184.
\]

Consequently

\[
\frac{\det I(X(u,v))}{\det I(B)}
=\frac{N(p,y)}{\delta(p,y)^{14}}.
\]

The numerator has 631 invariant monomials; every recovered coefficient is a
nonzero integer and the largest has 27 decimal digits. The reconstruction
artifact records all coefficients individually and hashes their canonical
ordering.

### 2.1 Discovery and recovery layer

On a scaling line \((u,v)=(s,ks)\),

\[
p=(3+k^2)s^2,
\qquad
y=\sqrt2(k^2-1)s^3.
\]

The weighted-degree-84 numerator becomes an ordinary degree-84 polynomial in
\(s\). Twelve prime fields reconstruct its coefficients by scale and direction
interpolation. Thirty directions give exact holdouts at every weight. CRT and
Wang reconstruction recover integer coefficients with a 96-digit modulus.
An unused prime, \(90000767\), independently reproduces all 631 residues.

This layer alone is deliberately not called a characteristic-zero proof. Its
artifact retains
`"characteristic_zero_identity_certified": false`.

### 2.2 Deterministic characteristic-zero lift

The final certificate checks the raw degree-216 identity, not merely the
reduced interpolation formula. For each of twenty primes it evaluates both
square roots of \(2\), forty invariant-separating directions, and 217 scale
nodes. The direction matrices have full rank at every invariant weight; the
largest invariant layer has dimension 37. Therefore the finite grid proves the
raw polynomial identity over each prime field.

Both roots of \(2\) are necessary. A raw coefficient lies in
\(\mathbb Z[\sqrt2]\); checking both embeddings separately forces both its
rational and radical components to vanish modulo the prime.

For the submultiplicative norm

\[
\lVert a+b\sqrt2\rVert_q=|a|+2|b|,
\]

the certificate reconstructs all 36 row coefficient norms of \(J\). The
product-of-row-sums determinant bound and the independently bounded candidate
give

\[
\lVert\det J-\det J(0,0)\delta^{22}N\rVert_q
\leq
250465232697927055725525937889410429424952321908313156346741615961051689201309551138418219837919430352909935634650339669430443359604788125293503823092908032.
\]

The product of the twenty checked primes is

\[
1215859028498992118841818725682807147578157438271029852486215817650875617732127294954792641147955506967167946889478298216647645131797500812773631854575021016599,
\]

which exceeds twice that bound. Hence every rational and radical coefficient
of the residual is zero in characteristic zero.

## 3. Exact positivity on the orbit domain

A rational fundamental chamber is

\[
0\leq k\leq3,
\qquad r\geq0,
\]

with

\[
p=\frac{(3+k^2)r^2}{2},
\qquad
y=\frac{(k^2-1)r^3}{2}.
\]

Set \(k=3z\) and compactify \(r=x/(1-x)\). The target gap becomes the
integer polynomial

\[
H(x,z)=(1-x)^{84}\bigl(101\delta^{14}-100N\bigr),
\qquad (x,z)\in[0,1]^2.
\]

It has multidegree \((84,84)\) and 1,849 nonzero power coefficients. Its
native tensor-product Bernstein representation is not positive: it contains
1,792 negative controls and 85 zero controls. The native failure is retained
as an explicit rejected proof route.

Five alternating midpoint splits produce the following exact cover:

| \(x\)-interval | \(z\)-interval | leaf depth |
|---|---|---:|
| \([1/2,1]\) | \([0,1]\) | 1 |
| \([0,1/2]\) | \([0,1/2]\) | 2 |
| \([1/4,1/2]\) | \([1/2,1]\) | 3 |
| \([0,1/4]\) | \([1/2,3/4]\) | 4 |
| \([0,1/8]\) | \([3/4,1]\) | 5 |
| \([1/8,1/4]\) | \([3/4,1]\) | 5 |

Every exact Bernstein control on every leaf is strictly positive. Bernstein
basis functions are nonnegative and form a partition of unity, so
\(H(x,z)>0\) on the complete closed square. Since \(\delta>0\) for every real
finite graph frame,

\[
\boxed{
\frac{\det I(P_{V_5})}{\det I(P_0)}<\frac{101}{100}
}
\]

for every pure \(V_5\) graph over \(P_0\).

## 4. Independent ray checks and candidate comparison

The generic numerator restricts exactly to both earlier independently derived
ray numerators. On the axisymmetric boundary,

\[
N(3t^2,-\sqrt2t^3)=N_{\rm axis}(t)(1+10t^2)^2.
\]

On the zero-cubic ray, with \(p=t^2/2,y=0\),

\[
N(t^2/2,0)=N_{\rm zero}(t)(1+t^2)^2(1+2t^2)^2.
\]

The extra factors are the cancellations specific to those singular orbit
shapes. These identities provide independent characteristic-zero checks on
two decisive boundary families.

The algebraic symmetric candidate satisfies

\[
\frac{\det I(P_\star)}{\det I(P_0)}>\frac{101}{100}.
\]

It therefore strictly beats the entire pure-\(V_5\) graph family.

## 5. Claim boundary

Proved here:

- the exact invariant determinant formula on the complete \(V_5\) Cartan
  section;
- its characteristic-zero identity, including the \(\delta^{22}\)
  cancellation;
- strict \(101/100\) upper control on every pure \(V_5\) graph;
- strict separation of this family from the algebraic symmetric candidate.

Not proved here:

- a finite-radius theorem coupling the \(V_1\) curve coordinate to \(V_5\);
- a theorem on the second supported \(V_5\) in the full frame tangent;
- global optimality on \(G_3(\mathbb R^{16})/\operatorname{Spin}(9)\);
- global optimality on the unrestricted rank-three frame stratum;
- uniqueness of the algebraic candidate modulo Spin(9).

The next exact frontier is the coupled \(V_1\oplus V_5\) Grassmann slice. The
full nonpolar quotient remains a separate global problem.

## 6. Reproduction

From the repository root:

```powershell
$env:PYTHONPATH = "src"

python -m spin9_v5_cartan_reconstruction `
  --output artifacts/spin9_v5_cartan_reconstruction_20260811.json

python -m spin9_v5_cartan_certificate `
  --output artifacts/spin9_v5_cartan_certificate_20260811.json

python -m pytest tests/test_spin9_v5_cartan_certificate.py -q
```

The full certificate command performs 347,200 exact raw determinant
evaluations and takes about eight minutes on the recorded Windows workstation.
The JSON artifacts are:

- [spin9_v5_cartan_reconstruction_20260811.json](../../artifacts/spin9_v5_cartan_reconstruction_20260811.json),
  SHA-256 `a7904ca0f405a3484a1bc6b12843676149768114e7a37e66c4c817f4d252fa78`;
- [spin9_v5_cartan_certificate_20260811.json](../../artifacts/spin9_v5_cartan_certificate_20260811.json),
  SHA-256 `e905ff7454f8a703d8898ffae8ebdfabb680c24643a65dfb29489c62765ad449`.

The certificate stores the exact coefficient bound, prime product, all prime
outcomes, compact power and Bernstein hashes, six leaf boxes, and a digest of
every leaf control tensor. The tests reconstruct the mathematical layers
rather than treating a stored `passed` flag as proof.
