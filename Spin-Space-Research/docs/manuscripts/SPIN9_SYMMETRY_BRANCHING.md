# The hidden \(\mathfrak{so}(3)\) behind the Spin(9) spectral factors

**Exact theorem note — 2026-08-07**
**Status:** exact curve stabilizer theorem and branching theorem
**Certificate:** [spin9_three_spinor_symmetry.py](../../src/spin9_three_spinor_symmetry.py)

## Abstract

The information operator on the symmetric three-spinor curve has characteristic
polynomial

\[
\chi_{I(c)}(x)=2^{-43}Q_1(x,c)^7Q_2(x,c)^5Q_4(x,c)^3.
\]

The powers \(7,5,3\) are not merely numerical multiplicities. Every interior
three-plane on the curve has a three-dimensional stabilizer acting as the full
\(\mathfrak{so}(3)\) algebra of that plane. At the exact Cayley-null point
\(c=0\), the pointwise stabilizer of the three spinors is trivial and the
adjoint module branches as

\[
\mathfrak{spin}(9)
\downarrow_{\mathfrak{so}(3)}
\cong 2V_7\oplus2V_5\oplus4V_3,
\]

where \(V_d\) denotes the unique real irreducible \(\mathfrak{so}(3)\)-module
of dimension \(d\). The factor degrees \(2,2,4\) are therefore the
multiplicity-space dimensions, and the powers \(7,5,3\) are the irreducible
dimensions. This reverses a tempting but incorrect reading in which the powers
would count copies of two-, two-, and four-dimensional modules.

The exact stabilizer embedding varies with \(c\); the claim is a common
stabilizer *type*, not one fixed subgroup acting on every plane. Since the open
parameter interval is connected and compact \(\mathfrak{so}(3)\)-module types
are discrete, the displayed branching persists along the interior curve.

## 1. Plane stabilizer versus pointwise stabilizer

Let \(B\in\mathbb R^{16\times3}\) contain the three orthonormal spinors at
\(c=0\), and let \(P=BB^{\mathsf T}\) be their three-plane projector. For
\(X\in\mathfrak{spin}(9)\), the plane-stabilizer equation is

\[
(I-P)XB=0.
\]

The maintained exact matrices give rank \(33\) for this linear system, hence a
three-dimensional kernel. The stronger pointwise equation

\[
XB=0
\]

has rank \(36\), hence zero kernel. Thus the plane has continuous symmetry,
but no nonzero infinitesimal action fixes all three chosen spinors.

For a primitive integer basis \(H_0,H_1,H_2\) of the plane stabilizer, direct
matrix commutation gives

\[
[H_0,H_1]=H_2,
\qquad
[H_0,H_2]=-2H_1,
\qquad
[H_1,H_2]=H_0.
\]

After the rescaling

\[
E_0=H_0,
\qquad E_1=\sqrt2H_1,
\qquad E_2=H_2,
\]

these are the standard cyclic \(\mathfrak{so}(3)\) brackets with common
structure constant \(\sqrt2\).

## 2. Exact Casimir decomposition

Let \(A_r=\operatorname{ad}(H_r)\) on
\(\mathfrak{spin}(9)\cong\Lambda^2\mathbb R^9\). The normalized quadratic
Casimir is

\[
\mathcal C=-\frac12\left(A_0^2+2A_1^2+A_2^2\right).
\]

Its characteristic polynomial is exactly

\[
\chi_{\mathcal C}(\lambda)
=(\lambda-12)^{14}(\lambda-6)^{10}(\lambda-2)^{12}.
\]

For an \(\mathfrak{so}(3)\) irrep of angular degree \(\ell\), the normalized
Casimir eigenvalue is \(\ell(\ell+1)\) and its real dimension is
\(2\ell+1\). Consequently:

\[
\begin{array}{c|c|c|c}
\ell & \ell(\ell+1) & \text{total dimension} & \text{multiplicity}\\ \hline
3&12&14&2\\
2&6&10&2\\
1&2&12&4
\end{array}
\]

which proves

\[
\mathfrak{spin}(9)\cong2V_7\oplus2V_5\oplus4V_3.
\]

## 3. The stabilizer along the complete curve

Put

\[
u^2=\frac{1-c}{1+c},
\qquad 0<u<\sqrt3.
\]

Without changing the plane, scale its three spanning vectors so that their
Gram matrix is

\[
\operatorname{diag}\bigl(1,1+u^2,4(1+u^2)\bigr).
\]

Solving \(XB=BA\) exactly for
\(X\in\mathfrak{spin}(9)\) and \(A\in\mathfrak{gl}(3)\) produces three
independent solutions. Their induced matrices are skew with respect to the
displayed Gram metric and span its complete three-dimensional skew algebra.
Thus the connected infinitesimal plane stabilizer is the full
\(\mathfrak{so}(3)\) acting irreducibly on the plane.

This is not only a generic symbolic rank calculation. A fixed \(42\times42\)
minor of the \(48\times45\) stabilizer system has determinant

\[
\frac{u^{10}(u^2-3)^4(u^2+1)^{13/2}}
{256\sqrt{3-u^2}},
\]

which is nonzero throughout \(0<u<\sqrt3\). The three explicit solutions give
nullity at least three; the minor gives rank at least \(42\), hence nullity at
most three. Therefore the stabilizer dimension is exactly three over the
whole open curve.

## 4. Recovery of all three spectral factors

The information operator at \(c=0\) commutes exactly with all three stabilizer
actions. Restricting it to the three Casimir eigenspaces gives

\[
\begin{aligned}
\chi_{12}(x)&=2^{-21}(8x^2-8x+1)^7,\\
\chi_{6}(x)&=2^{-10}(4x^2-7x+2)^5,\\
\chi_{2}(x)&=2^{-12}
(16x^4-60x^3+64x^2-16x+1)^3.
\end{aligned}
\]

Their product is the complete \(c=0\) specialization of the curve theorem.
This is an exact branching explanation of the full multiplicity pattern at the
Cayley-null point.

At \(c=0\), the Casimir computation labels the three isotypic pieces exactly.
The stabilizer embeddings and their adjoint representations vary continuously
with \(u\), while finite-dimensional \(\mathfrak{so}(3)\) multiplicities are
discrete. The branching type is consequently constant on the connected open
interval. Equivariance makes \(I(c)\) commute with the corresponding
stabilizer at every point, yielding the Schur tensor form

\[
(I_7\otimes A_2(c))\oplus(I_5\otimes B_2(c))
\oplus(I_3\otimes C_4(c)).
\]

This is the representation-theoretic origin of
\(Q_1^7Q_2^5Q_4^3\).

## 5. Claim boundary

This result proves more than a degree count and less than a global orbit
classification.

It proves:

- an exact three-dimensional \(\mathfrak{so}(3)\) plane stabilizer throughout
  the open curve;
- its exact \(\mathfrak{so}(3)\) bracket structure;
- the exact \(2V_7\oplus2V_5\oplus4V_3\) branching;
- exact commutation of the information operator with that stabilizer;
- exact recovery of the quadratic and quartic spectral factors.

It does not prove:

- that one fixed embedded subgroup fixes every point of the family;
- that the one-parameter family exhausts the full singular orbit-type stratum;
- that the curve maximizer is globally optimal among all three-spinor frames.

Those require an orbit-slice analysis, not multiplicity numerology.

## 6. Reproduction

Run

```powershell
$env:PYTHONPATH = "src"
python -m spin9_three_spinor_symmetry `
  --output artifacts/spin9_three_spinor_symmetry_20260807.json
python -m pytest tests/test_spin9_three_spinor_symmetry.py -q
```

All ranks, brackets, Casimir eigenvalues, commutators, and restricted
characteristic polynomials are reconstructed from exact SymPy matrices.

The generated artifact is
[spin9_three_spinor_symmetry_20260807.json](../../artifacts/spin9_three_spinor_symmetry_20260807.json),
with SHA-256 digest
`a77781a615c4ebd368cee9c1d81e7faf618c1dbb025c6a2bdcfcbb85bd48c1c9`.
