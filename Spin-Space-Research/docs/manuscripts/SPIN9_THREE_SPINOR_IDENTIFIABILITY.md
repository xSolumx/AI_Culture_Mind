# Three generic spinors identify a Spin(9) action

**Theorem note — 2026-08-07**
**Status:** human proof with independent exact infinitesimal witnesses
**Machine certificate:** [`spin9_dirac_clifford.py`](../../src/spin9_dirac_clifford.py)

## Abstract

Let \(S\cong\mathbb R^{16}\) be the real spin representation of
\(\operatorname{Spin}(9)\). The diagonal action on \(S^2\) has generic
stabilizer \(\operatorname{SU}(3)\), and therefore two generic spinor probes do
not identify a shared Spin(9) action. The diagonal action on \(S^3\), however,
is generically free: three generic spinors have trivial common stabilizer.
Consequently three is the sharp generic probe count for global recovery in the
faithful real spin representation.

The proof follows the octonionic orbit normal form of Bryant and the standard
stabilizer chain

\[
\operatorname{Spin}(9)
\supset\operatorname{Spin}(7)
\supset G_2
\supset\operatorname{SU}(3).
\]

An independent integer certificate obtains infinitesimal ranks \(15,28,36\)
for one, two, and three frozen rational probes over three prime fields.

## 1. Setting

Write \(S\cong\mathbb O^2\) for the real spin representation. Its action is
faithful: the nontrivial central element of \(\operatorname{Spin}(9)\) acts as
\(-I_{16}\), and hence fixes no nonzero spinor.

For a tuple \(\mathbf s=(s_1,\ldots,s_k)\), define its pointwise stabilizer

\[
G_{\mathbf s}
=\{g\in\operatorname{Spin}(9):gs_j=s_j\text{ for every }j\}.
\]

The word *generic* means outside a proper real-algebraic subset. Equivalently,
the required nonvanishing and linear-independence conditions hold on an open
dense set.

## 2. The sharp stabilizer theorem

### Theorem

For the diagonal action of \(\operatorname{Spin}(9)\) on powers of its real
spin representation:

1. a generic nonzero spinor has stabilizer \(\operatorname{Spin}(7)\);
2. a generic ordered pair has stabilizer \(\operatorname{SU}(3)\);
3. a generic ordered triple has trivial stabilizer.

In particular, three is the least generic probe count that globally identifies
a shared Spin(9) action.

### Proof

The action of \(\operatorname{Spin}(9)\) on the unit sphere in
\(S\cong\mathbb O^2\) is transitive. Fixing the first nonzero spinor therefore
reduces the stabilizer to \(K\cong\operatorname{Spin}(7)\).

Bryant's octonionic normal form for a generic ordered pair reduces it to

\[
\left(
\begin{pmatrix}a1\\0\end{pmatrix},
\begin{pmatrix}c1+d u\\b1\end{pmatrix}
\right),
\qquad a,b,d>0,
\]

where \(u\in\operatorname{Im}\mathbb O\) is a fixed unit imaginary octonion.
Inside \(K\), fixing the unit in the second octonion summand leaves \(G_2\).
Fixing \(u\) inside its seven-dimensional standard representation leaves
\(\operatorname{SU}(3)\). Hence the generic pair stabilizer is exactly
\(\operatorname{SU}(3)\), proving that two probes are insufficient.

It remains to restrict the third spinor to this stabilizer. The standard
branching rules along the same chain are

\[
S\!\downarrow_{\operatorname{Spin}(7)}
\cong \mathbb R\oplus\mathbb R^7\oplus\Delta_7,
\]

\[
\Delta_7\!\downarrow_{G_2}
\cong\mathbb R\oplus\mathbb R^7,
\qquad
\mathbb R^7\!\downarrow_{\operatorname{SU}(3)}
\cong\mathbb R\oplus(\mathbb C^3)_{\mathbb R}.
\]

Therefore

\[
S\!\downarrow_{\operatorname{SU}(3)}
\cong
4\mathbb R\oplus
(\mathbb C^3)_{\mathbb R}\oplus
(\mathbb C^3)_{\mathbb R}.
\]

The nontrivial part of a generic third spinor is thus an ordered pair
\((z,w)\in\mathbb C^3\oplus\mathbb C^3\) of linearly independent complex
vectors. If \(g\in\operatorname{SU}(3)\) fixes both, it fixes their complex
two-plane pointwise. It acts on the one-dimensional orthogonal complement by a
scalar \(\lambda\); the determinant-one condition forces \(\lambda=1\).
Thus \(g=I\), and the generic triple stabilizer is trivial. \(\square\)

## 3. Independent infinitesimal certificate

Let \(P_0,\ldots,P_8\) be the nine symmetric Clifford involutions constructed
from the maintained Spin(8) gamma matrices and chirality. For
\(G_{ij}=P_iP_j\), stack the columns \(G_{ij}s_r\) over all probes. The frozen
integer witnesses have ranks

\[
\operatorname{rank}J(s_1)=15,
\qquad
\operatorname{rank}J(s_1,s_2)=28,
\qquad
\operatorname{rank}J(s_1,s_2,s_3)=36
\]

over each of three prime fields. These equal the orbit dimensions predicted by
the theorem. A dependent third probe \(s_1+2s_2\) leaves the rank at 28.

The finite-field calculation is not used to infer the global stabilizer.
Conversely, the human proof does not ask the reader to trust a numerical rank.
They are independent checks of the same boundary.

## 4. Consequence for sensing

Suppose an unknown \(g\in\operatorname{Spin}(9)\) is observed through exact
input-output pairs \((s_j,gs_j)\). Three generic pairs determine \(g\)
uniquely. Two generic pairs determine it only up to a residual
\(\operatorname{SU}(3)\) action.

This statement concerns exact supplied probes and noiseless observations.
Conditioning, active probe selection, noisy recovery, and learned semantic
queries are separate problems.

## 5. Nonclaims

- The result does not say three arbitrary probes suffice. Degenerate triples
  can retain a positive-dimensional stabilizer.
- It does not establish optimal conditioning of any triple.
- It does not imply that Spin(9) memory outperforms direct or generic
  orthogonal memory.
- The underlying orbit and branching facts are classical. Any novelty claim
  must concern the sharp sensing formulation, independent certificate, or its
  use in a recurrent architecture, subject to a fuller novelty search.

## References

1. R. L. Bryant, *Notes on spinors in low dimension*, arXiv:2011.05568,
   especially Sections 2--3.
2. M. Parton and P. Piccinni, *The Role of Spin(9) in Octonionic Geometry*,
   Axioms 7 (2018), 72; arXiv:1810.06288.
3. D. Ferus, H. Karcher, and H.-F. Muenzer, *Clifford Algebras and New
   Isoparametric Hypersurfaces*, Math. Z. 177 (1981), translated at
   arXiv:1112.2780.
