# The frame-operator geometry of Spin(9) spinor sensing

**Exact theorem note — 2026-08-07**
**Status:** exact algebraic reduction and exact approximate-design optimum
**Certificate:** [spin9_frame_operator.py](../../src/spin9_frame_operator.py)

## Abstract

For spinor probes \(s_1,\ldots,s_k\in\mathbb R^{16}\), the infinitesimal
information operator of a shared \(\operatorname{Spin}(9)\) action depends
on the probes only through the frame operator

\[
M=\sum_{r=1}^k s_rs_r^{\mathsf T}.
\]

The symmetric Clifford decomposition

\[
\operatorname{Sym}(16)
=\Lambda^0\mathbb R^9\oplus\Lambda^1\mathbb R^9
 \oplus\Lambda^4\mathbb R^9
\]

then reveals a precise nine-dimensional gauge: the information map annihilates
the entire vector summand and is injective on the scalar and four-form
summands. This reduces the unrestricted three-probe problem exactly to a
rank-at-most-three positive-semidefinite optimization with trace three.

The convex approximate-design relaxation is solved globally. Its optimal
information matrix is \((3/4)I_{36}\), its determinant is \((3/4)^{36}\),
and its complete set of optimal frame operators is

\[
M=\frac3{16}I_{16}+\sum_{i=0}^8v_iP_i,
\qquad \lVert v\rVert\leq\frac3{16}.
\]

Every such frame has rank \(8\) or \(16\), whereas an exact three-probe frame
has rank at most \(3\). The approximate optimum is therefore unattainable by
three exact probes. The global exact three-probe optimum remains open.

The reduction also changes the relevant dimension count. Ordered unit triples
have a nine-dimensional generic quotient by Spin(9), but the objective forgets
a generic one-dimensional family of unit-column decompositions. The generic
frame-objective quotient is therefore eight-dimensional. If orthonormality can
be proved, the problem reduces further to the cohomogeneity-three Spin(9)
action on \(G_3(\mathbb R^{16})\). These counts and their boundaries are
recorded in the
[quotient audit](../SPIN9_QUOTIENT_DIMENSION_AUDIT_2026-08-07.md).

## 1. Information factors through the frame operator

Let \(P_0,\ldots,P_8\) be a symmetric Clifford system on the real spin module
\(S\cong\mathbb R^{16}\), and put

\[
G_{ij}=\frac12P_iP_j,
\qquad 0\leq i<j\leq8.
\]

For probes \(s_r\), define

\[
I_{(ij),(k\ell)}
=\sum_r\langle G_{ij}s_r,G_{k\ell}s_r\rangle.
\]

Linearity of the trace gives the exact identity

\[
I_{(ij),(k\ell)}
=\operatorname{tr}\!\left(MG_{ij}^{\mathsf T}G_{k\ell}\right),
\qquad
M=\sum_rs_rs_r^{\mathsf T}.
\]

Thus two probe collections with the same frame operator have identical
information matrices. No individual probe label survives this reduction.

For three unit probes, the image of the reduction is exactly

\[
\mathcal F_3
=\{M\succeq0:\operatorname{tr}M=3,\ \operatorname{rank}M\leq3\}.
\]

The forward inclusion is immediate. Conversely, let the nonzero eigenvalues
of such an \(M\) be
\(\lambda_1\geq\lambda_2\geq\lambda_3\geq0\). They sum to three, so

\[
\lambda_1\geq1,
\qquad
\lambda_1+\lambda_2=3-\lambda_3\geq2.
\]

Hence \((\lambda_1,\lambda_2,\lambda_3)\) majorizes \((1,1,1)\).
The Schur--Horn theorem supplies a \(3\times3\) Gram matrix with this spectrum
and unit diagonal. Factoring that Gram matrix produces three unit vectors
whose frame operator is \(M\).

## 2. The invisible vector gauge

The \(136\)-dimensional space of real symmetric endomorphisms of \(S\) has the
orthogonal Clifford basis

\[
I,
\qquad P_i,
\qquad P_iP_jP_kP_\ell\quad(i<j<k<\ell),
\]

of dimensions \(1+9+126=136\). Clifford anticommutation shows that every
\(P_i\) is annihilated by \(M\mapsto I(M)\). The scalar coefficient appears on
the diagonal of \(I(M)\), while every four-form coefficient appears in an
off-diagonal entry indexed by two disjoint bivectors. Therefore

\[
\ker I=\operatorname{span}\{P_0,\ldots,P_8\},
\qquad
\operatorname{rank}(M\mapsto I(M))=127.
\]

The machine certificate independently obtains rank \(127\) over three prime
fields. This gauge is not an optimization accident: it is an exact
representation-theoretic blindness of the information operator.

Two moment identities follow. With

\[
t=\operatorname{tr}M,
\qquad
h_i=\operatorname{tr}(MP_i),
\]

one has

\[
\operatorname{tr}I(M)=9t
\]

and

\[
\boxed{
\operatorname{tr}(I(M)^2)
=\frac{15}{8}t^2+6\operatorname{tr}(M^2)
-\frac38\lVert h\rVert^2
}.
\]

The second formula is Parseval's identity in the grade-\(0,1,4\) Clifford
decomposition, followed by the fact that the information map sees only grades
zero and four.

## 3. The Dirac four-form spectral reduction

The visible \(126\)-dimensional component has an intrinsic description. Define
the four-form \(q(M)\in\Lambda^4\mathbb R^9\) by

\[
q_{ijkl}(M)=\operatorname{tr}(MP_iP_jP_kP_\ell).
\]

It induces a symmetric operator \(K_q\) on
\(\Lambda^2\mathbb R^9\cong\mathbb R^{36}\) through

\[
\langle e_i\wedge e_j,\,
K_q(e_k\wedge e_\ell)\rangle
=q_{ijkl}.
\]

Repeated indices give zero by alternation. For disjoint pairs, Clifford
anticommutation gives

\[
G_{ij}^{\mathsf T}G_{k\ell}
=-\frac14P_iP_jP_kP_\ell,
\]

while \(G_{ij}^{\mathsf T}G_{ij}=I/4\). Therefore the entire information
operator is

\[
\boxed{
I(M)=\frac14\left(tI_{\Lambda^2}-K_{q(M)}\right),
\qquad t=\operatorname{tr}M.
}
\]

Equivalently,

\[
\boxed{
\det I(M)
=4^{-36}\det\!\left(tI_{36}-K_{q(M)}\right).
}
\]

Thus the unrestricted D-optimality problem is a spectral extremum for a
four-form subject to the condition that its scalar, vector, and four-form
components recombine into a positive-semidefinite rank-three spinor frame.
The certificate constructs \(K_q\) independently from the Clifford traces and
checks the boxed matrix identity entry by entry.

## 4. Global solution of the approximate-design relaxation

Relax the rank constraint and optimize over

\[
\mathcal F_{\mathrm{rel}}
=\{M\succeq0:\operatorname{tr}M=3\}.
\]

Every feasible information matrix has trace \(27\). The arithmetic--geometric
mean inequality for its \(36\) eigenvalues gives

\[
\det I(M)
\leq\left(\frac{27}{36}\right)^{36}
=\left(\frac34\right)^{36},
\]

with equality exactly when \(I(M)=(3/4)I_{36}\). The kernel result shows that
this is equivalent to

\[
M=\frac3{16}I_{16}+P(v),
\qquad
P(v)=\sum_i v_iP_i.
\]

Since \(P(v)^2=\lVert v\rVert^2I\), the eigenvalues of \(M\) are

\[
\frac3{16}+\lVert v\rVert
\quad\text{and}\quad
\frac3{16}-\lVert v\rVert,
\]

each with multiplicity eight. Positivity is therefore equivalent to
\(\lVert v\rVert\leq3/16\). Interior optimizers have rank \(16\); boundary
optimizers have rank \(8\). None belongs to \(\mathcal F_3\).

This proves a strict exact-versus-approximate design gap, but it does not give
the size of that gap.

## 5. The rank-two boundary has exact order sixteen

Let \(J(s)\) denote the \(16\times36\) infinitesimal observation matrix of one
spinor. A generic pair has

\[
\operatorname{rank}
\begin{bmatrix}J(s_1)\\J(s_2)\end{bmatrix}
=28,
\]

so its information matrix has nullity eight. Let \(v\) be a transverse third
direction for which the three-spinor observation has full rank \(36\), and
consider a duplicate probe leaving the boundary,

\[
s_3(t)=\frac{s_1+tv}{\lVert s_1+tv\rVert}.
\]

Every \(36\times36\) maximal minor of the stacked observation matrix vanishes
to order at least eight: the first two probes supply only 28 independent
columns, so eight independent contributions must come from the term linear in
\(t\). Full rank of the transverse triple guarantees that at least one maximal
minor has order exactly eight. Cauchy--Binet writes the information determinant
as the sum of the squares of these maximal minors. Consequently

\[
\boxed{\det I(t)=C\,t^{16}+O(t^{17}),\qquad C>0.}
\]

The normalization denominator is analytic and nonzero at \(t=0\), so it does
not change the leading order. The exact witness ranks \(28\) and \(36\) are
replayed over three prime fields. A separate float64 check along the canonical
curve gives successive log--log slopes converging to \(15.9999999\); that
number is a diagnostic consequence of the proof, not evidence used by it.

## 6. What remains open

The exact three-probe problem is now the sharply stated optimization

\[
\max\{\det I(M):M\succeq0,\ \operatorname{tr}M=3,
\ \operatorname{rank}M\leq3\}.
\]

It is still necessary to prove or falsify that the algebraic point found in
the companion one-parameter theorem is globally optimal on this nonconvex
rank-three boundary. The frame reduction removes probe-label redundancy and
identifies the nine invisible directions; it does not solve that final gate.

The sharp next dichotomy is:

- prove that an optimizer can be chosen as a rank-three projector, reducing
  the problem to a three-dimensional Grassmannian quotient; or
- find a nonorthogonal challenger and retain the full eight-dimensional
  frame-objective quotient.

## 7. Certificate boundary

The certificate verifies:

- the complete orthogonal grade-\(0,1,4\) basis of
  \(\operatorname{Sym}(16)\);
- direct-versus-frame information equality;
- the Dirac four-form operator and complete characteristic-determinant
  reduction;
- information-map rank \(127\) over three primes;
- the nine zero grade-one images;
- the rank-\(28\) pair and rank-\(36\) transverse triple underlying the
  order-sixteen boundary theorem;
- the trace and second-moment identities on an exact integer witness;
- the scalar optimum and a nontrivial positive vector-gauge deformation.

The Schur--Horn converse and the global approximate-design argument are short
human proofs stated above; the artifact is an independent finite-dimensional
check, not a substitute for those arguments.

The exact report is
[spin9_frame_operator_20260807.json](../../artifacts/spin9_frame_operator_20260807.json).
Its SHA-256 digest is
df09d424e81802a9ce7393c74c6ff2ff052687b69fc6291743216501041b786c.

## References

1. D. Ferus, H. Karcher, and H.-F. Muenzer, *Clifford Algebras and New
   Isoparametric Hypersurfaces*, Math. Z. 177 (1981), 479--502;
   arXiv:1112.2780.
2. R. Bhatia, *Matrix Analysis*, Springer, 1997, Chapter II (Schur--Horn
   theorem and majorization).
