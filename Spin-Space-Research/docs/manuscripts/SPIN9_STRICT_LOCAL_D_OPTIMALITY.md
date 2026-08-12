# Strict local D-optimality of the symmetric Spin(9) three-spinor sensor

**Exact theorem note — 2026-08-07**
**Status:** exact strict local theorem on the full rank-three frame stratum
**Certificate:** [spin9_local_hessian.py](../../src/spin9_local_hessian.py)

> **Representation-interface extension (2026-08-11).** The later exact
> [slice-to-isotypic bridge](../experiments/SPIN9_SLICE_ISOTYPIC_BRIDGE_RESULTS.md)
> constructs the concrete \(\mathbb Q(\sqrt2)\) map from the Cayley-null
> Grassmann slice, matches the supported coefficient basis, and compiles the
> resulting \(V_1\oplus2V_5\) action. It independently closes the displayed
> representation-coordinate interface; the Hessian signs and theorem domain
> remain certified here.

## Abstract

Let \(I(M)\) be the \(36\times36\) information operator associated with a
positive-semidefinite rank-three frame operator
\(M\in\operatorname{Sym}(16)\), normalized by
\(\operatorname{tr}M=3\). The symmetric Spin(9) curve has the exact interior
maximizer

\[
c_\star=\frac{-17+\sqrt{241}}{24}.
\]

This note proves that its frame operator \(M_\star\) is a strict local
maximizer of \(\log\det I(M)\) on the complete smooth \(44\)-dimensional
rank-three frame stratum, modulo the natural Spin(9) symmetry. The proof joins
three earlier exact reductions: fixed-plane strict concavity, the
\(33\)-dimensional Spin(9) orbit, and the
\(V_1\oplus V_5\) Grassmann normal slice. The only possible unresolved
second-order interaction is between two equivalent \(V_5\) modules. Exact
arithmetic in a quadratic tower gives their coupled multiplicity block:

\[
H_5=
\begin{pmatrix}
\lambda_o&\rho\\
\rho&\lambda_s
\end{pmatrix},
\]

where

\[
\lambda_o=-\frac{706+8\sqrt{241}}{15},\qquad
\lambda_s=\frac{127\sqrt{241}-7717}{600},
\]

\[
\rho^2=\frac{449999-28919\sqrt{241}}{2250},\qquad
\det H_5=\frac{7563+195\sqrt{241}}{20}>0.
\]

Both diagonal entries are negative, so \(H_5\) is negative definite. The
remaining scalar normal direction also has negative curvature. This proves
strict local optimality. It does **not** prove that \(M_\star\) is the global
rank-three optimum.

## 1. The objective and its domain

The Spin(9) frame-to-information map is linear:

\[
I:\operatorname{Sym}(16)\longrightarrow\operatorname{Sym}(36).
\]

On the open set where \(I(M)\) is positive definite, define

\[
F(M)=\log\det I(M).
\]

The smooth rank-three, trace-three stratum is

\[
\mathcal F_3^†
=\{M\succeq0:\operatorname{rank}M=3,\ \operatorname{tr}M=3\}.
\]

Its dimension is

\[
16\cdot3-\frac{3(3-1)}2-1=44.
\]

Every member of this stratum is realizable by three unit spinor probes. This
is an exact frame-operator formulation of the three-probe problem, not a
continuous approximate-design relaxation. For completeness, if the nonzero
eigenvalues of \(M\) are
\(\lambda_1\geq\lambda_2\geq\lambda_3>0\), then their sum is three and

\[
\lambda_1\geq1,
\qquad
\lambda_1+\lambda_2=3-\lambda_3\geq2.
\]

Thus \((\lambda_1,\lambda_2,\lambda_3)\) majorizes \((1,1,1)\).
Schur--Horn supplies a \(3\times3\) positive-semidefinite Gram matrix with
that spectrum and unit diagonal; factoring it gives three unit probes with
frame operator \(M\). This is the converse proved in detail in the
[frame-operator reduction](SPIN9_FRAME_OPERATOR_REDUCTION.md), not an
assumption of the local theorem.

## 2. Tangent-space decomposition

Write \(M_\star=P_W\), the orthogonal projector onto the candidate
three-plane \(W\). Tangent directions split into support motion and spectral
motion:

\[
T_{M_\star}\mathcal F_3^†
\cong T_WG_3(\mathbb R^{16})\oplus\operatorname{Sym}_0(W),
\]

with dimensions \(39+5\).

The candidate plane has a three-dimensional \(\operatorname{SO}(3)\)
stabilizer. Its Spin(9) orbit has dimension \(36-3=33\). The exact normal-slice
theorem gives

\[
T_WG_3(\mathbb R^{16})
=T_W(\operatorname{Spin}(9)\!\cdot\!W)
\oplus V_1\oplus V_5^{(o)}.
\]

Here \(V_1\) is tangent to the symmetric \(c\)-curve and \(V_5^{(o)}\) is the
five-dimensional orientation slice. The supported traceless symmetric
directions form a second copy,

\[
\operatorname{Sym}_0(W)\cong V_5^{(s)}.
\]

Consequently

\[
\boxed{
T_{M_\star}\mathcal F_3^†
=T_{M_\star}(\operatorname{Spin}(9)\!\cdot\!M_\star)
\oplus V_1\oplus V_5^{(o)}\oplus V_5^{(s)}.
}
\]

The dimensions are \(33+1+5+5=44\).

The slice representation was reconstructed exactly at \(c=0\). Its transport
to \(c_\star\) uses more than a bare dimension count. The exact
curve-stabilizer matrix has constant rank on the connected interior interval,
so its kernels form a smooth rank-three Lie-algebra bundle
\(\mathfrak h_c\). The induced action on \(W(c)\) is faithful and spans
\(\mathfrak{so}(W(c))\), identifying every fibre with
\(\mathfrak{so}(3)\). After a local smooth choice of normalized bracket basis,
the Casimir on the Grassmann normal bundle varies continuously. Its
eigenvalues are drawn from the discrete finite set \(j(j+1)\) allowed in a
six-dimensional real orthogonal representation; therefore their integer
multiplicities are locally constant. Connectedness then transports the exact
\(V_1\oplus V_5\) type from \(c=0\) to \(c_\star\). This is the precise
bundle argument used here; constant stabilizer dimension alone would not
justify the conclusion.

## 3. Stationarity

There are four types of tangent direction.

1. \(F\) is constant on the Spin(9) orbit.
2. The exact curve determinant has derivative zero at \(c_\star\).
3. The fixed-plane theorem proves stationarity against every member of
   \(\operatorname{Sym}_0(W)\).
4. A nontrivial irreducible \(V_5\) has no invariant linear functional, so
   stabilizer invariance forces the first derivative to vanish on
   \(V_5^{(o)}\).

Thus \(M_\star\) is a critical point on the full stratum. The exact verifier
also evaluates the representative first variations in both \(V_5\) channels
and obtains zero.

## 4. Reduction of the Hessian

The Hessian annihilates orbit directions, including mixed orbit--slice terms.
Indeed, for every infinitesimal action field \(\xi^\#\), invariance gives
\(dF(\xi^\#)=0\) identically. Differentiating this identity at the critical
point in an arbitrary tangent direction \(v\) gives
\(\operatorname{Hess}F(v,\xi^\#)=0\); the term involving
\(dF(\nabla_v\xi^\#)\) vanishes by stationarity. There is no invariant
bilinear coupling between \(V_1\) and \(V_5\). On the two equivalent \(V_5\)
copies, Schur's lemma gives

\[
\operatorname{Hess}_{V_5^{(o)}\oplus V_5^{(s)}}F
=I_{V_5}\otimes H_5,
\qquad
H_5=
\begin{pmatrix}
\lambda_o&\rho\\
\rho&\lambda_s
\end{pmatrix}.
\]

This cross term cannot be omitted: negative curvature on each \(V_5\) copy
separately would not rule out a positive mixed direction.

For a smooth frame path \(M(t)\), exact differentiation gives

\[
\frac{d^2}{dt^2}F(M(t))\bigg|_{t=0}
=\operatorname{tr}\!\left(I^{-1}I''\right)
-\operatorname{tr}\!\left(I^{-1}I'I^{-1}I'\right).
\]

The certificate evaluates this identity directly. It does not fit a
quadratic to samples of a trigonometric geodesic.

## 5. Exact quotient construction

The visible frame coordinates move both in the quotient and along the group
orbit. Treating their raw derivative as a normal-slice vector gives the wrong
curvature. The certificate therefore performs the following operations in
exact arithmetic:

1. construct all \(36\) infinitesimal Spin(9) frame directions;
2. select and verify a nonsingular \(33\times33\) orbit Gram system;
3. project both the symmetric-curve derivative and a trial orientation
   derivative off the orbit;
4. remove the quotient curve component from the orientation derivative;
5. verify horizontality, orbit-normality, and curve-normality;
6. normalize with the intrinsic Grassmann metric;
7. evaluate the diagonal Hessians and the complete mixed coupling.

This order matters. Projecting away the displayed curve derivative before
removing its hidden orbit motion produces a false candidate identity. The
exact acceptance test rejects that construction.

## 6. Exact values and signs

The scalar curve direction, in the coordinate \(c\), has curvature

\[
\mu
=F''(c_\star)
=-\frac{11809+137\sqrt{241}}{540}<0.
\]

For unit quotient tangents in the two \(V_5\) copies, the diagonal entries and
the squared intertwiner coefficient are

\[
\lambda_o=-\frac{706+8\sqrt{241}}{15}<0,
\]

\[
\lambda_s=\frac{127\sqrt{241}-7717}{600}<0,
\]

\[
\rho^2=\frac{449999-28919\sqrt{241}}{2250}>0.
\]

Most importantly,

\[
\lambda_o\lambda_s-\rho^2
=\frac{7563+195\sqrt{241}}{20}>0.
\]

The \(2\times2\) Sylvester criterion therefore proves \(H_5\prec0\). Together
with \(\mu<0\), the Hessian is negative definite on a complement to the group
orbit. Hence:

> **Theorem.** The symmetric frame \(M_\star\), with
> \(c_\star=(-17+\sqrt{241})/24\), is a strict local maximizer of
> \(\log\det I(M)\) on \(\mathcal F_3^\dagger\), modulo Spin(9).

Equivalently, every sufficiently small non-orbit perturbation of the complete
three-unit-probe frame decreases the determinant.

## 7. Proof object

The exact point lives in the quadratic tower

\[
\mathbb Q(q,d,b,z),
\]

with

\[
q^2=241,
\qquad d^2=\frac{7+q}{48},
\qquad b^2=\frac{41-q}{48},
\qquad z^2=\frac{-247+17q}{32}.
\]

The verifier represents every field element in the explicit \(16\)-element
basis and performs fraction arithmetic only. It reconstructs the frame,
information operator, orbit projection, second variations, and signs. The
closed forms above are checked against those independently computed field
elements. No floating-point arithmetic, PSLQ relation, or interpolated
rational function is accepted as proof.

An independent Hessian-path numerical falsifier is also maintained in
[spin9_local_hessian_independent.py](../../src/spin9_local_hessian_independent.py).
It does not import the exact certificate or reuse its quotient projection.
Instead it differentiates the direct information determinant on a fresh
44-coordinate rank-three chart. It still shares the foundational Spin(9)
generator constructor, so it is not an independent implementation of the base
Clifford system. Float64 autodiff finds 11 negative directions, 33 symmetry
zeros, and no positive direction. This is a regression check for sign, gauge,
normalization, and dimension errors; it is deliberately not used as exact
evidence.

## 8. Reproduction

```powershell
$env:PYTHONPATH = "src"
python -m spin9_local_hessian `
  --output artifacts/spin9_local_hessian_exact.json
python -m pytest tests/test_spin9_local_hessian.py -q
```

The generated artifact is
[spin9_local_hessian_exact.json](../../artifacts/spin9_local_hessian_exact.json),
with SHA-256 digest
`a156c032c67bc218cdca56815df5c3e9ef4b7b670b2d8a6c913a027443fd98db`.

## 9. Open problem and nonclaims

This theorem is local. It does not prove any of the following:

- global optimality over all rank-three frame operators;
- uniqueness of the global optimum modulo Spin(9);
- equality between exact three-probe and approximate-design optima;
- a sequence-model or memory advantage for Spin(9).

The remaining global problem is an eight-dimensional quotient problem for
the objective: three Grassmann orbit coordinates plus five spectral
coordinates. The present theorem establishes that the symmetric candidate is
an isolated strict local maximum in that quotient. A global proof still needs
a domain-wide invariant inequality, a complete quotient parametrization, or a
certified exclusion of all other critical or boundary strata.
