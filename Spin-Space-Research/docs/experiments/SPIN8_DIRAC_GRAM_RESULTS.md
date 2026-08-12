# What Would Dirac Say? The Spin(8) Dirac--Gram Gate

**Date:** 2026-08-03

**Preregistration:** `SPIN8_DIRAC_GRAM_PREREGISTRATION.md`

**Harness:** `../spin8_dirac_gram.py`

**Raw artifact:** `spin8_dirac_gram_20260803.json`

**Artifact SHA-256:**
`c659243e70a5f4e7dc1181c8b515013bfc4dbaf867f283b892405e2b156722ca`

## Result in one sentence

The missing QR lemma has been sharpened to the Gram-volume conjecture

\[
\det I(X)\leq \det(XX^T)^3\det I(Q),
\]

proved exactly on two nontrivial Cayley--Gram slices and attacked without a
counterexample over one million fresh frames, 64 direct adversaries, and a
separate whitening-flow campaign; the complete six-correlation theorem is not
yet proved.

## The Dirac reduction

A unit probe in any of the three triality representations contributes a
rank-seven orthogonal projector `P_r(x)` on the 28-dimensional Lie algebra.
Exact symbolic contraction gives

\[
\begin{aligned}
P_r(x)^2&=P_r(x), &\operatorname{rank}P_r(x)&=7,\\
\operatorname{tr}(P_r(x)P_r(y))
&=1+6\langle x,y\rangle^2, &&r=s,\\
\operatorname{tr}(P_r(x)P_s(y))
&=\frac74, &&r\ne s,
\end{aligned}
\]

for unit `x,y`. Thus different triality views are exactly isoclinic at the
level of Hilbert--Schmidt overlap, while correlations inside one view are
penalized quadratically. The information operator is simply

\[
I=\sum_{k=1}^{5}P_{r_k}(x_k).
\]

This is the useful Dirac viewpoint: prove a statement about a sum of geometric
projectors, not about a 28-by-28 determinant expanded without structure.

## Strengthened theorem target

Fix the singleton vector query and collect the other four unit probes into
`X`. Let

\[
G=XX^T,\qquad \Delta=\det G,
\qquad Q=G^{-1/2}X.
\]

Row whitening preserves the oriented four-plane, so it preserves the
normalized Cayley coordinate

\[
c=\frac{\Phi(X)}{\sqrt\Delta}.
\]

The previously proved orthonormal theorem therefore gives

\[
\det I(Q)=\frac{(1-c^2)^3(9-c^2)^2}{1024}.
\]

The new conjecture is

\[
\boxed{\det I(X)\leq\Delta^3\det I(Q)}.
\tag{DG}
\]

Since unit rows imply `Delta<=1` by Hadamard's inequality, `(DG)` is strictly
stronger than `det I(X)<=det I(Q)`. In invariant form it is

\[
1024\Delta^2\det I(X)
\leq(\Delta-\Phi^2)^3(9\Delta-\Phi^2)^2.
\tag{IG}
\]

The exponent three is now a precise, falsifiable structural claim. It is not
being inferred merely from the three-dimensional rank loss at a calibrated
endpoint.

## Exact two-slice proof

Let `u` be the square of the single active frame correlation and `z=c^2`.
For both the same-view and cross-view deformations, exact symbolic elimination
gives

\[
\Delta^3\det I(Q)-\det I(X)
=u(1-u)^3(1-z)^3 R(u,z)/C.
\]

The constants are `C=2048` and `C=4096`. The residual polynomials are

\[
\begin{aligned}
R_{\rm same}={}&u^2z^2-8u^2z+16u^2-4uz^2+42uz-104u\\
&+5z^2-70z+225,\\
R_{\rm cross}={}&-u^2z+3u^2-4uz^2+25uz-45u\\
&+8z^2-96z+216.
\end{aligned}
\]

Their exact tensor-product Bernstein coefficient matrices on `[0,1]^2` are

\[
B_{\rm same}=\begin{pmatrix}
225&190&160\\
173&297/2&127\\
137&119&103
\end{pmatrix},\qquad
B_{\rm cross}=\begin{pmatrix}
216&168&128\\
387/2&607/4&116\\
174&138&106
\end{pmatrix}.
\]

Every coefficient is strictly positive and every Bernstein basis function is
nonnegative on the unit square. Therefore both residuals are strictly positive
throughout the box. This is an exact proof of `(DG)` on both slices. Equality
comes from the extracted factor `u=0` or from a degenerate boundary, not from a
hidden zero Bernstein coefficient.

Positive Bernstein coefficients are a sufficient positivity certificate;
they are not claimed to characterize all positive polynomials.

## Fresh global attacks

The confirmatory seeds were fixed after the exploratory discovery of `(DG)`.

### One million random frames

- samples: `1,000,000` float64 unit four-frames;
- violations: **0**;
- largest log ratio:
  `log det I(X) - log det I(Q) - 3 log det G = -0.0053622286`;
- corresponding ratio: `0.9946521225`;
- corresponding Gram determinant: `0.9936518139`.

The best random frame was already close to orthonormal, as the conjecture
predicts.

### Sixty-four gradient adversaries

All 64 searches maximized the strengthened log ratio for 2,000 steps.

- violations: **0**;
- largest final ratio: `1 + 1.42e-13`, numerical equality;
- Gram determinant at the maximizer: `1.0`;
- maximum final off-diagonal Gram entry across all restarts: `0.01139`;
- best log ratio moved from `-0.1006` at step 1 to `-1.53e-12` at step 200
  and numerical zero thereafter.

The optimizers did not stall at an arbitrary negative value. They converged to
the proposed orthonormal equality manifold.

## Exact Dirac--Schur reduction

Relative to the fixed query's decomposition

\[
\mathfrak{so}(8)=\mathbb R^7\oplus\mathfrak{spin}(7),
\]

every moving unit-query projector has the exact graph form

\[
P(V)=\frac14
\begin{pmatrix}
I_7&V^T\\ V&VV^T
\end{pmatrix},
\qquad V^TV=3I_7.
\]

This was checked symbolically for both moving triality views with generic
eight-coordinate probes, not inferred from numerical samples. For the four
moving queries, put

\[
S=\sum_{i=1}^4V_i,
\qquad T=\sum_{i=1}^4V_iV_i^T.
\]

The complete information operator and its exact Schur determinant become

\[
I=\begin{pmatrix}
2I_7&S^T/4\\ S/4&T/4
\end{pmatrix},
\qquad
\boxed{\det I=2^7 32^{-21}\det(8T-SS^T)}.
\]

This replaces an opaque 28-dimensional determinant by a 21-dimensional
Dirac--Schur operator assembled from four isometric seven-frames. It does not
by itself prove the global upper bound, but it is an exact structural reduction
that the remaining proof must exploit.

## Coupled whitening: exact invariant and corrected limitation

Coordinate-by-coordinate orthogonalization is not monotone in general; direct
exploratory counterexamples rule out that tempting shortcut. The object that
survived is the coupled projected frame-potential flow

\[
\dot x_i=-\sum_{j\ne i}\langle x_i,x_j\rangle x_j
+x_i\sum_{j\ne i}\langle x_i,x_j\rangle^2.
\]

Write `H=G-I` and

\[
A=-H+\operatorname{diag}(H^2),\qquad \dot X=AX,
\qquad E=\|H\|_F^2.
\]

Exact symbolic calculation gives

\[
\operatorname{tr}A=E,
\qquad
\frac{d}{dt}\log\Delta=2E,
\qquad
\frac{d}{dt}\log|\Phi|=E.
\]

Consequently the normalized Cayley coordinate

\[
c=\Phi/\sqrt\Delta
\]

is **exactly conserved** by the flow. This is useful, but it also sharpens the
required Lyapunov inequality. Monotonicity of the strengthened ratio needs

\[
\frac{d}{dt}\log\det I(X(t))\geq6E,
\]

not merely a nonnegative derivative.

Fresh results:

- `100,000` random frames: zero negative derivatives;
- smallest raw derivative: `0.1622159109`;
- smallest frame-potential-normalized derivative: `6.0131156302`;
- 32 second-order adversaries: zero negative derivatives;
- adversarial normalized infimum observed: `5.9214371088`.

The adversarial minimizer approached a rank-three Gram boundary: its smallest
Gram eigenvalue was `5.41e-8`, while the other eigenvalues approached
approximately `1.2640, 1.2642, 1.4718`. Its normalized Cayley value is unstable
as `det G` tends to zero, so `5.9214371088` is recorded as a numerical boundary
infimum, **not** promoted to a new exact invariant. More importantly,
`5.9214371088 < 6`, so the observed boundary family falsifies this whitening
flow as a route to the strengthened cubic inequality. The flow may still be
relevant to the weaker unnormalized QR inequality because its determinant
derivative remained positive, but that is a separate open claim.

## Exact negative result: the standard elegant shortcut fails

The Kiefer--Wolfowitz approximate-design equivalence theorem would require the
inverse-information sensitivity of every admissible probe to be at most

\[
\frac{28}{5}.
\]

The exact maximum sensitivities of the balanced five-probe design are

\[
15,\qquad \frac{175}{12},\qquad \frac{175}{12}
\]

in the vector, positive-spinor, and negative-spinor views. The certificate is
therefore violated symbolically. The theorem is for an approximate design
measure; it cannot certify this exact five-probe optimum.

Simple spectral majorization is also false on general frames. The determinant
inequality, if true, is subtler than eigenvalue-by-eigenvalue ordering.

## Calibration is not information

The classical Cayley calibration and this information objective extremize
different functionals. A Cayley-calibrated plane (`|c|=1`) is special in
calibrated geometry, but it is the **informationally worst** endpoint here: the
information rank falls from 28 to 25. The Cayley-null orbit (`c=0`) is the
information-determinant maximizer on the orthonormal balanced information family.

This is compatible with the unit-comass Cayley form and its `Spin(7)`
stabilizer established in the triality construction of
[Katz and Shnider](https://arxiv.org/abs/0801.0283); it is not a claim that one
extremal problem supersedes the other.

## What is proved now

1. Every triality query contributes an exact rank-seven orthogonal projector.
2. Same-view and cross-view projector overlaps obey the exact formulas above.
3. The strengthened Gram-volume inequality is proved on two complete
   two-parameter slices by exact Bernstein positivity.
4. The approximate-design equivalence shortcut is exactly ruled out.
5. Fresh large-scale falsifiers found no counterexample to `(DG)` or to plain
   determinant growth along the whitening flow.
6. The exact graph-projector and 21-dimensional Schur reduction above hold for
   generic probes in both moving triality views.

## What remains conjectural

- `(DG)` over the full six-correlation feasible Gram--Cayley domain;
- global monotonicity of the plain information determinant along the coupled
  whitening flow (which is insufficient for `(DG)` even if true);
- the four nonbalanced allocation upper bounds;
- global five-query D-optimality of `81/1024`.

No numerical campaign, including the present one, closes those statements.

## Next proof gate

The whitening route does not prove `(DG)`. The immediate exact gate is instead
the signed star family preregistered in `SPIN8_DIRAC_STAR_PREREGISTRATION.md`,
followed by the three residual Cholesky correlations needed to reach a general
four-frame. The Schur operator `8T-SS^T` is the natural object on which to seek
a coupled matrix or Bernstein certificate.

That is the honest remaining proof—not another language-model run and not
another million random frames.
