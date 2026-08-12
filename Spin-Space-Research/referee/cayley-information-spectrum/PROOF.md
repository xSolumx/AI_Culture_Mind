# Human-readable proof

The proof is divided into structural reductions and elementary exact algebra.
This division matters: the minimal verifier recomputes the algebraic half but
does not silently claim to prove the representation-theoretic normal form.

## 1. One probe contributes a rank-seven projector

For a unit probe `x`, the map

\[
X\in\mathfrak{spin}(8)\longmapsto \rho_\alpha(X)x
\]

has kernel equal to the Lie algebra of the `Spin(7)` stabilizer of `x`.
Because `dim Spin(8)-dim Spin(7)=28-21=7`, its Jacobian has rank seven.
With compatible orthonormal generator conventions,
`P_alpha(x)=J_alpha(x)^T J_alpha(x)` is the corresponding information
projector.

## 2. Same-view basis changes are a gauge

The Jacobian is linear in its probe. If `(x,y)` is orthonormal and

\[
\begin{pmatrix}x'\\y'\end{pmatrix}
=
\begin{pmatrix}r&t\\-t&r\end{pmatrix}
\begin{pmatrix}x\\y\end{pmatrix},
\qquad r^2+t^2=1,
\]

then expanding the two quadratic terms gives

\[
P_\alpha(x')+P_\alpha(y')
=P_\alpha(x)+P_\alpha(y).
\]

The mixed terms cancel. Therefore the information operator depends on each
same-view pair only through its oriented two-plane, not through a chosen basis
of that plane.

## 3. The orbit-normal-form bridge

Fixing the vector probe leaves a `Spin(7)` action. Under Clifford
multiplication by that probe, the two restricted chiral modules are identified
with a common real eight-dimensional module. The four chiral probes then form
an oriented orthonormal four-frame.

The standard orbit description of the spin representation records that the
`Spin(7)` action on the oriented Grassmannian of four-planes in `R^8` has
cohomogeneity one; the Cayley four-form supplies its orbit coordinate. This
description is recalled explicitly in Section 4 of Berndt and Tamaru,
*Cohomogeneity one actions on noncompact symmetric spaces of rank one*, while
the Cayley calibration is classical in Harvey and Lawson.

The additional object here is a `2+2` split of the four-plane. Exact isotropy
calculations in the full repository show that, on a rational principal
representative, the plane stabilizer has dimension six and restricts onto all
of `so(4)`. Its action is therefore transitive on oriented orthogonal `2+2`
splittings, with residual `SO(2) x SO(2)`. The same restriction rank is checked
on both endpoint representatives. Together with Step 2, this leaves the single
Cayley coordinate `c`, and reflection shows that the spectrum depends only on
`z=c^2`.

This is the load-bearing external/structural bridge. The compact verifier takes
the resulting canonical family as input. The full repository certificate
recomputes the stated isotropy ranks, but a referee should assess the global
orbit argument as ordinary mathematics rather than infer it from a JSON flag.

## 4. Four invariant blocks

For the canonical family, a fixed permutation of the 28 bivector coordinates
splits the exact information matrix into blocks of dimensions `8,8,8,4`.
Writing `lambda` for the characteristic variable, the exact block polynomials
are

\[
\begin{aligned}
\chi_0(\lambda)
={}&-\frac14(\lambda-1)^2
(2c\lambda-c-2\lambda^3+8\lambda^2-6\lambda+1)\\
&\quad\cdot
(2c\lambda-c+2\lambda^3-8\lambda^2+6\lambda-1),
\end{aligned}
\]

\[
\begin{aligned}
\chi_1(\lambda)=\chi_2(\lambda)
={}&\frac1{16}(c-2\lambda^2+4\lambda-1)
(c-2\lambda^2+6\lambda-3)\\
&\quad\cdot(c+2\lambda^2-6\lambda+3)
(c+2\lambda^2-4\lambda+1),
\end{aligned}
\]

and

\[
\chi_3(\lambda)=(\lambda-1)^2(\lambda^2-3\lambda+1).
\]

The two middle blocks are not merely isospectral: the full certificate gives
a fixed signed permutation intertwining them. Their equality is therefore
structural rather than a numerical coincidence.

The maintained SymPy proof constructs the `28 x 28` matrix from the rational
triality generators and verifies these block identities modulo
`c^2+s^2=1`. The compact verifier deliberately starts here and reconstructs
everything that follows with a different, standard-library arithmetic path.

## 5. Determinant

Set `z=c^2`. Evaluating the four characteristic polynomials at zero gives

\[
\det I^{(0)}_8=\frac{1-z}{4},
\qquad
\det I^{(1)}_8=\det I^{(2)}_8
=\frac{(1-z)(9-z)}{16},
\qquad
\det I_4=1.
\]

Multiplication yields

\[
\det I_c=\frac{(1-z)^3(9-z)^2}{1024}.
\]

Differentiation gives

\[
\frac{d}{dz}\det I_c
=-\frac{(1-z)^2(9-z)(29-5z)}{1024},
\]

which is strictly negative for `0 <= z < 1`.

## 6. Direct and inverse spectral moments

Let

\[
p(\lambda)=\det(\lambda I-I_c)
=\lambda^{28}-e_1\lambda^{27}+e_2\lambda^{26}-\cdots.
\]

Multiplying the four exact block polynomials gives

\[
e_1=35,
\qquad
e_1^2-2e_2=67.
\]

These are respectively `tr(I_c)` and `tr(I_c^2)`. At `lambda=0`, logarithmic
differentiation gives

\[
\operatorname{tr}(I_c^{-1})=-\frac{p'(0)}{p(0)},
\]

and

\[
\operatorname{tr}(I_c^{-2})
=-\frac{p''(0)p(0)-p'(0)^2}{p(0)^2}.
\]

Exact substitution produces the two rational functions stated in the theorem.
Their derivatives reduce to

\[
\frac{96(z^2-6z+21)}{(1-z)^2(9-z)^2}>0
\]

and

\[
\frac{16(12609+336z-630z^2-8z^3-19z^4)}
{(1-z)^3(9-z)^3}>0.
\]

For the first numerator,
`z^2-6z+21=(z-3)^2+12`. For the second, the exact degree-four
Bernstein coefficients on `[0,1]` are

\[
12609,\quad12693,\quad12672,\quad12544,\quad12288,
\]

all strictly positive. This proves simultaneous D-, A-, and
inverse-Frobenius optimality at `z=0` within the family.

## 7. Endpoint rank and conditioning

At `c=1`, the first three block polynomials factor as

\[
\chi_0
=\lambda(\lambda-1)^3
(\lambda^2-4\lambda+2)(\lambda^2-3\lambda+1),
\]

\[
\chi_1=\chi_2
=\lambda(\lambda-2)^2(\lambda-1)^3
(\lambda^2-3\lambda+1).
\]

Each contributes one simple zero; the four-dimensional block is nonsingular.
Thus the endpoint rank is `28-3=25`.

For a simple zero branch `lambda_j(c)`, implicit differentiation gives

\[
\lambda_j'(1)
=-\frac{\partial_c\chi_j(1,0)}
{\partial_\lambda\chi_j(1,0)}=-\frac14.
\]

Since `1-c^2=-2(c-1)+O((c-1)^2)`, all three branches satisfy

\[
\lambda_j(z)=\frac{1-z}{8}+O((1-z)^2).
\]

The surviving factor `lambda^2-4lambda+2` supplies the largest root
`2+sqrt(2)`. The other possible large quadratic root is
`(3+sqrt(5))/2`, which is smaller. The condition-number and inverse-moment
asymptotics follow by inverting the three small branches.

## 8. Conclusion of the proof

Steps 4–7 prove the canonical-family theorem by exact finite algebra. Steps
1–3 promote it to the complete orthonormal balanced orbit when the global
normal-form bridge is accepted. Neither argument covers nonorthogonal frames,
other probe allocations, or sequence-model performance.
