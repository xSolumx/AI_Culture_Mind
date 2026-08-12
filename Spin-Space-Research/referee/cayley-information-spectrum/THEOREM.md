# Precise theorem statement

## Information operator

Let `V`, `S+`, and `S-` be the vector and two real chiral-spinor
representations of `Spin(8)`. Fix the same orthonormal basis
`E_1,...,E_28` of the action algebra in all three views. For a unit probe `x`
in view `alpha`, define

\[
J_\alpha(x)
=\begin{bmatrix}
G^{(\alpha)}_1x&\cdots&G^{(\alpha)}_{28}x
\end{bmatrix},
\qquad
P_\alpha(x)=J_\alpha(x)^{\mathsf T}J_\alpha(x).
\]

For the balanced allocation containing one vector probe, two positive-chiral
probes, and two negative-chiral probes, put

\[
I=P_V(v)+P_+(p_1)+P_+(p_2)+P_-(n_1)+P_-(n_2).
\]

Each `P_alpha(x)` is positive semidefinite of rank seven.

## Canonical-family theorem

Consider the canonical orthonormal family

\[
\mathcal D_c=(e_0;e_0,e_1;e_2,ce_3+se_4),
\qquad c^2+s^2=1,
\]

under the fixed triality identifications used to define the three generator
families. Write `I_c` for its information operator and `z=c^2`.

### Theorem

For `0 <= z <= 1`, the information operator has a fixed decomposition of
dimensions `8+8+8+4`. Its determinant is

\[
\det I_c=\frac{(1-z)^3(9-z)^2}{1024}.
\]

For every `0 <= z <= 1`,

\[
\operatorname{tr}I_c=35,
\qquad
\operatorname{tr}(I_c^2)=67.
\]

For `0 <= z < 1`,

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

The determinant decreases strictly with `z`, while both inverse moments
increase strictly. Hence `z=0` is the unique D-, A-, and inverse-Frobenius
optimum in the canonical family.

At `z=1`, `rank(I_c)=25`. Exactly three eigenvalue branches vanish, and

\[
\lambda_j(z)=\frac{1-z}{8}+O((1-z)^2),
\qquad j=1,2,3.
\]

The largest surviving endpoint eigenvalue is `2+sqrt(2)`.

## Orbit-completeness corollary

If the balanced-flag normal-form proposition is accepted—namely, that every
orthonormal balanced design is information-spectrally equivalent to one
`D_c` after the shared `Spin(8)` action and the two same-view `O(2)` basis
gauges—then the theorem gives the complete spectrum and the same three aligned
optima on the full orthonormal balanced orbit.

This corollary has a different proof dependency from the canonical-family
algebra. The minimal verifier proves the latter and does not certify the global
group-action classification.
