# Exact determinant bounds on two complete Spin(9) spin-two rays

**Computer-assisted theorem note — 2026-08-11**  
**Status:** exact all-parameter bounds on two named Grassmann normal-slice rays  
**Certificate:** [spin9_v5_ray_certificate.py](../../src/spin9_v5_ray_certificate.py)

> **Scope update, 2026-08-11.** This boundary-ray theorem remains valid, but
> its statement that intermediate cubic shapes are open has been superseded by
> the [full \(V_5\) Cartan certificate](SPIN9_V5_CARTAN_CERTIFICATE.md).
> The first complete finite-radius coupled (V_1\oplus V_5) slice has since
> been proved in
> [the coupled reconstruction theorem](SPIN9_V1_V5_RECONSTRUCTION.md).
> Exact candidate optimality, the second supported (V_5), and the global
> Grassmann quotient remain open.

## Abstract

At the Cayley-null three-plane, the normal slice to the Spin(9) action on

\[
G_3(\mathbb R^{16})
\]

is (V_1\oplus V_5), with (V_5\cong\operatorname{Sym}_0(3)). The earlier
strict-local theorem controls the Hessian on this slice but does not give a
finite-radius inequality. This note constructs two explicit graph-chart rays
in (V_5): a zero-cubic ray and the axisymmetric ray fixed by an
(\operatorname{SO}(2)) subgroup. Exact block determinants and Sturm root
counts prove, for every real parameter (t),

\[
\frac{\det I(P_{\rm zero}(t))}{\det I(P_0)}\leq1
\]

and

\[
\frac{\det I(P_{\rm axis}(t))}{\det I(P_0)}<\frac{101}{100}.
\]

The algebraic symmetric candidate satisfies

\[
\frac{\det I(P_\star)}{\det I(P_0)}>\frac{101}{100},
\]

so it strictly beats both complete rays. The axisymmetric ray nevertheless
contains exact points that beat the Cayley-null base; (t=-50) is one such
point. Thus the stronger idea that the Cayley-null plane maximizes every
(V_5) ray is false.

This theorem covers three distinguished normalized cubic shapes: the
zero-cubic shape and both signs of the axisymmetric shape, because (t) ranges
over all of (\mathbb R). It does not cover intermediate cubic shapes, couple
the (V_1) curve coordinate to (V_5), or settle the global Grassmannian or
rank-three design problem.

## 1. The Cayley-null frame and its (V_5) slice

Use the maintained spinor basis (e_0,\ldots,e_{15}) and the orthonormal
Cayley-null frame

\[
B=\begin{bmatrix}
e_0 & \frac{e_1+e_8}{\sqrt2} &
\frac{-e_2+e_{12}}{\sqrt2}
\end{bmatrix},
\qquad P_0=BB^{\mathsf T}.
\]

The exact normal-slice construction is replayed from the 36 Spin(9)
generators. Its six-dimensional Casimir has spectrum

\[
0^1,\qquad 6^5,
\]

so the eigenvalue-six space is the spin-two module (V_5). In the canonical
basis used by the certificate, its cubic invariant is

\[
\begin{aligned}
p_3(x)=-\frac14\bigl(&2x_0^2x_4-4x_0x_1x_3
+\sqrt2x_1^2x_2+2x_2^2x_4\\
&-2\sqrt2x_2x_3^2+2\sqrt2x_2x_4^2-4x_3^2x_4\bigr).
\end{aligned}
\]

The verifier differentiates this polynomial against all three infinitesimal
stabilizer actions and obtains zero identically. The selected zero-cubic vector
has (p_3=0). The selected axisymmetric vector has (p_3=-2), squared slice
norm (24), and is killed by one stabilizer generator, which certifies its
(\operatorname{SO}(2))-fixed orbit type.

## 2. Two exact graph families

For a horizontal variation (Z), define

\[
X_Z(t)=B+tZ,
\qquad
P_Z(t)=X_Z(t)\bigl(X_Z(t)^{\mathsf T}X_Z(t)\bigr)^{-1}
X_Z(t)^{\mathsf T}.
\]

The zero-cubic variation has columns

\[
Z_{\rm zero}=
\begin{bmatrix}
e_{10}+e_{13} &
-\frac{e_2+e_{12}}{\sqrt2} &
\frac{e_1-e_8}{\sqrt2}
\end{bmatrix}.
\]

Its Gram denominator is

\[
\det\bigl(X_{\rm zero}^{\mathsf T}X_{\rm zero}\bigr)
=(1+2t^2)(1+t^2)^2.
\]

The axisymmetric variation has columns

\[
\begin{aligned}
z_1&=\sqrt2(-2e_9+e_{14}),\\
z_2&=2e_1-e_6-2e_8+e_{15},\\
z_3&=e_2-e_5-e_{11}+e_{12}.
\end{aligned}
\]

They are mutually orthogonal, with squared norms (10,10,4), and therefore

\[
\det\bigl(X_{\rm axis}^{\mathsf T}X_{\rm axis}\bigr)
=(1+10t^2)^2(1+4t^2).
\]

Both graph families consist of rank-three orthogonal projectors for every
finite real (t). They are finite-radius Grassmann families, not merely
tangent vectors or Taylor expansions.

## 3. Exact Dirac information blocks

Let (G_1,\ldots,G_{36}) be the doubled integer Spin(9) generators. For one
column (x(t)=b+tz), its scaled information contribution is the quadratic
matrix polynomial

\[
\bigl[G_i x(t)\bigr]_{i=1}^{36\,\mathsf T}
\bigl[G_i x(t)\bigr]_{i=1}^{36}.
\]

Columns with equal (\lVert z\rVert^2) share one Gram denominator. Combining
these groups before taking determinants gives a polynomial information matrix
over (\mathbb Q(\sqrt2)[t]). The zero-cubic ray splits into blocks of sizes
(16+20). The axisymmetric stabilizer refines the second family into

\[
6+10+10+10.
\]

After cancelling common Gram factors, write the determinant ratios as

\[
R_{\rm zero}(t)=\frac{N_0(t)}{D_0(t)},
\qquad
R_{\rm axis}(t)=\frac{N_a(t)}{D_a(t)}.
\]

The reduced degrees are (76/76) and (80/80), respectively. For the
zero-cubic ray,

\[
D_0(t)=(1+t^2)^{26}(1+2t^2)^{12},
\]

and the complete factored numerator is stored in the proof artifact. For the
axisymmetric ray,

\[
D_a(t)=(1+4t^2)^{14}(1+10t^2)^{26}.
\]

The exact numerator factorizations and the four individual axis block factors
are deliberately kept in the machine-readable artifact rather than expanded
into a 144-degree display here.

## 4. Positivity certificates

### 4.1 Zero-cubic ray

Exact cancellation gives

\[
D_0(t)-N_0(t)=t^2Q_0(t),
\]

where (Q_0\in\mathbb Q[t]) has degree (72) and (Q_0(0)=110). An exact
Sturm count gives

\[
\#\{t\in\mathbb R:Q_0(t)=0\}=0.
\]

Hence (Q_0(t)>0) on the real line and

\[
R_{\rm zero}(t)\leq1,
\]

with the only finite equality at (t=0).

### 4.2 Axisymmetric ray

Define

\[
G(t)=101D_a(t)-100N_a(t)=A(t)+\sqrt2B(t),
\qquad A,B\in\mathbb Q[t].
\]

If (G) had a real zero, its rational norm

\[
H(t)=A(t)^2-2B(t)^2
\]

would vanish there. The verifier reconstructs (H) exactly, finds degree
(160), and obtains the exact Sturm count

\[
\#\{t\in\mathbb R:H(t)=0\}=0.
\]

Since (G(0)=1), continuity proves (G(t)>0) for every real (t). Therefore

\[
R_{\rm axis}(t)<\frac{101}{100}.
\]

The exact evaluation at (t=-50) also gives

\[
R_{\rm axis}(-50)>1
\]

through a positive quadratic-field square margin. Numerically, only for
orientation, the ratio is approximately (1.00406085684). The exact sign,
not this decimal, is the theorem evidence.

## 5. Comparison with the algebraic candidate

For

\[
c_\star=\frac{\sqrt{241}-17}{24},
\]

the symmetric-curve determinant formula gives

\[
\frac{\det I(P_\star)}{\det I(P_0)}
=-
\frac{9635337157579438428125}{212986666247081951232}
+
\frac{635045396822374553125}{212986666247081951232}\sqrt{241}.
\]

Subtracting (101/100) yields a quadratic-field numerator

\[
-246261342262224779971733
+15876134920559363828125\sqrt{241}.
\]

The exact squared margin is

\[
99801371016262592182041340677947456105742336>0,
\]

so the numerator is positive. This proves the strict sandwich

\[
\boxed{
\det I(P_{\rm zero}(t))
\leq\det I(P_0)
<\frac{101}{100}\det I(P_0)
<\det I(P_\star)
}
\]

and

\[
\boxed{
\det I(P_{\rm axis}(t))
<\frac{101}{100}\det I(P_0)
<\det I(P_\star)
}
\]

for every (t\in\mathbb R).

## 6. Claim boundary and next gate

This theorem is the first maintained finite-radius determinant certificate in
the nontrivial (V_5) Grassmann slice. It strengthens the local Hessian result
on two complete rays and falsifies a tempting Cayley-null radial-maximality
lemma.

It does **not** prove any of the following:

- the same bound for intermediate normalized cubic shapes in (V_5);
- a bound after coupling the (V_1) curve coordinate to (V_5);
- global optimality on (G_3(\mathbb R^{16})/\operatorname{Spin}(9));
- orthonormality or global optimality on the full rank-three frame stratum.

The sharp next target is a two-variable certificate over radius and normalized
cubic shape, followed by its coupling to the (V_1) coordinate. Nonpolarity
of the global Spin(9) action remains a warning that this local invariant chart
cannot simply be declared global.

## 7. Reproduction and trust boundary

```powershell
$env:PYTHONPATH = "src"
python -m spin9_v5_ray_certificate `
  --output artifacts/spin9_v5_ray_certificate_20260811.json
python -m pytest tests/test_spin9_v5_ray_certificate.py -q
```

The verifier reconstructs the Clifford generators, Cayley-null frame, normal
slice, Casimir, cubic invariant, both graph projectors, all information blocks,
the reduced determinant ratios, both Sturm obligations, the exact challenger,
and the algebraic candidate comparison. It does not trust a stored `passed`
field.

The generated artifact is
[spin9_v5_ray_certificate_20260811.json](../../artifacts/spin9_v5_ray_certificate_20260811.json),
with SHA-256 digest
`48147ab8edf5f25b5723d10b0d674f81ac869c5d0dcf76c35c764a18e322961f`.

The exact arithmetic depends on SymPy's polynomial-domain determinant and
Sturm implementations. Independent external replay or a second computer
algebra system remains a publication-strengthening step.
