# Exact Spin(9) slice-to-isotypic compiler bridge

**Exact implementation and representation certificate — 2026-08-11**

**Status:** the concrete Cayley-null Grassmann slice over
\(\mathbb Q(\sqrt2)\) is exactly conjugate to the rational
\(V_1\oplus V_5\) model; after adjoining the supported
\(\operatorname{Sym}_0(3)\) coefficient module, the complete
\(V_1\oplus2V_5\) representation passes the reducible isotypic compiler

**Code:**
[`spin9_slice_isotypic_bridge.py`](../../src/spin9_slice_isotypic_bridge.py)

**Artifact:**
[`spin9_slice_isotypic_bridge_20260811.json`](../../artifacts/spin9_slice_isotypic_bridge_20260811.json)

**Replay:**
[`test_spin9_slice_isotypic_bridge.py`](../../tests/test_spin9_slice_isotypic_bridge.py)

## The interface gap

The normal-slice theorem had already proved the branching

\[
N_{P_0}\cong V_1\oplus V_5
\]

from a Casimir spectrum. Separately, the reducible compiler certified a
rational fixture of type \(V_1\oplus2V_5\). What had not been checked was that
the compiler fixture was exactly the concrete stabilizer action produced from
the Spin(9) Clifford matrices. The geometric nullspace basis contains
\(\sqrt2\), while the compiler intentionally accepts rational generators.
A dimension match did not close that scalar-field gap.

The Grassmann constructor is now a reusable exact API. It returns the frame,
orbit matrix, horizontal constraints, normal basis, normal metric, stabilizer
matrices, slice actions, Casimir, and curve tangent from one source of truth.
The normal metric in its natural basis is

\[
G_N=\operatorname{diag}(2,4,4,2,8,4).
\]

## Standard stabilizer normalization

The raw stabilizer generators obey

\[
[H_0,H_1]=H_2,\qquad
[H_0,H_2]=-2H_1,\qquad
[H_1,H_2]=H_0.
\]

Thus

\[
F_0=H_0/\sqrt2,\qquad F_1=H_1,\qquad F_2=H_2/\sqrt2
\]

obey the standard \(\mathfrak{so}(3)\) brackets. Their six-dimensional slice
actions have entries in \(\mathbb Q(\sqrt2)\), remain skew for \(G_N\), and
satisfy

\[
-\sum_i\rho(F_i)^2=\mathcal C.
\]

The exact complementary Casimir projectors are

\[
P_1=I-\mathcal C/6,\qquad P_5=\mathcal C/6,
\]

with ranks one and five.

## Exact intertwiner, not a fitted basis

Let \(R_i=0\oplus\rho_2(F_i)\) be the rational action on
\(V_1\oplus\operatorname{Sym}_0(3)\). The certificate solves all 108 scalar
equations

\[
T\rho(F_i)=R_iT,\qquad i=0,1,2,
\]

over \(\mathbb Q(\sqrt2)\). The solution space has dimension two. Its two
canonical nullspace maps have ranks one and five, exactly exposing the two
inequivalent irreducibles. Their sum is

\[
T=
\begin{pmatrix}
\frac{\sqrt2}{2}&0&0&\frac{\sqrt2}{2}&0&1\\
\frac{\sqrt2}{3}&0&0&-\frac{\sqrt2}{6}&0&-\frac13\\
\frac{\sqrt2}{6}&0&0&-\frac{\sqrt2}{3}&0&\frac13\\
0&0&\frac{\sqrt2}{2}&0&0&0\\
0&\frac{\sqrt2}{2}&0&0&0&0\\
0&0&0&0&1&0
\end{pmatrix},
\qquad
\det T=-\frac14.
\]

Every conjugacy identity \(T\rho(F_i)T^{-1}=R_i\) is checked symbolically.
The exact curve tangent becomes

\[
T\dot P_0=\left(\frac{3\sqrt2}{8},0,0,0,0,0\right)^{\mathsf T},
\]

so the known curve is not merely in the Casimir kernel: it is explicitly the
canonical \(V_1\) coordinate.

## The second spin-two copy

The local-Hessian verifier uses the supported coefficient basis

\[
\operatorname{diag}(1,-1,0),\quad
\operatorname{diag}(1,1,-2),\quad
E_{12}+E_{21},\quad E_{13}+E_{31},\quad E_{23}+E_{32}.
\]

Its Frobenius squared norms are exactly \((2,6,2,2,2)\), matching that
verifier. The rational coordinate change to the compiler's canonical
\(\operatorname{Sym}_0(3)\) basis is

\[
U=
\begin{pmatrix}
1&1&0&0&0\\
0&2&0&0&0\\
0&0&1&0&0\\
0&0&0&1&0\\
0&0&0&0&1
\end{pmatrix},
\qquad \det U=2.
\]

It intertwines the supported coefficient action with the same rational
spin-two generators. Therefore \(T\oplus U\), whose determinant is
\(-1/2\), conjugates the concrete algebraic direct sum to the exact rational
fixture \(V_1\oplus2V_5\).

The reducible compiler then returns two real-type isotypic blocks:

| irreducible | multiplicity | block commutant |
|---|---:|---:|
| \(V_1\) | 1 | \(\mathbb R\), dimension 1 |
| \(V_5\) | 2 | \(\operatorname{Mat}_2(\mathbb R)\), dimension 4 |

All central-projector, inter-copy intertwiner, aligned-basis, center, and
double-centralizer gates pass.

## What is proved and what remains open

Established:

- the concrete Cayley-null normal-slice action lies in
  \(\mathbb Q(\sqrt2)\) and is exactly \(V_1\oplus V_5\);
- the full algebraic intertwiner space has dimensions and ranks \(1+5\);
- the displayed invertible intertwiner rationalizes the concrete slice;
- the curve tangent maps exactly to the canonical trivial coordinate;
- the supported local-Hessian coefficient basis is rationally the same
  \(V_5\);
- the combined concrete representation compiles as
  \(V_1\oplus2V_5\), with complete commutant
  \(\mathbb R\oplus\operatorname{Mat}_2(\mathbb R)\).

Not established:

- a general decomposition algorithm over arbitrary algebraic number fields;
- a single global chart on the nonpolar Grassmann quotient;
- the full finite-radius coupled determinant identity or its positivity;
- transport of this displayed matrix \(T\) as one global trivialization along
  the complete symmetric curve;
- any memory-capacity, scanner-throughput, or model-quality consequence.

The bridge transfers a representation object between Programmes 05 and 01.
It does not transfer the sensing theorem's domain or the compiler's systems
claims across that boundary.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
python -m spin9_slice_isotypic_bridge `
  --output artifacts/spin9_slice_isotypic_bridge_20260811.json
python -m pytest tests/test_spin9_grassmann_slice.py `
  tests/test_spin9_slice_isotypic_bridge.py `
  tests/test_reducible_isotypic_decomposition.py -q
```

The published artifact SHA-256 is
`1bc37f48e90d41f9bda8139033b7a39a3b2427d5931b41613214370a9491532f`.
