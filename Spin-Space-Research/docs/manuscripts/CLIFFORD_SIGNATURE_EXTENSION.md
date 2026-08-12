# An exact \(\mathrm{Cl}(3,0)\) subalgebra inside \(\mathrm{Cl}^0(1,4)\)

From the Spin(9) Clifford system

**Exact algebra and branching theorem — 2026-08-11**

**Status:** exact faithful \(\mathrm{Cl}(1,4)\) matrix construction, even
subalgebra and volume-sector classification, injective
\(\mathrm{Cl}(3,0)\) embedding, and Spin(8) triality controls

**Design:** [algebraic extension rationale](../ALGEBRAIC_EXTENSION_DESIGN.md)

**Certificate:**
[`clifford_signature_extension.py`](../../src/clifford_signature_extension.py)

**Artifact:**
[`clifford_signature_extension_20260811.json`](../../artifacts/clifford_signature_extension_20260811.json)

**SHA-256:**
`2c3c6b812e2468449be70b066cce4aafcfa278f6852747beaa073706239b47cc`

**Replay:**
[`test_clifford_signature_extension.py`](../../tests/test_clifford_signature_extension.py)

## Theorem

Let \(P_0,\ldots,P_4\) be five of the nine symmetric positive involutions in
the maintained real Spin(9) Clifford system. Set

\[
e_0=P_0,
\qquad
e_i=P_0P_i\quad(1\le i\le4).
\]

Then the five matrices anticommute and have signature

\[
e_0^2=I,
\qquad
e_1^2=e_2^2=e_3^2=e_4^2=-I.
\]

Their 32 ordered blade products are linearly independent. Hence the resulting
map

\[
\mathrm{Cl}(1,4)\longrightarrow\operatorname{Mat}_{16}(\mathbb R)
\]

is injective.

The volume element \(\omega=e_0e_1e_2e_3e_4\) is a central involution. Its
projectors

\[
E_\pm=\frac12(I\pm\omega)
\]

both have rank eight. The two restricted full-algebra modules are
quaternionic real Schur type and have zero intertwiner space between them.
Thus the matrix certificate recovers

\[
\mathrm{Cl}(1,4)
\cong
\operatorname{Mat}_2(\mathbb H)
\oplus
\operatorname{Mat}_2(\mathbb H).
\]

The 16 even blade images are independent. On the two volume sectors, the even
algebra still has quaternionic Schur type, but the cross-intertwiner space now
has dimension four. Therefore

\[
\mathrm{Cl}^0(1,4)
\cong\operatorname{Mat}_2(\mathbb H)
\]

acts as two equivalent quaternionic irreducibles on \(\mathbb R^{16}\).

Finally, define

\[
c_i=e_0e_i=P_i,
\qquad i=1,2,3.
\]

The \(c_i\) are positive anticommuting involutions. Their eight blade products
are independent and lie in the even blade span. Consequently

\[
\boxed{
\mathrm{Cl}(3,0)
\hookrightarrow
\mathrm{Cl}^0(1,4)
\hookrightarrow
\mathrm{Cl}(1,4).
}
\]

On the faithful 16-dimensional module, the embedded
\(\mathrm{Cl}(3,0)\cong\operatorname{Mat}_2(\mathbb C)\) acts as four copies
of its four-real-dimensional complex-type irreducible. The exact commutant has
dimension 32, matching \(\operatorname{Mat}_4(\mathbb C)\).

## Spin(8) control

The theorem is checked alongside the exact Spin(8) restriction already
contained in the same matrix system. The seven adjacent coordinate rotations
generate a 28-dimensional Lie algebra under exact brackets. The vector and
two chiral modules are all real Schur type and have Hom-space dimensions

\[
\begin{array}{c|ccc}
 &8_v&8_+&8_-\\\hline
8_v&1&0&0\\
8_+&0&1&0\\
8_-&0&0&1
\end{array}.
\]

The same table is recomputed after dense \(\mathbb Q(\sqrt2)\) changes of
basis. Chirality supplies two exact rank-eight projectors for

\[
S_9\big|_{\operatorname{Spin}(8)}=8_+\oplus8_-.
\]

Thus the scalar-extension work preserves rather than collapses triality's
three inequivalent irreducibles.

## Relation to the maintained \(\mathrm{Cl}(3,0)\) state

Two representation objects must not be conflated.

1. The maintained rotor model stores all eight coefficients of the algebra
   \(\mathrm{Cl}(3,0)\). Under Spin(3) conjugation this regular state is
   \(2V_0\oplus2V_1\).
2. The signature theorem restricts a faithful 16-dimensional
   \(\mathrm{Cl}(1,4)\) module to four complex spinors of
   \(\mathrm{Cl}(3,0)\).

The algebra embedding is exact, but it is not an eight-coordinate same-state
embedding of the current recurrent model. Moving to a full
\(\mathrm{Cl}(1,4)\) algebra state would increase algebra dimension from 8 to
32. Moving only to the displayed faithful module would use 16 real
coordinates. Neither change has a demonstrated memory, compute, or model
advantage.

## Dimension and claim ledger

| Object | Real dimension | Certified module statement |
|---|---:|---|
| \(\mathrm{Cl}(3,0)\) | 8 | four complex spinor copies on the 16D restricted module |
| \(\mathrm{Cl}^0(1,4)\) | 16 | two equivalent quaternionic 8D modules |
| \(\mathrm{Cl}(1,4)\) | 32 | two inequivalent quaternionic 8D modules |

Established:

- all Clifford relations and blade ranks above;
- both volume projectors and their module types;
- the injective \(\mathrm{Cl}(3,0)\) even-subalgebra map;
- the Spin(8) chirality and triality Hom-space controls;
- invariance of the Spin(8) classification under the declared algebraic field.

Not established:

- a trained or benchmarked \(\mathrm{Cl}(1,4)\) recurrent model;
- a same-state advantage over the maintained \(\mathrm{Cl}(3,0)\) model;
- a new Spin(8) or Spin(9) sensing optimum;
- a global Spin(9) determinant theorem;
- novelty relative to the classical abstract classification of real Clifford
  algebras. The contribution here is the exact connection to this repository's
  maintained matrices and compiler artifacts.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
python -m clifford_signature_extension `
  --output artifacts/clifford_signature_extension_20260811.json
python -m pytest tests/test_clifford_signature_extension.py -q
```
