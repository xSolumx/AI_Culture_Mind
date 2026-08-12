# Exact isotypic decomposition over \(\mathbb Q(\sqrt2)\)

**Exact compiler extension — 2026-08-11**

**Status:** certified real Schur detection and reducible isotypic decomposition
for matrices and witnesses in the declared field \(\mathbb Q(\sqrt2)\), under
the existing complete-reducibility assumption

**Design:** [algebraic extension rationale](../ALGEBRAIC_EXTENSION_DESIGN.md)

**Code:**
[`exact_real_scalar_field.py`](../../src/exact_real_scalar_field.py),
[`algebraic_isotypic_decomposition.py`](../../src/algebraic_isotypic_decomposition.py)

**Artifact:**
[`algebraic_isotypic_decomposition_20260811.json`](../../artifacts/algebraic_isotypic_decomposition_20260811.json)

**SHA-256:**
`0616bf5ae49343e1a9340e865b5e17f817f9e4401224a41478c57bb27731c049`

**Replay:**
[`test_algebraic_isotypic_decomposition.py`](../../tests/test_algebraic_isotypic_decomposition.py)

## Result

The maintained rational Schur and reducible compilers now accept an explicitly
declared positive quadratic generator. Membership, signs, nullspaces,
commutants, centers, intertwiners, minimal-polynomial factorization, CRT
projectors, real Schur type, and aligned isotypic coordinates remain exact.

The core obstruction control is

\[
A=\begin{pmatrix}0&2\\1&0\end{pmatrix}.
\]

It remains unresolved over \(\mathbb Q\), as required. Over
\(\mathbb Q(\sqrt2)\), the compiler reconstructs the two rank-one projectors

\[
E_\pm=\frac12I\pm\frac{\sqrt2}{4}A
\]

and the two real lines with eigenvalues \(\pm\sqrt2\).

The nonsplit control \(B^2=-2I\) remains one complex-type real irrep. Dense
algebraic conjugacies of canonical real, complex, and quaternionic irreps
retain their Schur types. An undeclared \(\sqrt3\) entry is rejected.

Most importantly for the active Spin programme, the concrete Spin(9) matrices
now compile directly:

| Concrete module | Recovered isotypic type | Commutant dimension |
|---|---|---:|
| Cayley-null Grassmann slice | \(V_1\oplus V_5\) | 2 |
| Full local quotient module | \(V_1\oplus2V_5\) | 5 |

The earlier rationalizing intertwiner is no longer required as compiler input;
it remains an independent closed-form cross-check.

## Direct falsifiers

| Failure mode | Required outcome | Result |
|---|---|---|
| run the genuine split over \(\mathbb Q\) | unresolved, not guessed | passed |
| negative-square quadratic | complex division type, not two lines | passed |
| \(\sqrt3\) under a \(\mathbb Q(\sqrt2)\) declaration | reject field drift | passed |
| omit complete reducibility | preserve existing refusal | passed through core suite |
| rerun rational artifacts | byte-stable reports | passed through regression suite |

## Boundary

Implemented:

- the rational field and the declared positive quadratic field
  \(\mathbb Q(\sqrt2)\);
- exact order in the chosen real embedding;
- sparse polynomial-domain nullspaces for the algebraic path;
- direct Spin(9) decomposition over the native field.

Open:

- automatic field discovery;
- arbitrary number fields or quadratic towers;
- selecting and certifying a real embedding for a general primitive
  polynomial;
- approximate or noisy generator recovery;
- any global Spin(9) determinant, Spin(8) sensing, memory, or model claim.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
python -m algebraic_isotypic_decomposition `
  --output artifacts/algebraic_isotypic_decomposition_20260811.json
python -m pytest tests/test_algebraic_isotypic_decomposition.py `
  tests/test_schur_type_detector.py `
  tests/test_reducible_isotypic_decomposition.py -q
```
