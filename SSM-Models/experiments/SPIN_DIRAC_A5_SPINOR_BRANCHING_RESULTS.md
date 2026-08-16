# Exact spinor branching over the global `2.A5` component atlas

**Status:** exact character-ring and Clifford-module calculation.

**Executed:** 2026-08-16 18:12:53 +02:00 (`Africa/Johannesburg`).

**Artifact:**
[`spin_dirac_a5_spinors_20260816.json`](artifacts/spin_dirac_a5_spinors_20260816.json),
SHA-256 `2cfe3573b49bcb2a217bcd7545e4629c7e6fa8cc50be61137c0b9f9f9864f21e`.

## Result

The spinor or half-spinor representation has now been restricted to `2.A5`
for every one of the 245 orthogonal representation types in the global atlas.
All irreducible multiplicities are exact nonnegative integers.

| `n` | orthogonal types checked | Spin components represented | types with invariant spinors | orientation splits | splits distinguished by chiral character |
|---:|---:|---:|---:|---:|---:|
| 3 | 3 | 3 | 1 | 0 | 0 |
| 8 | 25 | 32 | 16 | 7 | 7 |
| 9 | 32 | 32 | 16 | 0 | 0 |
| 10 | 42 | 42 | 21 | 0 | 0 |
| 11 | 59 | 59 | 29 | 0 | 0 |
| 12 | 84 | 98 | 44 | 14 | 14 |

“Invariant spinor” here means a trivial `2.A5` summand in the restricted
finite-dimensional spinor representation. It does **not** mean a Dirac zero
mode on an unspecified manifold.

## Irreducible building blocks

Write `S` for the unique odd-dimensional spinor and `S+`, `S-` for the two
even-dimensional half-spinors. The executable derives the following base
branchings.

| real vector block | real dimension | spinor branching |
|---|---:|---|
| `1_R` | 1 | `S = 1` |
| `3_R` | 3 | `S = 2` |
| `3p_R` | 3 | `S = 2_prime` |
| `5_R` | 5 | `S = 4_spin` |
| `4_R` | 4 | `S+ = 2`, `S- = 2_prime` |
| `2_H` | 4 | `S+ = 2*1`, `S- = 2` |
| `2p_H` | 4 | `S+ = 2*1`, `S- = 2_prime` |
| `4_H` | 8 | `S+ = 3*1 + 5`, `S- = 2*4_spin` |
| `6_H` | 12 | `S+ = 4*1 + 2*4_vector + 4*5`, `S- = 2*4_spin + 4*6_spin` |

The `+/-` convention fixes one orientation. Reversing orientation swaps the
two rows of the half-spin pair.

The quaternionic rows are not guessed from the final character table. For a
quaternionic SU(2) module of highest weight `m = 1, 3, 5`, the code enumerates
all signed half-sums of its weights, separates even and odd sign parity, and
decomposes the resulting weight multisets into SU(2) irreducibles before
restricting them to `2.A5`.

For example, the real eight-dimensional `4_H` block gives

\[
S^+\downarrow_{2.A_5}=3\cdot 1+5,
\qquad
S^-\downarrow_{2.A_5}=2\cdot 4_{\mathrm{spin}}.
\]

Its two orientation components exchange these characters. One orientation
therefore has three invariant positive-chirality spinors, while the reversed
orientation has three invariant negative-chirality spinors.

## Fixed Spin(3) ladder

For the original component

\[
V_n=3_{\mathbb R}+1_{\mathbb R}^{\oplus(n-3)},
\]

the previous qualitative “isotypic spinor” statement is now an exact branching
formula:

| `n` | restricted spinor or each half-spinor |
|---:|---|
| 3 | `2` |
| 8 | `4*2` |
| 9 | `8*2` |
| 10 | `8*2` |
| 11 | `16*2` |
| 12 | `16*2` |

No rung of this fixed component contains a trivial `2.A5` spinor summand.

## Independent Clifford verification

The branching calculation and its verification use different constructions.

1. The candidate spinors are assembled from irreducible blocks using the
   graded Clifford direct-sum rules.
2. Independently, for every group element and every vector component, Newton
   identities reconstruct all exterior-power characters from the power traces
   of the vector representation.
3. For even `n`, the code verifies exactly

   \[
   (S^+\otimes S^+)+(S^-\otimes S^-)=\Lambda^{\mathrm{even}}V,
   \]

   \[
   (S^+\otimes S^-)+(S^-\otimes S^+)=\Lambda^{\mathrm{odd}}V.
   \]

4. For odd `n`, it verifies

   \[
   S\otimes S=\Lambda^{\mathrm{even}}V
   =\Lambda^{\mathrm{odd}}V.
   \]

All `245` component records pass. Every unsplit even-dimensional type has equal
`S+` and `S-` branching, as required by its orientation-reversing commutant.
Every orientation-split type in dimensions 8 and 12 has unequal chiral
characters, so the branching detects all 21 split pairs rather than merely
repeating the vector-component count.

## Claim boundary

Proved by the executable exact calculation:

- spinor/half-spinor irreducible multiplicities for all 245 vector types;
- the SU(2)-weight derivation of quaternionic base blocks;
- all independent exterior-algebra Clifford identities;
- invariant-spinor multiplicities and chirality exchange under orientation;
- the exact isotypic formula for the original Spin(3) ladder.

Not proved or claimed:

- a geometric Dirac spectrum, eta invariant, or index without specifying a
  manifold, bundle, connection, boundary conditions, and Dirac operator;
- that a representation-theoretic invariant spinor is automatically a zero
  mode on an arbitrary geometry;
- an ML, memory, or production-kernel advantage;
- triality outside Spin(8).

## Replay

```powershell
python SSM-Models\spin_dirac_a5_spinors.py `
  --output SSM-Models\experiments\artifacts\spin_dirac_a5_spinors_20260816.json
python -m unittest discover -s SSM-Models `
  -p "test_spin_dirac_a5_spinors.py" -v
```

The full exact replay takes approximately 32 seconds. All five focused tests
pass.
