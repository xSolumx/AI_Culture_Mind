# Exact `2.A5` Spin-component atlas through dimension 12

**Status:** exact finite-table and character calculation, with one explicitly
identified standard theorem input.

**Executed:** 2026-08-16 17:59:40 +02:00 (`Africa/Johannesburg`).

**Artifact:**
[`spin_dirac_a5_components_20260816.json`](artifacts/spin_dirac_a5_components_20260816.json),
SHA-256 `ed0097caf2e605b0462a94910567471f12c1d095e57a89a7fcdaabfb170895ef`.

## Result

The global compact-real quotient

\[
\operatorname{Hom}(2.A_5,\operatorname{Spin}(n))/\operatorname{Spin}(n)
\]

is now enumerated for `n = 3, 8, 9, 10, 11, 12`.  The enumeration includes
representations on which the binary center acts nontrivially in the vector
module; it is not restricted to the earlier `A5`-projecting locus.

| `n` | orthogonal isomorphism types | Spin-conjugacy components | orientation-split types | `A5`-projecting types | faithful Spin components |
|---:|---:|---:|---:|---:|---:|
| 3 | 3 | 3 | 0 | 3 | 2 |
| 8 | 25 | 32 | 7 | 13 | 22 |
| 9 | 32 | 32 | 0 | 18 | 22 |
| 10 | 42 | 42 | 0 | 22 | 31 |
| 11 | 59 | 59 | 0 | 27 | 48 |
| 12 | 84 | 98 | 14 | 35 | 82 |

The fixed ladder embedding from the preceding experiments is the component

\[
3_{\mathbb R}\oplus 1_{\mathbb R}^{\oplus(n-3)}.
\]

It is faithful as a Spin homomorphism, sends the binary central element to
`-1`, and has centralizer Lie algebra `so(n-3)`, exactly reproducing the earlier
rigidity artifact.

## Exact character certificate

The complete 120-element quaternion table has nine conjugacy classes of sizes

\[
1,1,20,30,12,12,20,12,12.
\]

The internal `A/B` labels for equal-order classes are sorted by decreasing
defining trace; they need not match another database's arbitrary class suffixes.

All nine characters are reconstructed from the defining trace
`x(g) = 2 Re(g)`, its `sqrt(5)` Galois conjugate, and the symmetric-power
recurrence.  Exact inner products give the `9 x 9` identity matrix, the degree
squares sum to 120, and all 81 tensor products reconstruct with nonnegative
integral multiplicities.

| complex character | complex dimension | Frobenius--Schur | irreducible real dimension | commutant | binary-center action |
|---|---:|---:|---:|---|---:|
| `1` | 1 | +1 | 1 | `R` | +1 |
| `2` | 2 | -1 | 4 | `H` | -1 |
| `2_prime` | 2 | -1 | 4 | `H` | -1 |
| `3` | 3 | +1 | 3 | `R` | +1 |
| `3_prime` | 3 | +1 | 3 | `R` | +1 |
| `4_vector` | 4 | +1 | 4 | `R` | +1 |
| `4_spin` | 4 | -1 | 8 | `H` | -1 |
| `5` | 5 | +1 | 5 | `R` | +1 |
| `6_spin` | 6 | -1 | 12 | `H` | -1 |

Tensoring by the defining `2` produces the affine `E8` McKay graph encoded in
the artifact. The graph check is structural: it is a connected nine-node tree
with one trivalent node and arm lengths `1, 2, 5`. This is a
representation-ring statement, not a claim that an `E8` architecture has been
implemented or trained.

The exact table independently agrees with the official
[GAP Character Table Library entry for `2.A5`](https://www.math.rwth-aachen.de/~Thomas.Breuer/ctbllib/ctbltoc/data/2.A5.html),
which records order 120 and nine classes.  GAP's
[`ExtensionInfoCharacterTable` documentation](https://www.math.rwth-aachen.de/homes/Thomas.Breuer/ctbllib/doc/chap3.html#X7A05A9937F165A15)
records the multiplier of `A5` as `2` and names `2.A5` as its universal
covering group.

## Why this classifies Spin components

The logical chain has two separately labelled parts.

1. **Computer-assisted exact:** the table is perfect; its nine irreducible
   characters are complete; their Frobenius--Schur types determine every real
   irreducible; and every multiplicity vector of the requested total dimension
   is exhaustively enumerated.
2. **Standard theorem input:** `2.A5` is the universal central extension of
   `A5`.  It is therefore superperfect, so
   `H^1(2.A5; Z/2) = H^2(2.A5; Z/2) = 0`.
3. Perfectness makes every real representation oriented.  The two mod-2
   vanishings then make its lift through `Spin(n) -> SO(n)` exist and be unique.
4. An orthogonal isomorphism type splits into two `SO(n)`--and hence two
   `Spin(n)`--conjugacy types exactly when its commutant contains no
   orientation-reversing element.  For the certified real modules, this occurs
   precisely when no odd-dimensional real-type summand is present.
5. The earlier characteristic-zero `H1=0` result makes each conjugacy orbit
   locally rigid.  Thus the finite global list is a component list, not merely
   a sample of representation points.

The type counts are checked twice: direct multiplicity enumeration and the
independent coefficient of

\[
\prod_i(1-x^{d_i})^{-1}.
\]

The orientation-split counts are independently reproduced by the generating
function containing only even-dimensional/quaternionic module types.

## Central signatures

For an `A5`-projecting vector module, every nontrivial real `A5` irreducible
contributes the nontrivial spin-extension class.  Hence the lifted binary center
is `-1` exactly when the total number of nontrivial irreducible blocks is odd.

For genuinely binary vector modules the artifact instead records the dimension
of the `-1` eigenspace of the center.  If that eigenspace is the whole space,
the two orientation-split components send the center to `+volume_n` and
`-volume_n`.  In a mixed `(+1,-1)` vector decomposition, the two signs of the
corresponding Clifford involution are Spin-conjugate.

## Claim boundary

Proved by the executable exact calculation:

- group perfectness, conjugacy classes, and the complete character table;
- real/quaternionic Schur type and center action of every irreducible;
- the complete tensor ring and defining-representation McKay graph;
- every real-module multiplicity type in the six requested dimensions;
- centralizer Lie dimensions and the exact `O`-to-`SO` splitting criterion;
- the component counts in the table above, conditional only on the explicitly
  separated standard universal-cover theorem.

Not proved or claimed:

- that characteristic-zero `H2=0` implies the mod-2 lifting theorem;
- triality outside Spin(8);
- the unrestricted Spin(8) Dirac--Gram inequality;
- any sequence-model quality, memory, or systems advantage.

## Replay

```powershell
python SSM-Models\spin_dirac_a5_components.py `
  --output SSM-Models\experiments\artifacts\spin_dirac_a5_components_20260816.json
python -m unittest discover -s SSM-Models `
  -p "test_spin_dirac_a5_components.py" -v
```

The exact replay completes in approximately six seconds.  All four focused
tests pass.

## Subsequent spinor gate

The component list is now the input to
[`SPIN_DIRAC_A5_SPINOR_BRANCHING_RESULTS.md`](SPIN_DIRAC_A5_SPINOR_BRANCHING_RESULTS.md),
which computes and independently verifies the spinor or half-spinor branching
for every one of the 245 orthogonal types.
