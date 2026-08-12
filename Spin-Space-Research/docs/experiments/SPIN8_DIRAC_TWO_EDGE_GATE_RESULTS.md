# Variable-Cayley Two-Edge Initial Gate Results

**Date:** 2026-08-06

**Preregistration:** `SPIN8_DIRAC_TWO_EDGE_PREREGISTRATION.md`

**Exact anchor artifact:**
`../../artifacts/spin8_dirac_two_edge_anchor_20260806.json`

**CUDA falsifier artifact:**
`../../artifacts/spin8_dirac_two_edge_attack_20260806.json`

**Exact degree artifact:**
`../../artifacts/spin8_dirac_two_edge_degree_20260806.json`

**Exact common-factor artifact:**
`../../artifacts/spin8_dirac_two_edge_amplitude_20260806.json`

## Result in one sentence

The clean second-residual bridge passed its first four frozen gates: exact
common-triality symmetry permits eight Walsh sectors, both disjoint rational
anchors activate all eight, a 249,152-sample plus 32-restart CUDA attack found
no violation, and 144 exact degree slices passed all 576 disjoint-node checks.
This licenses an amplitude-factor derivation; it does not prove the inequality.

## Exact sign quotient

For parameter order `(a,d,e,g,i,c)`, exact common adjoint conjugacies induce

\[
(t_1,t_2,t_1t_2,t_4,t_2t_4,t_3t_4).
\]

The induced group has eight elements. Its annihilator in the 64-character
Walsh group has exactly eight elements:

```text
000000  001101  010110  011011
100011  101110  110101  111000
```

This derivation was completed before determinant anchors were evaluated. It
proves that every other character vanishes globally on this family.

## Exact anchor reconnaissance

Two disjoint rational-circle anchors were each crossed with all 64 sign
patterns. Results:

- exact direct determinants: `128`;
- nonzero sectors at anchor 0: `8`;
- nonzero sectors at anchor 1: `8`;
- supports equal between anchors: yes;
- both supports equal the complete symmetry annihilator: yes;
- exact target margins at both anchors: strictly positive.

The artifact stores all direct rational determinants and a digest over the
rows. A test recomputes the common symmetry, rebuilds the stored support, and
directly reevaluates one displayed `28 x 28` determinant.

This does not prove that all eight sector amplitudes are generically nonzero,
but matching full support at two unrelated interior points is strong
reconnaissance for the degree audit.

## Frozen CUDA falsifier

The prospective numerical gate used the local RTX 2070 SUPER in float64:

- uniform interior samples: `200,000`;
- boundary-biased samples: `49,152` (`4,096` per signed coordinate face);
- Adam restarts: `32`;
- steps per restart: `1,000`;
- optimizer parameter cap: `0.98`.

Results:

| Search | Maximum log advantage |
|---|---:|
| Uniform random | `-0.0125707` |
| Boundary-biased | `-0.0126687` |
| Optimized | `8.04e-14` |

The optimized point returned to the known equality manifold: `a,d,g,i` were
approximately zero and the two existing residual-scale values were near
`5e-8`. The tiny positive value is below the frozen `1e-8` violation threshold
by more than five orders of magnitude and is consistent with float64 roundoff.

The PyTorch objective was also checked against a direct exact determinant at a
rational anchor and agreed to 13 decimal places. This closes the risk that the
GPU search was optimizing a subtly different quantity from the exact program.

## Exact degree and endpoint audit

The crash-safe audit evaluated `2,736` exact `28 x 28` determinants:

- three unrelated rational base points;
- six squared-coordinate axes `(a,d,e,g,i,c)`;
- eight symmetry-allowed Walsh sectors;
- fifteen interpolation nodes per slice;
- four disjoint confirmation nodes per slice.

All `144` univariate reconstructions passed all `576/576` confirmation checks.
The maximum observed degree was `10`, below the predeclared conservative bound
of `14`. Per-axis maxima were:

| Squared coordinate | `a^2` | `d^2` | `e^2` | `g^2` | `i^2` | `c^2` |
|---|---:|---:|---:|---:|---:|---:|
| Maximum observed degree | 6 | 6 | 6 | 6 | 5 | 10 |

For every nontrivial character, the audited quantity is the sector squared
after division by the square of the character-forced signed coordinates. The
large and repeatable multiplicities at coordinate value one expose candidate
complement factors for the next exact derivation. They are not yet called
global factors: three generic slices can discover a pattern, but only a
quotient-ring or exact coefficient argument can prove it over the whole cube.

A post-gate exact factor audit sharpened this observation. After removing each
slice's character-forced signed-coordinate square and its full endpoint factor,
all `126/126` nontrivial slices are exact polynomial squares. Their square-root
degrees are:

- `39` linear;
- `81` quadratic;
- `6` cubic.

This is a highly structured discovery, not a global theorem. It proves the
factorization on three unrelated base slices per axis with disjoint exact
confirmation nodes. A multivariate quotient-ring derivation or complete exact
coefficient reconstruction is still required before the seven sector
amplitudes may be declared globally polynomial.

The missing global polynomial ansatz is now supplied by an exact extension of
the sign audit. Instead of tracking only `(a,d,e,g,i,c)`, it tracks all twelve
circle coordinates `(a,A,d,D,e,E,g,G,i,I,c,s)`, while including the independent
sign gauge of each probe. The induced chart-sign group has order `512`; its
annihilator contains exactly eight characters. Each of the original eight
lower-coordinate characters has exactly one compatible complement character:

| Lower mask | Complement mask `(A,D,E,G,I,s)` |
|---|---|
| `000000` | `000000` |
| `001101` | `001110` |
| `010110` | `011100` |
| `011011` | `010010` |
| `100011` | `100010` |
| `101110` | `101100` |
| `110101` | `111110` |
| `111000` | `110000` |

Consequently every sector is globally its displayed forced monomial times a
polynomial in `(a^2,d^2,e^2,g^2,i^2,c^2)`. This proves the minimal radical
ansatz. It does not yet prove the extra even complement powers suggested by the
endpoint atlas, nor the signs of the final orientation margins.

The normalization is now formal on the complete bridge. Both branches of all
five diagonal boundaries `A,D,E,G,I=0` have exact Jacobian rank `25` and
nullity `3`. The rank-32 circle normal form and coprimality therefore prove

\[
A^6D^6E^6G^6I^6=\Delta^3
\]

divides every raw determinant coefficient. Combined with the universal
rank-seven Cauchy--Binet bound, this yields a conservative residual degree
certificate with per-coordinate degrees no larger than four. Reconstructing
the eight sectors separately at those bounds requires `61,321` grid points.
That is the frozen proof-safe ceiling; the smaller slice degrees are not being
silently substituted for it.

In plain language, the determinant could have hidden high-degree behavior
between the original samples. It did not: independently chosen rational test
points landed exactly on the same low-degree polynomials. The remaining hard
problem is no longer guessing the algebra's size; it is proving the common
factors and certifying positivity of the resulting eight-sector matrix.

## First global amplitude factor

The first candidate factor is now proved globally, rather than inferred from
the slice atlas. At the two Cayley boundary branches `c=+1` and `c=-1`, the
symbolic `40 x 28` observation Jacobian has exact rank `25` and nullity `3`.
Every maximal minor is consequently order at least three in the transverse
coordinate `s`, so Cauchy--Binet makes the information determinant order at
least six.

In the quotient ring

\[
\mathbb{Q}[c,s]/(c^2+s^2-1),
\]

every element has the unique form `F0(s)+c F1(s)`. Applying the rank argument
on both branches and adding/subtracting them proves that both coefficients are
divisible by `s^6`. Therefore every direct determinant, and hence every Walsh
sector, contains the exact common factor

\[
s^6=(1-c^2)^3.
\]

This cancels the complete cubic Cayley-boundary factor from the target, leaving
`(9-c^2)^2`. It also explains the slice atlas exactly: the trivial sector's
degree in `c^2` drops from 5 to 2, while the squared nontrivial sector measures
drop from degrees 8 or 10 to 2 or 4 after the common squared factor is removed.

## First full-sector reconstruction

Sector `110101`, selected prospectively because its proof-safe grid was the
smallest, has now passed a complete exact reconstruction:

- two disjoint `4^6=4096` rational grids;
- `65,536` direct determinants across those grids;
- identical 243-term coefficient maps;
- observed multidegree `(2,2,2,2,1,1)`;
- 32/32 fresh exact off-grid matches from another 256 determinants.

Exact polynomial division exposed the additional global factor

\[
1-a^2=A^2.
\]

After division, the core has 162 terms and multidegree `(1,2,2,2,1,1)`.
Together with the character-forced `aA`, the complete sector amplitude contains
`aA^3`, precisely matching the earlier endpoint slice pattern. This is the
first time that one of those extra slice factors has been promoted to a global
six-variable identity.

The quotient has a further exact boundary-ideal decomposition:

\[
Q=Q|_{i^2=c^2=0}+(1-d^2)(1-e^2)(1-g^2)R,
\]

where the base has 42 terms and `R` has 28 terms and is multilinear. Therefore
the late coordinates `i^2,c^2` couple into this sector only through the three
intermediate complement energies `D^2E^2G^2`. This is a promising structural
route to the remaining tensors, not a positivity result by itself.

## Scientific status

The family has passed Gates 1--4, its global amplitude theorem, and complete
reconstruction of all eight sectors under the separately frozen shared-grid
protocol. It has **not** passed global positivity. In particular:

- numerical non-violation is not a theorem;
- two anchor supports do not establish radical amplitude factorizations;
- two disjoint grids recover identical 6,664-coefficient maps and all 256 fresh
  exact sector holdouts pass;
- all slice-predicted endpoint factors are now exact global identities;
- every endpoint-reduced sector obeys an exact bridge to the proved one-edge
  family, and the orientation algebra splits into paired four-sector blocks;
- the unrestricted family still has the `h` residual beyond this bridge.

The next action is an exact equality-kernel compatibility falsifier for the odd
first-order block, followed—if it survives—by a paired group-circulant Schur or
boundary-adapted positivity certificate. See
[`SPIN8_DIRAC_TWO_EDGE_ALL_SECTOR_RESULTS.md`](SPIN8_DIRAC_TWO_EDGE_ALL_SECTOR_RESULTS.md).
