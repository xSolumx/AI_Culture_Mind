# The Eight-Sector Amplitude Theorem

> **Subsequent result (2026-08-07).** The amplitude theorem remains the exact
> reconstruction foundation. Positivity on the complete frozen `h=0` family
> is now certified by the
> [two-edge triangular atlas](experiments/SPIN8_DIRAC_TWO_EDGE_ATLAS_RESULTS.md).

## Scope

This note proves the global algebraic form of the eight orientation sectors in
the preregistered `h=0` variable-Cayley two-edge family. It does not prove that
the final orientation margins are nonnegative.

Use the circle coordinates

\[
(a,A),(d,D),(e,E),(g,G),(i,I),(c,s),
\qquad x^2+X^2=1,
\]

and the frame

\[
\begin{aligned}
x_1&=e_0,\\
x_2&=a e_0+A e_1,\\
x_3&=d e_0+D(e e_1+E e_2),\\
x_4&=g e_0+G(i e_2+I(c e_3+s e_4)).
\end{aligned}
\]

## Theorem 1: the complete chart-sign quotient

Common diagonal triality actions and the independent sign gauge of each probe
induce a group of order `512` on the twelve chart coordinates

```text
(a,A,d,D,e,E,g,G,i,I,c,s).
```

The annihilator of this group has order eight. Projecting each character onto
the six lower coordinates gives the eight previously proved Walsh sectors, and
each lower character has exactly one compatible complement character:

| Lower `(a,d,e,g,i,c)` | Complement `(A,D,E,G,I,s)` | Forced monomial |
|---|---|---|
| `000000` | `000000` | `1` |
| `001101` | `001110` | `e E g G I c` |
| `010110` | `011100` | `d D E g G i` |
| `011011` | `010010` | `d D e i I c` |
| `100011` | `100010` | `a A i I c` |
| `101110` | `101100` | `a A e E g G i` |
| `110101` | `111110` | `a A d D E g G I c` |
| `111000` | `110000` | `a A d D e` |

### Proof

For each exact common triality sign action, allow an independent overall sign
on `x2`, `x3`, and `x4`; projectors cannot see those three signs. Equating the
resulting component signs in the displayed frame gives all `512` induced chart
actions. Exhausting the `2^12` characters leaves exactly the eight rows above.

The computation is exact over signs, not sampled determinant evidence. The
maintained common-adjoint-conjugacy certificate proves that every induced
action preserves the information determinant.

After its forced monomial is divided out, a sector is even in every circle
coordinate. The six relations `X^2=1-x^2` therefore reduce what remains to an
ordinary polynomial

\[
H_m(a^2,d^2,e^2,g^2,i^2,c^2).
\]

## Theorem 2: the common Cayley-boundary factor

Every direct normalized determinant, and therefore every one of the eight
Walsh sectors, is divisible by

\[
s^6=(1-c^2)^3.
\]

### Proof

At `s=0`, both circle branches `c=+1` and `c=-1` give an exact symbolic
`40 x 28` observation Jacobian of rank `25` and nullity `3`. Under a transverse
perturbation in `s`, every maximal minor is consequently order at least three.
Cauchy--Binet writes the information determinant as a sum of squares of those
minors, so it is order at least six.

In

\[
\mathbb Q[c,s]/(c^2+s^2-1),
\]

every element has unique form `F0(s)+c F1(s)`. Applying the order-six result on
both branches, then adding and subtracting, proves that `s^6` divides both
coefficients. This is quotient-ring divisibility, not a floating-point boundary
limit.

## Final global ansatz

For every allowed character `m`, the exact sector has the form

\[
S_m=s^6 M_m H_m(a^2,d^2,e^2,g^2,i^2,c^2),
\]

where `M_m` is the forced monomial in the table and `H_m` is an ordinary
rational polynomial.

## Exact normalization and degree ceiling

The apparent normalization by `Delta^3` is also proved legitimate on this
larger family. At both branches of each boundary

\[
A=0,\ D=0,\ E=0,\ G=0,\ I=0,
\]

the symbolic observation Jacobian has rank `25` and nullity `3`. Applying the
same both-branch normal-form argument independently to the five coprime
complement variables proves

\[
A^6D^6E^6G^6I^6=\Delta^3
\]

divides the raw determinant.

Each varying query block has universal rank at most seven because all 84
maintained generators are exactly skew-symmetric. Cauchy--Binet therefore
bounds raw degree in any state-coordinate pair by `14`; after its sixth-power
diagonal division the first five chart pairs have degree at most `8`.
Removing the forced character and using evenness gives the following
conservative multidegree ceilings for the residual `H_m`:

| Lower mask | Upper bound in `(a2,d2,e2,g2,i2,c2)` |
|---|---|
| `000000` | `(4,4,4,4,4,4)` |
| `001101` | `(4,4,3,3,3,3)` |
| `010110` | `(4,3,3,3,3,4)` |
| `011011` | `(4,3,3,4,3,3)` |
| `100011` | `(3,4,4,4,3,3)` |
| `101110` | `(3,4,3,3,3,4)` |
| `110101` | `(3,3,3,3,3,3)` |
| `111000` | `(3,3,3,4,4,4)` |

Separate tensor reconstruction at these proof-safe bounds would require
`61,321` grid points. That is a valid mathematical count but not the efficient
experiment: the same eight direct determinants recover all eight sectors at
each point. A common `5^6=15,625` grid covers the componentwise maximum bound
and reduces the one-grid determinant work from `490,568` to `125,000`. The
shared-grid protocol was frozen before those evaluations. The smaller degrees
seen in the slice atlas remain discovery evidence until their higher
coefficients are globally annihilated.

The first complete reconstruction annihilated those coefficients for sector
`110101`. Two disjoint `4^6` exact grids produced the same 243-term polynomial
with observed multidegree `(2,2,2,2,1,1)`, and 32 fresh exact holdouts matched.
Exact division then proved

\[
H_{110101}=(1-a^2)Q_{110101},
\]

where `Q_110101` has 162 terms and multidegree `(1,2,2,2,1,1)`. Since this
sector's forced monomial already contains `aA`, its full amplitude contains
`aA^3`. This promotes one endpoint multiplicity from slice evidence to a
global sector theorem; it does not promote analogous factors in the other
seven sectors without their own exact argument.

The complete shared-grid campaign has now promoted that milestone to all eight
sectors. Two disjoint `5^6` grids required 250,000 direct determinants and
recovered identical complete maps containing 6,664 nonzero rational
coefficients. Another 256 direct determinants supplied 256 fresh exact sector
equalities. Every endpoint factor predicted by the slice atlas divides its
full six-variable sector polynomial exactly.

There is also a universal second compression. After each sector's exact
endpoint factors are removed, all dependence beyond its `i^2=c^2=0` face lies
in the ideal generated by

\[
(1-d^2)(1-e^2)(1-g^2)=D^2E^2G^2.
\]

For `110101`, the correction has only 28 terms and is multilinear in all six
squared coordinates. More generally, the stronger exact identity

\[
Q_m=Q_m|_{i^2=0}+i^2(1-d^2)(1-e^2)(1-g^2)T_m
\]

holds for every sector. Its `i^2=0` base is precisely the already-proved
variable-Cayley one-edge family.

Finally, the four `i`-even and four `i`-odd characters share the same exact
order-four Hadamard table. Pairing orientations that differ only in the sign
of `i` gives

\[
\lambda_{r,+}=W_r(E+O),\qquad
\lambda_{r,-}=W_r(E-O).
\]

Thus the final target is positivity of two commuting `4 x 4`
group-circulants deforming the proved one-edge matrix, not eight unrelated
sector inequalities.

The same quotient has exact closed transverse faces

\[
Q|_{d^2=1}=3(a^2-1)(g^2-1)^2,
\qquad
Q|_{g^2=1}=3(a^2-1)(d^2-1)^2(e^2-1)(3e^2+1).
\]

Their signs are opposite on the unit cube. Thus no proof strategy should ask
the individual Walsh sectors to be nonnegative; only the eight orientation
margins—the character sums of those sectors—have that target property.

Two independent evidence layers now agree with this theorem:

- both rational anchors activate all eight sectors;
- all `126/126` nontrivial exact one-dimensional slices become perfect
  polynomial squares after the corresponding squared amplitude is removed,
  with all `576/576` disjoint interpolation checks passing.

The slice square result is still described as slice evidence; the global
polynomial ansatz comes from the exact sign quotient, not from interpolation.

## Why this matters

Before this result, a full reconstruction appeared to involve radicals and 64
independent orientation cases. It now involves eight low-degree polynomials in
six squared variables, with one universal boundary factor already cancelled.
This is the algebraic compression needed before a crash-safe coefficient or
positivity certificate is sensible.

## Plain-language version

The determinant seemed to change in dozens of unrelated ways when signs were
flipped. In fact, 512 exact renamings say that only eight sign patterns can
carry information. Each pattern comes with a fixed collection of square-root
factors. Once those known factors are peeled away, there is no radical left—
only a small polynomial in six numbers between zero and one.

## Remaining gates

- check exact compatibility of the odd first-order block with every equality
  kernel of the proved one-edge matrix;
- certify positive semidefiniteness of both paired `4 x 4` group-circulants;
- only then promote the two-edge inequality.

The complete reconstruction is documented in
[`SPIN8_DIRAC_TWO_EDGE_ALL_SECTOR_RESULTS.md`](experiments/SPIN8_DIRAC_TWO_EDGE_ALL_SECTOR_RESULTS.md).
