# Exact A5/2.A5 tangent rigidity along the Spin ladder

**Status:** passed computer-assisted exact theorem for the fixed standard
icosahedral embedding at
`n = 3, 8, 9, 10, 11, 12`.

**Executed:** 2026-08-16 17:33:40 +02:00 (`Africa/Johannesburg`).

**Artifact:**
[`spin_dirac_a5_rigidity_20260816.json`](artifacts/spin_dirac_a5_rigidity_20260816.json),
SHA-256 `e5c742c29ad7b9de044df7efc7d3836852d93bcc37d293d8824d61de29a3da66`.

> **Degree-two follow-up.** Exact averaging on the complete 120-element group
> table now proves `H1=H2=0` for every characteristic-zero linear `2.A5`
> module. See
> [`SPIN_DIRAC_A5_COHOMOLOGY_RESULTS.md`](SPIN_DIRAC_A5_COHOMOLOGY_RESULTS.md).

## The theorem proved by the executable certificate

Fix the icosahedral representation in the first three vector coordinates and
let the remaining `n-3` coordinates be trivial. For each listed `n`, the
tangent kernel of the exact `(2,3,5)` relation map equals the image of
infinitesimal `SO(n)` conjugation. Consequently,

\[
H^1\!\left(A_5,\mathfrak{so}(n)_{\operatorname{Ad}\rho}\right)=0
\]

for this fixed embedding. Equivalently, it has no infinitesimal deformations
other than conjugacy.

The binary lift has the same tangent calculation. Its relations in `Spin(n)`
have central target `-1`, and the covering `Spin(n) -> SO(n)` is a local
isomorphism of Lie groups.

This is an infinitesimal-rigidity theorem. It is not a classification of all
`2.A5` embeddings, global uniqueness up to conjugacy, or an obstruction/derived
theorem.

## Exact group certificate

The quaternion generators are already defined over
`Q(sqrt(5))`; no floating-point approximation is required:

\[
a=(0,0,0,1),\qquad
b=\frac12(1,\varphi-1,0,-\varphi),\qquad
\varphi=\frac{1+\sqrt5}{2}.
\]

Exact Hamilton multiplication verifies

\[
a^2=b^3=(ab)^5=-1.
\]

Breadth-first closure contains exactly 120 quaternions. Quotienting by
`q ~ -q` leaves exactly 60 classes. The central `-1` is explicitly present,
and exact quaternion-to-vector projection reproduces the two matrices used by
the relation Jacobian. This promotes the previous float64 120/60 enumeration
to an exact algebraic certificate; the earlier artifact remains an independent
numerical cross-check.

## Rank certificate

Write

\[
J_n:\mathfrak{so}(n)^2\longrightarrow\mathfrak{so}(n)^3
\]

for the derivative of
`(A,B) -> (A^2, B^3, (AB)^5)` under right-trivialized perturbations, and

\[
C_n:\mathfrak{so}(n)\longrightarrow\mathfrak{so}(n)^2
\]

for infinitesimal simultaneous conjugation.

The program constructs both matrices over `Q(sqrt(5))` and checks
`J_n C_n = 0` exactly. The untouched lower block supplies an explicit
`so(n-3)` kernel of `C_n`. Three independent specializations of `sqrt(5)` in
finite prime fields then produce nonzero pivot minors at primes
`1000039`, `1000081`, and `1000099`.

| n | dim so(n) | dim centralizer | rank C | rank J | dim ker J | dim H1 |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 3 | 0 | 3 | 3 | 3 | 0 |
| 8 | 28 | 10 | 18 | 38 | 18 | 0 |
| 9 | 36 | 15 | 21 | 51 | 21 | 0 |
| 10 | 45 | 21 | 24 | 66 | 24 | 0 |
| 11 | 55 | 28 | 27 | 83 | 27 | 0 |
| 12 | 66 | 36 | 30 | 102 | 30 | 0 |

The proof is dimension-theoretic but exact:

1. `im(C_n) <= ker(J_n)` supplies the upper bound on `rank(J_n)`.
2. The explicit `so(n-3)` kernel supplies the upper bound on `rank(C_n)`.
3. Each recorded nonzero modular pivot minor supplies the matching lower rank
   over `Q(sqrt(5))`.
4. Therefore `dim im(C_n) = dim ker(J_n)`. Together with containment, the two
   spaces are equal.

The JSON artifact retains every pivot row, pivot column, field root, prime,
and nonzero minor determinant needed to replay the lower-bound certificates.

## Trust boundary

- Exact matrices use SymPy's algebraic field `QQ<sqrt(5)>`.
- Matrix identities and `J_n C_n = 0` are tested before specialization.
- Modular determinants are recomputed from the recorded pivot minors, not
  inferred from an optimizer or floating rank threshold.
- A nonzero specialized minor proves the source minor is nonzero; the three
  primes are redundant independent cross-checks.
- The theorem still depends on the correctness of Python, SymPy algebraic-field
  arithmetic, and the small local elimination implementation.

## What this changes

The algebraic-geometry bridge is no longer merely a numerical centralizer
observation. For the listed embeddings, the complete first-order quotient
tangent has now been closed exactly.

It does **not** upgrade the Spin(8) Dirac--Gram inequality, Spin(9)
D-optimality, or any SSM result. Those are separate programmes.

## Next exact gate — superseded twice

The requested exact low-degree contracting homotopy now passes, and `H2=0`.
It also proves that the positive raw three-relator cokernel consists of
presentation syzygy redundancy rather than group-cohomological obstruction.
The then-remaining global component gate was subsequently completed by
[`SPIN_DIRAC_A5_COMPONENT_ATLAS_RESULTS.md`](SPIN_DIRAC_A5_COMPONENT_ATLAS_RESULTS.md),
which enumerates all conjugacy components of

\[
\operatorname{Hom}(2.A_5,\operatorname{Spin}(n))/\operatorname{Spin}(n).
\]

The rigidity certificate itself still claims only the fixed component. The
later atlas retains its `so(n-3)` stabilizer and supplies the global
classification through dimension 12.

## Replay

Full exact replay:

```powershell
python SSM-Models\spin_dirac_a5_rigidity.py `
  --output SSM-Models\experiments\artifacts\spin_dirac_a5_rigidity_20260816.json
```

Focused fast replay:

```powershell
python SSM-Models\spin_dirac_a5_rigidity.py --dimensions 3 8
python -m unittest discover -s SSM-Models -p "test_spin_dirac_a5_rigidity.py" -v
```

The full exact run completed in approximately 82 seconds on the recorded local
Windows environment. The three focused tests pass.
