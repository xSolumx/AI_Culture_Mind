# Exact low-degree 2.A5 cochain contraction

**Status:** passed computer-assisted exact theorem over `Q(sqrt(5))`.

**Executed:** 2026-08-16 17:41:32 +02:00 (`Africa/Johannesburg`).

**Artifact:**
[`spin_dirac_a5_cohomology_20260816.json`](artifacts/spin_dirac_a5_cohomology_20260816.json),
SHA-256 `c96315dd1338c66da5f58e4e0edf70b60ae4b85e9873dc741d76cad77b24db81`.

## Result

Let `G = 2.A5` be the exact 120-element binary icosahedral group constructed
over `Q(sqrt(5))`, and let `V` be any linear `G`-module over that field. The
inhomogeneous group-cochain complex has

\[
H^1(G,V)=H^2(G,V)=0.
\]

The result therefore applies to every adjoint module
`so(n)_Ad rho` in the Spin ladder. The fixed embeddings have neither
first-order deformations modulo conjugacy nor degree-two group-cohomological
obstructions.

This does not remove their stabilizers. The degree-zero invariant algebra is
the previously certified `so(n-3)` centralizer.

## Exact contraction

For positive degree, use averaging over the final group variable:

\[
(h_n f)(g_1,\ldots,g_{n-1})
=\frac{(-1)^n}{|G|}\sum_{k\in G}
f(g_1,\ldots,g_{n-1},k).
\]

The executable verifies the two identities needed for deformation and primary
obstruction theory:

\[
d h_1+h_2d=1\quad\text{on }C^1(G,V),
\]

\[
d h_2+h_3d=1\quad\text{on }C^2(G,V).
\]

The verification is universal in the module action: coefficients multiplying
the formal operators `rho(g)` and the identity are collected separately. It
does not test a few chosen cocycles.

## Finite certificate

The program:

1. reconstructs all 120 exact quaternions in deterministic BFS order;
2. builds the complete `120 x 120` multiplication table;
3. checks identity, inverses, both Latin-square directions, and all
   `120^3 = 1,728,000` associativity instances;
4. checks all `120^2 = 14,400` exact quaternion-to-vector homomorphism pairs;
5. verifies the degree-one contraction at all 120 output group elements;
6. verifies the degree-two contraction at all 14,400 output pairs.

Both contraction gates leave zero unexpected formal terms. The surviving
identity coefficient is exactly 120 before division by the group order.

The exact multiplication-table SHA-256 is
`ee26aff5719e54cf28eccc7a0259c79eeb27c7e06bb18134bf7ca910773f58c9`.

## Why the presentation cokernel is not H2

The earlier three-relator Jacobian

\[
(A,B)\longmapsto(A^2,B^3,(AB)^5)
\]

has a positive-dimensional cokernel on every rung. The exact cochain result
proves that this cokernel cannot be labelled `H2`: genuine group-cohomological
`H2` is zero. The extra directions record dependencies and higher syzygies in
the raw presentation complex. A presentation-only derived calculation must
include those identities among relators before interpreting its cohomology.

This distinction is enforced by
[`test_spin_dirac_a5_cohomology.py`](../test_spin_dirac_a5_cohomology.py).

## Claim boundary

Proved by the executable certificate:

- the exact 120-element table and exact vector action;
- the universal degree-one and degree-two contracting identities;
- `H1=H2=0` for every `Q(sqrt(5))`-linear `2.A5` module;
- formal rigidity and absence of primary group-cohomological obstructions for
  the fixed Spin-ladder embeddings.

Not proved by this cohomology certificate alone:

- classification of all global representation components;
- global uniqueness of the standard embedding up to Spin conjugacy;
- absence of the `so(n-3)` stabilizer;
- an equivalence between a truncated raw presentation scheme and the full
  derived mapping stack;
- any ML or SSM advantage.

## Next gate — completed by the component atlas

The local deformation/obstruction problem is closed. The subsequent
[`SPIN_DIRAC_A5_COMPONENT_ATLAS_RESULTS.md`](SPIN_DIRAC_A5_COMPONENT_ATLAS_RESULTS.md)
classifies the conjugacy components of

\[
\operatorname{Hom}(2.A_5,\operatorname{Spin}(n))/\operatorname{Spin}(n)
\]

through `n=12` by the proposed route: exact real irreducibles, a separately
stated universal-cover lifting theorem, exhaustive orthogonal sums,
centralizers, and central signatures. The standard `3 + trivial` embedding is
one faithful rigid component among several, not a globally unique embedding.

## Replay

```powershell
python SSM-Models\spin_dirac_a5_cohomology.py `
  --output SSM-Models\experiments\artifacts\spin_dirac_a5_cohomology_20260816.json
python -m unittest discover -s SSM-Models `
  -p "test_spin_dirac_a5_cohomology.py" -v
```

The exact replay completed in approximately 12 seconds. All three focused
tests pass.
