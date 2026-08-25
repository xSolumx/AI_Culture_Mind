# Program 08: Spin--Dirac branching and finite subgroup lifts

**Research author:** Hayden Austin

## Scope

Complex Clifford spinor modules along
`Spin(3) -> Spin(8) -> Spin(9) -> Spin(10) -> Spin(11) -> Spin(12)`, together
with one fixed icosahedral `A5 < SO(3)` action and its binary lift
`2.A5 < Spin(3)`.

This programme is separated from Spin(8) triality, Spin(9) sensing, and rotor
SSM training.  It asks which representation, branching, center, and subgroup
properties actually survive the dimensional ladder.

## Core Questions

1. Which fixed `2.A5` lifts and spinor branches occur from Spin(3) through
   Spin(12)?
2. Which cohomology and compact-real component statements are exact for the
   finite representation problem?
3. What additional manifold, bundle, connection, and operator data would be
   required before the finite census becomes a geometric Dirac problem?

## Proven / Established Results

The executable ladder starts from the maintained octonionic Spin(8) gamma
system and extends it by chirality and graded Pauli doubling.  Exact
Gaussian-integer matrix checks pass through Spin(12).  The abstract A5
permutation presentation generates 60 elements; the fixed quaternion lift
numerically generates 120 elements, has 60 projective classes, and acts with
the correct central sign on every spinor module.

Restriction to the embedded Spin(3) is an isotypic sum of the two-dimensional
spinor.  In even dimensions each Weyl half is checked separately.  Triality's
8D vector/8D/8D coincidence occurs only at Spin(8).

The follow-up exact rigidity gate promotes the binary-group order,
projective quotient, centralizer, and tangent-rank calculations from float64
to `Q(sqrt(5))`. For every listed rung, the exact relation kernel equals the
infinitesimal conjugacy image, so the fixed embedding has `H1 = 0` and no
first-order deformation modulo conjugacy.

The complete exact 120-element table now also carries an explicit low-degree
averaging contraction. Universal coefficient cancellation proves `H1=H2=0`
for every `Q(sqrt(5))`-linear `2.A5` module. Thus the fixed embeddings are
formally rigid and have no primary group-cohomological obstruction, while
retaining their `so(n-3)` stabilizers.

The global component atlas reconstructs all nine irreducible complex
characters, their five real-type and four quaternionic-type real modules, and
the affine-`E8` McKay graph directly from the same exact table. Combining exact
multiplicity enumeration with the standard universal-cover theorem for `2.A5`
classifies all Spin-conjugacy components in the six requested dimensions. The
component counts are `3, 32, 32, 42, 59, 98`; dimensions 8 and 12 contain 7 and
14 additional oriented components caused by `O`-to-`SO` splitting.

The exact spinor atlas now branches the Spin(n) spinor or half-spinors over all
245 orthogonal types. Quaternionic blocks are derived from signed SU(2) weight
sums, and every result passes independent exterior-algebra Clifford identities.
All 21 orientation-split types exchange distinct chiral characters. The fixed
`3 + trivial` ladder is an exact isotypic sum of the defining binary spinor and
contains no invariant spinors.

Canonical evidence:

- [`spin_dirac_a5_ladder.py`](../../SSM-Models/spin_dirac_a5_ladder.py)
- [`test_spin_dirac_a5_ladder.py`](../../SSM-Models/test_spin_dirac_a5_ladder.py)
- [`SPIN_DIRAC_A5_LADDER_RESULTS.md`](../../SSM-Models/experiments/SPIN_DIRAC_A5_LADDER_RESULTS.md)
- [`spin_dirac_a5_ladder_20260816.json`](../../SSM-Models/experiments/artifacts/spin_dirac_a5_ladder_20260816.json)
- [`spin_dirac_a5_rigidity.py`](../../SSM-Models/spin_dirac_a5_rigidity.py)
- [`test_spin_dirac_a5_rigidity.py`](../../SSM-Models/test_spin_dirac_a5_rigidity.py)
- [`SPIN_DIRAC_A5_RIGIDITY_RESULTS.md`](../../SSM-Models/experiments/SPIN_DIRAC_A5_RIGIDITY_RESULTS.md)
- [`spin_dirac_a5_rigidity_20260816.json`](../../SSM-Models/experiments/artifacts/spin_dirac_a5_rigidity_20260816.json)
- [`spin_dirac_a5_cohomology.py`](../../SSM-Models/spin_dirac_a5_cohomology.py)
- [`test_spin_dirac_a5_cohomology.py`](../../SSM-Models/test_spin_dirac_a5_cohomology.py)
- [`SPIN_DIRAC_A5_COHOMOLOGY_RESULTS.md`](../../SSM-Models/experiments/SPIN_DIRAC_A5_COHOMOLOGY_RESULTS.md)
- [`spin_dirac_a5_cohomology_20260816.json`](../../SSM-Models/experiments/artifacts/spin_dirac_a5_cohomology_20260816.json)
- [`spin_dirac_a5_components.py`](../../SSM-Models/spin_dirac_a5_components.py)
- [`test_spin_dirac_a5_components.py`](../../SSM-Models/test_spin_dirac_a5_components.py)
- [`SPIN_DIRAC_A5_COMPONENT_ATLAS_RESULTS.md`](../../SSM-Models/experiments/SPIN_DIRAC_A5_COMPONENT_ATLAS_RESULTS.md)
- [`spin_dirac_a5_components_20260816.json`](../../SSM-Models/experiments/artifacts/spin_dirac_a5_components_20260816.json)
- [`spin_dirac_a5_spinors.py`](../../SSM-Models/spin_dirac_a5_spinors.py)
- [`test_spin_dirac_a5_spinors.py`](../../SSM-Models/test_spin_dirac_a5_spinors.py)
- [`SPIN_DIRAC_A5_SPINOR_BRANCHING_RESULTS.md`](../../SSM-Models/experiments/SPIN_DIRAC_A5_SPINOR_BRANCHING_RESULTS.md)
- [`spin_dirac_a5_spinors_20260816.json`](../../SSM-Models/experiments/artifacts/spin_dirac_a5_spinors_20260816.json)
- [`benchmark_pure_rotor_2a5.py`](../../SSM-Models/benchmark_pure_rotor_2a5.py)
- [`PURE_ROTOR_2A5_CENTER_PROTOCOL.md`](../../SSM-Models/experiments/PURE_ROTOR_2A5_CENTER_PROTOCOL.md)
- [`PURE_ROTOR_2A5_CENTER_PILOT300_RESULTS.md`](../../SSM-Models/experiments/PURE_ROTOR_2A5_CENTER_PILOT300_RESULTS.md)
- [`pure_rotor_2a5_center_pilot300.json`](../../SSM-Models/experiments/artifacts/pure_rotor_2a5_center_pilot300.json)
- [`PURE_ROTOR_2A5_UNSEEN_RELATION_RESULTS.md`](../../SSM-Models/experiments/PURE_ROTOR_2A5_UNSEEN_RELATION_RESULTS.md)
- [`pure_rotor_2a5_unseen_relation_exploratory.json`](../../SSM-Models/experiments/artifacts/pure_rotor_2a5_unseen_relation_exploratory.json)
- [`benchmark_spin_multirelation_2a5.py`](../../SSM-Models/benchmark_spin_multirelation_2a5.py)
- [`SPIN_2A5_MULTIRELATION_PROTOCOL.md`](../../SSM-Models/experiments/SPIN_2A5_MULTIRELATION_PROTOCOL.md)
- [`SPIN_2A5_MULTIRELATION_RESULTS.md`](../../SSM-Models/experiments/SPIN_2A5_MULTIRELATION_RESULTS.md)
- [`spin_2a5_multirelation_pilot300.json`](../../SSM-Models/experiments/artifacts/spin_2a5_multirelation_pilot300.json)

## Open Claims

The local cohomology, global compact-real component, and finite-group spinor
branching gates are closed through `n=12`. A geometric Dirac theorem now needs
new input that is not present in a finite representation: a specific spin
manifold or orbifold, bundle, connection, operator, and boundary conditions.
Only after those are fixed can the invariant-spinor census be promoted to a
kernel, spectrum, eta-invariant, or index calculation.

### ML gate and remaining frontier

The first binary-center-sensitive prefix gate is complete. A frozen
parameter-near three-seed screen compares exact and projective oracles, Pure
Rotor v2.1, its identity ablation, an explicit Spin quaternion product scan,
and Transformers Mamba-2. Training excludes `a,a`; paired evaluation contrasts
`a,a=z` with `b,b^-1=e` inside shared contexts. The Spin scan alone passes the
registered central-margin gate in every seed and length, retaining 100% direct
target-versus-central-partner preference through L128. Its mean exact L128
accuracy is 62.20%; the other learned candidates are at 1.89--4.56% and near
chance on the center metrics.

This closes one finite-group mechanism screen, not the full ML programme. The
same frozen checkpoints also pass a deterministic length-11 relation pair that
is absent from all realized training schedules: Spin retains 100% center margin
through L128 and 59.68% mean exact L128 accuracy. That follow-up is exploratory
because its selector was designed after the pilot.

The preregistered successor now performs the stronger external-model and
relation-family falsifier. It withholds all three central presentation words,
adds DeltaProduct plus an exact regular PD ceiling, and repeats identical token
schedules under three inner-conjugate generator coordinates. Spin passes all
18 early-L64/L128 center splits with a 99.50% minimum margin and uniquely wins
long exact accuracy in every split. The learned alternatives fail the center
gate. Because this is seed 0 only, the next ML gate is fresh-seed validation of
the unchanged architecture, followed by state-/compute-matched sweeps and only
then a preregistered real-sequence hybrid.

## Dependencies

- Programme 01 supplies exact finite-group and representation tooling.
- Programme 04 owns triality statements at the Spin(8) stage only.
- Programme 06 owns any separately trained model experiment using these
  representations.

## Nonclaims

- No triality is claimed outside Spin(8).
- This programme does not prove the unrestricted Spin(8) Dirac--Gram
  inequality or global D-optimality.
- The algebraic gates themselves are not trained-model results; the separate
  `2.A5` pilot is an empirical benchmark with its own scope.
- A larger spinor dimension is not evidence of better memory or efficiency.
