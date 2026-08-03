# Table-blind finite-action compiler: family baselines

Date fixed: 2026-08-03, after the Spin(8) table-blind seed-19 smoke and
seeds-20--28 cohort completed, and before a table-blind quaternion or
Householder checkpoint was compiled.

## Question

Which part of the new result belongs to endpoint-algebra recovery, and which
part belongs specifically to the positive-chiral Spin(8) chart?

The same anonymous L15/L16 calibration corpus, transition-vote thresholds,
regular-group recovery, observer transport, dense evaluation, and long
evaluation are applied to:

- `pure_quaternion_spinor`, using one exact recovered four-dimensional real
  irrep shared across its two quaternion blocks;
- `pure_householder4_shared`, using the same recovered four-dimensional irrep
  but an unrestricted learned O(4) gauge factored into four reflections.

No supplied Q8 table, inverse pairing, or target labels may enter either
compiler. Q8 isomorphism and behavioral evaluation remain post hoc only.

## Family-specific exact retraction

For quaternion spinors, inverse pairs are inferred from the recovered token
permutations. The two learned antipodal axis pairs are jointly orthogonalized;
their four targets are then one exact quaternion frame. No token is normalized
independently.

For shared Householder actions, the unique recovered real 4D irrep is extracted
from the recovered regular table. One simultaneous orthogonal intertwiner per
channel aligns the complete learned token family; each exact SO(4) action is
then factored into four Householder reflections. The candidate representation
is never taken from Q8 constants.

Both compilers keep the learned initial state, build its exact recovered-group
orbit, and transport the observer by the same minimum-change section map.

## Prospective correction after the seed-0 implementation fixtures

The first draft incorrectly required exact orbit rank eight and exact transport
of all eight teacher-logit directions. Both seed-0 fixtures exposed the error
before any reliability run: a cyclic orbit in repeated copies of Q8's faithful
real quaternionic irrep has theoretical rank four. Tiny action-factorization
errors can numerically inflate that rank, but they do not create four new exact
representation directions. The original rank-eight gate is impossible for the
intended baseline and is withdrawn rather than treated as a family failure.

For the formal baseline compiler, effective rank is measured at a fixed 1e-5
singular-value tolerance and must equal four. Teacher logits are orthogonally
projected onto the realizable row space of the exact rank-four endpoint orbit.
The minimum-change observer must reproduce those projected logits to 1e-5 and
their anonymous endpoint argmax must remain 100% correct. Discarded teacher
logit RMS, minimum projected margin, and nonzero-spectrum condition number are
reported. This is the unique least-squares realizable observer target; it uses
no hidden labels or table.

The seed-0 results produced with the impossible rank-eight transport remain
preserved as invalidated implementation artifacts. They are not baseline
evidence and are not overwritten.

After the corrected seed-0 fixtures passed, seeds 29--37 were reserved as the
unchanged fresh reliability cohort for both families. Neither their training
nor compiler results had been generated when this assignment was recorded.
The formal reliability gate is at least 8/9 all-gate passes per family; 9/9 is
reported separately. No family-specific threshold changes are allowed.

## Gates and interpretation

The Spin(8) discovery thresholds and post-hoc behavioral gates are unchanged.
Additionally, family action reconstruction and recovered-table homomorphism
RMS must each be at most 1e-5, effective exact orbit rank must be four, the
projected observer target must retain all anonymous class decisions, and
observer transport error against that realizable target must be at most 1e-5.

Existing seed-0 checkpoints are implementation fixtures, not reliability
evidence. If both compile, a later fresh cohort is required before comparing
reliability. A pass by all families means table-blind algebra discovery is
architecture-agnostic on this task; it does not erase Spin(8)'s separate value
as an 8D irreducible chart or its prospective triality coupling. A failure is
classified as discovery, representability/factorization, observer transport,
or behavior rather than reported as a generic family loss.
