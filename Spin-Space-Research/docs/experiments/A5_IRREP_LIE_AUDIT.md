# A5 Irrep-Branch and Defect Lie-Closure Audit

## Questions

1. Do the two learned anchor-generator defects remain in one rotation plane, or
   do they generate a full-dimensional subgroup of SO(3)?
2. Did different seeds or non-anchor channels learn the two inequivalent real
   3D irreducible representations of A5?

This is a read-only audit of the ten deterministic saved checkpoints. No model
was retrained.

## Method

Both degree-three A5 characters are now supported by
`a5_orthogonal_irrep(group, branch=0|1)`. The two branches swap the golden-ratio
character values on the two five-cycle conjugacy classes.

For every seed and channel:

1. extract the 3x3 vector-grade rotation block for all four learned tokens;
2. align each exact irrep branch to the learned actions with one global SO(3)
   conjugation jointly fitted across all tokens;
3. compare branch alignment RMS and the trace of the learned five-cycle;
4. measure maximum commutator and cyclic/mixed-relator residuals;
5. for the anchor and its best branch, form per-generator residuals
   `D_s = L_s (Q O_s Q^T)^T`, take their principal SO(3) logs, and compute the
   Lie closure using the cross-product bracket of so(3).

The synthetic test suite verifies that the alignment is invariant to an
arbitrary global conjugation, distinguishes the two exact branches, and gives
rank three for two small nonparallel residual rotations.

## Irrep result

All ten anchors match branch 0:

- learned token-2 five-cycle trace: `1.613573-1.621233`;
- branch-0 exact trace: `phi = 1.618034`;
- branch-1 exact trace: `phi' = -0.618034`;
- branch-0 joint alignment RMS: `0.002244-0.005468`;
- branch-1 joint alignment RMS: `0.540088-0.540893`.

Exactly one channel per seed passes the predeclared representation-like screen
(alignment RMS at most 0.05, commutator at least 0.5, and mean relator RMS at
most 0.1), and it is always the previously identified anchor. No non-anchor
channel matches either irrep. The closest non-anchor branch-0 alignment is
`0.168111` with relator RMS about `0.69`; the closest branch-1 alignment is
`0.546515`.

There is therefore no evidence that an auxiliary channel learned the other A5
irrep. The stable decoder ensemble uses non-representational residual features,
not a direct sum of the two faithful 3D irreps.

## Crucial cap confound

The 10/10 branch agreement is not a free SGD choice:

- branch 0 maps the audited five-cycle to a 72-degree rotation,
  `1.256637` radians;
- branch 1 maps it to a 144-degree rotation, `2.513274` radians;
- every checkpoint uses `max_rotor_angle = 2.2` radians.

The convergence-stabilizing cap admits branch 0 and excludes an exact branch-1
token action. This architecture preselects the observed irrep. A future branch-
selection experiment must use a chart that represents both branches, for
example a cap at least `pi` together with an initialization/trust-region scheme
that does not reopen the earlier rotor convergence failure.

## Defect Lie closure

After aligning each anchor to branch 0, independent generator residual angles
are small but nonzero:

- order-3-token residual: approximately `0.0070-0.0172` radians;
- order-5-token residual: approximately `0.00032-0.00447` radians.

Their residual axes are nonparallel in every seed (axis dot products range
from about `-0.984` to `0.359`). In so(3), the bracket is the vector cross
product. Two resolved nonparallel log axes plus their bracket span all three
dimensions. The normalized Lie-closure rank is therefore `3/3` in all ten
seeds, not the earlier single-plane rank-1 case.

The appropriate mathematical picture is stronger than circle
equidistribution. The closure of a two-generator SO(3) subgroup is constrained
to finite/polyhedral, axis-preserving O(2)-type, or full SO(3)-dense cases. Two
small non-half-turn rotations about nonparallel axes are numerically outside
the axis-preserving exceptional form, so the measurements indicate the generic
full-SO(3) risk. See the topological subgroup classification in
[Ando (2025)](https://arxiv.org/abs/2507.20593) and the special finite-order
two-generator analysis in
[Radin and Sadun (1997)](https://arxiv.org/abs/math/9706203).

This is not presented as a formal density proof for the learned model:

- floating-point output cannot establish irrationality;
- the residuals use nearest proper rotations after numerical alignment;
- actual word error is a cocycle interleaving residuals with oracle actions,
  not an unconstrained product of two residual matrices.

It is nevertheless a decisive falsification of the one-fixed-plane mental
model. Generic long-word drift has three rotational degrees of freedom and can
approach decoder boundaries from many orientations.

## Consequence for rounding

Independent per-token angle snapping is not enough. A safe intervention must
jointly round the token actions onto one exact A5 representation, up to one
global conjugation, and then verify all Cayley relations. Exact joint rounding
makes every residual holonomy identity; merely shrinking two nonparallel
defects can extend the horizon while leaving the full-dimensional closure
mechanism intact.

The immediate controlled comparison should be:

1. the current learned anchor;
2. independently angle-rounded tokens;
3. jointly aligned-and-rounded exact branch-0 actions;
4. the same frozen soft decoder gate in all cases.

Require dense transfer, relation residuals, and decoder margins to be reported
together. The exact construction—not finite-length accuracy—is the only member
of this comparison with an infinite-horizon group-action guarantee.

This comparison is now complete; see `JOINT_A5_ROUNDING_RESULTS.md`. The
prediction was borne out sharply. Independent angle rounding passes through
L256 but fails three seeds at L4096 on the untouched changed-order-3 alphabet.
Joint exact projection passes all ten, with a `96.88%` population floor and
float32 homomorphism RMS below `2.4e-7`.

## Artifact

`a5_anchor_representation_audit_10seeds.json`
