# Partial-Cayley inverse-cover retraction: prospective contract

Date fixed: 2026-08-03, before the ten-seed GPU cohort.

## Question

Can the self-compiling rotor recurrence recover an exact shared A5 action
family when the compiler observes only a little more than half of the unique
state-token transitions?

This is a partial-supervision identifiability experiment, not arbitrary table
completion and not unsupervised group discovery. Prefix-state labels remain
available to the task loss. A masking environment first constructs a
reverse-edge cover from the complete deterministic transition record; the
compiler receives only the masked record.

## Structural protocol

The four input tokens form two unknown inverse pairs. For each of the 120
undirected reverse-edge pairs, the compiler observes exactly one direction.
For exactly one edge pair in each inverse-token family, it observes both
directions. Thus it receives 122 of 240 unique directed edges (50.8333%).

The compiler is **not** given the inverse-token pairing. It must:

1. enumerate perfect matchings of the four tokens;
2. reject any matching contradicted by observed two-step transitions;
3. identify the unique matching supported by the two bidirectional calibration
   pairs;
4. complete all hidden reverse edges using that one shared involution;
5. reconstruct the regular permutation action and its multiplication table;
6. discover a real 3D irrep from the regular representation;
7. retract all learned token actions jointly onto one shared conjugacy orbit.

The masking environment's reverse-edge cover is an explicit structural design
assumption. The claim will not be generalized to uniformly random missingness.

## Fixed controls

- Zero calibration: 120/240 edges. Expected to be refused as inverse pairing is
  underidentified.
- Equal-budget uniform random mask: 122/240 edges. Expected to be refused when
  entire reverse-edge pairs remain unseen.
- Supervision curve: 0, 1, 2, 3, and 6 bidirectional calibration pairs per
  inverse-token family, over 100 deterministic mask seeds.
- Gauge audit: six base states crossed with all 24 token closure orders. Every
  recovered table must be exactly isomorphic after quotienting the base-state
  coset; the representation compiler must remain exact for every closure order.

The CPU controls were executed before this document was written only to select
the minimum nonzero identifiable budget. Their artifact is
`partial_cayley_supervision_audit.json`. No GPU cohort result had been observed.

## Frozen training protocol

- Seeds: 0 through 9.
- Architecture: four-channel pure Cl(3) rotor recurrence.
- Training: 2,000 steps, batch 256, length 16, deterministic CUDA.
- Optimizer and losses: unchanged from the latent-Cayley cohort.
- Ambient gradients remain unconstrained until discovery. After discovery, the
  anchor token family is jointly retracted; tokens are never normalized or
  rounded independently.
- Inverse-cover calibration fraction: `1/60` per inverse-token family.
- Mask seed: `910001 + training_seed`.
- Untouched macro alphabet: changed-generator selection index 44,
  `(24315, 41325, 23514, 41253)`. It is disjoint from the training tokens.

## Evaluation and gates

Each seed must satisfy all of the following:

1. infer inverse tokens `(1, 0, 3, 2)` without receiving that mapping;
2. complete all 118 hidden directed edges and replay the complete transition
   action exactly;
3. generate a group of order 60 and compile a 3D candidate with invariance and
   homomorphism RMS below `1e-10`;
4. trigger shared-manifold retraction;
5. obtain at least 90% anchor-only accuracy at every dense length 16, 32, ...,
   256 on both the original and untouched index-44 alphabets;
6. obtain at least 90% at L4096 on both alphabets and at L16384 on the untouched
   alphabet;
7. retain the original strict raw-homomorphism diagnostic separately. Passing
   the exact compiled-manifold gate must not be described as making the
   pre-retraction learned ambient operators exact.

The cohort passes only if all ten seeds pass every item. Any changed threshold,
mask, generator index, or selected seed after results are visible is a new
experiment.

## Intended claim if it passes

The strongest permitted claim is:

> Under a reversible-edge-cover missingness design, two calibration edge pairs
> identify the shared inverse-token involution and reduce explicit transition
> supervision from 240 to 122 edges without degrading exact representation
> compilation or long-horizon recurrence.

This would be a sharp sample-complexity result for structured partial action
recovery. It would not show arbitrary partial Cayley completion, endpoint-only
learning, noisy-label robustness, or group discovery without dense state labels.
