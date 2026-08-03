# Generic exact-half recovery and the adversarial boundary

Date: 2026-08-03.

## Result

Under uniformly sampled reverse-cover orientations, the latent-action compiler
exactly reconstructs all 240 directed transitions of the four-token A5 action
from **120 observed transitions (50%)**, with no supplied inverse-token map and
no bidirectional calibration edge: 1,000/1,000 sampled masks pass.

This is not universal over all reverse-cover orientations. An exact 2-SAT
search constructs masks for each wrong inverse-token matching under which that
matching is also globally feasible. The learner safely refuses these masks as
ambiguous. Therefore 120 edges is a generic-case result under the sampled mask
distribution, not a worst-case identifiability boundary.

This was discovered after the 122-edge GPU cohort in
`PARTIAL_CAYLEY_RETRACTION_PREREGISTRATION.md` had started. It is reported as a
separate strengthening; the active cohort's fixed protocol was not changed.

## Why exact half is sufficient

Let `T_a` be the permutation assigned to token `a`, and let `iota` be the
unknown fixed-point-free involution satisfying

```text
T_iota(a)(T_a(s)) = s.
```

The masking protocol retains exactly one member of every reverse-edge pair

```text
(s, a) <-> (T_a(s), iota(a)).
```

For four tokens there are only three perfect matchings. The learner enumerates
them, propagates every observed edge through the candidate inverse relation,
and retains a matching only when the jointly completed family is
contradiction-free, complete, and permutation-valued. Across 1,000 random masks,
the true matching is the unique feasible completion. The inverse map
`(1, 0, 3, 2)` is therefore an output, not an input in those runs. The
adversarial audit demonstrates that feasibility alone need not make that
output unique for every orientation.

This is stronger than scoring isolated two-step identity examples. It uses the
global compatibility of all token actions, which is precisely the principle
behind joint representation-manifold retraction.

## Audit

Artifact: `partial_cayley_supervision_audit.json`.

| Mask | Edges | Exact recovery |
|---|---:|---:|
| Reverse-edge cover minus one observed orbit | 119/240 | 0/1,000 |
| Reverse-edge cover, no calibration | 120/240 | 1,000/1,000 |
| Adversarial 2-SAT reverse cover | 120/240 | refused as ambiguous |
| Reverse-edge cover, one calibration pair | 121/240 | 1,000/1,000 |
| Reverse-edge cover, two calibration pairs | 122/240 | 1,000/1,000 |
| Uniform random | 120/240 | 0/1,000 |
| Uniform random | 121/240 | 0/1,000 |
| Uniform random | 122/240 | 0/1,000 |

Even when the random masks are granted the true inverse pairing for diagnostic
purposes, 120 observed edges leave 42--76 directed edges missing after inverse
propagation (mean 59.648). The comparison therefore isolates **coverage of every
reverse orbit**, not raw edge count.

The completed action was also reconstructed under six base states and all 24
token closure orders. All 144 gauges pass exact post-hoc isomorphism after
base-coset normalization; all 24 compiler orderings recover both real 3D
irreps with maximum invariance RMS `9.89e-15` and maximum homomorphism RMS
`8.09e-16`.

Artifact `inverse_cover_adversarial_audit.json` records the 120-variable 2-SAT
construction. Both wrong perfect matchings admit a complete-permutation mask,
both have zero two-step identity support in the full action, and both are
refused safely. Revealing one true reverse pair adds one directed edge and a
strict two-step-identity score: the true matching scores positively, both wrong
matchings remain at zero, and the final unpaired tokens are forced. Thus
**121/240 is the worst-case-safe threshold for this matching-based protocol**.
All 120 possible choices of the single calibration pair resolve each of the two
adversarial masks (240/240 exact completion checks).

The exact 2-SAT feasibility decision is complete over the full `2^120`
orientation space for each of the only two wrong matchings, but solves that
decision in polynomial time by implication-graph strongly connected
components; it does not brute-force-enumerate `2^120` masks. It returns one
witness rather than all adversarial mask shapes. Correspondingly, the 240 repair checks are exhaustive
over calibration choice for those witnesses, not over every satisfying mask.
Worst-case 121-edge safety follows from the identity-support proof in
`INVERSE_COVER_IDENTIFIABILITY_THEOREM.md`.

## Precise scope

The random-mask experiment attains the inverse-propagation cover count, and the
sampled 119-edge ablation fails 0/1,000. It does not prove universal uniqueness
at 120; the adversarial audit explicitly falsifies that statement. Nor is 121
a general information-theoretic lower bound once additional group relations or
priors are allowed. These are conditional results for the stated
matching-and-cover protocol, not arbitrary missing-edge completion.
The mask constructor uses the true reversible pairing to guarantee coverage,
while the learner receives neither that pairing nor any hidden transition.
Dense prefix-state labels are still used by the surrounding task training.
The number 121 is specific to this A5 regular action and its four-token,
two-inverse-pair alphabet; it must not be ported to other groups without a new
identifiability analysis.

The next information-reduction gate is therefore no longer “remove table
entries.” It is to remove dense prefix labels themselves: recover word
equivalence and the shared action from endpoint-only or noisy observations.
