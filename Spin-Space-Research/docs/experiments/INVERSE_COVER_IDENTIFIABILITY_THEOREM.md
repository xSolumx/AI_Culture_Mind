# Inverse-cover identifiability: theorem, counterexample, and scope

Date: 2026-08-03.

## Definitions

Let `S` be a finite state set and `A` an even-sized token alphabet. Each token
`a` acts by a permutation `T_a` of `S`. An inverse-token structure is a
fixed-point-free involution `iota` on `A` satisfying

```text
T_iota(a) = inverse(T_a).
```

It induces the reverse-edge involution

```text
rho(s, a) = (T_a(s), iota(a)).
```

Every `rho` orbit has two directed edges. A reverse cover is an observation set
containing at least one member of every orbit.

## Proposition 1: completion with known inverse tokens

Given `iota` and a reverse cover, the entire transition family is uniquely
determined by inverse propagation.

**Proof.** For any unobserved `(s, a)`, its reverse-orbit partner is observed,
say `(u, iota(a)) -> s`. Invertibility forces `T_a(s) = u`. Applying this to
every missing member completes the family. No alternative value is compatible
with the observed reverse transition. QED.

The cover contains at least `|S||A|/2` edges. This is a lower bound for this
inverse-propagation argument, not for every possible learner supplied with
additional group relations or priors.

## Proposition 2: four unknown tokens need one calibration pair in the worst case

Assume four distinct tokens form two unknown inverse pairs and act regularly by
group multiplication. Assume no wrong token pair multiplies to identity. Given
a reverse cover plus both directions of one reverse orbit, the true inverse
matching is uniquely identified.

**Proof.** There are three perfect matchings of four tokens. The calibration
orbit supplies two observed directed compositions returning to their source,
so the true matching receives positive two-step-identity support. In a regular
group action, multiplication by a nonidentity element has no fixed point.
Every wrong token pair therefore has zero two-step-identity support. The true
matching is the unique maximizer. Once one pair is known, the two remaining
tokens must pair. Proposition 1 then completes the action. QED.

For the A5 harness this uses 120 cover edges plus one additional reverse
direction: **121/240 edges**. The full transition audit confirms 240 directed
two-step identities for the true matching and zero for each wrong matching.

## Why 120 is not universally sufficient when `iota` is unknown

The absence of a calibration edge cannot be replaced by a universal appeal to
global feasibility. `inverse_cover_adversarial_audit.py` reduces the choice of
one orientation per reverse orbit to 120 Boolean variables. For each candidate
matching it adds two kinds of 2-SAT clauses:

1. coverage: a transition or its candidate reverse must be observed;
2. consistency: two observations that would force different inverse values
   cannot both be selected.

Both wrong matchings have satisfying assignments. Under either assignment, the
true and wrong matchings both complete to permutation families, so the learner
refuses the mask as ambiguous. This falsifies universal exact-half
identifiability. It does not contradict 1,000/1,000 success under random
reverse-cover orientations; that is a distributional fact.

The exact 2-SAT algorithm is complete over the full `2^120` orientation space
for each of the only two wrong perfect matchings, so it decides the existence
question “does any adversarial mask exist for this matching?” It does so in
polynomial time using implication-graph strongly connected components, not by
brute-force enumeration. It returns one witness, not an enumeration or
classification of every satisfying mask shape.

Adding any one of the 120 possible true calibration directions resolves each
of the two adversarial witnesses: 240/240 checks recover the exact action. This
is exhaustive over calibration choice for those two witnesses only. Universal
121-edge safety under the stated assumptions follows from Proposition 2's
identity-support argument, not from treating 240 computational checks as an
enumeration of every adversarial orientation.

## What the theorem does not say

- The masking environment uses the true reverse-orbit structure to guarantee
  coverage; the learner does not receive the pairing.
- Dense exact prefix-state labels still create the source transition record.
- Randomly observing half the entries is insufficient: uniform masks at 120,
  121, and 122 edges recover 0/1,000.
- Self-inverse tokens, noisy aliases, more than two inverse pairs, nonregular
  actions, or endpoint-only supervision require different identifiability
  conditions.
- The number 121 is specific to this 60-state, four-token, two-inverse-pair
  regular action. Q8, D4, S3, other token alphabets, and groups lacking a real
  3D irrep require their own bounds and adversarial audits.
- This theorem concerns recovery of a finite action. It does not establish
  language-model quality or a general advantage of Clifford rotors over other
  parameterizations.
