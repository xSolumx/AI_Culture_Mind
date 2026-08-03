# Spin(8) table-blind path compiler: prospective contract

Date fixed: 2026-08-03, after the table-aware path-section compiler passed
9/9 untouched seeds 10--18 and before any result from seed 19 or later was
inspected. Seed 10 may be used only as an implementation fixture. Seed 19 is
the first prospective smoke; seeds 20--28 are reserved for an unchanged
reliability cohort if the smoke passes.

## Question

Can a trained positive-chiral Spin(8) recurrence recover and retract its own
finite token-action algebra without receiving the Q8 multiplication table,
element names, token-to-element map, inverse pairs, identity label, target
labels, or a group-aware calibration sampler?

This is a table-blind, decoder-labeled compiler. It is not unsupervised
discovery: training used eight endpoint classes and the compiler is told that
there are eight anonymous states and four tokens.

## Frozen compiler inputs

The compiler receives only:

- the trained checkpoint;
- uniformly sampled token strings over the four-token alphabet;
- the model's own recurrent states and eight-way predictions;
- the fixed cardinalities, eight states and four tokens.

It must not import or consult Q8 constants until the post-compilation scoring
phase. Calibration uses 32 batches of 512 strings at each of L15 and L16.
The random-token sampler is independent of the hidden task generator.

## Frozen discovery and compilation

1. Group final recurrent states by the model's predicted anonymous class.
2. For every sampled state and each token, apply one recurrent step and decode
   the successor. Accumulate all `class x token -> successor class` votes.
3. Require all eight classes to occur, every edge to have at least 128 votes,
   winner fraction at least 0.99, and winner-minus-runner-up fraction at least
   0.98.
4. Require every recovered token action to be a permutation. Generate the
   permutation group from these actions and require it to be regular and have
   exactly eight elements. Choose anonymous class 0 as a gauge base; no
   semantic identity label is supplied.
5. Order the eight learned class centroids by the recovered regular action.
   Project each complete 8x8 orbit Gram matrix onto the commutant of the
   recovered regular representation. No rank threshold is used.
6. Conjugate the recovered token permutations into each channel, take the real
   positive-chiral Spin(8) logarithm, and jointly replace the complete token
   family. Tokens are never normalized independently.
7. Transport the frozen linear observer by the minimum-change section map.
   No target one-hot vectors and no post-compilation gradient steps are used.

## Frozen gates

An implementation or formal seed passes only if:

- all discovery gates above pass;
- maximum channel centroid projection RMS is at most 0.03;
- commutant residual is at most 1e-10;
- Spin(8) action reconstruction is at most 1e-5;
- recovered-table homomorphism RMS is at most 1e-5;
- exact section rank is eight;
- observer logit transport error is at most 1e-5;
- recurrent streaming state error is at most 1e-5 and logit error at most
  1e-4;
- post hoc only, the recovered table is isomorphic to Q8;
- post hoc dense L15--L256 and long L4095/L4096/L16383/L16384 central-pair
  evaluations both pass at 99% per checkpoint.

The frozen outcome taxonomy is: discovery rejection, accepted wrong group,
exact compilation failure, behavioral failure, or all-gate pass. A
post-hoc-isomorphic table is never fed back into compilation.

## Falsifiers and claim boundary

Persist the complete vote tensor and margins. As negative controls, permuting
successor labels independently per token must either fail regular closure or
recover a different structure, and collapsing two anonymous labels must fail
coverage/permutation closure. The main claim, if successful, is narrow:
network-predicted endpoint equivalence plus long-path transition observations
are sufficient to identify an exact finite action and compile it onto a shared
Spin(8) representation manifold. It is not a claim that state cardinality or
equivalence classes were inferred from raw language data.
