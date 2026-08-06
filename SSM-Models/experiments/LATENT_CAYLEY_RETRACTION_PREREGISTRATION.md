# Latent Cayley Recovery and Retraction Preregistration

## Status

Written on 2026-08-03 after the transition-recovery unit test and before any
table-blind model is trained or evaluated. Changed-generator classes 0, 1, 2,
11, and 22 have been observed previously. Class 33 is prospectively reserved
as the untouched primary alphabet. It introduces a fourth order-3 inverse
pair, `14352`, `15324`, combined with `23514`, `41253`.

## Gate being removed

The previous self-compiling result removed supplied irrep matrices, character
values, branch choice, alignment, and channel choice, but still gave the
compiler the A5 Cayley table and mapping from input tokens to group elements.

This experiment withholds both from every learning-side algebraic operation.
The true table may exist only in:

- the external synthetic data generator; and
- the final held-out evaluator.

It may not be passed to transition recovery, holonomy construction,
representation compilation, trigger selection, or post-trigger retraction.

## Table-blind recovery

The learner observes only token IDs and the prefix class labels already used
by cross-entropy. It accumulates deterministic triples

`(previous prefix label, token ID, next prefix label)`.

No numerical or permutation meaning is assigned to a label. Once all
state-token edges are observed:

1. each token column must be a permutation of the 60 label classes;
2. the four permutations are closed under composition;
3. closure must contain exactly 60 permutations and act regularly on every
   chosen base label;
4. an arbitrary base label fixes a gauge between state labels and recovered
   elements;
5. the multiplication table is reconstructed from permutation composition;
6. every observed edge must be reproduced exactly by the recovered table.

Conflicting edges, incomplete coverage, non-bijective token actions,
non-regular closure, or more/fewer than 60 generated permutations are hard
failures. Nothing is filled by the true table.

The recovered table then supplies the regular-representation compiler from the
previous experiment. Multi-scale holonomy targets are translated into the
recovered label gauge and use only this inferred table.

## Frozen training protocol

- model, seeds, training data, held-out bigram, optimizer, rotor cap, and
  2,000-step schedule match `SELF_COMPILING_RETRACTION_PREREGISTRATION.md`;
- four unconstrained rotor channels search before compilation;
- transition evidence is accumulated online from training batches only;
- regular-irrep candidates are constructed only after the inferred
  permutation action passes all closure checks;
- the same step-250-to-1500 compiler audit, `0.08` fit, `0.20` branch gap, and
  `0.50` commutator thresholds are retained;
- pre-trigger multi-scale holonomy at steps greater than 750 uses only the
  recovered table and recovered target gauge;
- post-trigger training uses the discovered anchor alone and applies one joint
  conjugacy retraction after every ambient optimizer step.

No restart, manual label gauge, channel choice, candidate choice, or table
repair is permitted.

## Evaluation

For all ten seeds, report:

- step of full edge coverage and minimum evidence count;
- recovered closure order, regularity, associativity, and edge replay error;
- compiler trigger step/channel/candidate and separation;
- float32 homomorphism and mixed-relator residuals;
- original and untouched class-33 dense L16-L256 accuracy;
- original and class-33 L4096 accuracy, batch 512; and
- untouched class-33 L16384 accuracy, batch 256, plus sequential-versus-direct
  canonical state drift.

## Claim gates

All must pass without changing the protocol:

1. conflict-free complete transition recovery and a 60-element regular group
   in all ten seeds;
2. automatic representation trigger by step 1500 in all ten seeds;
3. float32 homomorphism RMS at most `1e-5`, maximum at most `1e-4`, in all ten;
4. at least 90% at every original and class-33 dense length in all ten;
5. at least 90% on original and class-33 L4096 in all ten;
6. at least 90% on untouched class-33 L16384 in all ten; and
7. nonzero post-trigger ambient updates and tangent motion in all ten.

The experiment may be called table-blind because the compiler never receives
the true table. It may not be called unsupervised: prefix state labels provide
strong transition-equivalence supervision.
