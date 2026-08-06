# Self-Compiling Joint Retraction Preregistration

## Status

Written on 2026-08-03 before training the self-compiling cohort or evaluating
any resulting model. Changed-generator classes 0, 1, 2, and 11 have been
observed in earlier experiments. Class 22 is prospectively reserved as the
untouched primary alphabet for this experiment.

Class 22 is selected structurally, not by model performance. It is the first
repository-ordered macro class after the previously observed families with a
third order-3 inverse pair: `13542`, `15243`, combined with order-5 pair
`23514`, `41253`.

## Hypothesis

A write-free rotor recurrence can search with independent unconstrained token
tangent updates, recognize when one channel has approached a faithful finite-
group action, compile the corresponding exact irrep directly from the group
table, and thereafter retract every ambient update of that token family
through one shared conjugation. This should combine SGD discovery with exact
mixed-relation closure and remove the oracle matrices used in the previous
joint-rounding intervention.

## Representation compiler

The compiler may use:

- the finite group multiplication table;
- the identities of the four training tokens as group elements;
- the learned 3x3 token actions; and
- a deterministic random seed fixed here.

It may not use:

- `a5_orthogonal_irrep` or any supplied icosahedral matrices;
- A5 character values, including `phi` or `phi'`;
- class-22 sequences, states, labels, margins, or accuracy; or
- post-hoc choice of an irrep branch.

The exact candidates are derived from the regular representation. A generic
symmetric element of the right-regular algebra commutes with the left-regular
action. Its dimension-three eigenspaces are therefore exact invariant copies.
Equivalent copies are deduplicated by their computed character vectors; the
learned action family selects the nearest candidate and one global SO(3)
conjugation.

## Training protocol

- family: `pure_ga_rotor`;
- seeds: 0 through 9;
- training alphabet: `23145`, `31245`, `23451`, `51234`;
- training split: the established held-out ordered bigram split;
- channels during search: 4;
- steps: 2,000, batch size 256, sequence length 16;
- AdamW learning rate `3e-3`, rotor cap `2.2` radians;
- before compilation: the established multi-scale holonomy objective begins at
  step 750 with weights `0.01`/`0.1`, ramp 500, scales 2/3/4/5;
- compiler audits: every 50 steps from step 250 through step 1,500;
- compile trigger: joint alignment RMS at most `0.08`, runner-up separation at
  least `0.20`, and maximum token commutator at least `0.50`;
- after compilation: task logits use only the compiled channel, holonomy loss
  is disabled, and training continues through step 2,000.

At every post-compile optimizer step, the four anchor token parameters first
receive ordinary independent ambient gradients. The resulting action family
is then jointly retracted onto the selected exact conjugacy orbit. One
least-squares tangent update determines one shared SO(3) conjugation for all
four tokens; no token is normalized or rounded independently. Auxiliary
channels receive no task gradient after compilation and are excluded from
evaluation.

If no channel satisfies the trigger by step 1,500, that seed is a discovery
failure and is not rescued manually.

## Evaluation

The compiled anchor alone is evaluated on:

- original training generators;
- untouched changed-generator class 22;
- every multiple of 16 from L16 through L256, two deterministic batches of
  512 per length; and
- L4096, one deterministic batch of 512 for both alphabets.

Report trigger step/channel/candidate, branch character only after selection,
pre-trigger fit and runner-up separation, retraction residuals, exact
homomorphism and mixed-relator residuals, state-norm error, and accuracy.

## Gates

The major claim is supported only if:

1. all ten seeds trigger without manual branch or channel selection;
2. all ten compiled models have vector homomorphism RMS at most `1e-5` and
   maximum at most `1e-4` in float32;
3. every seed scores at least 90% at every dense class-22 length and at L4096;
4. the original-alphabet L4096 gate also passes all ten seeds; and
5. saved traces confirm that post-trigger ambient updates are nonzero while
   every retracted family remains exact.

No threshold, compiler seed, training hyperparameter, or evaluation batch is
changed after the first cohort result is read.
