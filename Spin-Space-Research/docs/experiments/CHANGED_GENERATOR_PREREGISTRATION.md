# Changed-Generator Transfer Preregistration

## Question

Does a frozen learned action transfer from its original two A5 generators and
their inverses to a genuinely different generating pair, without fitting new
token operators or changing the decoder?

## Construction fixed before evaluation

1. Exclude the four original input elements.
2. Enumerate A5 elements in repository order.
3. Select the first order-3 element and first order-5 element whose generated
   subgroup is all 60 elements and whose inverses produce four distinct macro
   tokens.
4. Compile each new element and inverse into its canonical BFS word over the
   original four tokens.
5. Compose the frozen learned token matrices along those words to obtain four
   macro-actions. No parameter is trained or edited.
6. Generate new random sequences directly over the macro alphabet and evaluate
   L2/L4/L8 plus every multiple of 16 through L256 with the frozen decoder.

The algorithmic first-valid-pair rule prevents choosing a favorable pair after
looking at results. Macro compilation is the only coherent meaning of an
unseen token without learning a new operator: a token with neither a learned
nor compiled action is undefined.

## Models

- GA seed 1: uniformly clean causal-irrep control;
- GA seed 6: nuisance-channel dense failure;
- Householder seed 0: clean holonomy control;
- Householder seed 3: holonomy-rescued drift case.

For multi-channel models, report both the full decoder and the strongest
noncommutative channel alone. The latter is a diagnostic fixed by the original
generator commutator, not selected on changed-generator accuracy.

## Gate

A model passes changed-generator transfer only if final-position accuracy is at
least 90% at every dense L16-L256 length. L2/L4/L8 and means are descriptive.
Passing named checkpoints while failing an interior length is failure. Exact
streaming parity and unchanged checkpoint hashes are mandatory sanity checks.

This is a zero-shot transfer test over compiled group elements. It does not
claim that the model can infer an operator for an arbitrary new symbol from its
name.

## Prospective extension: second macro alphabet and subset audit

This extension was written on 2026-08-03 after seeing the first-pair four-seed
results and before evaluating any checkpoint on the second pair. It is not
retroactively part of the original preregistration.

The first-pair audit showed that channel effects can reverse between the
original and changed generator distributions. One macro alphabet is therefore
insufficient to label a channel robust or harmful. The extension fixes the
following procedure:

1. Treat a generator and its inverse as one equivalence class, so swapping a
   macro token with its inverse does not create a nominally new generator set.
2. Enumerate disjoint order-3/order-5 A5 generating-pair equivalence classes in
   repository order.
3. Retain the original first class as index 0 and select index 1 as the second
   macro alphabet. No model output participates in selection.
4. Evaluate all ten deterministic GA checkpoints without retraining at every
   multiple of 16 from L16 through L256.
5. For all 15 non-empty channel subsets, report the dense minimum and, relative
   to the one fixed max-commutator anchor channel, per-length repair rate
   (anchor wrong, subset correct) and damage rate (anchor correct, subset
   wrong).
6. Keep the 90%-at-every-length gate unchanged and report each macro alphabet
   separately. Do not choose the better alphabet or subset as the headline.

The subset attaining the highest evaluated dense minimum is an oracle
diagnostic because it uses evaluation labels. It may motivate a later training
objective, but it is not a deployable selection rule. `Error-correcting` is
reserved for a later claim only if repairs materially exceed damages and a
follow-up margin/drift analysis shows systematic compensation rather than a
generic decoder shift.
