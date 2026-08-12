# Joint A5 Rounding Preregistration

## Status

Written on 2026-08-03 after the irrep/Lie-closure audit and before evaluating
any checkpoint on generator-class index 11. Classes 0, 1, and 2 have already
been observed and are regression tests; class 11 is the untouched primary
distribution.

## Question

Can the dominant learned anchor be projected onto one exact A5 representation,
eliminating its rank-three defect closure while preserving the frozen soft
decoder ensemble and zero-shot changed-generator accuracy?

## Frozen contract

For every deterministic GA checkpoint:

- retain the learned initial state, three auxiliary channel actions, output
  head, bias, logit scale, and previously selected robust scalar gates;
- retain the fixed max-commutator anchor identity;
- fit no parameter using class-11 states or labels;
- align exact branch 0 to the four learned anchor token actions with one global
  SO(3) conjugation computed only from the actions themselves.

## Variants

1. `learned`: unchanged anchor actions.
2. `independent_angle`: preserve each learned rotation axis but replace its
   angle with the exact branch-0 angle for that labeled token. This enforces
   individual generator orders without enforcing mixed relations.
3. `joint_exact`: replace all four anchor actions by the single globally
   aligned exact branch-0 A5 anti-representation.

Every replacement is converted back through the Cl(3) rotor chart, so vector
and bivector grades remain one coherent rotor action. The same frozen soft gate
is used for all variants, alphabets, and lengths.

## Evaluation

- alphabets: original and changed-generator equivalence classes 0, 1, 2, and
  11;
- class 11 is the first repository-ordered class with a different order-3
  inverse pair: `13425`, `14235`, combined with order-5 pair `23514`, `41253`;
- dense lengths: every multiple of 16 from L16 through L256;
- two deterministic batches of 512 per length;
- long stress: L4096 on original and class 11, one batch of 512;
- report anchor vector-grade homomorphism RMS/max, single-generator relators,
  mixed relators, orthogonality, and dense/long accuracy.

## Gates

The primary class-11 behavioral gate is at least 90% accuracy at every dense
length and at L4096. The exact-mechanism gate for `joint_exact` is vector-grade
homomorphism RMS at most `1e-5` and maximum at most `1e-4` in float32 execution.

The result clears the claim gate only if `joint_exact` passes both gates in all ten
seeds without reducing any seed's class-11 dense floor by more than one
percentage point relative to `learned`. Independent angle rounding is expected
to pass cyclic relations but is not credited unless it also passes mixed and
homomorphism gates.

No hyperparameter is adjusted after class-11 evaluation.
