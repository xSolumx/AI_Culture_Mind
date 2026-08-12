# Spin(8) blind shared-action gate preregistration

Date frozen: 2026-08-03, before implementation or results.
Baseline commit: 315db25.

## Pre-execution mask correction

The first implementation smoke, before any training result, found that the
original four-column mask has structural Jacobian rank 25 for every sampled
teacher. Resampling therefore cannot satisfy the frozen rank-28 validity
condition. A rank sweep found that five columns in the vector and
positive-chiral representations generically have rank 28 while still hiding
three columns and the complete negative-chiral action.

The executable protocol is corrected prospectively from e0--e3 to e0--e4.
The failed four-column proposal remains recorded here rather than silently
disappearing. No variant accuracy or long-composition result was observed
before this correction.

A subsequent seed-0 implementation smoke found that one-shot polar projection,
although faithful to the requested independent-normalization control, moves
the visible columns away from their fitted endpoints. Before the reliability
run, a stronger independently optimized Lie control is therefore added. It
uses a separate 28-coordinate SO(8) exponential for every token and every
representation, fits the same visible endpoints, and shares no coordinates
across representations. This addition makes failure harder to attribute to
the projection algorithm. The smoke is not part of the reliability cohort.

## Question

Can a jointly retracted token-action family recover the unobserved chiral
representation and long-composition dynamics from incomplete endpoint
observations, without receiving the teacher action matrices or bivector
coefficients?

## Teacher and observation boundary

Each seed samples four noncommuting Spin(8) token actions from hidden
28-dimensional bivector coefficients. The learner receives only:

- each token acting on basis vectors e0 through e4;
- only the vector and positive-chiral outputs;
- no negative-chiral endpoint labels;
- no teacher matrices, coefficients, logarithms, or unobserved columns.

The numerical Jacobian from one token's 28 coefficients to these 64 observed
coordinates must have rank 28. Seeds failing this design audit are invalid and
must be resampled before training, not discarded after seeing evaluation.

## Variants

1. exact supplied-action oracle;
2. unconstrained 8 by 8 token matrices fitted to the observed columns;
3. independent polar projection of every unconstrained token/representation
   matrix;
4. independent Lie retraction with separate token/representation tangents;
5. joint diagonal-triality retraction: all tokens are optimized together on
   the shared manifold

[
{(\rho_V(g),\rho_+(g),\rho_-(g)):g\in\mathrm{Spin}(8)},
]

using one 28-coordinate tangent per token and the same coordinate vector in
all three fixed representations.

The joint retraction may consume only the fitted vector/positive observed
columns. It may not inspect hidden teacher quantities. No token is normalized
or projected independently in that row.

## Optimization

- unconstrained fit: Adam, 500 steps;
- joint retraction: Adam, 1500 steps followed by a deterministic LBFGS
  refinement;
- seeds 0 through 9;
- float64 retraction and evaluation;
- local GPU when available.

## Evaluation

- observed-column MSE;
- random one-step states in all three representations;
- random held-out compositions of lengths 8, 32, 128, 512, and 2048;
- state-direction cosine and accumulated log-norm drift;
- triality-equivariance residual on every token;
- vector-action commutator magnitude and separation from the oracle;
- logarithmic-depth parallel scan versus recurrent execution.

Composition evaluation may renormalize the diagnostic state after each step
while accumulating its removed log norm. This prevents overflow without
concealing norm instability.

## Frozen gates

A joint-retraction seed passes only if:

- observed-column MSE is below 1e-8;
- mean cosine is at least 0.9999 for every representation in the one-step
  random-state test;
- mean cosine is at least 0.99 for every representation and every held-out
  composition length, including 2048;
- maximum absolute accumulated log-norm drift is below 1e-5;
- triality-equivariance maximum error is below 1e-8;
- parallel/recurrent maximum error is below 1e-10;
- its vector commutator magnitude is at least 90% of the oracle magnitude.

The reliability gate is at least 8 of 10 passing seeds.

Strong support additionally requires:

- the unconstrained and independent-Lie controls fit the observed columns
  below 1e-6;
- joint retraction beats independent Lie retraction on the completely
  unobserved negative-chiral representation in every seed;
- joint retraction beats both controls at length 2048 in at least 8 of 10
  seeds.

## Interpretation boundary

Passing establishes blind completion of a shared continuous action family from
incomplete endpoints. The Spin(8) generator algebra and triality embedding are
still architectural priors. This is not discovery of Spin(8) from raw language,
learned addressing, or a downstream sequence-model result.
