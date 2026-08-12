# Positive-half-spin versus generic SO(8): preregistration

Date frozen: 2026-08-03, before training the generic SO(8) baseline.

## Algebraic prediction

The 28 positive-half-spin generators and the 28 standard elementary skew
matrices are orthogonal bases of the same vector space `so(8)`. The coefficient
change between them must therefore be an orthogonal 28x28 matrix `M` satisfying

```text
G_positive[a] = sum_b M[a,b] G_standard[b].
```

Consequently, for every positive-chart coefficient vector `beta`, the generic
SO(8) coefficients `beta @ M` produce exactly the same tangent and action
matrix. The two single-representation transition families have equal capacity.

## Empirical question

Run `pure_spin8_positive` and `pure_so8_exponential` with identical recurrent
state width, 28 action coordinates per token/channel, initial function, data,
and training budget. Any difference under AdamW is an optimization-coordinate
effect, not a difference in representable transitions. AdamW is not invariant
under a dense orthogonal rotation of parameter coordinates, so exact seedwise
trajectory equality is not predicted.

## Frozen paired cohort

- Fresh seeds: `60, 61, 62, 63, 64`.
- Families: `pure_spin8_positive` and `pure_so8_exponential`.
- Training: the unchanged 2,000-step Q8 endpoint curriculum, four channels,
  batch size 256, AdamW at `3e-3`.
- Evaluation: the unchanged dense central-pair lengths
  `15,16,31,32,63,64,127,128,255,256`, raw homomorphism diagnostics, streaming
  equivalence, elapsed time, and paired per-seed optimization trajectories.
- Both charts must have identical parameter counts, identical initial model
  function under the same seed, and 28 action coordinates per token/channel.

Interpretation is fixed before results:

1. Equal performance supports the exact chart-equivalence result.
2. A difference is an AdamW coordinate/conditioning result, not a capacity
   result, because the action families are already proved identical.
3. No five-seed outcome licenses a uniquely spinorial-mechanism claim.

## Claim boundary

A positive-spin result may be distinctive through its center/kernel semantics
or through a coupled vector/positive/negative triality construction. It must not
be attributed to a smaller or richer 8D transition family than generic SO(8).
