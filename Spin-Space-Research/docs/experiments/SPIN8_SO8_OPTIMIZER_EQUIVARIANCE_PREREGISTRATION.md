# Spin(8)/SO(8) optimizer-equivariance audit: preregistration

Date frozen: 2026-08-03, before executing the audit.

## Causal question

The positive-half-spin and standard SO(8) generators are related by an exact
orthogonal 28x28 coefficient map `M`. Does an optimizer preserve this chart
equivalence during training?

For identical batches and mapped parameters `alpha = beta M`, the losses obey
`L_so8(alpha) = L_spin(beta)`. Their gradients transform covariantly. Plain SGD
with scalar learning rate is equivariant under the orthogonal map, so the mapped
parameters, actions, logits, and all non-action parameters should remain equal.

AdamW applies coordinatewise first/second-moment normalization. A dense
orthogonal rotation does not commute with that operation, so the equivalence is
predicted to break after the first adaptive update even though the initial
gradient is correctly covariant.

## Frozen protocol and gates

- Float64 CPU, two channels, Q8 endpoint loss, 12 identical L16 batches of 64.
- Learning rate `3e-3`, zero weight decay for both optimizers.
- SGD must keep mapped coefficients, action matrices, and post-update logits
  within `1e-10` throughout.
- AdamW must exceed `1e-4` in mapped-coefficient and post-update-logit error.
- AdamW's first pre-update gradient covariance error must remain below `1e-10`.

Passing establishes optimizer-induced chart bias. It does not establish that
either chart is universally easier to optimize.
