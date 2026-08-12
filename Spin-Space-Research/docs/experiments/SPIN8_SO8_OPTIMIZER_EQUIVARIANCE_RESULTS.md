# Spin(8)/SO(8) optimizer-equivariance audit: results

Date completed: 2026-08-03.

All prospectively frozen checks passed.

## Exact chart equivalence

The positive-half-spin and standard skew-generator bases are connected by a
28x28 orthogonal coefficient map with determinant `1.0000000000000007`. The
basis-change orthogonality and generator-reconstruction errors are exactly zero;
five random exponentials agree to `4.44e-16`.

Thus a single positive-chiral 8D stream and a generic dense SO(8) exponential
have the same transition family. No capacity interpretation can separate them.

## Optimizer intervention

Two identically initialized models received identical Q8 endpoint batches. The
generic coefficients were maintained in the mapped chart `alpha = beta M`.

| Maximum across 12 updates | SGD | AdamW |
|---|---:|---:|
| Gradient covariance error | 1.67e-16 | 1.50e-1 |
| Mapped coefficient error | 1.73e-18 | 3.75e-2 |
| Action-matrix error | 3.47e-18 | 3.77e-2 |
| Post-update logit error | 1.11e-16 | 1.52e-1 |
| Non-action parameter error | 1.08e-19 | 4.31e-2 |

AdamW's first gradient covariance error was `3.47e-17`, proving that the models
and coefficient map began correctly aligned. Divergence appeared only when the
coordinatewise adaptive update was applied.

## Finding

Plain SGD is equivariant under the exact orthogonal chart change; AdamW is not.
Therefore any single-stream positive-Spin(8) versus generic-SO(8) training gap
under AdamW is an optimizer-coordinate effect, not a difference in expressible
recurrent transitions. The direction of that bias may still be empirically
useful, but it cannot establish a uniquely spinorial mechanism.

The remaining distinctive Spin(8) hypothesis is coupled triality or a task that
observes differing global center kernels—not the isolated 28-DOF local chart.

Artifacts:

- `spin8_so8_chart_equivalence.json`
- `spin8_so8_optimizer_equivariance.json`
