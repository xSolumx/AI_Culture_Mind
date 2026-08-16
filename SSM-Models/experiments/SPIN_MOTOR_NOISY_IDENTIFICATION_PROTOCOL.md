# Spin-Motor Noisy Identification Protocol

Protocol frozen: **2026-08-16T20:27:39+02:00**
Exact 3x3 replication artifact SHA-256:
`97ffc994889278b21da7482ff49a597d9799f4ac38472e43b459465468a00aa5`

## Question

Does local motor identification remain useful when every supervised training
prefix pose is independently corrupted, or was the exact 3x3 result only an
exact-arithmetic lookup effect?

## Frozen data and noise model

Use the unchanged coordinate-`e`, schedule-seed-0 legal training inputs:

- 300 batches, batch size 16, sequence length 16;
- zero occurrences of `a^2`, `b^3`, and `(ab)^5`;
- 76,800 prefix observations;
- clean evaluation on the same 18 held-out-relation splits through L128.

For each true signed prefix pose `(q,t)`, generate

`q_noisy = q exp(0.5 theta u)`

with an independent isotropic axis `u` and zero-mean Gaussian angle `theta`,
and

`t_noisy = t + epsilon`,

with independent isotropic Gaussian translation noise. Four tiers are fixed:

| tier | rotation standard deviation | per-axis translation standard deviation |
|---|---:|---:|
| clean | 0 degrees | 0 |
| low | 1 degree | 0.01 |
| medium | 5 degrees | 0.05 |
| high | 15 degrees | 0.15 |

Run noise seeds 0--4 for every tier, for 20 identified models. All use the same
unmodified per-token arithmetic mean estimator; there is no tier-specific
tuning or robust loss. Save and hash every 49-parameter checkpoint.

## Critical sign boundary

Noise is applied to the **signed** quaternion and never introduces an
independent antipodal flip. This is intentional and must remain explicit.
Physical `SO(3)` observations alone cannot identify which binary lift generated
a central relation: independently replacing `q` by `-q` destroys exactly the
bit measured by this benchmark. The existing quotient oracle establishes that
boundary. This experiment tests metric noise conditional on signed-lift
supervision, not recovery of an unobserved central sign.

## Frozen gates

Every run must preserve the split/training audits and finite metrics.

The clean, low, and medium tiers pass only if every one of their 15 runs has:

- every long split signed-hemisphere accuracy at least 90%;
- every long split joint signed-pose accuracy at least 80%; and
- every split paired double-cover pose accuracy at least 80%.

The high tier is a preregistered stress measurement, not a required success
gate. Report its complete worst-run metrics without dropping failures.

Expected outcome: averaging thousands of repeated token observations should
make low and medium noise pass. Failure at low noise would show that differencing
adjacent noisy poses is too unstable without a joint estimator. Success does
not establish robustness to correlated drift, outliers, sign flips, continuous
inputs, or sparse/final-only supervision.
