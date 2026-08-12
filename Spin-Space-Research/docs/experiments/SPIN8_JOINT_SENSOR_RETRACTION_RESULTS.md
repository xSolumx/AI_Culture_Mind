# Spin(8) Joint Sensor Retraction Results

**Date:** 2026-08-03

**Preregistration:** `SPIN8_JOINT_SENSOR_RETRACTION_PREREGISTRATION.md`

**Untouched cohort:** seeds 20-29

**Raw artifact:** `spin8_joint_sensor_retraction_seeds20_29.json`

**Artifact SHA-256:** `4d8b9bf7da3ce22bce143f1e312e699158fae026963f961635d396710878ff07`

## Result in one sentence

Soft query-family continuation followed by joint hard retraction and continuous
polish reaches the balanced D-optimal Spin(8) sensor in 10/10 untouched seeds,
versus 6/10 for the fresh hard baseline, while the prospectively frozen exact
degree-28 spectral law replicates in 10/10; the strict joint-versus-independent
causal gate fails at 4/10 because independent argmax was already optimal in the
other six seeds.

## Frozen gate outcomes

| Gate | Outcome |
|---|---:|
| Retraction validity | **10/10 pass** |
| Conditioning reliability | **10/10 joint versus 6/10 hard: pass** |
| Strict joint-over-independent causal comparison | **4/10 fail** |
| Noisy recovery | **10/10 pass** |
| Single-query projector theorem | **pass** |
| Fresh exact-spectrum replication | **10/10 pass** |

The failed causal gate is retained. It demanded strict joint improvement in
8/10 seeds. Independent argmax from the soft checkpoint was already at the
global numerical optimum in six seeds, making strict improvement impossible.
Joint retraction repaired all four remaining independent failures and harmed
none. That `4/4` opportunity-conditioned repair is an informative post-hoc
decomposition, not a replacement preregistered gate.

## Why `trace(I)=35` is exact for every design

For one unit probe `x` in representation `r`, let

\[
P_{r,x}=J_{r,x}^{\mathsf T}J_{r,x}.
\]

In the maintained orthonormal bivector coordinates, `P_{r,x}` is an orthogonal
projector onto the seven tangent directions that move `x`; its 21-dimensional
kernel is the `Spin(7)` stabilizer algebra. Therefore

\[
P_{r,x}^2=P_{r,x},\qquad \operatorname{rank}P_{r,x}=7,
\qquad \operatorname{tr}P_{r,x}=7.
\]

Across 300 deterministic fresh probes in the three triality representations:

- maximum idempotence error: `4.44e-16`;
- rank range: exactly `7–7`;
- maximum trace error from seven: `3.55e-15`.

For any five-query design, optimal or not,

\[
\operatorname{tr}(I)=\sum_{k=1}^5\operatorname{tr}P_k=35.
\]

Thus the sensor problem is not acquiring more total information trace. It is
arranging five fixed-rank projector subspaces so that information is distributed
well across all 28 Lie-algebra directions.

## Prospectively replicated exact spectrum

The characteristic factorization discovered after seeds 10-19 was frozen
before seeds 20-29:

\[
\chi_I(\lambda)=\frac{1}{1024}
(\lambda-1)^4
(\lambda^2-3\lambda+1)
(2\lambda^2-6\lambda+3)^4
(2\lambda^2-4\lambda+1)^4
(2\lambda^3-8\lambda^2+6\lambda-1)^2,
\]

where adjacent factors are multiplied. Every new oracle optimum matches this
law:

- spectral pass count: `10/10`;
- maximum factor residual: `1.38e-13`;
- maximum relative characteristic-coefficient error: `1.31e-14`;
- maximum determinant error from `81/1024`: `1.36e-15`;
- maximum trace error from `35`: `1.42e-14`;
- maximum inverse-trace error from `43`: `3.56e-14`.

The polynomial makes the recurring invariants algebraically transparent:

\[
\det(I)=\frac{81}{1024},\qquad
\operatorname{tr}(I)=35,\qquad
\operatorname{tr}(I^{-1})=43.
\]

The projector argument proves the trace identity. The fresh polynomial
replication is extremely strong evidence for one exact balanced-spectrum law,
but a global inequality proving that no other physical five-query design has
larger determinant remains open.

## Joint continuation results

The soft optimizer used no balance, diversity, rank, or allocation target. The
joint retraction evaluated all 243 hard assignments in every seed.

| Family | Balanced allocations | Strict conditioning passes | Log-det range | `trace(I^-1)` range |
|---|---:|---:|---:|---:|
| fresh hard straight-through | 6/10 | 6/10 | `-2.773–-2.537` | `43.0–45.50` |
| soft independent argmax | 6/10 | 6/10 | `-3.466–-2.537` | `43.0–57.50` |
| soft joint retracted | 10/10 | 10/10 after polish | `-2.545–-2.537` | `43.0–43.034` |
| soft joint polished | 10/10 | **10/10** | `-2.537049–-2.537023` | `43.0–43.00015` |
| oracle | 10/10 | 10/10 | numerical constant `-2.537023` | numerical constant `43` |

Soft continuation alone solves six seeds. Joint family selection repairs the
remaining four by choosing a balanced assignment from the complete learned
vector bank. Continuous polish closes the small residual vector-geometry gap.
The total intervention improves strict reliability from 6/10 to 10/10, meeting
the frozen requirement by four seeds.

## Noisy held-out recovery

With matched endpoint noise `sigma=1e-3`:

| Sensor | Least-observed one-step cosine | Worst-representation L2,048 cosine |
|---|---:|---:|
| polished joint | `0.999993–0.999996` | `0.9526–0.9834` |
| oracle | `0.999993–0.999996` | `0.9614–0.9880` |
| random mixed | `0.999981–0.999993` | `0.9191–0.9792` |

The polished joint sensor beats matched random sensing in every seed:

- minimum one-step advantage: `2.13e-6`;
- minimum length-2,048 advantage: `0.00222`;
- maximum gap behind the oracle at length 2,048: `0.00899`.

All ten pass the frozen noisy-recovery gate.

## Numerical contracts

- maximum triality-equivariance error: `1.45e-15`;
- maximum parallel-prefix/recurrent error: `1.38e-14`;
- maximum absolute log-norm drift through length 2,048: `4.47e-13`;
- exactly 243 hard assignments evaluated per joint retraction;
- complete learned, retracted, polished, and oracle probe vectors are retained
  in the raw artifact.

## What is established

1. Every unit query contributes an exact rank-seven information projector, so
   total information trace is fixed at 35 for all five-query sensors.
2. The balanced oracle spectrum obeys one prospectively replicated exact
   degree-28 factorization across ten fresh optimizations.
3. Soft continuation plus joint family retraction removes all four discrete
   allocation failures observed in the fresh hard baseline.
4. The total intervention improves strict D-optimal reliability from 6/10 to
   10/10 without an explicit balance or diversity target.
5. Better conditioning transfers to noisy long-composition recovery.

## What is not established

- The preregistered strict joint-over-independent gate failed at 4/10 due to
  six ceiling cases.
- The exact characteristic polynomial is prospectively replicated numerical
  evidence, not yet a symbolic global-optimality proof.
- Exhaustive 243-way retraction does not scale directly to large query budgets.
- No language-model or semantic-memory claim follows from this experiment.

## Next mathematical target

Prove the balanced spectrum analytically. The natural route is to treat each
query as a rank-seven projector onto the orthogonal complement of a
`Spin(7)` stabilizer, derive the allowed principal-angle relations between
projectors in distinct triality views, and maximize `log det(sum P_k)` under
those intersection constraints. A successful proof should recover the exact
factor polynomial and show that `(2,2,1)` is globally D-optimal up to triality
permutation and Spin(8) gauge.
