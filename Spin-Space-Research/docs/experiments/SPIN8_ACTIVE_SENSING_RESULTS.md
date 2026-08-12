# Spin(8) Active Triality Sensing Results

**Date:** 2026-08-03
**Preregistration:** `SPIN8_ACTIVE_SENSING_PREREGISTRATION.md`
**Untouched cohort:** sensor seeds 10-19
**Raw artifact:** `spin8_active_sensing_seeds10_19.json`
**Artifact SHA-256:** `0b5f168e53beeff3f3c66007c6c4f3fe62f798bafb408deb7c56ebe522d9a388`

## Result in one sentence

The local information geometry of five Spin(8) triality queries is exactly
independent of the unknown action, an exhaustive numerical oracle selects a
balanced `(2,2,1)` query allocation with invariant information volume, and a
hard learned selector discovers an identifying rank-28 sensor in 10/10 seeds
but reaches the strict D-optimal basin in only 6/10; despite that optimization
failure, it beats matched random sensing under noise in all ten seeds and
passes the frozen practical recovery gate in 9/10.

## Frozen gate outcomes

| Gate | Outcome |
|---|---:|
| Action-independent information | **10/10 pass** |
| Learned strict near-oracle design | **6/10 fail** |
| Noisy held-out recovery | **9/10 pass** |
| Structural controls | **10/10 pass** |

The design failure and recovery pass answer different questions. Every learned
sensor is identifying, but four hard selectors settle into suboptimal discrete
allocations. Their conditioning is still sufficient to outperform the matched
random sensor on the frozen noisy task.

## The local adaptivity theorem

For a query `(r,x)` and unknown action `Q_r`, a right-invariant perturbation has
Jacobian

\[
J_{Q}[:,a]=Q_rG_{r,a}x.
\]

Orthogonality gives

\[
J_Q^{\mathsf T}J_Q
=J_I^{\mathsf T}Q_r^{\mathsf T}Q_rJ_I
=J_I^{\mathsf T}J_I.
\]

Across every teacher, token, and sensor variant, the measured maximum
information-matrix difference was `2.22e-15`; the maximum spectral difference
was `7.44e-15`. Thus a policy that adapts its next query to observed endpoints
cannot improve the local Fisher information objective over the best universal
design. This does not rule out global adaptive benefits away from the local
chart.

## The balanced triality sensor

The exhaustive oracle optimized all 21 five-query representation allocations
with four deterministic continuous restarts per allocation. In all ten seeds,
the optimum used a permutation of `(2,2,1)`:

- information rank: `28`;
- `log det(I)`: `-2.537022650927`;
- `det(I)`: numerically `0.0791015625 = 81/1024`;
- `trace(I)`: numerically `35`;
- `trace(I^-1)`: numerically `43`.

The same spectrum and invariants recur across independently optimized probe
frames. This is strong numerical evidence for an algebraic balanced-sensor
optimum, but `det(I)=81/1024` and global optimality remain a conjectural exact
formula until an analytic inequality proves them.

The relative orientations matter. Random within-view orthonormal pairs do not
generally achieve this spectrum; the optimizer discovers cross-triality
alignment in addition to the balanced allocation.

## Learned sensor optimization

The hard-forward selector received no diversity, rank, or mixed-view penalty.
Every seed nevertheless selected at least two views and achieved rank 28.

| Seed | Learned allocation | Log-det gap to oracle | `trace(I^-1)` ratio | Strict design pass |
|---:|---:|---:|---:|---:|
| 10 | `(1,2,2)` | `1.36e-7` | `1.00000002` | yes |
| 11 | `(1,3,1)` | `0.18375` | `1.0561` | no |
| 12 | `(2,1,2)` | `9.19e-8` | `1.00000001` | yes |
| 13 | `(4,1,0)` | `0.92883` | `1.3372` | no |
| 14 | `(1,1,3)` | `0.18233` | `1.0558` | no |
| 15 | `(2,2,1)` | `1.28e-6` | `1.00000017` | yes |
| 16 | `(2,2,1)` | `1.82e-4` | `1.000025` | yes |
| 17 | `(1,2,2)` | `2.41e-10` | `1.00000000` | yes |
| 18 | `(2,0,3)` | `0.23557` | `1.0581` | no |
| 19 | `(2,2,1)` | `3.62e-9` | `1.00000000` | yes |

This is a discrete allocation-basin failure, not an identifiability failure.
The straight-through hard selector learns “use multiple triality views” more
reliably than it learns the globally balanced experiment.

## Noisy held-out action recovery

Each frozen sensor observed five endpoints with matched isotropic noise
`sigma=1e-3`, then fit the same shared 28-parameter action family. All results
use unseen teacher families and dense word lengths through 2,048.

| Sensor | Info rank | `log det(I)` range | `trace(I^-1)` range | Least-observed one-step cosine | Worst-representation L2,048 cosine |
|---|---:|---:|---:|---:|---:|
| learned hard | 28 | `-3.466–-2.537` | `43.0–57.5` | `0.999991–0.999995` | `0.9593–0.9853` |
| oracle D-optimal | 28 | exactly `-2.537` numerically | exactly `43` numerically | `0.999994–0.999996` | `0.9664–0.9861` |
| random mixed | 28 | `-11.527–-3.343` | `48.7–148.7` | `0.999980–0.999993` | `0.8684–0.9724` |
| fixed `(1,4,0)` | 28 | `-11.468–-5.444` | `65.4–243.7` | `0.999962–0.999993` | `0.8841–0.9641` |
| single-view D-optimal | 25 | singular | singular | `0.9477–0.9894` | `-0.0667–0.0077` |

The learned sensor beats the matched random sensor in least-observed one-step
cosine and length-2,048 worst-representation cosine in **10/10** seeds. It is
within the frozen `0.02` oracle gap in 9/10. Seed 11 misses only that threshold:
`0.020923` versus `0.020000`. It remains recorded as a failure.

The old `(1,4,0)` design is a particularly useful correction to the preceding
five-probe result: it is sufficient for noiseless identifiability but often
poorly conditioned. Sharp rank is not sharp statistical efficiency.

## Numerical contracts

- maximum triality-equivariance error: `1.78e-15`;
- maximum parallel-prefix/recurrent error: `1.65e-14`;
- maximum absolute log-norm drift through length 2,048: `4.24e-13`;
- all five sensor variants receive the identical query count and matched noise
  tensor within each seed;
- the artifact includes every learned and oracle probe vector, not only
  allocations and summary spectra.

## What is established

1. Local D-optimal query design is independent of the unknown Spin(8) action;
   local endpoint adaptivity is unnecessary.
2. The numerically optimal five-query design is balanced across triality views
   and has a reproducible algebraic-looking information spectrum.
3. A hard learned policy discovers rank-28 multiview sensing without being told
   to diversify in 10/10 seeds.
4. Global D-optimal convergence remains unreliable at 6/10 because hard view
   choices lock into imbalanced allocation basins.
5. Better information conditioning produces a real finite-noise and
   long-composition advantage over random sensing.

## What is not established

- The exact determinant `81/1024` and global optimality of the balanced frame
  are not yet analytically proven.
- Action-independent local Fisher information does not prove that adaptive
  querying is globally useless.
- The strict learned-design gate failed; the hard selector is not yet a
  reliable D-optimal optimizer.
- No semantic, language-model, or natural-data result follows from this gate.

## Next gate

The next justified intervention is a **joint query-family continuation**:
optimize soft representation mass for the complete five-query family, then
retract the family jointly to five hard physical queries only after the
information geometry is well-conditioned. Compare that continuation against
the current straight-through hard policy without adding an explicit balance or
diversity target. The frozen objective remains D-optimality; the question is
whether joint late retraction eliminates the four discrete allocation traps.
