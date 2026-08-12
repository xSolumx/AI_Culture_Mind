# Research phase 2: 1000-step recurrence ladder

## What the original Q8 run actually says

The seed-0 report uses 1,000 training steps, length 16, two layers, four
channels, 13,512 parameters, and a 64-scalar recurrent cache for every family.
All candidates have the same parameter shapes, initial parameter values,
initial function, training batches, validation batches, optimizer, and state
width. Streaming logits agree with full-sequence logits within `3.41e-5`.

| Family | validation loss | L16 final | L32 final | steps/s |
|---|---:|---:|---:|---:|
| real selective | 0.822 | 25.90% | 11.62% | 64.75 |
| complex unitary | **0.413** | 49.12% | 29.69% | 24.88 |
| quaternion even | 0.546 | 48.24% | 11.18% | 11.84 |
| selective GA rotor | 0.528 | **51.15%** | **36.79%** | 16.89 |
| static GA rotor | 0.946 | 19.19% | 12.48% | 17.06 |

The honest reading is mixed. Complex phases learn the prefixes best and are
about 1.47 times faster than GA. The selective GA rotor is narrowly best at
the training horizon and clearly best at twice the training length. The static
rotor ablation fails, so token-selective rotation matters. Q8 is nevertheless
structurally aligned with quaternionic/rotor machinery, and one seed is not
enough to establish a general advantage.

## Multi-group, three-seed focused replication

The complex and selective-GA families were repeated at seeds 0, 1, and 2 on
Q8, D4, and S3 without changing the 1,000-step protocol. Values are mean plus
or minus sample standard deviation. The machine-readable aggregation is in
`recurrence_ladder_multigroup_1000_summary.json`; the generated report is
`RECURRENCE_LADDER_MULTIGROUP_1000.md`.

| Group | Family | loss | L16 final | L32 final | steps/s |
|---|---|---:|---:|---:|---:|
| D4 | complex | 0.594 +/- 0.335 | 43.16 +/- 17.34% | 21.88 +/- 4.81% | 22.77 |
| D4 | selective GA | 0.559 +/- 0.355 | 46.69 +/- 22.03% | 24.99 +/- 14.03% | 15.93 |
| Q8 | complex | **0.503 +/- 0.079** | **48.80 +/- 0.79%** | 19.88 +/- 11.75% | 23.86 |
| Q8 | selective GA | 0.554 +/- 0.051 | 46.38 +/- 6.62% | **29.78 +/- 6.55%** | 15.79 |
| S3 | complex | 0.859 +/- 0.182 | 27.86 +/- 9.28% | 20.62 +/- 4.82% | 24.20 |
| S3 | selective GA | **0.497 +/- 0.473** | **54.02 +/- 34.20%** | **26.66 +/- 14.74%** | 15.81 |

Selective GA has higher mean length-32 accuracy on all three groups. Q8 is the
most repeatable result: GA wins L32 on every seed, by 9.90 points on average.
D4 and S3 have large seed variance and individual reversals. On S3, GA finds
an excellent solution at seeds 0 and 2 but fails at seed 1. The current signal
is therefore long-horizon potential combined with an optimization/reliability
problem, not universal dominance.

## Grade-preserving multi-decay

`ga_rotor_grade_decay` gives grades 0, 1, 2, and 3 distinct token-selective
decays. It is mathematically scan-compatible because rotor conjugation
preserves grade. On seed-0 Q8 it reaches 47.07% at L16 but only 18.16% at L32,
versus 51.15% and 36.79% for the shared-decay rotor. Throughput falls from
16.89 to 13.63 steps/s. Grade preservation is still exact: both rotor
conjugation and the implemented grade-diagonal damping preserve grade, and the
two operations commute. The result therefore does **not** show cross-grade
leakage or damaged grade preservation. It shows only that independently
rescaling the four grade subspaces hurt this run while adding controller cost.
Relative-grade anisotropy and harder optimization are plausible hypotheses,
but the recorded measurements do not isolate a cause. This unconstrained
variant should not advance without a targeted diagnostic or regularizer.

## Fixed-width complex/GA direct sum

`hybrid_complex_ga` assigns half the channels to complex phases and half to GA
rotors. It preserves the parameter count, 64-scalar cache, exact initial
function, norm-preserving sub-actions, and streaming interface.

| Group, seed 0 | complex L32 | GA L32 | hybrid L32 | hybrid steps/s |
|---|---:|---:|---:|---:|
| Q8 | 29.69% | **36.79%** | 24.88% | 10.31 |
| D4 | **24.15%** | 11.96% | 16.50% | 10.31 |
| S3 | 26.12% | **43.53%** | 27.69% | 10.35 |

The direct sum is slower than both parents and does not inherit their best
behavior. At fixed width it appears to starve each representation while paying
for both kernels. Retain it as a negative control; do not scale it before a
better routing or capacity argument exists.

## Research decision

The next high-value work is not Cl(8,0) scale-up yet. First:

1. Run at least five seeds for complex and selective GA on more groups and
   order-sensitive recall tasks.
2. Diagnose the rotor's bimodal optimization with per-layer angle, decay,
   gradient, and state-norm trajectories.
3. Add dense orthogonal and contractive dense 8-by-8 baselines. These test
   whether the gain comes from orthogonality/noncommutativity rather than GA.
4. Train at multiple sequence lengths and test 2x, 4x, and 8x extrapolation.
5. Advance to Spin(8) chiral generators only if the selective-rotation signal
   survives those controls.

That sequence is less glamorous than immediately building a 28-generator
model, but it is the shortest route to knowing whether there is a real result.
