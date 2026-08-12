# Spin(8) Q8 path-section compiler: untouched validation results

Date: 2026-08-03

The preregistered untouched cohort passed 9/9 seeds. The machine-readable
record is `spin8_q8_path_validation_seeds10_18_compiled.json`; raw results are
in `spin8_q8_path_validation_seeds10_18_raw.json`.

## Validated result

Seeds 10--18 were excluded from every design decision. Each raw model used the
unchanged four-channel positive-chiral Spin(8) recurrence, unconstrained 28D
tangent parameters, and the parity-complete 2,000-step Q8 curriculum.

All nine raw models achieved 100% on the held-out L15/L16 calibration section,
yet all nine eventually reached 0% minimum joint accuracy in the dense
L15--L256 sweep. Raw full homomorphism RMS ranged from `0.4334` to `0.6701`.
Thus the compiler did not repair an already-perfect long-horizon cohort.

After path-section compilation, all nine seeds achieved:

- 100% member and joint central-pair accuracy at every dense length through
  256;
- 100% member and joint accuracy at 4,095, 4,096, 16,383, and 16,384;
- every exact-family, observer-transport, and streaming gate;
- zero gradient steps after compilation.

| Metric across seeds 10--18 | Minimum | Maximum |
|---|---:|---:|
| Full homomorphism RMS | `5.817e-7` | `7.363e-7` |
| Spin(8) action reconstruction max | `4.993e-7` | `8.210e-7` |
| Centroid-logit transport max | `3.555e-8` | `8.354e-8` |
| Per-channel centroid projection RMS | `0.001316` | `0.009359` |
| Exact-section condition number | `52.74` | `148.32` |
| Observer displacement RMS | `0.01699` | `0.05624` |
| Raw tangent norm maximum | `3.071` | `3.464` |

Every calibration endpoint received between 3,976 and 4,225 examples. Raw
tangent norms were moderate rather than divergent; the maximum stayed well
below `2 pi`. This does not eliminate all exponential-chart conditioning
questions, but it falsifies tangent-norm explosion as the cause of this
cohort's raw long-horizon failures.

## The mechanism discovered

An approximate recurrent model does not necessarily assign one state to one
group element. It defines a path-dependent bundle: many word histories with
the same endpoint can occupy nearby but different states. A shortest-word BFS
choice is an arbitrary section of that bundle and can be behaviorally wrong.

The validated compiler performs four coupled operations:

1. estimate one behaviorally valid state per endpoint from disjoint L15/L16
   path ensembles;
2. project the complete eight-state Gram matrix onto the commutant of the exact
   Q8 regular action, preserving every supported irrep without rank selection;
3. map the resulting exact orthogonal token family through a real positive-
   chiral Spin(8) logarithm;
4. transport the linear observer on the reachable section by a minimum-change
   pseudoinverse solution.

This is a state-space realization compiler: transition `A`, reachable state
section, and observer `C` are compiled together. Normalizing tokens separately
or transporting the decoder by a guessed global rotation is insufficient.

## Negative and corrective results retained

The route to the validated method matters:

- The first rank-four, frozen-decoder retraction passed 8/9 fresh seeds but
  damaged seed 4. That prospective 8/9 result remains unchanged.
- Preserving only a fixed mean and faithful variation amplitude still failed
  seed 4 with a parity split.
- Full regular state retraction removed the rank cutoff, but its first run was
  invalidated by the reconstruction gate: the complex principal logarithm
  mishandled the even-dimensional `-1` eigenspace.
- A real Schur logarithm with paired pi-rotation planes fixed that numerical
  error, after which state-only regular retraction still failed behaviorally.
- Transporting shortest-word canonical logits exactly also failed because raw
  seed 4's shortest-word canonical accuracy was only 50%, despite 100% L15/L16
  accuracy.
- Replacing the arbitrary shortest-word section with held-out path centroids
  fixed seed 4 prospectively, then passed all nine untouched seeds.

## Reviewer-limit audit

1. **Oracle dependence:** real. The Q8 multiplication table groups endpoints
   and supplies the regular action. This is a table-aware offline compiler, not
   a blind symmetry-discovery engine.
2. **Decoder alignment:** resolved generally on the reachable section. A
   global `W U^T` update applies only to a pure coordinate gauge; the actual
   nonlinear orbit projection needs pseudoinverse observer transport.
3. **SVD threshold:** eliminated. The validated method uses the full regular
   representation and no spectral cutoff.
4. **Forgetting:** outside this write-free falsifier. The full Spin(8) SSM has
   independent scalar decay and affine writes.
5. **Matrix-exponential gradients:** skew tangents do not have generic
   non-normal exponential blow-up. Large angles can still cause differential
   attenuation and aliasing; the cohort logs tangent norms, which remained
   moderate here.

## Claim boundary and next gate

The result establishes a reliable table-aware compiler for Q8 on nine
untouched Spin(8) seeds. It does not establish table-blind discovery, a
Spin(8)-specific advantage over quaternion spinors or capable Householders, or
language-model benefit.

The next discriminating gate is cross-group and table-blind: infer the endpoint
algebra from path clusters before compilation, then compare the chiral family
against lower-dimensional faithful actions at matched state and controller
budgets. Triality-coupled token writes should remain deferred until that
baseline question is answered.
