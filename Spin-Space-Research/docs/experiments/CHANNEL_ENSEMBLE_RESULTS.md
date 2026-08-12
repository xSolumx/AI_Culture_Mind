# Stable Channel-Ensemble Results

## Status and provenance

This is a post-hoc mechanistic audit, not a replacement for the original raw
homomorphism gate. The second macro-alphabet rule and the repair/damage
decomposition were recorded prospectively in
`CHANGED_GENERATOR_PREREGISTRATION.md` before evaluating the second alphabet.
The channel subsets themselves are evaluation-label oracles and are not a
deployable selection mechanism.

All ten GA checkpoints were regenerated with deterministic CUDA algorithms and
saved. Their original-generator dense minima reproduce the earlier ten-seed
table, including seed 0 `99.61%@L208`, seed 4 `96.39%@L240`, seed 5
`98.24%@L32`, seed 7 `93.55%@L240`, and seed 8 `98.05%@L176`. This independently
checks the reproducibility fix before using the saved tensors.

The frozen models were evaluated on two ranked changed-generator equivalence
classes:

- pair 0: `12453`, `12534`, `23514`, `41253`;
- pair 1: `12453`, `12534`, `24153`, `31524`.

Pair 1 is the next distinct order-3/order-5 class after quotienting inverse
re-labellings. It shares the first order-3 inverse pair with pair 0 but uses a
different order-5 inverse pair. Neither test retrains actions or the decoder.

## Ten-seed subset lattice

For each seed, the table reports the subset maximizing the minimum accuracy
over both macro alphabets and every L16-L256 multiple of 16. This is an oracle
diagnostic. `anchor` is fixed from the original-token maximum commutator and is
never reselected by length, alphabet, or accuracy.

| seed | anchor | anchor joint floor | full joint floor | joint-oracle subset | subset joint floor |
|---:|---:|---:|---:|---|---:|
| 0 | 1 | 79.98% | 97.66% | 0+1+2+3 | 97.66% |
| 1 | 0 | 97.75% | 100.00% | 0+1 | 100.00% |
| 2 | 0 | 91.99% | 92.77% | 0+1+2 | 99.41% |
| 3 | 2 | 98.05% | 99.80% | 0+2 | 100.00% |
| 4 | 1 | 93.16% | 98.54% | 0+1+2+3 | 98.54% |
| 5 | 2 | 99.80% | 98.83% | 0+1+2 | 100.00% |
| 6 | 2 | 90.33% | 76.76% | 2+3 | 97.17% |
| 7 | 2 | 98.63% | 97.56% | 1+2+3 | 98.73% |
| 8 | 0 | 95.12% | 97.66% | 0+2+3 | 99.71% |
| 9 | 3 | 94.73% | 94.92% | 0+2+3 | 99.80% |

Every joint-oracle subset contains the fixed anchor. Seven of ten seeds select
the same standalone best subset on both alphabets. In the remaining three,
the pair-0 subset still transfers to pair 1 strongly. Selecting the subset on
pair 0 and evaluating pair 1 gives ten of ten dense passes; the lowest pair-1
floor is `97.75%` (seed 6). Thus the useful redundancy is a stable per-seed
property across these two macro distributions, not merely a pair-0 accident.

The full code passes the dense gate in nine of ten seeds on each alphabet;
seed 6 fails both. The anchor passes nine of ten on pair 0 and ten of ten on
pair 1. The oracle subset passes ten of ten on both and has a worst observed
joint floor of `97.17%`. These are distinct facts: the oracle number does not
constitute a learned selection method.

## Outcome decomposition

The decoder is linear across channel blocks, so channel logits add before
`argmax`. Relative to the fixed anchor, every subset prediction falls into one
of five exhaustive outcomes: both correct, repair, damage, both wrong with the
same prediction, or both wrong with different predictions.

Across the 320 equally weighted seed/alphabet/length strata for each seed's
joint-oracle subset:

| outcome | mean rate |
|---|---:|
| anchor and subset both correct | 97.66% |
| anchor wrong, subset correct (`repair`) | 2.08% |
| anchor correct, subset wrong (`damage`) | 0.18% |
| both wrong, same prediction | 0.07% |
| both wrong, different predictions | 0.02% |

Repairs exceed damages in 19 of 20 seed/alphabet aggregates. The subset is not
merely inert: in some runs it repairs a substantial fraction of examples
(about 11% for seed 0 on pair 0 and 5% for seed 6) while remaining mostly safe.

## What kind of correction is this?

Three diagnostics distinguish increasingly strong meanings of correction.
They are restricted to examples where the learned canonical anchor prototype
decodes correctly but the path-evolved anchor fails.

1. **Correct class-boundary direction: supported.** The robust residual's
   true-versus-actual-wrong-class margin is positive in `98.6%` of 249
   evaluable seed/alphabet/length strata on average; the median is `100%`.
2. **Magnitude proportional to anchor defect: rejected.** Across 229 strata,
   Pearson correlation between projected defect magnitude and residual
   compensating magnitude has mean `-0.155` and median `-0.234`. Repaired
   anchor margins are usually shallow (roughly `-0.4` to `-1.3`) while residual
   margins are much larger (roughly `+3` to `+7`).
3. **Full logit-vector cancellation: unsupported.** The mean cosine between
   the centered residual vector and the negative centered anchor-defect vector
   is `0.040` across 249 strata; the median is `0.022`.

The precise conclusion is therefore:

> Auxiliary channels form a stable, transferable decoder ensemble around a
> dominant approximate irrep. They provide selective class-boundary correction
> but do not proportionally estimate or cancel the anchor's geometric/logit
> drift.

`Geometric error-correcting code` is not supported. `Pure nuisance channels` is
also false. The narrower phrase `stable decoder ensemble` is justified.

## Structural regularity and limits

The selected auxiliaries are not explained by one simple scalar diagnostic.
They are not consistently the next-largest commutator, lowest homomorphism
error, largest decoder-energy channel, or largest orbit-energy channel. The
stable subset is real, but its membership remains an SGD-dependent property of
the joint learned states and decoder.

This audit covers A5, ten seeds, two macro alphabets that share their order-3
inverse pair, and one fixed trained architecture. Dense length testing ends at
256. It does not satisfy the original raw homomorphism threshold and does not
establish infinite-horizon correctness.

## Next falsifier

Do not hard-prune and do not deploy the oracle subset. Freeze the recurrent
actions and learn only three bounded auxiliary gate scalars around the fixed
anchor (`alpha_anchor = 1`, `alpha_aux in [0, 1]`) using original words plus
pair 0 under a worst-stratum decoder-margin objective. Fix pair 1 for model
selection and prospectively reserve generator class index 2 as the untouched
test alphabet. Compare:

1. anchor only;
2. all channels;
3. learned fixed gates;
4. pair-0 oracle subset as a labeled upper diagnostic, never as a baseline
   available to training.

The learned gate is successful only if one fixed gate per seed passes every
dense length on the untouched third alphabet and does not reduce the original
or pair-1 dense floor by more than one percentage point. This tests whether the
stable ensemble can be identified without evaluation-label subset search.

This experiment is now complete and passed both criteria `10/10`; see
`ROBUST_CHANNEL_GATING_RESULTS.md`. The next remaining falsifier changes the
order-3 inverse pair rather than only the order-5 pair.

That falsifier is also complete; see `JOINT_A5_ROUNDING_RESULTS.md`. Frozen
soft gates retain 10/10 dense transfer on the changed-order-3 class, but the
learned anchors fail L4096 in all ten seeds. Joint exact anchor projection
passes L4096 in all ten, showing that decoder ensembling and exact transition
closure address separate failure modes.

## Artifacts

- `changed_generator_channel_subset_lattice_a5_pair0_10seeds.json`
- `changed_generator_channel_subset_lattice_a5_pair1_10seeds.json`
- `mechanistic_a5_ga_holonomy_multiscale_channel_audit_seed{0..9}_1500.pt`
