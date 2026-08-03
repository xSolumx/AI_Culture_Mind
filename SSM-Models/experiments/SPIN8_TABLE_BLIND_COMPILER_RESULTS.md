# Spin(8) table-blind path compiler: results

Date completed: 2026-08-03.

## Result

The table-blind compiler passes the prospective seed-19 smoke and all nine
untouched seeds 20--28. It receives no Q8 table, inverse pairing,
token-to-element map, identity label, target label, or group-aware calibration
batch. It uses only uniformly sampled token strings, recurrent states, model
predictions as anonymous endpoint labels, and the fixed cardinalities of eight
states and four tokens.

| Cohort | All-gate passes | Post-hoc Q8-isomorphic | Dense and L16K |
|---|---:|---:|---:|
| Prospective smoke, seed 19 | 1/1 | 1/1 | 1/1 |
| Untouched seeds 20--28 | 9/9 | 9/9 | 9/9 |

Every raw seed has minimum central-pair accuracy 0% at one of the frozen dense
checkpoints before compilation, with raw homomorphism RMS 0.560--0.729. Thus
the result is not merely preserving already-exact long-horizon behavior.

## What was recovered

For each seed, all 32 anonymous `state x token` transitions have 100% winner
fraction and 100% winner-minus-runner-up vote gap over 32,768 randomly sampled
L15/L16 paths. The inferred token columns are permutations and generate a
regular group of order eight. Only after the compiled checkpoint is persisted
is that anonymous table compared with hidden Q8; all ten prospective tables
are isomorphic.

The exact action is constructed from the recovered regular representation,
not Q8 constants. Across untouched seeds 20--28:

| Metric | Range |
|---|---:|
| Maximum channel centroid projection RMS | 0.00323--0.00842 |
| Spin(8) action reconstruction max | 4.66e-7--6.97e-7 |
| Recovered-table homomorphism RMS | 5.67e-7--7.12e-7 |
| Observer logit transport max | 6.05e-8--1.48e-7 |
| Exact section rank | 8 in 9/9 |
| Exact section condition number | 26.6--134.9 |

Every seed remains 100% on all dense L15--L256 central-pair checkpoints and
at L4095, L4096, L16383, and L16384. Streaming recurrent-state parity remains
exact within the frozen numerical tolerances. The collapsed-label and
independently scrambled-successor controls reject in every recorded run.

## The new mechanism statement

This closes the oracle gap in the earlier path-section result. Endpoint
equivalence learned by the network is enough to recover its finite transition
algebra, and that recovered algebra is enough to jointly retract an
unconstrained token-action family onto a shared exact Spin(8) representation
manifold. No token is rounded or normalized independently, and no gradients
are taken after compilation.

The claim remains decoder-labeled rather than unsupervised. The compiler is
told there are eight anonymous endpoint classes and four tokens, and training
itself consumed exact endpoint labels. It does not infer state cardinality,
discover equivalence from raw language, or establish a language-model gain.

The next strict gate is therefore state-only recovery: cluster recurrent
states without consulting decoder argmax, select or infer the number of stable
endpoint classes, recover transitions between those clusters, and reserve the
observer only for post-recovery readout transport. Only after that should the
same compiler be tested on a group not used to design the Q8-specific
center-fidelity curriculum.

Artifacts:

- `spin8_table_blind_smoke_seed19_raw.json`
- `spin8_table_blind_smoke_seed19_compiled.json`
- `spin8_table_blind_validation_seeds20_28_raw.json`
- `spin8_table_blind_validation_seeds20_28_compiled.json`
- `spin8_table_blind_smoke_checkpoints/`
- `spin8_table_blind_validation_checkpoints/`
- `spin8_table_blind_validation_compiled/`
