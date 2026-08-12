# Table-blind finite-action compiler: family baseline results

Date completed: 2026-08-03.

## Outcome

The anonymous transition/table recovery is not specific to Spin(8). Under the
corrected prospective rank-four contract, quaternion spinors pass 9/9 fresh
seeds and shared four-reflection Householder actions pass 8/9.

| Family | Fresh all-gate passes | Exact cyclic section rank | Uniform 9/9 |
|---|---:|---:|---:|
| Positive-chiral Spin(8), seeds 20--28 | 9/9 | 8 | Yes |
| Quaternion spinor, seeds 29--37 | 9/9 | 4 | Yes |
| Shared Householder O(4), seeds 29--37 | 8/9 | 4 | No |

All passing models are 100% at every dense and long checkpoint through
L16,384 after exact compilation. Their recovered-table homomorphism errors are
on the order of 1e-7.

## The seed-34 Householder failure

Seed 34 is a discovery/optimization failure, not an irrep or factorization
failure. Its final L16 training batch has 100% accuracy and loss 0.000188, but
its raw central-pair minimum is 0%, raw homomorphism RMS is 0.8166, and its
anonymous transition winner fraction is 0.987214, below the frozen 0.99 gate.
The compiler rejects before exact representation extraction. This is evidence
that perfect endpoint training accuracy does not guarantee a deterministic
long-path quotient action.

## Full-rank versus minimal sufficient sections

The most informative difference is not final task accuracy. Q8's faithful real
quaternionic irrep supplies a rank-four cyclic endpoint section. Before
observer transport, teacher logits must therefore be projected onto its
four-dimensional realizable row space. Among passing fresh seeds:

| Metric | Quaternion | Householder |
|---|---:|---:|
| Discarded teacher-logit RMS, range | 0.00224--0.57594 | 0.77200--1.53901 |
| Discarded teacher-logit RMS, median | 0.36871 | 1.30021 |
| Projected minimum margin, range | 2.338--3.463 | 0.410--2.283 |
| Observer transport max, range | 5.90e-8--9.07e-8 | 1.95e-7--3.88e-7 |

Despite that loss, the projected logits retain every anonymous class decision
for passing seeds. Spin(8)'s regular eight-dimensional section instead has
rank eight and preserves the complete centroid-logit geometry directly, with
no rank-four information projection.

This yields a sharper conclusion than “Spin(8) solves Q8”: quaternion and
Householder actions can solve and compile Q8 too. The demonstrated Spin(8)
advantage is a linearly complete exact endpoint section and correspondingly
less destructive observer transport. Quaternion is the most compact and
uniform minimal realization in this cohort; Householder is capable but has one
long-path transition-consistency failure and substantially greater discarded
logit energy.

## Harness interruption

The first combined raw-training command reached the ten-minute shell ceiling
after saving all nine quaternion checkpoints and Householder seed 29, before
writing its aggregate JSON. Householder seeds 30--37 were resumed unchanged in
a second deterministic command. File and checkpoint metadata confirm exactly
one checkpoint for each family/seed pair. This interruption affects only the
missing combined raw trajectory artifact, not the blinded compiler cohort.

Artifacts:

- `table_blind_family_validation_seeds29_37_compiled.json`
- `table_blind_householder_validation_seeds30_37_raw.json`
- `table_blind_family_validation_checkpoints/`
- `table_blind_family_validation_compiled/`
- invalidated rank-eight fixtures:
  `table_blind_quaternion_development_seed0.json` and
  `table_blind_householder4_development_seed0.json`
- corrected fixtures:
  `table_blind_quaternion_development_seed0_corrected.json` and
  `table_blind_householder4_development_seed0_corrected.json`
