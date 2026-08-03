# Recurrence ladder: 1000-step multi-group replication

All rows use the same optimizer, batches per seed, eight-real state width,
parameter shapes, initial parameters, and initial function. Values are mean
+/- sample standard deviation across seeds 0, 1, and 2.

| Group | Family | Loss | L16 final accuracy | L32 final accuracy | steps/s |
|---|---|---:|---:|---:|---:|
| D4 | `complex_unitary` | 0.594 +/- 0.335 | 43.16 +/- 17.34% | 21.88 +/- 4.81% | 22.77 |
| D4 | `ga_rotor_selective` | 0.559 +/- 0.355 | 46.69 +/- 22.03% | 24.99 +/- 14.03% | 15.93 |
| Q8 | `complex_unitary` | 0.503 +/- 0.079 | 48.80 +/- 0.79% | 19.88 +/- 11.75% | 23.86 |
| Q8 | `ga_rotor_selective` | 0.554 +/- 0.051 | 46.38 +/- 6.62% | 29.78 +/- 6.55% | 15.79 |
| S3 | `complex_unitary` | 0.859 +/- 0.182 | 27.86 +/- 9.28% | 20.62 +/- 4.82% | 24.20 |
| S3 | `ga_rotor_selective` | 0.497 +/- 0.473 | 54.02 +/- 34.20% | 26.66 +/- 14.74% | 15.81 |

## GA minus complex margin

- D4: L16 +3.53 points; L32 +3.11 points.
- Q8: L16 -2.43 points; L32 +9.90 points.
- S3: L16 +26.16 points; L32 +6.04 points.

## Interpretation

The selective GA rotor has the higher mean L32 final accuracy in all
three groups, but variance is large and individual D4/S3 seeds reverse
the ordering. This is promising evidence for long-horizon behavior, not
yet a robust universal win. Q8 is the most consistent result: GA wins L32
on every seed. More seeds and harder tasks remain necessary.

The grade-decay family is intentionally excluded from this two-family
replication summary because only seed 0 has been run.

## Source reports

- `SSM-Models/experiments/recurrence_ladder_q8.json`
- `SSM-Models/experiments/recurrence_ladder_q8_seed1_1000_focused.json`
- `SSM-Models/experiments/recurrence_ladder_q8_seed2_1000_focused.json`
- `SSM-Models/experiments/recurrence_ladder_d4_seed0_1000.json`
- `SSM-Models/experiments/recurrence_ladder_d4_seed1_1000_focused.json`
- `SSM-Models/experiments/recurrence_ladder_d4_seed2_1000_focused.json`
- `SSM-Models/experiments/recurrence_ladder_s3_seed0_1000.json`
- `SSM-Models/experiments/recurrence_ladder_s3_seed1_1000_focused.json`
- `SSM-Models/experiments/recurrence_ladder_s3_seed2_1000_focused.json`
