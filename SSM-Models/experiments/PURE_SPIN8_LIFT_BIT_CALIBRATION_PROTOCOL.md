# Pure Spin(8) adaptive lift-bit calibration protocol

Protocol frozen: **2026-08-17T07:05:22.8794763+02:00**

## Question

Can the exact hidden-lift failures of vector-only endpoint supervision be
closed by the minimum lift-odd information: one binary sign? What calibration
chart makes that bit numerically stable?

This follows the failed endpoint-observability cohort without changing its
teacher, noisy injective input chart, L16 training length, excluded adjacent
center relation, 2,000-update budget, candidates, optimizer, or L128 tests.

## Fiber theorem and robust chart

For a double-cover fiber `{y,-y}`, any nonzero odd scalar `l(y)` gives opposite
sign bits on the two lifts. At least `ceil(log2(2))=1` binary measurement is
necessary, and one such bit is sufficient away from `l(y)=0`.

A fixed coordinate is not uniformly stable: its zero set is a codimension-one
hyperplane. The development audit found fixed-probe magnitudes as small as
`2.38e-6` in training and `0.00189` on L128 evaluation.

The adaptive chart uses

`j = argmax_k |y_k|`, `b = sign(y_j)`.

The address `j` is invariant under `y -> -y`, while `b` flips. For a unit
eight-real spinor,

`|y_j| >= 1/sqrt(8) = 0.353553...`.

The transmitted calibration therefore contains a three-bit, lift-invariant
chart address plus exactly one lift-odd bit. It is not one total bit. The claim
being tested is that one bit of *lift information*, supplied in a robust
quotient-invariant chart, is sufficient.

## Frozen modes and controls

Every mode uses the identical precomputed schedule and identical candidate-
specific initialization:

| Mode | Endpoint supervision |
|---|---|
| `vector_only` | eight real `8v` scalars |
| `vector_plus_positive_bit` | `8v` plus one fixed-coordinate sign bit |
| `vector_plus_adaptive_lift_bit` | `8v` plus three-bit address and one sign bit |
| `positive_only` | eight real `8s+` scalars |
| `full_triality` | all 24 real scalars |

Each mode trains the 930-parameter shared Pure Spin(8) router and the
957-parameter, exactly 24-state independent `SO(8)^3` family. The bit loss is
`0.1 * BCEWithLogits(8 * selected_prediction, bit)`. For adaptive calibration,
only the final vector block, integer address, and one bit are transferred to
the accelerator-side loss. A unit test requires gradient support only on the
final vector block and the one addressed `8s+` scalar.

## Development evidence used to freeze gates

Seed 0 uses 64,000 endpoints and 1,024,000 unique observations. All eight chart
addresses occur, the adaptive bit has positive fraction `0.53266`, no address
ties occur at recorded precision, and minimum selected training margin is
`0.39705`, above the exact `1/sqrt(8)` bound up to float tolerance.

The naïve fixed probe is a negative development result. Shared training reaches
only vector MSE `0.01604`, batch bit accuracy `0.96875`, action RMSE
`0.08537--0.08724`, and late-L128 center accuracy `0.9141--0.9219`. It is worse
than vector-only because near-zero probe labels create an ill-conditioned
classification surface.

The adaptive chart changes the outcome:

| Development metric | Shared adaptive | Independent adaptive |
|---|---:|---:|
| final vector MSE | 0.00201 | 0.03226 |
| final adaptive-bit accuracy | 1.0 | 1.0 |
| action RMSE range across views | 0.01312--0.01312 | 0.06865--0.21668 |
| early-L128 MSE range | 0.01035--0.01165 | 0.08891--0.25032 |
| late-L128 MSE range | 0.01922--0.02438 | 0.14657--0.24764 |
| L128 adaptive-bit accuracy | 1.0 / 1.0 | 0.6875 / 0.6094 |
| L128 spinor center accuracy | 1.0 in all rows | 0.4922--0.7031 |

The adaptive shared row is also strictly better than its identically initialized
vector-only row in every action view and every L128 view.

Development artifacts:

- fixed-probe comparison SHA-256:
  `5fded3a079877fabaec4d8dc7b7ae7d6d99b0ad8310f97c1a3d4911a2bfe8c53`
- adaptive comparison SHA-256:
  `9565026805afb2948b18be08a1ce684b3048bf83c412435707650abe223aefef`
- strict adaptive replay SHA-256:
  `88c38c8a819e191e99f74862ef8d568c10db47978155dc126e2893f6c6372757`

## Frozen fresh cohort and gates

Untouched seeds **4, 5, and 6** run all five modes and both candidates. Every
seed must independently satisfy every gate; no median rescue is allowed.

1. Schedule, observation, address, bit, evaluation, source, and checkpoint
   hashes reproduce; every checkpoint strictly reloads and every metric is
   finite and replayed.
2. Adaptive training bit accuracy is exactly `1.0`.
3. Shared adaptive action RMSE is at most `0.04` in every view and each L128
   per-view MSE is at most `0.05`.
4. Shared adaptive L128 bit accuracy and both spinor center accuracies are
   exactly `1.0`; relation pairs retain one address and opposite bits.
5. Shared adaptive is strictly better than identically initialized vector-only
   in every action view and every L128 view.
6. Shared adaptive is strictly better than independent adaptive in every action
   view and every L128 view.
7. Independent adaptive remains a capable vector control: vector action RMSE is
   at most `0.18` and both vector L128 MSEs are at most `0.22`.
8. Shared positive-only and full-triality references each have action RMSE at
   most `0.05` and every L128 per-view MSE at most `0.07`.
9. Training schedules, observation systems, adaptive addresses, adaptive bits,
   and all 18 evaluation schedules are distinct across seeds.

No mode, chart, loss weight, logit scale, address rule, optimizer, threshold,
step count, candidate, or gate may change after this freeze.

## Commands

For each `SEED` in `4,5,6`:

```powershell
python benchmark_pure_spin8_lift_bit_calibration.py --seed SEED --steps 2000 --batch-size 32 --training-length 16 --evaluation-pairs 64 --evaluation-lengths 16,64,128 --evaluation-microbatch-size 32 --modes vector_only,vector_plus_positive_bit,vector_plus_adaptive_lift_bit,positive_only,full_triality --candidates shared_pure_spin8,independent_so8_triplet --device cuda --output experiments\artifacts\pure_spin8_lift_bit_calibration_validation_seedSEED.json --checkpoint-directory checkpoints\pure_spin8_lift_bit_calibration_validation
```

Then adjudicate with
[`validate_pure_spin8_lift_bit_calibration.py`](../validate_pure_spin8_lift_bit_calibration.py).

## Claim boundary

Passing would show that a robust chart address plus one lift-odd bit closes the
specific vector-only failures on this injective, seven-coordinate synthetic
teacher. It would not construct a global continuous section, remove the chart
address, solve physical unsigned sensing, cover all 28 trained coordinates,
or establish natural-task or generic SSM superiority.
