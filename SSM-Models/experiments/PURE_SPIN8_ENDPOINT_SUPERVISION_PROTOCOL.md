# Pure Spin(8) endpoint-only continuous-observation protocol

Protocol frozen: **2026-08-17T05:02:15+02:00**

## Question

Does the shared Pure Spin(8) action prior remain identifiable when training
provides only the final signed triality state, rather than every intermediate
prefix? Does it retain an advantage over a capable parameter-near, exactly
state-matched independent `SO(8)^3` action family?

This changes one axis of the passed continuous-observation cohort. Observation
charts, noise, architectures, initial teacher state, training length, excluded
adjacent center relation, and early/late L16/L64/L128 evaluations are unchanged.

## Anti-leakage contract

Each training batch has fields `observations`, `endpoint_targets`, `coordinates`,
and `events`. It has no intermediate-target field. The loss is

`MSE(predictions[:, -1], endpoint_targets)`.

The unit contract requires exactly zero gradient at every nonfinal prediction.
Coordinates and event labels are retained only for split auditing and are never
passed to a candidate. Each sequence exposes 24 supervised endpoint scalars,
not `16 * 24` prefix scalars.

## Frozen task and cohort

- 2,000 equal updates, batch 32, training length 16;
- AdamW, learning rate `3e-3`, weight decay `1e-4`, gradient clip 1;
- unique 12-real observations of seven hidden active Spin(8) coordinates;
- Gaussian observation noise `0.01` and a new nonlinear chart per seed;
- isolated half-center events are present, but adjacent half-center events are
  absent from training;
- 64 paired center/identity evaluations at early and late positions for lengths
  16, 64, and 128; and
- candidates and parameter/state counts exactly match the primary continuous
  cohort: shared Spin(8), independent `SO(8)^3`, Mamba-2, parameter-near GRU,
  observation-only MLP, and exact-state GRU.

## Development evidence used only to freeze gates

Seed 0 used 1,024,000 unique observations and 64,000 endpoint targets. Shared
Spin(8) reached action RMSE `0.01117`, perfect L128 relation correctness, and
early/late L128 post-relation MSE `0.00800/0.01614`. Independent `SO(8)^3`
reached action RMSE `0.06973`, perfect L128 relation correctness, and
`0.05256/0.09986`. Mamba-2, both GRUs, and observation-only remained near
chance at L128.

The source timestamp label was initially written as 05:02 despite the artifact
being completed earlier. Before protocol freeze, only that metadata was
corrected to a 04:54 start and a correction time was added; model metrics and
checkpoints were unchanged.

Development artifact:
`artifacts/pure_spin8_endpoint_supervision_development_seed0.json`<br>
SHA-256:
`87626dd9ac5a4f81695999a8832cb6eb3fb58ef312e86edfe642d1b54163a1d2`

## Fresh seeds and immutable gates

Fresh seeds **1, 2, and 3** must each satisfy all of the following; no median
rescue is allowed:

1. teacher, uniqueness, no-intermediate-target, exclusion, pairing, finiteness,
   parameter, and state audits pass;
2. all six checkpoints rehash and strictly reload with endpoint-only metadata;
3. shared action RMSE is at most `0.03` and strictly below independent action
   RMSE;
4. independent action RMSE is at most `0.12`, preventing an incapable control;
5. every shared early/late L128 split has post-relation MSE at most `0.04`,
   exact classification and row correctness, both center signatures at most
   `0.05` RMSE, and lower MSE than every other candidate;
6. every independent early/late L128 split has post-relation MSE at most `0.15`,
   classification at least `0.95`, and each row correctness at least `0.90`;
7. all shared learned relation-action residuals are at most `0.04`, and all
   independent residuals are at most `0.08`; and
8. training and evaluation schedules and observation systems are distinct
   across seeds.

No architecture, threshold, schedule, optimizer, noise level, or supervision
rule may change after this freeze.

## Separately frozen measured-wall continuation

If the equal-update cohort passes, the corrected seed-0 model-update walls fix
this hardware-specific allocation:

| Candidate | Updates |
|---|---:|
| shared Pure Spin(8) | 2,000 |
| independent `SO(8)^3` | 1,558 |
| Mamba-2 | 2,811 |
| parameter-near GRU | 11,907 |
| observation-only MLP | 15,482 |
| state-matched GRU | 11,911 |

These are `round(2000 * shared_wall / candidate_wall)`. This continuation must
remain separate, cannot rescue a failed primary cohort, and is not a claim of
FLOP, energy, fused-kernel, or hardware-independent compute equality.

## Commands

For each `SEED` in `1,2,3`:

```powershell
python benchmark_pure_spin8_endpoint_supervision.py --seed SEED --steps 2000 --batch-size 32 --training-length 16 --evaluation-pairs 64 --evaluation-lengths 16,64,128 --evaluation-microbatch-size 32 --device cuda --candidates shared_pure_spin8,independent_so8_triplet,mamba2_parameter_near,gru_parameter_near,observation_only_ablation,gru_state_matched --output experiments\artifacts\pure_spin8_endpoint_supervision_validation_seedSEED.json --checkpoint-directory checkpoints\pure_spin8_endpoint_supervision_validation
```

Then adjudicate the three source artifacts with
[`validate_pure_spin8_endpoint_supervision.py`](../validate_pure_spin8_endpoint_supervision.py).

## Claim boundary

Passing supports endpoint-only system identification for the stated synthetic
signed-state task and a replicated shared-action inductive-bias advantage. It
does not establish natural-data utility, unsigned or partially observed pose
identification, all-28-coordinate coverage, fused-Mamba parity, or language-
model superiority.

## Post-freeze outcome

Fresh seeds 1--3 passed all 51 registered checks per seed without median
rescue. Shared Spin(8) recorded median L128 post-relation MSE `0.01296`, versus
`0.06268` for independent `SO(8)^3` at equal updates; every structured L128
row was classified correctly. The separately pre-frozen update-wall
continuation preserved the ordering at `0.01296` versus `0.09080`, with all
candidate walls within 1.97% of shared. Both validators strictly rehashed and
reloaded 18 checkpoints.

Primary aggregate SHA-256:
`1cf51a4af05303bc3ca9e781478e2352e8dbb077d1c9b367f46af2f384653880`<br>
Wall aggregate SHA-256:
`538a3bdbddfd76863a5bef5507a6d0019a114b35021d7b5b9d1223d31983ac64`

The authoritative interpretation, limitations, and reproduction details are
in
[`PURE_SPIN8_ENDPOINT_SUPERVISION_RESULTS.md`](PURE_SPIN8_ENDPOINT_SUPERVISION_RESULTS.md).
