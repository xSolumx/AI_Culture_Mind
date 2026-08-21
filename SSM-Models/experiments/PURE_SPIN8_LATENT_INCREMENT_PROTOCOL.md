# Pure Spin(8) Latent-Increment Validation Protocol

Protocol frozen: **2026-08-17T03:34:00+02:00**<br>
Development run completed: **2026-08-17T03:31:04+02:00**

## Question

Can a parameter-near learned Pure Spin(8) tracker identify token-local group
actions from every-prefix supervision, then compose the never-observed relation
`a_half_center * a_half_center` into the nontrivial central element at lengths
up to 128?

This is the direct next falsifier from the supplied-coordinate
Pure-Spin(8)-versus-Mamba-2 experiment. The 28 teacher Lie coordinates are now
latent: candidates receive only one of eight symbolic token IDs.

## Frozen task

The eight tokens have fixed hidden Spin(8) increments. `a_half_center` is a
`pi` rotation in one coordinate plane, so its square acts as `+I` in `8v` and
`-I` in both half-spin representations. `b` and `b_inverse` are a paired
identity control. Training:

- supervises all 24 scalars of every prefix in `(8v, 8s+, 8s-)`;
- contains every token and every ordered token pair except `(a, a)`; and
- uses length 16, 500 AdamW updates, batch 32, learning rate `3e-3`, weight
  decay `1e-4`, and gradient clipping at 1.

Evaluation forces matched-context `(a, a)` and `(b, b_inverse)` rows at early
and late positions, with 64 pairs at lengths 16, 64, and 128. Schedules and
targets are generated independently for each seed and recorded by SHA-256.

## Frozen candidates

| Candidate | Trainable parameters | Persistent state scalars | Role |
|---|---:|---:|---|
| latent Pure Spin(8) | 892 | 24 | 24-token embedding, `24 -> 28` router, frozen exact maintained Spin(8) transport |
| Transformers Mamba-2 | 891 | 160 | installed one-layer unfused/naive reference |
| GRU | 887 | 9 | recurrent nonlinear control |
| token-only MLP | 874 | 0 | no-prefix-state ablation |

The parameter spread is below 3%. State size and measured compute are
deliberately not called matched. The Pure Spin(8) candidate receives the
correct algebra and a fixed teacher initial probe, but not the hidden token
coordinates. The controls receive no Spin(8) structural prior.

## Development result used only to freeze gates

Seed 0 reached action RMSE `0.00408`. On both early and late length-128 splits
it achieved center and identity row accuracy `1.0`; post-relation MSE was
`1.57e-4` and `2.99e-4`. The recurrent controls remained near chance on center
classification. This result selected the thresholds below and is excluded
from the validation cohort.

Development artifact:
`artifacts/pure_spin8_latent_increment_development_seed0.json`<br>
SHA-256:
`8982c02285650669b5187176834b82f004e500ca5eb30e4be501f26166f59576`

## Fresh validation cohort and immutable gates

Run fresh seeds **1, 2, and 3** without changing model code, schedule size,
optimizer, or thresholds. Every seed must satisfy every gate:

1. teacher, training-split, evaluation-split, finiteness, and parameter-spread
   audits pass;
2. all 12 checkpoints rehash and reload strictly into their declared model;
3. Pure Spin(8) action RMSE is at most `0.02`;
4. each learned `a^2` vector/spinor residual and `b_inverse*b` residual is at
   most `0.01` RMSE;
5. for both early and late length-128 splits, Pure Spin(8):
   - has post-relation MSE at most `0.002`;
   - has center classification, center-row correctness, and identity-row
     correctness exactly `1.0`;
   - has predicted vector-pair and spinor-negation RMSE at most `0.02`; and
   - has lower post-relation MSE than both Mamba-2 and GRU.

The cohort passes only if all three seeds pass. No median-only rescue is
allowed. The validator is
[`validate_pure_spin8_latent_increment.py`](../validate_pure_spin8_latent_increment.py).

## Frozen commands

For each `SEED` in `1,2,3`:

```powershell
python benchmark_pure_spin8_latent_increment.py --seed SEED --steps 500 --batch-size 32 --training-length 16 --evaluation-pairs 64 --evaluation-lengths 16,64,128 --evaluation-microbatch-size 32 --device cuda --output experiments\artifacts\pure_spin8_latent_increment_validation_seedSEED.json --checkpoint-directory checkpoints\pure_spin8_latent_increment_validation
```

Then aggregate and independently enforce the frozen gates:

```powershell
python validate_pure_spin8_latent_increment.py experiments\artifacts\pure_spin8_latent_increment_validation_seed1.json experiments\artifacts\pure_spin8_latent_increment_validation_seed2.json experiments\artifacts\pure_spin8_latent_increment_validation_seed3.json --output experiments\artifacts\pure_spin8_latent_increment_validation_seeds1_3.json
```

## Claim boundary

Passing establishes replicated, checkpointed evidence for algebra-matched
latent action identification and unseen central-relation composition on this
synthetic every-prefix task under a parameter-near budget. It does not establish
state- or compute-matched superiority, natural-data usefulness, generic
language-model quality, a fused-Mamba throughput result, or a theorem about
Mamba-2, diagonal SSMs, triality, or unrestricted global optimality.
