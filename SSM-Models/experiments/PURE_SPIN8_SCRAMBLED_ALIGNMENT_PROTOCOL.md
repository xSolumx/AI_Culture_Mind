# Pure Spin(8) shared-latent scrambled-alignment protocol

Protocol frozen: **2026-08-17T08:40:40.5270068+02:00**

## Question

Does adaptive lift calibration succeed because one 28-coordinate bottleneck is
shared across views, or because those coordinates act through the **correct
Spin(8) triality representations**?

The earlier independent `SO(8)^3` control cannot answer this alone. Its
negative-specific head has exactly zero data gradient under vector-plus-bit
loss, while Pure Spin(8)'s one common head receives gradients. The exact trace
is recorded in
[`PURE_SPIN8_LIFT_GRADIENT_IDENTIFIABILITY_RESULTS.md`](PURE_SPIN8_LIFT_GRADIENT_IDENTIFIABILITY_RESULTS.md).

## Matched scrambled-alignment control

Both candidates use the same 12-to-22-to-28 observation router, identical
router initialization, the same 24-scalar recurrent state, the same endpoint
scan, and the same optimizer schedule.

The Pure Spin(8) candidate applies the common bivector coordinates through the
maintained aligned `(8v,8s+,8s-)` generators.

The control retains one common 28-coordinate head but conjugates the positive
and negative actions independently:

`R'_+(c)=Q_+ R_+(c) Q_+^T`, `R'_-(c)=Q_- R_-(c) Q_-^T`,

where each `Q` is a valid Spin(8) action parameterized by 28 trainable
coordinates. The vector alignment is fixed. The control therefore:

- composes legal orthogonal actions in all three views;
- has the same 24-state cache;
- has 986 trainable parameters versus 930 for Pure Spin(8), so it is slightly
  over-parameterized rather than disadvantaged;
- receives gradients in all 28 shared coordinate rows under adaptive loss;
- can exactly recover the aligned family at `Q_+=Q_-=I`;
- receives a positive-alignment gradient but exactly zero negative-alignment
  gradient under adaptive supervision;
- receives gradients in both alignments under full-triality supervision.

This isolates a supplied cross-view representation alignment from generic
coefficient sharing.

## Frozen task

The teacher, noisy seven-coordinate injective observation system, excluded
adjacent center relation, L16 endpoint-only training, 2,000 updates, and
L16/L64/L128 evaluation remain unchanged.

Only two supervision modes are needed:

| Mode | Purpose |
|---|---|
| `vector_plus_adaptive_lift_bit` | Test transfer from vector endpoint plus three-bit chart address and one lift-odd bit |
| `full_triality` | Prove the scrambled control can learn both hidden alignments when supervised |

The full calibration word remains four transmitted bits, only one of which is
lift odd. No intermediate targets, event labels, or latent coordinates enter
either model.

## Development evidence

Seed 0 used identical router hashes
`74478a6c2b0b752af1390ac9655caaba14696b7dce5875164c72d8aacc085479`
for both candidates. The automatic gradient route audit passed exactly.

| Mode / candidate | action RMSE `(v,+,-)` | L128 MSE range | negative center accuracy |
|---|---:|---:|---:|
| adaptive Pure Spin(8) | `(0.01398,0.01398,0.01398)` | `0.01021--0.02093` | `1.0 / 1.0` |
| adaptive scrambled | `(0.09896,0.04431,0.16523)` | `0.04793--0.17017` | `0.8516 / 0.8359` |
| full Pure Spin(8) | `(0.01087,0.01087,0.01087)` | `0.00728--0.01572` | `1.0 / 1.0` |
| full scrambled | `(0.07448,0.02242,0.02261)` | `0.03355--0.06683` | `1.0 / 1.0` |

The scrambled control fits the adaptive training bit exactly. Full supervision
then reduces its negative action RMSE from `0.16523` to `0.02261` and repairs
every center row, establishing bounded capability rather than an incapable
straw control.

Development artifacts:

- source SHA-256:
  `327e862e883a9add78331542155ae327962f46fb665e31226ad1793ff9bdc0e8`
- strict replay SHA-256:
  `7b19b034873fcccffa3cf25e0d7a013aa9eb44cb024d22b0bb60661c1d22bff4`

## Frozen fresh cohort

Untouched seeds **7, 8, and 9** must each pass every gate. No median rescue is
allowed.

1. All source, schedule, observation, address, bit, evaluation, and checkpoint
   hashes reproduce; every checkpoint strictly reloads and every metric is
   recomputed.
2. Router initialization is identical across candidates, both states contain
   24 scalars, and the control has exactly 56 additional parameters.
3. Adaptive loss gives the control a positive-alignment gradient, exactly zero
   negative-alignment gradient, and nonzero gradients in all 28 shared head
   rows. Full loss gives both alignments gradients.
4. Shared adaptive training-bit accuracy is `1.0`, every action RMSE is at most
   `0.04`, every L128 MSE is at most `0.05`, and all L128 lift/center rows are
   exact.
5. Scrambled adaptive training-bit accuracy is `1.0`; vector/positive action
   RMSE is at most `0.15/0.10`, and vector/positive L128 MSE is at most `0.13`.
6. Scrambled full supervision is capable: every action RMSE and L128 MSE is at
   most `0.10`, and all L128 spinor center rows are exact.
7. Shared adaptive is strictly better than scrambled adaptive in every action
   view and every L128 view.
8. Full supervision strictly improves scrambled negative action RMSE and both
   negative L128 rows over adaptive supervision.
9. The adaptive scrambled negative alignment is decay-only; full supervision
   data-updates both alignments.
10. Training schedules, observation systems, adaptive addresses/bits, and all
    18 evaluation schedules are distinct across seeds.

No candidate, alignment parameterization, initialization scale, mode, loss,
optimizer, threshold, seed, step count, or gate may change after this freeze.

## Commands

For each `SEED` in `7,8,9`:

```powershell
python benchmark_pure_spin8_scrambled_alignment.py --seed SEED --steps 2000 --batch-size 32 --training-length 16 --evaluation-pairs 64 --evaluation-lengths 16,64,128 --evaluation-microbatch-size 32 --modes vector_plus_adaptive_lift_bit,full_triality --candidates shared_pure_spin8,shared_latent_scrambled_alignment --device cuda --output experiments\artifacts\pure_spin8_scrambled_alignment_validation_seedSEED.json --checkpoint-directory checkpoints\pure_spin8_scrambled_alignment_validation
```

Then adjudicate with
[`validate_pure_spin8_scrambled_alignment.py`](../validate_pure_spin8_scrambled_alignment.py).

## Claim boundary

Passing would show that the correct supplied triality alignment adds transfer
beyond a matched shared latent bottleneck on this fixed synthetic task. It
would not show that the alignment was discovered from raw data, establish a
universal triality advantage, cover natural inputs or unknown initial lifts,
or compare fused production throughput with modern SSMs.
