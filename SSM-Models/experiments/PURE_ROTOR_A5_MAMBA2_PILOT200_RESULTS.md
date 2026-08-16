# Pure Rotor v2.1 / Mamba-2 A5 pilot: 200 updates

Completed **2026-08-16T16:43:51+02:00** (`Africa/Johannesburg`, UTC+02:00).

## Status

This is a completed three-seed, parameter-near **pilot**, not a completed
training-budget cohort. It is evidence against a reliable 200-update
short-composition advantage for the present Pure Rotor configuration. It does
not establish an SSM expressivity theorem, a language-model conclusion, or a
systems-throughput ranking.

All candidates train on the same hashed schedule within each seed. The
schedule excludes `a -> b`, every evaluation word contains it, all 60 A5 labels
occur in training, and an exact bounded-language audit finds a legal witness
for every label by length 12 (training length 16). This repairs the prior
random 200-step pilot, whose sample happened to cover only 59 labels and is
retained only in the execution log as non-comparable provenance.

## Configuration

- A5 presentation: `a = 13254`, `b = 24315`, with
  `a^2 = b^3 = (ab)^5 = e` and generated subgroup order 60.
- Inputs: `(a, b, b_inverse)`; only adjacent `a -> b` is excluded in training.
- 200 AdamW updates, batch 16, length 16, seeds 0–2.
- Four logical validation batches of 64 at lengths 2, 16, 64, and 128;
  evaluation microbatch size 16.
- Pure Rotor / identity / Mamba-2 raw parameters: 24,990 / 24,990 / 25,604.
  The identity ablation has 1,242 disabled rotor-controller parameters.
- Hardware: RTX 2070 SUPER. Mamba-2 uses Hugging Face Transformers without
  importable `mamba_ssm`; no native fused-kernel conclusion is permitted.

## Paired final forced-pair accuracy

Values are final-position percentages. Chance is 1.67% over 60 A5 labels.

| Seed | Candidate | L2 | L16 | L64 | L128 |
|---:|---|---:|---:|---:|---:|
| 0 | Pure Rotor | 100.00 | 19.53 | 2.34 | 3.12 |
| 0 | Identity transport | 0.00 | 3.91 | 1.95 | 2.34 |
| 0 | Mamba-2 | 0.00 | 5.47 | 1.56 | 2.73 |
| 1 | Pure Rotor | 0.00 | 2.34 | 2.34 | 1.56 |
| 1 | Identity transport | 0.00 | 3.91 | 1.95 | 2.73 |
| 1 | Mamba-2 | 0.00 | 4.69 | 3.52 | 1.56 |
| 2 | Pure Rotor | 0.00 | 2.73 | 1.95 | 1.95 |
| 2 | Identity transport | 0.00 | 1.17 | 0.39 | 1.56 |
| 2 | Mamba-2 | 0.00 | 4.69 | 2.34 | 0.78 |

Pure Rotor finds the missing length-2 composition only in seed 0. Its
three-seed L2 mean is 33.33% with a 57.74-point sample standard deviation;
identity and Mamba-2 are 0/3. At L64 and L128 every row is near chance. The
seed-0 event is therefore a lead for budget calibration, not a replicated
architecture result.

Mamba-2 has lower final training loss in every seed, but that does not transfer
to the forced unseen pair at this budget. Pure Rotor is about 2,329 training
tokens/s at 30 MiB peak CUDA allocation; Transformers Mamba-2 is about 4,559
tokens/s at 714 MiB. Those recorded numbers describe this small, unfused
implementation and batch schedule only.

## Artifacts

- [`pure_rotor_a5_mamba2_pilot200_coverage_batch16_seed0.json`](artifacts/pure_rotor_a5_mamba2_pilot200_coverage_batch16_seed0.json),
  SHA-256 `d56e2909af526800ee5ef3405b508b5cdaa7a0b4d7c29b85c4cb3f263ca9e651`.
- [`pure_rotor_a5_mamba2_pilot200_coverage_batch16_seeds1_2.json`](artifacts/pure_rotor_a5_mamba2_pilot200_coverage_batch16_seeds1_2.json),
  SHA-256 `0972f165f960ac4b751f73bfa64b601250f5da684ec0f62ec3ceb815988a6c4b`.

The checkpoint SHA-256 values are recorded inside each JSON result.

## Next falsifier

Run a seed-0 1,000-update budget calibration with the same coverage contract.
If it does not produce high L2 and L16 forced-pair accuracy, the immediate
rotor-transport hypothesis fails for this canonical model. If it does, rerun
seeds 1 and 2 before judging reliability, then assess L64/L128 separately.

See [`PURE_ROTOR_A5_MAMBA2_PROTOCOL.md`](PURE_ROTOR_A5_MAMBA2_PROTOCOL.md) for
the predeclared task contract and
[`PURE_ROTOR_A5_MAMBA2_EXECUTION_LOG.md`](PURE_ROTOR_A5_MAMBA2_EXECUTION_LOG.md)
for the interrupted full-batch run.
