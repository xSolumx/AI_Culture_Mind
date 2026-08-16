# Pure Rotor v2.1 / Mamba-2 A5 screen: 1,000 updates

Completed **2026-08-16T17:02:30+02:00** (`Africa/Johannesburg`, UTC+02:00).

## Result status

This completed three-seed screen finds a stronger but variable short-composition
signal for the canonical Pure Rotor model. It does **not** find long-horizon
group tracking: by lengths 64 and 128 all candidates are near the 1.67% chance
level. It is therefore evidence for a next persistence/optimization diagnostic,
not a claim that Pure Rotor has learned an exact A5 representation or beaten
Mamba-2 generally.

Every seed uses a precomputed and SHA-256-recorded schedule. All 60 A5 output
labels occur during training; the only absent input bigram is `a -> b`, every
evaluation word contains that pair, and every A5 label has a legal training
language witness by length 12. This is parameter-near (24,990 Pure Rotor,
24,990 identity, 25,604 Mamba-2), not state-matched.

## Final forced-pair accuracy

Values are final-position percentages. Training has 1,000 AdamW updates at
length 16 and batch 16; each evaluation has four logical batches of 64, run in
microbatches of 16.

| Seed | Candidate | L2 | L16 | L64 | L128 |
|---:|---|---:|---:|---:|---:|
| 0 | Pure Rotor | 100.00 | 66.02 | 7.81 | 1.95 |
| 0 | Identity transport | 0.00 | 1.95 | 1.95 | 1.17 |
| 0 | Mamba-2 | 0.00 | 8.20 | 2.73 | 1.56 |
| 1 | Pure Rotor | 100.00 | 12.11 | 0.39 | 1.17 |
| 1 | Identity transport | 0.00 | 5.47 | 1.95 | 1.17 |
| 1 | Mamba-2 | 0.00 | 8.20 | 0.78 | 0.78 |
| 2 | Pure Rotor | 0.00 | 8.98 | 1.17 | 0.39 |
| 2 | Identity transport | 100.00 | 5.47 | 1.95 | 1.95 |
| 2 | Mamba-2 | 0.00 | 7.81 | 2.34 | 1.95 |

| Candidate | L2 mean ± sample SD | L16 mean ± sample SD | L64 mean | L128 mean |
|---|---:|---:|---:|---:|
| Pure Rotor | 66.67 ± 57.74 | 29.04 ± 32.06 | 3.12 | 1.17 |
| Identity transport | 33.33 ± 57.74 | 4.30 ± 2.03 | 1.95 | 1.43 |
| Mamba-2 | 0.00 ± 0.00 | 8.07 ± 0.23 | 1.95 | 1.43 |

## Interpretation

- The 200-update seed-0 event was not a reliable claim. Increasing the budget
  turns it into an informative but incomplete pattern: Pure Rotor passes the
  unseen direct pair in two of three seeds and has the highest mean L16 score,
  but variation is large and the identity control also passes L2 in one seed.
- Mamba-2 reaches lower training loss (mean 0.37 versus 0.69 for Pure Rotor),
  yet it is 0/3 on the forced length-2 pair. This is an observation about this
  small Transformers Mamba-2 configuration, not an application of a theorem
  about diagonal SSMs to Mamba-2.
- Long-horizon retention is the hard failure: no candidate clears even a weak
  L64 gate. A short-pair fit cannot be promoted to compositional tracking.
- The recorded throughput/memory values are backend-specific. Transformers
  Mamba-2 runs at roughly 4,904 training tokens/s and 714 MiB peak allocation;
  Pure Rotor runs at roughly 2,356 tokens/s and 30 MiB. `mamba_ssm` was not
  importable, so this is not a fused-kernel comparison.

## Artifacts

- [`pure_rotor_a5_mamba2_budget1000_coverage_batch16_seed0.json`](artifacts/pure_rotor_a5_mamba2_budget1000_coverage_batch16_seed0.json),
  SHA-256 `8d4a9a70650511fcc241356edd92b18abb34394cdc13f954c78805a98e5dd194`.
- [`pure_rotor_a5_mamba2_budget1000_coverage_batch16_seeds1_2.json`](artifacts/pure_rotor_a5_mamba2_budget1000_coverage_batch16_seeds1_2.json),
  SHA-256 `73d30a8aa0af29a1e5413022a7afce726becc8c7e6fd8ac3b9e3371f20a244b5`.

Each result includes its model checkpoint SHA-256. The runner, task contract,
and lower-budget falsification are in
[`PURE_ROTOR_A5_MAMBA2_PROTOCOL.md`](PURE_ROTOR_A5_MAMBA2_PROTOCOL.md) and
[`PURE_ROTOR_A5_MAMBA2_PILOT200_RESULTS.md`](PURE_ROTOR_A5_MAMBA2_PILOT200_RESULTS.md).

## Most promising next experiment

Do not widen the state or claim an algebraic win yet. First separate inadequate
state retention from inadequate relation learning:

1. Measure each trained Pure Rotor layer's decay distribution and recurrent
   state drift on the forced-pair L16/L64 trajectories.
2. Run a length-curriculum control with the same total updates and all existing
   candidates. It must retain the same missing-pair and coverage audits.
3. Only if a curriculum improves L64/L128 in replicated seeds, introduce and
   ablate a generic near-unit transport memory lane. It must preserve the
   affine-scan, streaming, stability, and identity-ablation contracts.

This order turns the current long-horizon failure into a direct falsifier for
the proposed persistence upgrade rather than a post hoc architectural story.
