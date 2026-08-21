# Pure Spin(8) Noisy Continuous-Observation Validation Protocol

Protocol frozen: **2026-08-17T04:13:00+02:00**<br>
Corrected development run completed: **2026-08-17T04:17:46+02:00**

> **Pre-validation split correction, 2026-08-17T04:15:00+02:00.** The first
> runner used an unnecessary first-batch coverage overwrite. On seed 1 this
> could place a forced half-center event beside an already sampled half-center
> event, so the split audit refused to start training. No fresh model update or
> artifact was produced. The overwrite was removed, seed 0 was replayed, and
> the thresholds and architectures below were left unchanged. The corrected
> natural schedule still contains 43,751 half-center events and no excluded
> adjacency.

## Question

Can a maintained Pure Spin(8) tracker infer fresh noisy continuous local
actions online and compose a never-trained adjacent half-center relation? More
sharply, does requiring one shared Spin(8) element across `8v`, `8s+`, and
`8s-` improve identification against a parameter-near, exactly state-matched
tracker with three independent `SO(8)` actions?

## Frozen observation and target system

Each example contains fresh 12-dimensional real observations, not token IDs or
teacher coordinates. For a hidden seven-coordinate increment `u`, seed-specific
orthonormal mixing `P`, bias `b`, and independent Gaussian noise `epsilon`, the
model receives

`x = tanh(P (u + 0.15 u^3) + b) + epsilon`,

with noise standard deviation `0.01`. The chart is injective before noise on
the stated domain, nonlinear, and changes with each seed. No exact observation
is reused.

Regular increments use seven noncommuting teacher coordinates. Half-center
events use only plane 0 with angle `pi + delta`, where
`delta in [-0.25,0.25]`. Training forbids adjacent half-center events while
supervising all 24 scalars of every prefix in `(8v,8s+,8s-)`.

Evaluation forces fresh complementary observations with angles
`pi+delta, pi-delta`, whose actions compose to the nontrivial center. The
paired identity control uses `beta,-beta`. Early and late pairs share
byte-identical surrounding observations at lengths 16, 64, and 128.

## Frozen same-update cohort

All rows receive the same 800 precomputed updates, batch 32, length 16, AdamW
learning rate `3e-3`, weight decay `1e-4`, and gradient clipping at 1.

| Candidate | Parameters | State scalars | Structural role |
|---|---:|---:|---|
| shared Pure Spin(8) | 930 | 24 | one observation router and one shared faithful group element |
| independent `SO(8)^3` | 957 | 24 | capable state-matched control; separate action per triality stream |
| Mamba-2 | 931 | 160 | installed unfused parameter-near SSM reference |
| parameter-near GRU | 960 | 10 | recurrent nonlinear control |
| observation-only MLP | 949 | 0 | no-prefix-state ablation |
| state-matched GRU | 3,312 | 24 | exact-state, higher-parameter recurrent control |

The five parameter-near rows have 3.125% maximum spread. Shared Spin(8),
independent `SO(8)^3`, and the larger GRU are exactly state matched. The
independent orthogonal tracker is the decisive structural control: it can
represent the teacher without a shared-action constraint.

## Development result used only to freeze gates

Seed 0 produced action RMSE `0.01336` for shared Spin(8) and `0.03187` for the
independent tracker. Both achieved 100% center/identity classification through
L128. Shared Spin(8)'s early/late L128 post-relation MSE was `0.01190/0.02457`,
versus `0.05023/0.09652` for independent `SO(8)^3`. Long Mamba-2 and GRU
classification was chance-scale.

Development artifact:
`artifacts/pure_spin8_continuous_observation_development_seed0.json`<br>
SHA-256:
`64aed46a68bda6523690e14673000f1d916df51d62ccf99718526cf9afe67094`

## Fresh seeds and immutable gates

Run fresh seeds **1, 2, and 3**, each with its own observation chart, training
noise, evaluation noise, contexts, and initialization. Every seed must satisfy:

1. teacher, uniqueness, training exclusion, evaluation pairing, finiteness,
   parameter-spread, and state-match audits pass;
2. all 18 checkpoints rehash and strictly reload;
3. shared Spin(8) action RMSE is at most `0.03` and strictly below independent
   `SO(8)^3` action RMSE;
4. independent `SO(8)^3` action RMSE is at most `0.06`, preventing an incapable
   structural control;
5. every shared-action early/late L128 split has:
   - post-relation MSE at most `0.05`;
   - center classification, center-row correctness, and identity-row
     correctness exactly `1.0`;
   - vector-pair and spinor-negation RMSE at most `0.05`; and
   - lower post-relation MSE than independent `SO(8)^3`, Mamba-2, both GRUs,
     and observation-only MLP;
6. every independent `SO(8)^3` L128 split has center classification at least
   `0.95`, each row correctness at least `0.90`, and post-relation MSE at most
   `0.12`;
7. all shared learned center/inverse action residuals over every evaluation
   split are at most `0.04` RMSE, while the independent control residuals are
   at most `0.08`.

All three seeds must pass; no median-only rescue is allowed. No architecture,
threshold, schedule, noise level, or optimizer may change after this freeze.

## Separately frozen measured-wall continuation

Development wall time fixes a second, hardware-specific update allocation with
the shared 800-update row as target:

| Candidate | Allocated updates |
|---|---:|
| shared Pure Spin(8) | 800 |
| independent `SO(8)^3` | 636 |
| Mamba-2 | 1,134 |
| parameter-near GRU | 5,001 |
| observation-only MLP | 6,604 |
| state-matched GRU | 5,005 |

These counts are `round(800 * shared_development_wall / candidate_wall)` using
the corrected development replay.
This continuation must be reported separately because it changes data and
update counts. Its result cannot rescue a failed same-update cohort and may
not be described as hardware-independent compute matching.

## Commands

For each `SEED` in `1,2,3`:

```powershell
python benchmark_pure_spin8_continuous_observation.py --seed SEED --steps 800 --batch-size 32 --training-length 16 --evaluation-pairs 64 --evaluation-lengths 16,64,128 --evaluation-microbatch-size 32 --observation-noise-std 0.01 --device cuda --output experiments\artifacts\pure_spin8_continuous_observation_validation_seedSEED.json --checkpoint-directory checkpoints\pure_spin8_continuous_observation_validation
```

Then run
[`validate_pure_spin8_continuous_observation.py`](../validate_pure_spin8_continuous_observation.py)
over the three source artifacts.

## Outcome pointer

The untouched primary seeds and the separately frozen measured-wall
continuation both completed and were independently adjudicated. The protocol
above is unchanged; results, hashes, limitations, and interpretation are in
[`PURE_SPIN8_CONTINUOUS_OBSERVATION_RESULTS.md`](PURE_SPIN8_CONTINUOUS_OBSERVATION_RESULTS.md).

## Claim boundary

Passing establishes replicated online identification in a synthetic noisy,
injectively observed continuous system, plus a bounded shared-action advantage
against a capable state/parameter-near independent orthogonal control. It does
not establish natural-data utility, generic triality necessity, fused-Mamba or
training throughput, hardware-independent compute superiority, or a theorem
about Spin(8), Mamba-2, Dirac--Gram, or global optimality.
