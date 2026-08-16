# Spin / DeltaProduct / Mamba-2: multi-relation `2.A5` protocol

Protocol frozen **2026-08-16T19:20:57+02:00** (`Africa/Johannesburg`), before
the first optimizer step from
[`benchmark_spin_multirelation_2a5.py`](../benchmark_spin_multirelation_2a5.py).

**Status at freeze:** exact task/oracle/model-contract regression tests pass;
there is no training result yet. This document preregisters a one-initialization
coordinate-robustness pilot, not a multi-seed replication.

**Completed result:** the frozen pilot finished at
**2026-08-16T19:29:51+02:00**. The Spin quaternion scan passes both registered
gates in all 18 long-retention splits; no other learned candidate passes the
center-retention gate. See
[`SPIN_2A5_MULTIRELATION_RESULTS.md`](SPIN_2A5_MULTIRELATION_RESULTS.md).
The original freeze statement above is retained as provenance.

## Question

Can a learned sequence model infer and preserve the central sign of the binary
icosahedral group when **all three** central presentation words are absent from
training,

\[
a^2=b^3=(ab)^5=z,\qquad z^2=e,
\]

and does the result survive inner-conjugate changes of generators? In
particular, compare an explicit associative Spin quaternion product with the
maintained Pure Rotor sandwich transport, its identity ablation, Transformers
Mamba-2, and a modern DeltaProduct state-tracking architecture.

This is a finite-group mechanism test. It is not a language-model benchmark, a
state-matched benchmark, a theorem about SSM families, or an official fused
DeltaProduct systems result.

## Exact task and split

The fixed 120-state exact multiplication table has SHA-256
`ee26aff5719e54cf28eccc7a0259c79eeb27c7e06bb18134bf7ca910773f58c9`.
Inputs are `(a, b, b_inverse, e)`, where `e` is an explicit identity token.
Every prefix is labelled by its full `2.A5` product.

Training forbids these exact token substrings:

| Key | Central block | Equal-length identity control |
|---|---|---|
| `a_squared` | `a a` | `e e` |
| `b_cubed` | `b b b` | `e e e` |
| `ab_fifth` | `a b a b a b a b a b` | `e` repeated 10 times |

The relation-free language is not impoverished: exact breadth-first search
reaches every one of the 120 binary states, with maximum shortest-witness
length 10. The full shortest-length distribution is
`{0:1, 1:3, 2:6, 3:11, 4:15, 5:19, 6:17, 7:18, 8:18, 9:8, 10:4}`.
The fixed length-16 training schedule injects one canonical legal witness for
every state. The audit must confirm 120 binary targets, all 60 projective
targets, both center bits, and zero occurrence of every withheld substring.

## Conjugated-coordinate control

Repeat the task for `g = e, a, b` after replacing each generator by
`g^-1 x g`. The exact input-element indices are frozen as:

| Coordinate | `(a, b, b_inverse, e)` indices |
|---|---|
| `e` | `(1, 2, 4, 0)` |
| `a` | `(1, 17, 19, 0)` |
| `b` | `(24, 2, 4, 0)` |

The input token schedules, forced relation locations, contexts, candidate
initialization seed, and optimizer schedule are byte-identical across the
three coordinates. Exact target indices change according to conjugation. This
is a paired sensitivity control, but one initialization seed cannot establish
seed-level replication.

## Evaluation

For every relation, pair its central block with its identity block inside an
identical audited context. The center branch must contain exactly one occurrence
of the designated relation and none of the other two; the identity branch must
contain none. Score only positions at and after block completion.

- `early` at lengths 16, 64, and 128 tests retention;
- `late` at lengths 16, 64, and 128 tests acquisition;
- two batches of 32 word pairs per split;
- deterministic schedule namespace per initialization seed, relation, length,
  and position.

Every scored pair must pass exact central-partner and equal-projective-label
audits before metrics are interpreted.

Metrics are exact 120-state accuracy, projective 60-state accuracy, center-bit
accuracy, target-versus-central-partner margin accuracy (ties worth one half),
target probability conditional on the central pair, paired final exact
accuracy, and paired final structural center separation.

## Learned candidates

| Candidate | Persistent mechanism | Parameters | Real state scalars disclosed |
|---|---|---:|---:|
| Pure Rotor v2.1 | 3 maintained Cl(3) sandwich-transport layers | 29,370 | 216 |
| Identity ablation | same network, rotor angle fixed to zero | 29,370 | 216 |
| Spin quaternion scan | 8 Hamilton-product lanes | 29,624 | 32 |
| Transformers Mamba-2 | 3-layer fallback implementation | 29,300 | architecture-specific |
| DeltaProduct reference | 4 heads, 4 delta/Householder updates per token | 29,288 | 256 |

The maximum parameter gap is 1.13%. State sizes are deliberately **not**
matched.

The DeltaProduct reference implements the official state-tracking equation

\[
S \leftarrow S + k\,[\beta(v-k^\top S)]^\top,
\]

with normalized keys, four updates per token, `beta in (0,2)`, no short
convolution, no output gate, and no forget gate. Per-token updates are composed
as matrix-affine maps and scanned associatively. Forward, recurrent/parallel,
gradient, padding, streaming-cache, and CUDA contracts are regression tested.
It is pinned to official DeltaProduct commit
`d62241a81d07aa32b1b65e7d17377f6a7cd0a5d8`, but is an unfused PyTorch
reference because FLA/Triton is unavailable in this Windows runtime. Therefore
its throughput is **not** an official DeltaProduct kernel result. See the
[DeltaProduct paper](https://arxiv.org/abs/2502.10297) and
[official repository](https://github.com/automl/DeltaProduct).

Transformers Mamba-2 is retained as the widely used diagonal/selective SSM
control. The unavailable `mamba_ssm` fused extension is recorded, and no
fused-kernel throughput claim is allowed.

## Analytic/mechanistic controls

- `exact_table_oracle`: 100% exact throughout;
- `float64_quaternion_oracle`: 100% exact throughout;
- `projective_a5_oracle`: 100% projective but 50% exact, center-bit, and
  central-margin accuracy on balanced pairs;
- `exact_regular_pd_oracle`: exact 120-complex-state regular action with
  unit diagonal, 100% exact throughout.

The last control is the exact representability ceiling for the permutation-
diagonal mechanism reviewed at IBM PD-SSM commit
`8682e78101be84f67ceb64702855e5d9e820f7d2`. It is not a trained official-model
reproduction and has 240 real-equivalent state scalars. Prior learned PD-SSM
experiments remain documented separately. See the
[PD-SSM paper](https://arxiv.org/abs/2509.22284) and
[official repository](https://github.com/IBM/expressive-sparse-state-space-model).

Failure of any exact oracle or split audit invalidates the run.

## Frozen pilot budget

- initialization seed: `0`;
- conjugated coordinates: `e,a,b`;
- 300 AdamW updates per learned candidate and coordinate;
- batch 16, training length 16;
- learning rate `3e-3`, weight decay `0.01`, gradient clipping `1.0`;
- Schur-parallel Pure Rotor scan;
- parallel Hamilton-product scan;
- parallel matrix-affine DeltaProduct reference scan;
- CUDA on the recorded local device;
- 15 checkpoints, each with SHA-256.

## Predeclared interpretation

1. All three relation families are independent falsifiers. Report each; do not
   replace them by a pooled average.
2. A candidate passes the **coordinate-robust center-retention pilot gate** only
   if central-margin accuracy is above 75% at every `early_L64` and
   `early_L128` split, for all three relations and all three coordinates.
3. `late` success with `early` failure is acquisition without stable retention.
4. Projective success with chance center metrics is quotient-only tracking.
5. A declared learned-model winner must have the highest early-L64/L128 exact
   accuracy on every relation and coordinate; otherwise report mixed results,
   not a winner.
6. The explicit Spin scan is allowed to win by inductive bias; that would not
   make it a generic learned SSM. Conversely, failure would concern
   learnability under this budget, not exact representability.
7. DeltaProduct results are architecture-level empirical evidence only. They
   do not reproduce official fused kernels or source-scale hyperparameters.
8. Because seed 0 is the only initialization, passing the pilot demands a new
   frozen multi-seed validation cohort before a replicated claim.
9. Throughput and peak memory are local-backend diagnostics, not systems
   rankings.
10. No post-result architecture or threshold change may be folded into this
    artifact; it requires a new protocol or an explicit exploratory label.

## Frozen command

```powershell
python SSM-Models\benchmark_spin_multirelation_2a5.py --device cuda `
  --steps 300 --batch-size 16 --training-length 16 `
  --validation-batches 2 --validation-pairs-per-batch 32 `
  --evaluation-microbatch-size 16 --evaluation-lengths 16,64,128 `
  --seeds 0 --coordinates e,a,b `
  --candidates pure_rotor,identity_rotation_ablation,spin_quaternion_scan,mamba2_transformers,delta_product_reference `
  --rotor-scan-mode schur_parallel --quaternion-scan-mode parallel `
  --delta-scan-mode parallel `
  --checkpoint-directory SSM-Models\checkpoints\spin_2a5_multirelation_pilot `
  --output SSM-Models\experiments\artifacts\spin_2a5_multirelation_pilot300.json `
  --quiet-report
```
