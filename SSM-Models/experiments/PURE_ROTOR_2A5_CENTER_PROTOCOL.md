# Pure Rotor / Spin quaternion / Mamba-2: center-sensitive `2.A5` protocol

Protocol frozen **2026-08-16T18:25:33+02:00** (`Africa/Johannesburg`) before
the first multi-seed training artifact from
[`benchmark_pure_rotor_2a5.py`](../benchmark_pure_rotor_2a5.py).

**Completed result:** the authoritative three-seed replay finished at
**2026-08-16T18:39:55+02:00**. The Spin quaternion scan passes the registered
center-margin gate in every seed and tested retention length; the full metrics,
limitations, artifact hash, and follow-up are in
[`PURE_ROTOR_2A5_CENTER_PILOT300_RESULTS.md`](PURE_ROTOR_2A5_CENTER_PILOT300_RESULTS.md).

## Question

Can a learned sequence model preserve the binary central sign in `2.A5`, not
merely the projected `A5 < SO(3)` state? Does an explicit quaternion
composition scan provide the missing inductive bias relative to the maintained
Pure Rotor sandwich transport, its identity ablation, and parameter-near
Mamba-2?

This is a mechanism benchmark. It is not a language-model benchmark and it
does not turn a result for Transformers Mamba-2 into a theorem about diagonal
SSMs.

## Exact task

The exact 120-element table and quaternion generators satisfy

\[
a^2=b^3=(ab)^5=z,\qquad z^2=e.
\]

Inputs are `(a, b, b_inverse)` and every prefix is labelled by its full
120-state binary product. Training forbids `a -> a`, but the legal training
language reaches all 120 states within length 10. The fixed schedule injects a
legal witness for every state, and the artifact must record all 60 projective
labels and both center-bit labels as observed.

Evaluation is paired:

```text
central word:  context_left + a a         + context_right
identity word: context_left + b b_inverse + context_right
```

After the two-token block, the words have identical projected A5 trajectories
and exact `2.A5` targets differing by `z`. Contexts contain no additional
`a,a`. An `early` block tests retention; a `late` block tests acquisition.

## Metrics

Each split reports:

- exact 120-state accuracy;
- projective 60-state accuracy;
- center-bit accuracy;
- direct target-versus-central-partner logit-margin accuracy, with ties worth
  one half;
- target probability conditioned on the central pair;
- paired final exact accuracy;
- paired structural separation: predictions have the same projection and
  opposite center bits.

The margin metric isolates center knowledge even when the projected group state
is wrong. Chance is 50% for center-bit and central-margin accuracy, and
`1/120` for exact state accuracy.

## Candidates

| Candidate | Persistent mechanism | Parameters |
|---|---|---:|
| Pure Rotor v2.1 | maintained Cl(3) state, rotor sandwich transport | 29,370 |
| Identity ablation | identical model, rotation fixed to identity | 29,370 |
| Spin quaternion scan | 8 learned quaternion lanes, Hamilton prefix product | 29,592 |
| Mamba-2 | Transformers Mamba-2, state size 24 | 29,300 |

The maximum parameter gap is below 1%. States are intentionally not described
as matched: the models retain architecture-specific caches and representations.
The quaternion candidate is a sign-sensitive custom layer, not the canonical
Pure Rotor model.

The Spin quaternion recurrence has both recurrent and Hillis--Steele parallel
prefix implementations. Forward and gradient equivalence are regression
tested. Its mathematical update distinguishes `q` from `-q`; rotor sandwiching
does not, because both signs induce the same conjugation.

## Analytic controls

- `exact_table_oracle`: must score 100% everywhere.
- `float64_quaternion_oracle`: must score 100% exact top-1 through length 128.
- `projective_a5_oracle`: must score 100% projective, 50% exact/center, and 50%
  central-margin accuracy on paired post-relation positions.

Failure of any oracle contract invalidates the run before model interpretation.

## Frozen pilot

- seeds: `0,1,2`;
- 300 AdamW updates per candidate;
- train batch 16, length 16;
- evaluation lengths `2,16,64,128`;
- two batches of 32 relation pairs per split;
- early and late relations, except the identical length-2 case;
- checkpoints and SHA-256 hashes for every trained candidate;
- CUDA on the recorded local device;
- Transformers fallback Mamba-2, with no fused-kernel throughput claim.

## Predeclared interpretation

1. A center-sensitive success requires central-margin accuracy above 75% in
   all three seeds, not merely a single exact-accuracy spike.
2. Acquisition and retention are separate. Passing `late_L16` while failing
   `early_L64/L128` is a short-relation fit, not persistent binary tracking.
3. A Spin quaternion advantage must exceed both Pure Rotor and identity on the
   early long splits in every seed. Otherwise the custom lane has not supplied
   replicated evidence.
4. Projective accuracy without center accuracy is explicitly a quotient-only
   result.
5. The exact quaternion oracle proves representability, not learnability.
6. Training loss, throughput, and memory are backend-specific diagnostics.
7. Any post-pilot architecture change requires a new protocol or an explicit
   exploratory label.

## Command

```powershell
python SSM-Models\benchmark_pure_rotor_2a5.py --device cuda --steps 300 `
  --batch-size 16 --training-length 16 `
  --validation-batches 2 --validation-pairs-per-batch 32 `
  --evaluation-microbatch-size 16 --evaluation-lengths 2,16,64,128 `
  --seeds 0,1,2 `
  --checkpoint-directory SSM-Models\checkpoints\pure_rotor_2a5_center_pilot `
  --output SSM-Models\experiments\artifacts\pure_rotor_2a5_center_pilot300.json `
  --quiet-report
```
