# Pure Spin(8) versus Mamba-2 Triality-Transport Results

Protocol frozen: **2026-08-16T21:54:00+02:00**
Execution: **2026-08-16T21:57:06--22:00:42+02:00**

## Outcome

The maintained Pure Spin(8) v1.0 model passes every frozen gate in all three
seeds. On the center-sensitive terminal-only task, it extrapolates from the
length-2/4/8/16 curriculum to length 128 at MSE `5.81e-5`--`6.68e-5`, while
the unfused Transformers Mamba-2 and DeltaProduct references remain at
`0.132`--`0.135` and `0.144`--`0.155`.

This is approximately a **2,000x L128 MSE reduction versus Mamba-2** under the
frozen cohort. It is an algebra-matched synthetic transport result, not a claim
of generic language-model superiority.

## Architecture under test

`maintained_pure_spin8` is the actual
[`PureSpin8SSMLayer`](../pure_spin8_ssm/torch_backend.py), configured with:

- one shared `28 -> 28` bivector controller;
- the vector, positive-chiral, and negative-chiral eight-real actions;
- a 24-scalar faithful triality recurrent state;
- the batched Lie-exponential action chart;
- pure transport with no damping, drive, input normalization, or bilinear
  readout shortcut; and
- an ordered work-efficient affine/operator scan.

It has 836 trainable parameters. The configured Mamba-2 has 10,380 parameters
and 1,408 recurrent-state scalars; DeltaProduct has 12,512 parameters and 256
state scalars. These are deliberately reported rather than called matched.

## Frozen results

L128 terminal MSE:

| Seed | Pure Spin(8) | Transformers Mamba-2 | DeltaProduct | Mamba/Pure ratio |
|---:|---:|---:|---:|---:|
| 0 | `5.810e-5` | `0.132773` | `0.154836` | 2,285x |
| 1 | `6.478e-5` | `0.131957` | `0.145928` | 2,037x |
| 2 | `6.677e-5` | `0.134511` | `0.143661` | 2,014x |

Length extrapolation for Pure Spin(8):

| Seed | L16 MSE | L64 MSE | L128 MSE |
|---:|---:|---:|---:|
| 0 | `6.126e-6` | `2.809e-5` | `5.810e-5` |
| 1 | `7.162e-6` | `3.120e-5` | `6.478e-5` |
| 2 | `6.712e-6` | `3.213e-5` | `6.677e-5` |

## Central-sign test

Paired L128 examples differ only in the first token: identity versus a `2*pi`
rotation in one coordinate plane. The teacher's vector endpoints agree to at
most `2.98e-7`; both spinor endpoints are negatives to at most `4.77e-7`.

| Candidate | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| Pure Spin(8) center classification | **1.000** | **1.000** | **1.000** |
| Mamba-2 | 0.500 | 0.500 | 0.500 |
| DeltaProduct | 0.523 | 0.473 | 0.500 |

The Pure Spin(8) predicted spinor-negation RMSE is `4.66e-4`--`5.25e-4`.
This gate is why the maintained default uses all three triality views rather
than calling one `SO(8)`-type eight-vector a faithful Spin(8) state.

## Local systems observations

Median-style throughput was not preregistered, so the following are training
wall/peak-allocation diagnostics rather than kernel benchmarks:

| Candidate | training seconds per seed | peak allocated CUDA memory |
|---|---:|---:|
| Pure Spin(8) | 10.6--11.5 | 174.6 MB |
| DeltaProduct | 17.7--18.9 | 197.4 MB |
| Transformers Mamba-2 | 35.2--36.8 | 4.92 GB |

Mamba-2 and DeltaProduct are unfused transparent reference paths. The numbers
must not be generalized to optimized production kernels.

## Checkpoints and artifact

All nine learned checkpoints independently rehash and reload. Maintained-model
checkpoint hashes:

- seed 0:
  `959579ff670ca02883ff5cbaa20deae76c779325017465ce2d9b9cfd711ca209`;
- seed 1:
  `d41432877a5b1806143348a9e9765f2ed503ff75e686b4dad99a2262ba056c2e`;
- seed 2:
  `21f9b3cde0d21da09fd0849f11cf8581eae2fc938a4a37043cad1d34248dfd0a`.

Artifact:
`experiments/artifacts/pure_spin8_vs_mamba2_triality_transport1000.json`
SHA-256:
`d265e28a132c28261ae317958adfa34619c5dd0c58a0859b26e7afd653ad9876`

Protocol:
[`PURE_SPIN8_VS_MAMBA2_PROTOCOL.md`](PURE_SPIN8_VS_MAMBA2_PROTOCOL.md)
Runner:
[`benchmark_pure_spin8_vs_mamba2.py`](../benchmark_pure_spin8_vs_mamba2.py)
Model contract:
[`pure_spin8_ssm/CONTRACT.md`](../pure_spin8_ssm/CONTRACT.md)

## Claim ledger

Established:

- a separately maintained, checkpointed Pure Spin(8) v1.0 PyTorch model exists;
- its default three-view cache distinguishes all four center signatures;
- its associative work-efficient scan, recurrent stream, gradients, masking,
  cache continuation, norm bound, CUDA path, and checkpoint roundtrip pass;
- all three trained checkpoints pass the registered L128 and center gates; and
- under this frozen synthetic task/budget, it materially beats the installed
  Mamba-2 and DeltaProduct reference implementations.

Not established:

- generic sequence- or language-model superiority;
- parameter-, state-, or compute-matched superiority;
- a fused Spin(8)-versus-fused-Mamba kernel result;
- natural-task usefulness of triality coupling;
- a global optimization theorem; or
- the open unrestricted Dirac--Gram/global D-optimality theorem.

## Next falsifier

The next benchmark must leave the teacher's Lie coordinates latent. Use token
embeddings or natural continuous observations from which the local Spin(8)
increment is not supplied, retain odd/even center probes, and compare at
separately matched parameter, state, and measured-compute points. A byte-level
language cohort may follow, but only after the fused action/scan path is no
longer dominated by eager matrix exponentials.
