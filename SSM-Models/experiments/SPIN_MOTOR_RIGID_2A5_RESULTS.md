# Spin-Motor Rigid `2.A5` Results

Last reconciled: **2026-08-16T20:35:33+02:00**

## Bottom line

The rigid-motion benchmark produced one useful negative result and one narrow,
replicated positive result.

1. **End-to-end 300-step training failed.** Neither the 22k-parameter motor
   classifier nor Mamba-2, DeltaProduct, or the quaternion control passed the
   frozen long-context center or joint-pose gates. A 49-parameter direct motor
   also failed to optimize from random initialization.
2. **The exact motor state is realizable and locally identifiable.** Recovering
   each token increment from legal supervised prefix differences
   `inverse(M_prev) M_next`, without reading any evaluation relation, identifies
   all seven motors and yields 100% joint signed-pose and paired double-cover
   accuracy on every split through length 128.
3. **The identification result replicated 9/9.** Three conjugated generator
   coordinates times three independently sampled legal schedules all passed
   every gate. The matched direct-product ablation retained the center sign on
   every long split but failed translation, isolating the semidirect
   rotation-translation coupling.

This is a finite deterministic system-identification result under
every-prefix pose supervision. It is not a general SSM breakthrough, not an
end-to-end learning win, and not evidence on noisy or natural data.

## Task and falsifier

The vocabulary is `a, b, b_inverse, e, tx, ty, tz`. The three translation
tokens move by 0.25 along a body-frame axis, so

`t_next = t + R(q) delta_t`.

Training length is 16 and contains zero occurrences of `a^2`, `b^3`, or
`(ab)^5`; audited legal words still reach all 120 binary-icosahedral states.
Each evaluation central word is paired with an equal-width identity block in
the same context. The exact signed quaternions are antipodal while the physical
rotation and translation are identical.

The exact `SE(3)` quotient oracle therefore scores 100% physical pose but only
50% signed rotation and 0% paired antipodal recovery. The task really contains
one double-cover bit that an ordinary pose quotient cannot represent.

## Frozen 22k-parameter pilot

Execution: **20:04:53--20:05:53 +02:00** on an RTX 2070 SUPER. All candidates
were within a 0.35% parameter range. Mamba-2 used the Transformers unfused
Windows fallback; DeltaProduct was the pinned unfused equation reference.

| candidate | parameters | final train loss | mean joint pose L16 | mean joint pose L128 | long gates |
|---|---:|---:|---:|---:|---|
| quaternion scan | 22,111 | 0.0820 | 0.74% | 0.086% | fail/fail |
| motor classifier | 22,123 | 0.0800 | 1.00% | 0.123% | fail/fail |
| Transformers Mamba-2 | 22,132 | 0.1248 | 0.091% | 0.019% | fail/fail |
| DeltaProduct reference | 22,056 | 0.1711 | 0.216% | 0.018% | fail/fail |

All four also have essentially zero mean paired double-cover pose accuracy.
The low train losses do not imply geometric extrapolation: translation and
strict 15-degree rotation accuracy collapse outside the length-16 training
distribution. The motor's exact algebra is therefore insufficient when hidden
behind a generic learned decoder.

Artifact:
`experiments/artifacts/spin_motor_rigid_2a5_pilot300.json`
SHA-256:
`49d5d031a6e496e36e8404b666d07d49ff4af6469aa7b4a2af6f3f5c01e2d3e2`

The four checkpoint hashes, in table order, are:

- `4a1cc24a452d10fbd4bc93ab62874b69c98cddf62b1d3ef257401f5d4223a648`;
- `ee0d8c7e96443a3c9bec915412fbc7a6cc65a09e31a9bda99f12278e38bebd92`;
- `cbc79d5536fe67a898540fa39bd2048d632eab7bb9263ac042e01e4bb0ee281f`;
- `ef884b096fa78430a2046b5164522476401726e1975e2fae2d71885b3205b1a9`.

## Direct-state follow-up

Execution: **20:09:15--20:09:52 +02:00**. Both models have 49 parameters.

- The direct-product state `Spin(3) x R^3` achieved 100% signed-hemisphere
  accuracy on every L64/L128 split, but mean L128 joint-pose accuracy was only
  0.292%. Its translations do not rotate with the body frame.
- The randomly initialized direct motor did not fit the rotation generators in
  300 updates and failed both gates. This is an optimization failure, not a
  counterexample to motor realizability.

Artifact:
`experiments/artifacts/spin_motor_direct_readout_pilot300.json`
SHA-256:
`7a364b61ba51666db65f0ced909fc78d81855582fd14e9dd5e598d2d4d3ab1f2`

Direct-product checkpoint:
`8d2528f1504c821ff1e59ba4915cb87ffce28b59d91eec116094d2c369163cd3`
Direct-motor checkpoint:
`9f1fbaade8d7dd216b44319fc3d7f6503689c60b11967e1f926b76f25c6291c1`

## Local transition identification

For each supervised legal prefix, the estimator computes

`q_delta = conjugate(q_prev) q_next`

and

`t_delta = R(q_prev)^T (t_next - t_prev)`.

It averages these repeated observations by token and installs the seven
identified increments into `DirectMotorPoseTracker`. The estimator reads no
evaluation inputs or targets and the training audit confirms zero forbidden
relation occurrences.

The coordinate-`e`, seed-0 screening run passed every preregistered gate:

- maximum exact token quaternion error: 0 degrees at reported precision;
- maximum exact token translation error: below `1e-6`;
- minimum joint signed-pose accuracy over all splits: 100%;
- minimum paired double-cover pose accuracy: 100%;
- worst long-split mean translation error: `6.67e-7`.

Artifact:
`experiments/artifacts/spin_motor_identified_e_seed0.json`
SHA-256:
`df132600d8be86505a4e5156b161a7e2ee33ce84dc4dba1dae26d3207653c62b`
Checkpoint SHA-256:
`645ba46a373e012f14100998f7d825283a3d0326da8811aa1bdd51eac59cf9ab`

## Three-by-three replication

Execution: **20:15:58--20:17:33 +02:00**. The nine runs cover coordinates
`e`, `a`, and `b` and seeds 0, 1, and 2. All nine training schedule hashes are
unique. No run is averaged away: every run passes every individual gate.

Aggregate worst cases across all nine runs and all 162 evaluation splits:

- minimum joint signed-pose accuracy: **100%**;
- minimum paired double-cover pose accuracy: **100%**;
- maximum fitted token quaternion error: **0 degrees** at reported precision;
- maximum fitted token translation error: **2.66e-9**;
- maximum mean signed rotation error: **0.00433 degrees**;
- maximum mean translation error: **7.74e-7**;
- maximum paired mean translation difference: **4.60e-7**.

Artifact:
`experiments/artifacts/spin_motor_identification_replication_3x3.json`
SHA-256:
`97ffc994889278b21da7482ff49a597d9799f4ac38472e43b459465468a00aa5`

The artifact records all nine training/evaluation schedule hashes and all nine
checkpoint hashes.

## Signed-pose noise audit

Execution: **20:30:30--20:31:20 +02:00**. The unchanged arithmetic-mean
identifier was evaluated on the coordinate-`e`, seed-0 legal training schedule
with 76,800 independently corrupted signed prefix poses. Five independent
noise seeds were used at each of four frozen tiers. Evaluation remained clean.

| pose-noise tier | runs | worst token rotation | worst token translation | worst long rotation | worst long translation | minimum joint / paired accuracy |
|---|---:|---:|---:|---:|---:|---:|
| clean | 5 | 0 degrees | `1.84e-9` | 0.00237 degrees | `6.67e-7` | 100% / 100% |
| 1 degree, 0.01 translation | 5 | 0.0271 degrees | `3.65e-4` | 0.163 degrees | 0.00542 | 100% / 100% |
| 5 degrees, 0.05 translation | 5 | 0.1346 degrees | 0.00184 | 0.828 degrees | 0.0272 | 100% / 100% |
| 15 degrees, 0.15 translation | 5 | 0.3958 degrees | 0.00841 | 2.468 degrees | 0.1026 | 46.875% / 46.875% |

Clean, low, and medium noise pass every frozen gate in all 15 runs. All five
high-noise runs retain the central sign but fail the joint-pose and paired-pose
gates. This is a useful stress boundary, not a tuned threshold: the estimator
and sample schedule were unchanged across tiers.

The noise was applied to the **signed** quaternion without independent
antipodal flips. This result therefore does not recover a double-cover sign
from physical `SO(3)` pose observations, where that sign is absent. It also
does not test correlated drift, outliers, sparse supervision, or continuous
token encoders.

Artifact:
`experiments/artifacts/spin_motor_noisy_identification_4tiers_5seeds.json`
SHA-256:
`2adc41d821e110e8b8e05624a0587c2aaa04492c2c47f8eabd7e35251020a6d5`

All 20 checkpoint hashes are recorded in the artifact and were independently
replayed from disk with zero mismatches.

## What is established

### Exact/structural

- Unit dual quaternions form the required associative signed rigid-motion
  state, and the direct readout is the real quaternion plus extracted
  translation.
- The motor exactly realizes the benchmark transition law; the matched direct
  product does not realize body-frame translation.
- With signed pose at adjacent prefixes, a right-composed token motor is
  algebraically observable as a local group difference.

### Empirical

- Blind 300-step end-to-end optimization is inadequate for this strict pose
  benchmark, including for the motor candidate.
- The local estimator identifies the deterministic token transitions from
  relation-free data and extrapolates perfectly under all tested coordinates,
  schedules, relations, and lengths.

### Open

- robust identification under correlated drift, outliers, and quaternion-sign
  corruption; independent metric signed-pose noise is now bounded above;
- a learned continuous encoder `x_t -> Lie algebra motor increment`;
- final-only or sparse supervision, where local differences are unavailable;
- whether a transition-consistency auxiliary loss makes end-to-end motor
  training competitive without explicit identification;
- state-matched and fused-kernel comparisons on Linux/CUDA;
- natural sequence tasks where the motor group is an appropriate latent state.

## External-source boundary

The generic ordered Lie-group-product idea is not new; the closest reviewed
sequence-model precedent is the Path Development Network
<https://arxiv.org/abs/2204.00740>. Mamba-2 is
<https://arxiv.org/abs/2405.21060>. The reviewed DeltaProduct source commit is
`d62241a81d07aa32b1b65e7d17377f6a7cd0a5d8`.

The official Fixed-Point RNN source was found and reviewed at commit
`0cc1e3c520423e02674c20333fcf9dfa46b7d204`, but it pins Python below 3.12,
Torch 2.4.1, Triton 3.0, Mamba-SSM, and custom scan/causal-convolution paths.
This Windows/Python-3.12 checkout therefore excludes it instead of presenting
an inexact reimplementation as FP-Mamba.
