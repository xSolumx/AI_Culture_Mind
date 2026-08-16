# Spin-Motor Direct-Readout Follow-up Protocol

Protocol frozen: **2026-08-16T20:08:40+02:00**
Parent pilot artifact:
`experiments/artifacts/spin_motor_rigid_2a5_pilot300.json`
Parent SHA-256:
`49d5d031a6e496e36e8404b666d07d49ff4af6469aa7b4a2af6f3f5c01e2d3e2`

## Why this is a separate follow-up

The frozen 22k-parameter pilot completed before this protocol was written.
Every trained candidate failed both long-context gates. The motor classifier
also failed, despite its exact semidirect-product state. Its generic MLP
readout had to rediscover the dual-quaternion pose map and did not extrapolate.

This follow-up tests the smallest correction suggested by that negative result:
**read the structured state directly**. It does not alter, overwrite, or
retroactively reinterpret the parent run.

## Candidates

Both candidates have exactly 49 trainable scalars: one quaternion and one
translation increment for each of seven tokens.

1. `direct_motor_pose_scan` (state size 8): scan unit dual quaternions and emit
   the real quaternion plus the translation extracted by
   `2 q_d conjugate(q_r)`.
2. `direct_product_pose_scan` (state size 7): scan the same signed quaternions
   but add translations independently in a global vector state. Its group is
   `Spin(3) x (R^3,+)`, not the rigid-motion semidirect product.

The direct-product model is a matched structural falsifier. It retains the
central sign but cannot rotate local translations by the current orientation.

No token is initialized from the oracle. Both models retain the random
initialization used by their underlying trainable token layers.

## Frozen data, loss, metrics, and budget

The task, seed, schedules, audits, loss, thresholds, optimizer, batch size,
training length, evaluation lengths, and 300-step budget are byte-for-byte the
same deterministic construction as the parent protocol. Only the candidate
list changes to the two direct models.

The same long-context gates apply independently to every `L64` and `L128`
split:

- signed-hemisphere accuracy at least 90%;
- joint signed-pose accuracy at least 80%.

## Preregistered expectations

- Direct motor should pass both center and joint-pose gates if optimization
  learns the seven token increments.
- Direct product should pass the center gate but fail the joint-pose gate,
  specifically through translation, not quaternion sign.
- If direct motor fails training-length pose accuracy, the 300-step optimizer
  budget is inadequate and no algebraic conclusion is permitted.
- If direct motor fits `L16` but fails long contexts, learned token increments
  are not close enough to a true representation; inspect relation residuals
  before increasing capacity.
- If direct product matches the motor, the current translation distribution is
  not actually exposing the semidirect coupling despite the symbolic audit.

As before, this single-seed, single-coordinate follow-up cannot justify a
general superiority or “breakthrough” claim.
