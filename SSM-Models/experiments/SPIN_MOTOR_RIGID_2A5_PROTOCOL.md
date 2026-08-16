# Spin-Motor Rigid `2.A5` Protocol

Protocol frozen: **2026-08-16T20:03:22+02:00**
Runner: `SSM-Models/benchmark_spin_motor_rigid_2a5.py`
Status at freeze: implementation and contract smoke tests pass; the reported
300-step run had not started.

## Question

Can an associative unit-dual-quaternion state retain the central sign of
`Spin(3)` while also tracking non-commuting rigid translations, and does that
inductive bias outperform parameter-near sequence models on unseen central
relations beyond the training length?

This is a deliberately narrower claim than “a new general SSM.” Generic
Lie-group path development already constructs sequence features as ordered
products of exponentials. The candidate here is a **double-cover Euclidean
path-development specialization** whose state distinguishes `m` from `-m`
even though both induce the same `SE(3)` action.

## Exact task

The seven input tokens are

`a, b, b_inverse, e, tx, ty, tz`.

- `a`, `b`, and `b_inverse` are exact binary-icosahedral rotations.
- `e` is the identity.
- `tx`, `ty`, and `tz` translate by `0.25` along a local/body axis.
- A token is right-composed with the current pose. Translation therefore obeys
  `t_next = t + R(q) delta_t`; rotations and translations do not commute.
- Every prefix target is the seven-vector `[w,x,y,z,tx,ty,tz]`. Quaternion sign
  is retained, not canonicalized modulo `q ~ -q`.

Training excludes every occurrence of

- `a^2 = z`,
- `b^3 = z`, and
- `(ab)^5 = z`,

where `z=-1` is the non-trivial center of `2.A5`. The legal-language breadth
first audit must nevertheless reach all 120 binary rotation states. Canonical
legal witnesses are injected into the otherwise random frozen schedule.

For evaluation, one held-out central word is inserted early or late and paired
with an equal-width identity block in an identical context. After that block:

- exact signed quaternions are antipodal;
- projective rotations are equal;
- translations are equal; and
- the full physical `SE(3)` poses are equal.

The quotient oracle is therefore a direct falsifier: it has perfect physical
pose information but cannot score both signed targets in a pair.

## Frozen pilot budget

- generator coordinate: `e` only;
- seed: `0` only;
- train steps: `300`;
- batch size: `16`;
- train length: `16`;
- validation: `2` batches of `16` pairs per split;
- evaluation lengths: `16`, `64`, `128`;
- relation positions: early and late;
- total evaluation splits: `3 relations x 3 lengths x 2 positions = 18`;
- optimizer: AdamW, learning rate `3e-3`, weight decay `0.01`;
- pre-clip gradient norm threshold: `1.0`;
- loss: signed quaternion alignment + translation MSE + `0.01` quaternion
  output-norm penalty;
- scan mode: parallel for Spin, motor, and DeltaProduct;
- device: local CUDA if available.

This is a screening budget, not a converged reproduction of the much larger
training budgets in the external state-tracking literature.

## Trained candidates

| candidate | trainable parameters | recurrent cache scalars | status |
|---|---:|---:|---|
| Spin quaternion scan, 8 lanes | 22,111 | 32 | local experimental layer |
| Spin motor scan, 4 lanes | 22,123 | 32 | local experimental layer |
| Transformers Mamba-2 | 22,132 | 2,304 | released implementation, unfused Windows fallback |
| DeltaProduct reference | 22,056 | 256 | equation-faithful unfused local reference |

The parameter envelope is below 2%. State size is intentionally reported, not
hidden: the Spin and motor candidates receive a substantial cache-size
advantage. This pilot tests structured state efficiency, not state-matched
capacity.

The maintained Pure Rotor v2.1 model is not relabelled as the motor. Its
sandwich action is center-blind and its bounded affine multivector state is a
different research object.

## Analytic oracles

- `exact_spin_motor_oracle`: exact signed quaternion and exact translation;
- `se3_quotient_oracle`: exact translation and exact physical rotation after
  deterministic quaternion-sign canonicalization;
- `spin_only_oracle`: exact signed quaternion and zero translation.

Before training, the quotient oracle must score:

- physical rotation threshold accuracy `1.0`;
- translation threshold accuracy `1.0`;
- signed rotation threshold accuracy `0.5`; and
- paired antipodal recovery `0.0`.

## Frozen metrics and gates

- signed rotation success: signed quaternion angle at most `15 degrees`;
- physical rotation success: quotient angle at most `15 degrees`;
- translation success: endpoint L2 error at most `0.10`;
- joint signed pose: signed rotation and translation both succeed;
- paired antipodal recovery: predicted pair is antipodal within `15 degrees`;
- paired physical agreement: predicted projective rotations agree within
  `15 degrees` and predicted translations differ by at most `0.10`;
- paired double-cover pose: both branches match their signed rotations and the
  shared translation within thresholds.

The two long-context screening gates are evaluated separately on every `L64`
and `L128` split:

1. center gate: signed-hemisphere accuracy at least `90%`;
2. joint-pose gate: joint signed-pose accuracy at least `80%`.

No mean may conceal a failed split. A candidate passes a gate only if every
long split passes it.

## Hypotheses and falsifiers

Primary hypothesis: the motor is the only trained candidate that passes both
long-context gates, because its update is exactly closed under signed rigid
motion.

Useful negative outcomes:

- motor passes center but fails translation: the decoder/training recipe is not
  enough to expose the geometric state;
- motor and quaternion tie: compact product states can encode the finite pilot
  without requiring genuine Euclidean closure;
- Mamba-2 or DeltaProduct wins: the bespoke inductive bias is not useful at
  this budget;
- all candidates fail: increase optimization budget only after checking oracle,
  gradients, and split contracts; do not call the hypothesis disproved by an
  undertrained run.

A “breakthrough” label is forbidden from this pilot alone. The strongest
permitted claim is a single-seed mechanism result. A stronger claim requires
at least three seeds, all three generator coordinates, a matched non-group
rigid-motion distribution, and an official fused external baseline where the
platform permits it.

## External-source boundary

- Path Development Network: <https://arxiv.org/abs/2204.00740>
- Mamba-2: <https://arxiv.org/abs/2405.21060>
- DeltaProduct source commit:
  `d62241a81d07aa32b1b65e7d17377f6a7cd0a5d8`
- Fixed-Point RNN paper: <https://arxiv.org/abs/2503.10799>
- Fixed-Point RNN official source commit reviewed:
  `0cc1e3c520423e02674c20333fcf9dfa46b7d204`

The official Fixed-Point RNN source is excluded rather than approximated. It
pins Python below 3.12, Torch 2.4.1, Triton 3.0, Mamba-SSM, and custom
scan/causal-convolution paths; this checkout is Python 3.12, Torch 2.12, and
Windows. A simplified loop would not be an official FP-Mamba reproduction.

## Invalidating conditions

Do not interpret the run if any of the following fail:

- a forbidden relation appears in training;
- legal-language coverage is below 120 binary states;
- any translation token is absent;
- any evaluation pair is not an exact central partner after the block;
- any paired physical pose differs above float32 construction tolerance;
- candidate parameter gap exceeds 2%;
- any checkpoint or JSON artifact cannot be hashed; or
- a candidate produces a non-finite loss or metric.
