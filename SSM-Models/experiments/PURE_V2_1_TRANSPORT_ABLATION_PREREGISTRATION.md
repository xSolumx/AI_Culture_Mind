# Pure v2.1 transport-ablation preregistration

Frozen: 2026-08-06, before any v2.1 training or family outcome was observed.

## Question

Does input-controlled noncommutative rotor transport improve memory,
next-byte prediction, or quality-adjusted CUDA efficiency over simpler stable
transitions when the surrounding model is held fixed?

## Canonical refinement

Version 2.1.0 retains the v2.0 bounded recurrence

```text
h_t = d_t U_t h_(t-1) + (1-d_t) w_t z_t,
```

with `0<d<1`, `0<w<1`, `||z||<1`, and norm-nonexpansive `U`. Therefore
`||h_t|| <= max(||h_0||,1)` remains exact over finite real arithmetic.

The only mathematical change is the default rotor chart limit from `pi/2` to
`pi`. The v2.0 checkpoint had p95 angles at the old cap in every layer. The
new open chart covers every proper 3D rotation except the measure-zero exact
pi boundary while preserving the identity tangent and stability theorem.

## Transport families

All families store eight real state coordinates per channel and use the same
bounded candidate, write gate, positive decay, residual blocks, FFN, tied
decoder, training batches, optimizer, and loss. Only `U_t` changes.

1. `identity`: retrained `U_t=I`; the scalar selective-SSM baseline.
2. `real_diagonal`: input-selective positive diagonal contraction; commuting.
3. `complex_phase`: four input-selective commuting SO(2) phases.
4. `quaternion_left`: one input-selective unit quaternion acting on two
   quaternion state copies; noncommutative and center-faithful.
5. `rotor`: canonical input-selective Cl(3,0) rotor sandwich.
6. `fixed_rotor`: a learned token-independent rotor per channel.
7. `so8`: input-selective generic SO(8) action represented by eight paired
   Householder reflections and composed as 8x8 matrices.

Identity-clamping and time-shuffled-action evaluation are interventions on the
trained rotor checkpoint, not substitutes for retrained baselines.

## Fairness views

### State matched

Every family uses the same channels and exactly `8*C` recurrent scalars per
layer. This is the primary scientific comparison.

### Effective-parameter matched

No dummy/dead parameters count. Before training, enumerate integer channel
counts and choose the closest effective trainable parameter count to the
state-matched rotor target. Ties choose the smaller model.

### CUDA matched

Before outcome training, benchmark untrained forward/backward at the frozen
batch/context with synchronized CUDA events. Enumerate integer channel counts
from 1 through 96 in increasing order and select the family width whose median
step time is closest to the rotor target. The search may stop after two widths
meet or exceed the target, since all smaller integer widths have then been
observed; any out-of-memory width and the truncated suffix are recorded.
Report the residual mismatch; do not claim exact compute matching when no
integer width lies within 10%.

## Prediction protocol

- Data: WikiText-2 raw UTF-8 bytes, with hashes recorded in every report.
- State-matched model: 8 channels, 2 layers, expansion 2, context 128,
  batch 64, 300 updates, AdamW `3e-3`, dropout zero.
- Seeds: 0--4 with identical sampled windows across families within a seed.
- Development validation: the first 20 deterministic batches.
- Confirmation validation: the next 40 nonoverlapping batches, not consulted
  during training.
- Parameter- and CUDA-matched confirmations use the same protocol and seeds.

Primary endpoint: paired confirmation cross-entropy. Also report bits/byte,
train time, tokens/s, peak allocation, state size, effective parameters, action
activity, and full/chunk/recurrent parity.

## Memory protocol

Two endpoint-supervised tasks are used so language frequency cannot masquerade
as memory.

1. Associative recall: key/value pairs followed by a queried key; predict its
   value. Train at lengths through 64 and test at 64, 128, 256, and 512.
2. Q8 ordered product: predict the noncommutative product of generator tokens.
   Train through length 32 and test at 32, 64, 128, and 256.

State-matched families use four channels, two layers, five paired seeds, and
identical examples within each seed. Each task uses 800 AdamW updates at
`3e-3`, batch 128, gradient clipping at 1.0, and deterministic evaluation on
32 batches per length. Training lengths are sampled uniformly from even
lengths 4--64 for associative recall and integer lengths 4--32 for Q8. Report
endpoint accuracy at every length, not a cherry-picked maximum.

## Compute protocol

Use PyTorch 2.12 eager float32 on the same RTX 2070 SUPER. Record synchronized
CUDA-event medians after warmup for inference and forward/backward at contexts
64, 128, 256, and 512 with batch 8. CUDA-width matching itself uses the frozen
prediction batch 64 and context 128. Report tokens/s and peak allocated memory. The SO(8)
family uses `torch.linalg.matrix_exp` only as a correctness oracle; its training
path uses paired Householder reflections because `torch.linalg.solve` would
introduce a documented CUDA-to-CPU synchronization.

## Decision rules

- “Prediction benefit” requires a lower mean paired confirmation loss than
  retrained identity in at least four of five state-matched seeds. Confidence
  intervals and all seeds remain visible; this rule is descriptive, not a
  substitute for adequate power.
- “Memory benefit” requires higher accuracy than identity at every
  extrapolation length on at least one task and a positive five-seed mean at
  the longest length. A win only on Q8 is evidence for ordered algebra, not
  general memory.
- “Compute efficiency” requires a Pareto improvement: lower loss or higher
  memory accuracy at no greater measured step time, or equal quality with
  lower time/memory. Parallel-versus-recurrent speed within one family does
  not count.
- If the retrained identity family catches up, or trained-rotor identity
  clamping causes no degradation, rotor geometry is treated as nonessential.

## Pre-cohort calibration amendment

Amended 2026-08-06 21:31 SAST, after timing-only calibration and before any
outcome-training run: extend the CUDA width search ceiling from 24 to 96.
Widths 1--24 left the launch-bound identity, diagonal, and complex families far
below the C=8 rotor time, so the original ceiling could not implement the
stated measured-budget comparison. The increasing search still stops after two
widths meet/exceed the target or at out-of-memory, but never before evaluating
the state-matched C=8 width. The rotor row is definitionally fixed to its C=8
target measurement rather than re-searched. Two implementation-development
sweeps were discarded before outcome training: the first wrote no artifact;
the second exposed the rotor self-reference error and was superseded. No
family, outcome, training setting, split, metric, or decision rule changed.

## Integrity boundary

The v2.0 checkpoint and reports remain immutable historical evidence. Smoke
failures may fix implementation bugs before the cohort starts. Once seed 0 of
the main cohort begins, no architecture, hyperparameter, split, family, metric,
or decision rule changes without a dated amendment that labels all earlier
runs exploratory.
