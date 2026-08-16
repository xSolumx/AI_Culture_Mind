# Haar-Basis Final-Only Operator Protocol

Protocol frozen: **2026-08-16T21:23:00+02:00**

## Purpose

Remove the exact local-identification shortcut from the Haar-basis experiment.
Every training example supplies a length-16 observed token sequence

`x_t = Q u_t`

and only the terminal transported operator

`Y = Q L_(u_16) ... L_(u_1) Q^T`.

No intermediate prefix target is used in the loss. Evaluate terminal operators
on fresh fixed schedules at lengths 16, 64, and 128.

## Frozen cohort

- hidden deterministic Haar bases: seeds 0, 1, 2;
- 1,000 AdamW updates, batch 32, learning rate `3e-3`, weight decay `1e-4`;
- three independent model initializations for the 28-parameter learned-basis
  operator and the 512-parameter dense linear operator;
- one initialization for the unchanged 12,300-parameter Transformers Mamba-2
  and 13,192-parameter DeltaProduct references;
- identical per-basis token/target schedules for every candidate;
- learned-basis coordinates initialized with `N(0,0.05)` around identity;
- dense operator initialized at canonical left multiplication plus
  `N(0,0.01)` weight noise.

The structured and dense candidates both scan 64-scalar complete operators.
Mamba-2 and DeltaProduct retain their prior state accounting and unfused
reference implementations. This is architecture-quality evaluation, not a
kernel-throughput comparison.

## Frozen gates

For each Haar basis:

- exact transported oracle maximum MSE below `2e-12`;
- transported collapsed-octonion L128 MSE above `1e-2`;
- all learned metrics finite and all checkpoints reload/rehash;
- every learned-basis initialization has L128 MSE below `1e-3`;
- every learned-basis initialization beats the dense operator with the
  corresponding initialization at L128.

No dense, Mamba-2, or DeltaProduct absolute-accuracy win is preregistered. A
failed structured gate remains a negative result; no schedule, threshold,
optimizer, or initialization distribution may be changed after execution.

## Claim boundary

Passing would establish multi-basis, multi-initialization recovery of a latent
transported octonion composition law from final-only synthetic supervision. It
would not establish natural-task utility, a parameter/state/compute-matched SSM
win, triality-specific benefit, recovery of a time-varying law, or novelty of
the classical `G2` stabilizer theorem.
