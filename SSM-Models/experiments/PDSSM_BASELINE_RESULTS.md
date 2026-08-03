# Hard PD Baseline Results

## Comparison contract

The primary source defines PD-SSM transitions as a column one-hot matrix `P`
times a complex diagonal `D`, with an optimal `N`-state FSA construction using
state size `N` ([paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/77b830c18836a9b2e1395a4936dd687a-Paper-Conference.pdf)).
The reference forward pass hardens a softmax with a straight-through estimator
([IBM code](https://github.com/IBM/expressive-sparse-state-space-model/blob/main/state_tracking_PyTorch/models/pdssm.py)).

`pdssm_group_actions.py` specializes that mechanism to the same write-free
fixed-token A5 test used by the rotor and Householder models:

- one hard transition per token;
- unit-modulus complex diagonal;
- no affine writes, residual path, MLP, or contextual controller;
- all prefixes supervised on the same inverse-augmented 15/16-bigram split;
- dense L16-L256 evaluation and exact recurrent streaming;
- deterministic PyTorch/cuBLAS execution.

State width is deliberately not matched. The theorem's exact automaton ceiling
needs 60 complex coordinates for the 60-state task, or 120 real scalars, versus
the rotor's 32 real scalars. Hiding that difference would misstate the theorem.

## Results

| model | oracle? | hard structure | prefix val | final val | dense tail | transition audit |
|---|---:|---|---:|---:|---:|---|
| exact regular PD | yes | four exact permutations, `D=I` | 100% | 100% | 100% at every length | 60/60 targets, zero collisions |
| learned column-one-hot PD, constant LR | no | one hard output per input column | 39.46% | 12.11% | chance | 32-35 targets, 25-28 collisions |
| learned column-one-hot PD, source-style schedule | no | one hard output per input column | 27.11% | 12.89% | chance | 30-35 targets, 25-30 collisions |
| learned projected-permutation PD | no | Hungarian hard permutation, Sinkhorn gradient | 26.00% | 4.00% | chance | 60/60 targets, zero collisions |

All learned pilots use seed 0 and 1,500 update steps. The scheduled pilots use
LR 0.002, 10% warmup, cosine decay, and final LR `1e-5`. The exact and learned
models all have exact recurrent streaming parity to floating precision.

## Interpretation

The oracle proves that hard discrete structure is a perfect finite-state
ceiling when the transition table is supplied. It does not show that the table
is easy to discover.

At equal update budget, the official-style column-one-hot surrogate remains
collision-heavy and fails. Enforcing a true permutation removes collisions but
does not rescue optimization; the discontinuous assignment landscape is even
harder. Thus this experiment does **not** support the claim that any hard
finite-state architecture will trivially learn the task. The compact smooth
rotor chart gives SGD a substantially more navigable route to the 3D faithful
irrep while using far less state.

This is an equal-budget mechanistic result, not a reproduction of the paper's
end-to-end benchmark. The official repository trains a width-128, two-block
model with writes, residual machinery, and up to 100,001 steps. Those features
violate this experiment's write-free isolation contract and use roughly 67
times the updates. The learned PD variants are therefore not declared ruled out
in their source regime.

## Next

1. Keep the exact regular PD result as the discrete ceiling.
2. Keep the learned 1,500-step results as transparent optimization negatives.
3. If source-regime compute becomes scientifically important, preregister a
   separate 100k-step full-PD replication rather than moving the equal-budget
   goalpost.
4. Continue the current mechanism sequence with holonomy transfer to
   Householder and then changed generators.
