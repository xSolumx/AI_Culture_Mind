# Program 01: Selective rotor state-space models

## Object

A bounded recurrent state in the full eight-coordinate Clifford algebra
`Cl(3,0)`, transported by input-selective `Spin(3)` rotor conjugation and
trained through associative affine scans.

## Claims that currently survive

- The maintained recurrence has a hard state-norm bound under its stated
  finite-input assumptions.
- Its affine transition family is exactly associative in real arithmetic and
  supports constant-state recurrent inference.
- Learned rotor actions are causally active: post-training identity clamping
  and action shuffling damage prediction.
- At fixed recurrent width in the frozen five-seed v2.1 experiment, rotor
  transport improved prediction loss over identity.

## Negative results and boundaries

- Rotor was not the best state-matched transport; quaternion, commuting phase,
  and larger generic orthogonal rows performed better in the reported table.
- The preregistered associative-recall and Q8 memory gates failed.
- At matched measured eager-PyTorch CUDA cost, the wider identity model was
  substantially better.
- These experiments do not demonstrate a language-model scaling advantage or
  a `Spin(8)`/triality advantage.

## Canonical evidence

- [`SSM-Models/FOUNDATIONS.md`](../../SSM-Models/FOUNDATIONS.md)
- [`PURE_V2_1_TRANSPORT_ABLATION_RESULTS.md`](../../SSM-Models/experiments/PURE_V2_1_TRANSPORT_ABLATION_RESULTS.md)
- [`PURE_V2_1_RELATED_WORK_NOTES.md`](../../SSM-Models/experiments/PURE_V2_1_RELATED_WORK_NOTES.md)

## Next publishable question

Separate state efficiency from kernel efficiency. Compare optimized rotor,
quaternion, complex/MIMO, Householder-product, and identity transports at
matched state, parameter, and genuinely optimized hardware budgets. A result
must report all three rather than promoting one matching regime as universal.
