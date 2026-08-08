# Program 06: Spin(9) Dirac--Clifford sensing

## Object

Nine symmetric Clifford involutions acting on the real 16-dimensional spin
module, and the information geometry of one, two, or three spinor probes.

## Results that survive the audit

- exact Clifford relations and restriction to the maintained Spin(8) gamma
  system;
- exact generic identifiability of a shared action from three spinors;
- frame-operator reduction with an exact nine-dimensional kernel;
- exact symmetric-family spectrum and isotropy branching;
- an exact negative-definite quotient Hessian proving strict local
  D-optimality of the symmetric rank-three candidate modulo Spin(9).

The local theorem is internally replayed and now has a separate full-chart
float64 autodiff path that rederives the \(11\)-negative/\(33\)-zero signature
without importing the exact Hessian code. It shares the foundational Spin(9)
generator constructor, so it reduces one important class of shared-code risk
without becoming an independent base-algebra implementation or external peer
review. External review remains pending.

## Open gate

The unrestricted global exact three-spinor optimum is not proved. The bounded
multistart screen is counterexample-search evidence only. No result here
establishes a Spin(9) sequence-memory advantage.

## Canonical evidence

Use the theorem submodule's
[`Spin(9) sensing ledger`](../../Spin8-Triality-Research/programs/spin9-dirac-clifford-sensing/README.md).
