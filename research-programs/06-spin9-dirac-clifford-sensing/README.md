# Program 06: Spin(9) Dirac--Clifford sensing

**Research author:** Hayden Austin

**Status:** legacy route; current ownership is
[Programme 05](../05-information-geometry-and-dirac-gram-sensing/README.md).

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
- exact norm-preserving (9\to16) Clifford binding and algebraic unbinding,
  plus a Hopf-coordinate coarse-index construction for hierarchical routing.
- The exact `V1 + 2V5` quotient certificate is now consumable by the v2.1.1
  isotypic compiler without trusting descriptive labels: observed and expected
  block signatures must agree and each row must satisfy its double-centralizer
  dimension. This is a verified Programme 01 interface, not a Spin(9)
  Tensor-Core or model-performance result.
- The complete pure first-`V5` Cartan graph is reconstructed in the radial and
  cubic shape invariants and has an exact global `101/100` bound. The complete
  coupled first-`V1 + V5` finite-radius graph has a characteristic-zero
  determinant identity and exact global `21/20` bound. The pure-`V1` line is
  pulled back exactly to the symmetric curve and has four classified candidate
  preimages. Exact Cartan blow-up certificates now prove finite-radius collars
  around all four preimages. These are graph-slice theorems, not the
  unrestricted optimum.
- The spinor-probe ranks `15,28,36` give stabilizer dimensions `21,8,0`,
  consistent with the orbit ladder `Spin(7) -> SU(3) -> 1`. This ladder is
  representation-specific and is not the defining-vector `3 -> 5 -> 7`
  Stiefel tower in Spin(8).

The local theorem is internally replayed and now has a separate full-chart
float64 autodiff path that rederives the \(11\)-negative/\(33\)-zero signature
without importing the exact Hessian code. It shares the foundational Spin(9)
generator constructor, so it reduces one important class of shared-code risk
without becoming an independent base-algebra implementation or external peer
review. External review remains pending.

## Open gate

The unrestricted global exact three-spinor optimum is not proved. The remaining
gates are exact maximality of the algebraic symmetric candidate on the mixed
first-`V1 + V5` slice outside the four certified local collars, control of the
second supported `V5`, and passage from the graph chart to the unrestricted
quotient. The bounded multistart screen is
counterexample-search evidence only. The 9-to-16 binding
identity is an exact mechanism, but it expands width and therefore does not
establish a same-state Spin(9) memory or sequence-model advantage. Its current
programme role is a possible coarse router for hierarchical memory, not a
replacement for fine retrieval.

The later Spin(9)--Spin(12) Clifford branching implementation is tracked
separately in [Program 08](../08-spin-dirac-a5-branching/README.md). It reuses
this programme's Spin(9) algebra but does not inherit the sensing or
D-optimality claims.

## Canonical evidence

Use the exact theorem tree's
[`Spin(9) Dirac--Clifford gate`](../../Spin-Space-Research/docs/experiments/SPIN9_DIRAC_CLIFFORD_GATE.md).
The promoted finite-radius theorem and its precise exclusions are assembled by
[`spin9_v1_v5_theorem.py`](../../Spin-Space-Research/src/spin9_v1_v5_theorem.py),
with the proof narrative in
[`SPIN9_V1_V5_RECONSTRUCTION.md`](../../Spin-Space-Research/docs/manuscripts/SPIN9_V1_V5_RECONSTRUCTION.md).
The cross-representation probe atlas is in
[`SPIN8_PROBE_STABILIZER_TOWER.md`](../../Spin-Space-Research/docs/experiments/SPIN8_PROBE_STABILIZER_TOWER.md).
The near-candidate rational collar and the resulting local-chart proof route
are recorded without theorem promotion in
[`SPIN9_CANDIDATE_COLLAR_DIAGNOSTIC.md`](../../Spin-Space-Research/docs/experiments/SPIN9_CANDIDATE_COLLAR_DIAGNOSTIC.md).
The first candidate-centered bridge is now exact: the pure-line gap contains
the square of the `Q(sqrt(241))` quartic equality fiber, and the first mixed
`V5` radial coefficient is positive at all four roots. See
[`SPIN9_CANDIDATE_NORMAL_FORM.md`](../../Spin-Space-Research/docs/experiments/SPIN9_CANDIDATE_NORMAL_FORM.md).
The corresponding quantitative neighborhoods are certified in
[`SPIN9_CANDIDATE_EXPLICIT_COLLARS.md`](../../Spin-Space-Research/docs/experiments/SPIN9_CANDIDATE_EXPLICIT_COLLARS.md).
The compact atlas has also been lifted to the exact ordered field
`Q(sqrt(241))`; its strict leaves certify the irrational target directly and
its retained cells identify the still-open cusp handoff. See
[`SPIN9_CANDIDATE_QUADRATIC_ATLAS.md`](../../Spin-Space-Research/docs/experiments/SPIN9_CANDIDATE_QUADRATIC_ATLAS.md).
The four equality-edge handoffs are now closed by exact shape-uniform cusp
charts on macroscopic rational scalar intervals; the remaining first-graph
gate is the compact region above their explicit radial floors. See
[`SPIN9_CANDIDATE_CUSP_CHARTS.md`](../../Spin-Space-Research/docs/experiments/SPIN9_CANDIDATE_CUSP_CHARTS.md).
The memory boundary and executable gate are reported in
[`SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md`](../../Spin-Space-Research/docs/experiments/SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md).

The compiler interface and its bounded Spin(8) hardware application are
reported separately in
[`ISOTYPIC_TO_SILICON_COMPILER_V211.md`](../../SSM-Models/experiments/ISOTYPIC_TO_SILICON_COMPILER_V211.md).
