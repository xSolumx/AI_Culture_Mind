# AI Culture Mind

Enjoy this, learn from this, and then step outside and touch grass ;)

AI Culture Mind is the latter third of a machine learning and topology research workspace and it's progress, not one monolithic model claim. Its
work is separated into independently falsifiable programs with different
mathematical objects, evidence standards, and publication paths.

Start with the [research-program index](research-programs/README.md). It
distinguishes:

1. selective rotor state-space models;
2. shared-family representation learning;
3. triality memory and intertwiner scans;
4. Spin(8) sensing and Cayley design;
5. Spin(8) Dirac--Gram inequalities;
6. Spin(9) Dirac--Clifford sensing;
7. controlled model benchmarks;
8. Spin--Dirac branching and finite subgroup lifts;
9. the historical SpinorModel prototype.

The [foundational claim and logic audit](FOUNDATIONAL_CLAIM_AUDIT_2026-08-08.md)
records which mathematical and empirical claims survived an adversarial
definition--domain--evidence review, the corrections made, and the gates that
remain open. The
[2026-08-10 repository-wide documentation refresh](DOCUMENTATION_REFRESH_2026-08-10.md)
records the previous cross-program interpretation and validation scope. The
[2026-08-16 documentation reconciliation](DOCUMENTATION_REFRESH_2026-08-16.md)
records the current maintained-model, benchmark, and link-audit status.

## Current research frontier (reviewed 2026-08-16)

The triality-memory programme now has a completed matched memory-core campaign.
Under identical routers, direct slots and triality-coded slots have the same
ordinary overwrite capacity; the defensible Spin(8) contribution is a shared
cross-view action/routing prior, not extra same-state storage. A hierarchical
coarse-to-block router improves learned retrieval for both direct and delta
memories, and a co-moving change of coordinates maps general invertible value
transport to a standard gated-delta recurrence. On the recorded RTX 2070 SUPER
benchmark, the official FLA chunk implementation is about `5.20x` faster in
forward time and `5.30x` faster in combined forward/backward time than the
matched direct transported reference at length 4096, with the declared
`128`- versus `64`-scalar recurrent-state caveat. These are memory-core and
systems results, not language-model results or a proof of triality-specific
capacity. See the
[canonical result](Spin8-Triality-Research/docs/experiments/SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md)
and [Program 03](research-programs/03-triality-memory-and-intertwiner-scans/README.md).

No result in one program is evidence for another unless an explicit bridge
experiment or theorem is cited. In particular, the maintained language model
uses `Cl(3,0)`/`Spin(3)`, whereas the triality mathematics concerns three
eight-dimensional representations of `Spin(8)`.

The maintained Pure Rotor SSM is version 2.1.0. Its PyTorch backend now has an
opt-in, Schur-factored scan whose forward, cache, CUDA, and first-order-gradient
parity are tested; it is not yet a measured production kernel. A new direct
Pure Rotor versus Mamba-2 runner has completed only a one-step smoke artifact,
so it supplies execution/provenance evidence and no quality or systems claim.
See [Program 01](research-programs/01-rotor-state-space-models/README.md) and
[Program 07](research-programs/07-controlled-model-benchmarks/README.md).

The repository now also maintains a separate
[`pure_spin8_ssm`](SSM-Models/pure_spin8_ssm/) v1.0 family instead of breaking
the Pure Rotor v2.1 checkpoint contract. One shared 28-coordinate controller
acts on the vector and both chiral eight-real representations, giving a
faithful 24-scalar triality cache that distinguishes all four Spin(8) center
signatures. Its associative affine scan, gradients, cache continuation,
masking, state bound, CUDA path, and checkpoint roundtrip are tested. In a
frozen center-sensitive synthetic transport cohort, all three trained models
reach L128 MSE `5.81e-5`--`6.68e-5` and 100% center classification, versus
Mamba-2 at `0.132`--`0.135` and chance center classification. This is an
algebra-matched result, not generic language-model superiority. See
[`PURE_SPIN8_VS_MAMBA2_RESULTS.md`](SSM-Models/experiments/PURE_SPIN8_VS_MAMBA2_RESULTS.md).

The new [Program 08](research-programs/08-spin-dirac-a5-branching/README.md)
implements a checked Clifford ladder
`Spin(3) -> Spin(8) -> Spin(9) -> Spin(10) -> Spin(11) -> Spin(12)` and carries
the icosahedral vector action together with its binary spin lift. The exact
matrix core passes through Spin(12); the float64 lift enumerates 120 elements
and 60 projective classes while retaining the central `-1` on every spinor
module. This is an algebraic representation gate, not a trained-model result,
and triality remains confined to Spin(8).

The follow-up exact rigidity certificate now constructs the same binary group,
its projective quotient, and the complete `(2,3,5)` relation Jacobian over
`Q(sqrt(5))`. On every listed rung, the relation kernel equals the
infinitesimal conjugacy image, proving `H1=0` for the fixed embedding. This is
an exact first-order theorem, not global conjugacy classification or a derived
obstruction theorem.

An exact averaging contraction on the complete 120-element table now also
proves `H2=0` for every linear `2.A5` module over `Q(sqrt(5))`. This separates
genuine group-cohomological obstruction from the redundant cokernel of the raw
three-relator presentation.

The follow-up exact character atlas now completes the global compact-real
component classification through dimensions `3, 8, 9, 10, 11, 12`. It derives
all nine complex irreducibles, their real/quaternionic types, the affine-`E8`
McKay graph, every real-module multiplicity vector, Spin-center signature,
centralizer dimension, and `O`-to-`SO` orientation split. See
[`SPIN_DIRAC_A5_COMPONENT_ATLAS_RESULTS.md`](SSM-Models/experiments/SPIN_DIRAC_A5_COMPONENT_ATLAS_RESULTS.md).
Nontrivial stabilizers remain part of the classified quotient; no ML advantage
follows from the result.

The exact spinor follow-up now restricts the Spin(n) spinor or half-spinors on
all 245 orthogonal types and independently verifies every branching against
the even/odd exterior-algebra Clifford identities. All 21 orientation-split
types are distinguished by exchanged chiral characters. The fixed Spin(3)
ladder remains an isotypic sum of the defining binary spinor and has no
invariant spinors. See
[`SPIN_DIRAC_A5_SPINOR_BRANCHING_RESULTS.md`](SSM-Models/experiments/SPIN_DIRAC_A5_SPINOR_BRANCHING_RESULTS.md).

The first center-sensitive ML gate is now also complete under a frozen
three-seed, parameter-near protocol. On paired `2.A5` words that have the same
projected A5 path but differ by the binary center, the explicit Spin quaternion
composition scan preserves a 100% target-versus-central-partner margin through
length 128 in every seed. Mean exact 120-state accuracy at length 128 is
62.20%, versus 2.29% for Pure Rotor v2.1, 1.89% for its identity ablation, and
4.56% for Transformers Mamba-2. This is a finite-group mechanism result, not a
general SSM or language-model theorem. The successful scan is now a separate
experimental module and does not silently change v2.1. See
[`PURE_ROTOR_2A5_CENTER_PILOT300_RESULTS.md`](SSM-Models/experiments/PURE_ROTOR_2A5_CENTER_PILOT300_RESULTS.md).
The same frozen checkpoints then passed a deterministically selected length-11
identity/center relation pair absent from every training schedule, again with
100% central margin through L128; because that selector was designed after the
pilot, it is correctly labelled exploratory. See
[`PURE_ROTOR_2A5_UNSEEN_RELATION_RESULTS.md`](SSM-Models/experiments/PURE_ROTOR_2A5_UNSEEN_RELATION_RESULTS.md).

The next preregistered pilot withholds all three central presentation words
`a^2`, `b^3`, and `(ab)^5`, adds an equal-length identity-token control, and
repeats byte-identical schedules under three conjugated generating sets. It
also adds a pinned, equation-faithful DeltaProduct reference and an exact
regular-action PD oracle. Spin passes every long-retention center gate with a
99.50% minimum margin and is the unique exact-accuracy winner in all 18
registered long splits; the four other learned candidates fail the center
gate. This is one initialization across paired coordinates, not multi-seed
replication or a fused-kernel result. See
[`SPIN_2A5_MULTIRELATION_RESULTS.md`](SSM-Models/experiments/SPIN_2A5_MULTIRELATION_RESULTS.md).

The first geometric extension is now implemented as a separate unit-dual-
quaternion motor scan, covering the double cover of `SE(3)` rather than only
rotations. Its recurrent, parallel, cache, gradient, central-sign, and
homogeneous-matrix contracts pass through length 4096. This is a numerical
path-development gate, not a learned result. The local eager implementation is
about 8.6 times slower than batched 4 by 4 matrix scan, so a compact state has
not yet become a kernel advantage. See
[`MOTOR_PATH_DEVELOPMENT_RESULTS.md`](SSM-Models/experiments/MOTOR_PATH_DEVELOPMENT_RESULTS.md).

On the corresponding supervised rigid task, blind learned readouts fail, but
local motor increments identified only from legal adjacent training prefixes
pass all 162 coordinate/seed/split evaluations. A separately frozen 20-run
signed-pose noise audit retains perfect joint and paired accuracy through the
5-degree rotation/0.05 translation tier, then finds a clear failure boundary
at 15 degrees/0.15 despite preserving the center sign. This is conditional on
signed every-prefix supervision; it does not infer the binary lift from
ordinary `SO(3)` poses. See
[`SPIN_MOTOR_RIGID_2A5_RESULTS.md`](SSM-Models/experiments/SPIN_MOTOR_RIGID_2A5_RESULTS.md).

The octonion continuation now has a lawful parallel architecture rather than a
nonassociative pseudo-scan. Each unit octonion is lifted to its real `8 by 8`
multiplication operator; an ordered work-efficient tree composes those maps,
while streaming keeps only the acted-on eight-vector. The exact Lie certificate
has rank 28 and determinant `-2^49`, so products of the seven-dimensional leaf
family generate all `SO(8)`. A bounded affine layer passes forward, gradient,
cache, CUDA, and length-4096 gates. The optional WSL/Triton recurrence also
passes custom backward and runs 4.96x faster than the work-efficient operator
path at L4096 (1.805 versus 8.951 ms) on the RTX 2070 SUPER. This is an
experimental mechanism/systems result, not triality task superiority. On the
frozen continuous full-prefix-operator task, the 72-parameter algebra-matched
encoder reaches L128 MSE `1.85e-12`, versus `0.216` for the invalid collapsed
octonion, `0.124` for the unfused DeltaProduct reference, and `0.125` for the
unfused Transformers Mamba-2 control. That is a one-seed, coordinate-aligned
realizability result—not generic model superiority. A harder frozen replication
then hides the Fano coordinates behind three independent Haar `SO(8)` bases.
A 28-parameter learned gauge crosses its L128 gate in all three bases
(`1.54e-9`--`8.74e-8`) and recovers the hidden law up to numerical `G2`
automorphism residuals at most `2.17e-4`. The overall preregistered cohort still
fails because the gradient-trained dense control misses its gate and one
float32 oracle is narrowly above threshold; legal direct identification later
shows the dense map is realizable. This is structural gauge recovery, not a
generic leaderboard win. See
[`OCTONION_OPERATOR_SCAN_RESULTS.md`](SSM-Models/experiments/OCTONION_OPERATOR_SCAN_RESULTS.md).

The final-only successor initially fails under fixed L16 training, then
recovers all 9/9 structured and 9/9 dense runs under the frozen
L2-to-L16 composition-depth curriculum. The result exposed a sharper symmetry:
even-only terminal data identifies `G2 union -G2`, not `G2` alone. Four learned
gauges lie in the positive coset and five in the negative coset; a held-out L17
audit produces the predicted sign in all nine runs. The failed unsigned audit
is preserved, and the corrected interpretation is recorded in
[`OCTONION_FINAL_ONLY_RESULTS.md`](SSM-Models/experiments/OCTONION_FINAL_ONLY_RESULTS.md).

The learned rigid follow-up now combines all three held-out central relations
with body-frame translations and compares quaternion, motor, Transformers
Mamba-2, and DeltaProduct candidates at about 22k parameters. All four
300-step learned readout models fail the strict long-context joint-pose gate,
including the motor classifier. The productive result is more specific: local
group differences between supervised legal prefixes identify the seven token
motors without reading an evaluation relation. The resulting 49-parameter,
8-state-scalar tracker achieves 100% joint signed pose and paired double-cover
pose accuracy in all 9 generator-coordinate/schedule replications and all 162
splits through L128. A matched direct-product state retains the center sign but
fails translation, isolating the semidirect coupling. This is a replicated
finite deterministic identification result under every-prefix pose
supervision—not an end-to-end or natural-data breakthrough. See
[`SPIN_MOTOR_RIGID_2A5_RESULTS.md`](SSM-Models/experiments/SPIN_MOTOR_RIGID_2A5_RESULTS.md).

## Source layout

| Path | Role |
|---|---|
| [`research-programs/`](research-programs/README.md) | Public claim map, status ledgers, and reading order |
| [`SSM-Models/`](SSM-Models/) | Maintained Pure Rotor and Pure Spin(8) SSMs, controlled comparisons, and executable representation gates |
| [`Spin8-Triality-Research/`](Spin8-Triality-Research/) | Standalone theorem repository, linked as a Git submodule |
| [`Spin8-SSM-Benchmark/`](Spin8-SSM-Benchmark/) | Matched empirical benchmarks and controls |
| [`SpinorModel/`](SpinorModel/) | Preserved original prototype and a separate overhaul |

The existing project paths are retained deliberately: they are source and
provenance boundaries, and moving them would invalidate historical links and
artifact manifests. The new program layer reorganizes the scientific claims
without silently rewriting that history.

Large model weights, downloaded datasets, generated caches, and raw process
logs are intentionally excluded from Git. The compact 2026-08-16 frozen
checkpoint cohort is a deliberate exception so artifact rehash/reload tests
work from a clean clone. Reproducible conclusions should be backed by structured
artifacts, executable checks, and a concise interpretation that states both the
pass criteria and the nonclaims.
See the [public-release policy](PUBLICATION_SCOPE.md) for the complete boundary.

Clone the repository and its theorem submodule with:

```bash
git clone --recurse-submodules https://github.com/xSolumx/AI_Culture_Mind.git
```
