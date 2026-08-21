# How to read the experiment record

This directory is a chronological scientific record. It contains frozen
preregistrations, dated result reports, theorem certificates, negative results,
and later corrections. Those documents answer different questions and should
not be flattened into one undated narrative.

## Authority and chronology

Use this order when two statements appear to conflict:

1. the current repository [README](../../README.md) and
   [research map](../RESEARCH_MAP.md) for present status;
2. the [research audit and correction ledger](../RESEARCH_AUDIT_AND_NEXT_STRATEGY_2026-08-06.md)
   for known interpretive corrections;
3. the latest result or theorem document for the relevant family, together
   with its exact artifact and verifier;
4. the matching preregistration for what was committed before results were
   inspected;
5. earlier reports for the historical route by which the result was reached.

A dated sentence such as “the next gate is” means “the next gate at that
time.” A later addendum may supersede its roadmap without altering the original
record. A preregistered threshold remains part of history even if later work
adopts a separately named functional criterion.

The real-type limitation in the 2026-08-03 Spin(3) isotypic report is extended
by the later
[`DIVISION_SCHUR_SCAN_RESULTS.md`](DIVISION_SCHUR_SCAN_RESULTS.md). The new
report implements canonical complex- and quaternionic-type blocks but does not
retroactively make the earlier experiment a model-quality result.
The still-later
[`SCHUR_TYPE_DETECTION_RESULTS.md`](SCHUR_TYPE_DETECTION_RESULTS.md) removes
the need to name the type or basis for an exact irreducible input under
complete reducibility. The subsequent
[`REDUCIBLE_ISOTYPIC_DECOMPOSITION_RESULTS.md`](REDUCIBLE_ISOTYPIC_DECOMPOSITION_RESULTS.md)
recursively splits rational commutant idempotents, certifies irreducible
leaves, and reconstructs aligned isotypic blocks. It refuses unresolved
rational splittings and does not support noisy floating-point inputs.
The cross-program
[`SPIN9_SLICE_ISOTYPIC_BRIDGE_RESULTS.md`](SPIN9_SLICE_ISOTYPIC_BRIDGE_RESULTS.md)
then verifies an exact \(\mathbb Q(\sqrt2)\) rationalization of the concrete
Spin(9) Grassmann slice before invoking that compiler. It is a specialized
bridge, not a general extension of the compiler's scalar domain.
The later
[`ALGEBRAIC_ISOTYPIC_DECOMPOSITION_RESULTS.md`](ALGEBRAIC_ISOTYPIC_DECOMPOSITION_RESULTS.md)
does extend the compiler itself to the declared ordered field
\(\mathbb Q(\sqrt2)\). Its genuine split and nonsplit controls distinguish
coefficient-field extension from algebraic closure, and its native Spin(9)
replay no longer requires rationalization. Arbitrary number fields and noisy
floating-point inputs are still outside the maintained contract. The companion
[`CLIFFORD_SIGNATURE_EXTENSION.md`](../manuscripts/CLIFFORD_SIGNATURE_EXTENSION.md)
records the exact Spin(8) controls and
\(\mathrm{Cl}(3,0)\hookrightarrow\mathrm{Cl}^0(1,4)\) theorem; it is not a
model experiment.

For the memory line, the current empirical status document is
[`LARGE_SLOT_SEMANTIC_HIERARCHY_RESULTS.md`](LARGE_SLOT_SEMANTIC_HIERARCHY_RESULTS.md).
It extends
[`SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md`](SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md)
from a separable eight-slot proxy and idealized selected-state arithmetic to a
64-slot overlapping-semantic cohort and measured fused gathered-state kernel.
The earlier reports remain the authority for their own frozen cohorts,
full-transport compiler, and eager/reference timings.
The derived
[`MEMORY_BENCHMARK_ATLAS.md`](MEMORY_BENCHMARK_ATLAS.md) is the visual entry
point across these cohorts and the official FLA operator results. It is not a
new evidence source; its figure manifest hashes every frozen input.
The last Task B action row is completed separately by
[`TASK_B_PAIRED_ACTION_REPLICATION_RESULTS.md`](TASK_B_PAIRED_ACTION_REPLICATION_RESULTS.md).
Its positive prospective verdict does not overwrite the failed strict replay
recorded in
[`TASK_B_DELTA_ACTION_REPLAY_RESULTS.md`](TASK_B_DELTA_ACTION_REPLAY_RESULTS.md).

## Evidence vocabulary

The exact ordered-probe geometry and its representation-dependent stabilizer
boundaries are summarized in
[`SPIN8_PROBE_STABILIZER_TOWER.md`](SPIN8_PROBE_STABILIZER_TOWER.md). Its
`18+7+3` refinement is a defining-vector Stiefel theorem, not an empirical
model result or an `SU(n)` isotypic decomposition.

The current Spin(9) candidate-maximality handoff is recorded separately in
[`SPIN9_CANDIDATE_COLLAR_DIAGNOSTIC.md`](SPIN9_CANDIDATE_COLLAR_DIAGNOSTIC.md).
It is an exact partial atlas with retained boxes, not a promoted positivity
certificate.
The follow-on
[`SPIN9_CANDIDATE_NORMAL_FORM.md`](SPIN9_CANDIDATE_NORMAL_FORM.md) exactly
factors the candidate fiber and proves positive first mixed radial coefficient
at all four algebraic preimages. The quantitative continuation
[`SPIN9_CANDIDATE_EXPLICIT_COLLARS.md`](SPIN9_CANDIDATE_EXPLICIT_COLLARS.md)
uses an exact Cartan blow-up to certify explicit finite-radius collars around
all four preimages. Candidate maximality on the compact complement remains
open.

- **Theorem or exact identity:** derived symbolically or checked by exact
  arithmetic under an explicitly stated domain.
- **Computational certificate:** a finite exact object whose verifier checks the
  claimed identity or positivity condition. Artifact integrity, reconstruction,
  and mathematical sufficiency are separate obligations.
- **Numerical falsification:** a search that found no counterexample. It raises
  confidence but never proves global nonnegativity or optimality.
- **Empirical result:** a statement about the recorded seeds, budgets, hardware,
  and evaluation protocol. It is not automatically a reliability theorem.
- **Hypothesis or roadmap:** a proposed mechanism or next experiment, not an
  established result.

## Preservation rule

Historical observations, failed gates, and negative results are not silently
rewritten to match the current interpretation. When a later audit changes the
meaning of an earlier result, the correction belongs in a dated addendum or the
central correction ledger. Current manuscripts should state the corrected
interpretation directly and cite the historical path when it matters.

## Known examples of supersession

- Five probes established identifiability before the balanced sensor's
  conditioning and D-optimality questions were separated.
- Bigram generalization in fixed-token write-free actions tests whether
  optimization finds useful per-token operators; it is not the same falsifier
  as the earlier context-dependent model test.
- The behavioral functional gate is weaker than the original
  (10^{-3}) raw-homomorphism gate; the two counts must remain separate.
- The variable-Cayley one-edge theorem and the finite second-edge reduction
  supersede older documents that call the Cayley-null edge the current frontier.
- GPU sweeps over the finite two-edge gates are recorded as counterexample
  searches only; exact global positivity remains open.

This policy preserves history while preventing old, locally accurate language
from masquerading as the present theorem boundary.
