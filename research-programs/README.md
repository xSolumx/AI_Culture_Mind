# Research-program index

This directory is the scientific table of contents for the parent workspace.
It separates lines of enquiry by object, decisive evidence, and open gate.
Canonical code and artifacts remain in their source repositories; these
charters prevent results from drifting between unrelated papers.

For the cross-program adversarial review, see the
[foundational claim and logic audit](../FOUNDATIONAL_CLAIM_AUDIT_2026-08-08.md).
For the maintained-model and benchmark reconciliation, including the boundary
between completed artifacts, the Pure Rotor/Mamba-2 smoke artifact, and the
maintained Pure Spin(8) result, see
the [2026-08-16 documentation refresh](../DOCUMENTATION_REFRESH_2026-08-16.md).

| Program | Central question | Current boundary |
|---|---|---|
| [01 — Rotor state-space models](01-rotor-state-space-models/README.md) | Can selective noncommutative transport improve recurrent sequence models? | Pure Spin(8) passes a synthetic signed-transport cohort; Spin/motor mechanisms pass narrower gates, while broad learned and compute superiority remain open |
| [02 — Shared-family representation learning](02-shared-family-representation-learning/README.md) | Which relational constraints identify a family that independent fitting leaves underdetermined? | Strong controlled completion results; not a language-model claim |
| [03 — Triality memory and Intertwiner SchurScans](03-triality-memory-and-intertwiner-scans/README.md) | Which equivariant bilinear recurrences have finite associative lifts? | Exact algebra; matched hierarchical-routing and full-transport FLA benchmarks; no same-router Spin(8) capacity advantage |
| [04 — Spin(8) sensing and Cayley design](04-spin8-sensing-and-cayley-design/README.md) | How many mixed triality probes identify a shared action, and how should balanced probes be conditioned? | Exact generic, canonical-family, and local theorems; global exact design remains open |
| [05 — Spin(8) Dirac--Gram inequalities](05-spin8-dirac-gram-inequalities/README.md) | Does the cubic Gram penalty control every correlated four-probe frame? | Several exact family and boundary theorems; unrestricted positivity remains open |
| [06 — Spin(9) Dirac--Clifford sensing](06-spin9-dirac-clifford-sensing/README.md) | What can three real spinors identify, and how does their exact design problem reduce? | Exact local design theorem and exact 9-to-16 Clifford bind/unbind; neither proves a same-width memory advantage |
| [07 — Controlled model benchmarks](07-controlled-model-benchmarks/README.md) | How do proposed sequence models compare under explicit matching and artifact validation? | Three-seed Pure Spin(8) transport passes against unfused references; latent-input, state/compute matching, and natural tasks remain open |
| [08 — Spin--Dirac branching and finite subgroup lifts](08-spin-dirac-a5-branching/README.md) | Which Clifford, chirality, and central-sign properties survive from Spin(3) through Spin(12)? | Exact ladder, H1=H2=0, component/spinor atlases, and three center-sensitive 2.A5 screens pass; geometric Dirac data and fresh-seed ML validation remain open |
| [90 — Historical SpinorModel](90-historical-spinor-prototype/README.md) | What did the original compact prototype implement? | Preserved baseline, not the maintained model |

## Evidence vocabulary

- **Theorem:** a stated domain and conclusion with a human-readable proof.
- **Computer-assisted theorem:** a proof with explicit finite obligations and a
  declared trust boundary.
- **Exact reduction:** a reversible simplification of an open problem; it is
  not a sign certificate for the reduced conditions.
- **Empirical result:** a completed artifact under a stated protocol.
- **Falsification evidence:** a search that can find violations but cannot
  prove their absence.
- **Open:** no theorem or completed empirical gate settles the claim.

## Publication rule

Each paper should draw its headline from one program and one primary claim.
Cross-program material belongs in motivation, baselines, or an explicitly
proved bridge. The parent [public-release policy](../PUBLICATION_SCOPE.md)
governs what is committed; the standalone theorem repository has its own
stricter artifact manifest and program registry.
