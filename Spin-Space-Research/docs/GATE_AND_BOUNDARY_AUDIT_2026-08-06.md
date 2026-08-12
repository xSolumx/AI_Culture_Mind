# Gate and Boundary Audit

**Date:** 2026-08-06
**Scope:** every maintained test suite and every current promoted or open claim

> **2026-08-10 addendum.** This ledger was complete on its audit date. New
> classified gates now cover hierarchical matched retrieval, the Spin(9)
> Clifford memory boundary, the co-moving delta equivalence/stability tests,
> and frozen FLA implementation selection plus measurement aggregation. Their
> current interpretation is in
> [`SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md`](experiments/SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md);
> a further classified gate records the failed strict Task B replay and the
> successful prospective paired-action replication in
> [`TASK_B_PAIRED_ACTION_REPLICATION_RESULTS.md`](experiments/TASK_B_PAIRED_ACTION_REPLICATION_RESULTS.md).
> A third classified gate covers the later 64-slot semantic hierarchy, eager
> gathered-state control, and fused gathered-state CUDA result in
> [`LARGE_SLOT_SEMANTIC_HIERARCHY_RESULTS.md`](experiments/LARGE_SLOT_SEMANTIC_HIERARCHY_RESULTS.md).
> Older ledger rows below retain their dated meaning.

> **2026-08-12 addendum.** The dated unrestricted row predates the exact
> adjacent-octet determinant reconstruction and endpoint theorem. Both
> \(y=0\) and \(y=1\) determinant faces are now proved; only the open
> \(0<y<1\) determinant interior remains on that octet. See
> [`SPIN8_DIRAC_OCTET_DETERMINANT_ENDPOINT_RESULTS.md`](experiments/SPIN8_DIRAC_OCTET_DETERMINANT_ENDPOINT_RESULTS.md).

## Why this audit was necessary

A green test suite can mean several very different things. It may show that an
exact polynomial identity was recomputed, that a stored file still has the
published hash, that two floating-point execution paths agree within tolerance,
or that a historical cohort report still satisfies its frozen threshold. Those
are all useful checks, but they are not interchangeable evidence.

The maintained registry in
[`spin8_gate_contracts.py`](../src/spin8_gate_contracts.py) therefore assigns
every test suite to an explicit claim, status, evidence layer, replay tier,
boundary obligation, and limitation. The registry is itself checked by
[`test_gate_contracts.py`](../tests/test_gate_contracts.py). A new test file
cannot be added without being classified.

## Evidence classes

| Evidence class | What it can establish | What it cannot establish |
|---|---|---|
| Exact arithmetic or symbolic identity | A stated finite algebraic identity | A global sign unless sign is separately certified |
| Exact reconstruction | The polynomial in the declared finite ansatz | Positivity of that polynomial |
| Positivity certificate | Nonnegativity on the certificate's exact domain | A larger domain or another chart |
| Exact counterexample | Falsity of the precisely stated claim or method | Falsity of a broader theorem |
| Hybrid theorem | A theorem from exact local work plus a named classical input | A self-contained proof of the external theorem |
| Floating-point falsifier | A counterexample if one is found and made exact | A proof from failure to find one |
| Empirical cohort | Performance under the frozen protocol and seeds | Universal reliability or causal mechanism without controls |
| Artifact hash | Byte identity | Mathematical correctness |
| Implementation parity | Agreement of execution paths | Scientific usefulness |

## Current gate ledger

| Gate | Status | Acceptance boundary |
|---|---|---|
| Repository integrity | Operational | Hashes, links, syntax, and configured resource ceilings only |
| Rotor SSM streaming | Validated implementation | Full/chunk/token parity, identity tangent gradient, bounded long-scan error |
| Finite-group recurrence cohorts | Empirical | Frozen functional, margin, and mechanism metrics remain separate |
| Triangular SchurScan | Exact theorem | Two affine scans equal the sequential triangular recurrence; feedback remains excluded |
| Triality algebra and memory | Exact theorem | Equivariance and single-pair inversion; no claim of high-dimensional superposition capacity |
| Shared-family retraction | Empirical mechanism result | Independent controls fit observations; held-out relational completion is tested; direct-memory parity is retained |
| Four-versus-five identifiability | Hybrid theorem | Exact invariant/stabilizer calculations plus the compact principal-orbit theorem |
| Balanced Cayley information family | Hybrid theorem | Classical global orbit classification plus exact split-isotropy and block algebra |
| Balanced local exact design | Exact theorem | Strict quotient Hessian and finite tangent atlas; local only |
| Approximate-design optimum | Hybrid theorem in its domain | Exact constant sensitivity plus Kiefer--Wolfowitz equivalence; not the exact five-query problem |
| D4/24-cell bridge | Exact theorem | Bridge and non-equivalence statements checked separately |
| Signed-star Dirac--Gram | Exact theorem | Exact reconstruction, both signs, Bernstein positivity, complete equality set on the ansatz |
| Conditional decorrelation | Exact negative | Only the fixed-coordinate residual-removal map is falsified |
| Variable-Cayley one-edge | Exact theorem | Both Duffy charts, exact holdouts, positivity, equality set |
| Multiplicity gauge | Exact theorem | Gauge and physical ranks separated |
| Two-edge sector reconstruction | Exact reduction | All sectors and holdouts reconstructed; sign remains open |
| Two-edge local kernel | Exact local result and exact negative | Local transverse positivity; quadratic-Schur strategy falsified only |
| Two-edge finite gate | Exact reduction | Reversible radical removal and endpoint jets |
| Two-edge global positivity | **Exact theorem** | Complete 34-leaf chart cover, outward sign enclosures, and exact integer fallbacks on the frozen `h=0` domain |
| Unrestricted polynomial identity | Exact reduction, local theorem, global energy theorem, exact endpoint-face theorem, and adjacent-face Schur reduction | Sixteen exact seven-variable sectors from two disjoint grids; positive tangent and endpoint blow-up; exact all-sector RMS bound on the complete seven-cube; exact Klein-four positivity on the complete \(u_a=u_h=0,c^2=1\) face; exact five-variable first Schur block plus scalar, quadratic, and cubic second-block minors on \(u_a=0,c^2=1\); its determinant and other individual-sign domains remain open |
| Global exact five-query optimum | **Open** | Requires all allocations, nonorthogonal interiors, and singular boundaries |
| Unrestricted Dirac--Gram | **Open** | Requires a domain-wide sign certificate for all sixteen reconstructed margins, including non-vertex interiors and singular boundaries |
| Triality-specific ML advantage | **Open** | Requires matched modern baselines, state budgets, tokens, and throughput |
| Task B shared-action replication | Empirical mechanism result | Historical replay failure remains separate; prospective retained-parameter cohort passes action, routing, direct/delta, and scan controls |
| Large-slot semantic and fused-gather memory | Empirical quality and systems result | Supplied-frame shared routing and hierarchy pass 10/10; eager/fused parity precedes hardware-specific inference timing; no action-discovery, gradient, model-quality, or capacity claim |
| Independent FLINT crosscheck | Operational support | Reduces common-backend arithmetic risk; does not prove an inference |

The executable registry contains the precise source, artifact, test, and
limitation paths behind this compact table.

## Boundary policy

Every promoted theorem or implementation gate must name the boundary on which
it can fail. The present policy is:

1. **Algebraic domains:** test interior exact points, all symmetry sectors, and
   every singular face used in the proof. A generic rank calculation is not a
   boundary proof.
2. **Polynomial positivity:** reconstruct first, verify the identity on disjoint
   exact points, then certify sign. Native negative Bernstein coefficients
   reject that certificate basis, not the inequality.
3. **Orbit quotients:** separate local isotropy ranks from global orbit
   classification. If the global step is classical, name it as an external
   theorem.
4. **Numerical searches:** include non-vertex interior deformations,
   boundary-biased samples, optimizer restarts, and exact rationalization of
   any challenger before changing theorem status.
5. **Recurrences:** test sequential, parallel, chunked, and single-token paths;
   include the identity tangent and the longest declared horizon.
6. **Empirical gates:** freeze seeds, thresholds, data split, and baseline
   budget before evaluation. Report the complete distribution rather than only
   a mean or checkpoint-length pass count.
7. **Archives:** hashes preserve bytes. A verifier must recompute the proof
   object if the document says it replays a proof.

## Operational limitations that remain

- Historical empirical checkpoints were intentionally not all published.
  Their JSON cohorts are hash-verifiable, and the current harness semantics are
  tested, but the entire training history cannot be regenerated from this
  archive alone.
- CUDA tests skip when CUDA is unavailable. A CPU-only green suite must not be
  reported as a CUDA replay.
- The signed-star and one-edge full exact reconstructions are expensive and are
  not rerun inside every lightweight unit invocation. Lightweight verifier,
  complete reconstruction, and positivity replay remain distinct tiers.
- The balanced Cayley quotient uses a classical cohomogeneity-one theorem. The
  repository independently verifies the maintained normal form, local split
  isotropy, and information algebra, not the full classical classification.

## Reproduction

    $env:PYTHONPATH = "src"
    python -m spin8_gate_contracts
    python -m unittest tests.test_gate_contracts -v
    python tools/verify_artifact_manifest.py
    python tools/audit_math_docs.py

For a release claim, the bounded full suite is still required. Its pass means
that every listed check succeeded at its own evidence level; it does not erase
the distinctions in this audit.
