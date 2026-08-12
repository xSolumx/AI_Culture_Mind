# Programme 01: Associative scan algebra and compiler systems

## Scope

Associative composition laws for affine, semidirect-product, triangular, and
representation-factored recurrences, together with algorithms and compiler
forms that evaluate their ordered prefixes efficiently. The mathematical
object is the transition algebra; CPU/CUDA scanner measurements are systems
evidence about implementations of that algebra.

## Core Questions

1. Which recurrent transition families are closed under finite-dimensional
   associative composition?
2. Which factorizations preserve chronological order while reducing work,
   storage, or kernel-launch cost?
3. When can a transported recurrence be changed into coordinates supported by
   an existing fused primitive without changing the recurrence?

## Proven / Established Results

- Affine transitions compose by the exact semidirect-product law
  `(A2, b2) o (A1, b1) = (A2 A1, A2 b1 + b2)`.
- The maintained triangular bilinear recurrence has an explicit finite
  homogeneous lift and exact staged/recurrent equivalence. For the two generic
  cyclic-feedback graphs actually analysed, formal degree growth excludes a
  fixed finite monomial linear lift unless leading terms cancel.
- The ordered work-efficient tree performs `O(N)` compositions, supports
  irregular positive lengths and noncommuting actions, and passes recurrent,
  long-horizon, and gradient parity gates.
- The `Spin(3)` isotypic audit establishes an eight-dimensional commutant where
  the earlier grade-preserving family used only four dimensions. The factored
  Schur scan closes exactly for the implemented real-type blocks.
- Canonical complex- and quaternionic-type blocks are now implemented with
  right multiplicity matrices over \(\mathbb C\) and \(\mathbb H\). Exact
  centralizer ranks prove the displayed bases complete for the canonical
  irreps; chronological quaternion order, scan/recurrent parity, dense-real
  equivalence, and gradient parity pass their maintained gates.
- An exact detector now solves the commutant of supplied rational
  representation generators, distinguishes real, complex, and quaternionic
  irreducible type under complete reducibility, and extracts the full
  multiplication basis. Split dimension-two, repeated-irrep, missing-
  assumption, and rational-conjugacy controls pass their required outcomes.
- A reducible-input compiler now recursively splits exact rational commutant
  idempotents, certifies every irreducible leaf, groups equivalent copies by
  exact intertwiner spaces, and reconstructs aligned isotypic coordinates.
  Repeated real, complex, and quaternionic controls; a mixed-type control; the
  `Cl(3,0)` decomposition `2V0 + 2V1`; and the Spin(9) quotient model
  `V1 + 2V5` all satisfy their corner-algebra, center, and
  double-centralizer gates, including non-orthogonal rational conjugacies.
- A concrete cross-program bridge now solves the full
  \(\mathbb Q(\sqrt2)\) intertwiner space of the Cayley-null Spin(9)
  Grassmann slice, rationalizes it exactly to \(V_1\oplus V_5\), appends the
  supported \(\operatorname{Sym}_0(3)\) copy, and feeds the resulting
  \(V_1\oplus2V_5\) action through this compiler. This is a certified
  rationalization of one algebraic representation, not a general
  algebraic-number-field decomposition algorithm.
- The compiler itself now has a declared exact ordered-field backend for
  \(\mathbb Q(\sqrt2)\). A two-dimensional rational block with generator
  \(A^2=2I\) is correctly unresolved over \(\mathbb Q\) and splits into two
  real rank-one leaves over \(\mathbb Q(\sqrt2)\), with the closed-form
  projectors \((I\pm A/\sqrt2)/2\). The negative-square control
  \(B^2=-2I\) remains one complex-type real irreducible, so adjoining
  \(\sqrt2\) is not confused with algebraic closure. Dense algebraic
  conjugacies of the canonical real, complex, and quaternionic controls and a
  \(\sqrt3\)-outside-field refusal gate also pass exactly.
- Native algebraic matrices for the Spin(9) slice \(V_1\oplus V_5\) and full
  quotient \(V_1\oplus2V_5\) now pass the same compiler directly. This is
  independent of the earlier rationalizing bridge and checks that the scalar
  extension reaches a non-toy repository representation.
- A separate exact arithmetic audit identifies the correct invariant behind a
  supplied Machin/Alferov reformulation. The algebraic phase map is
  \(\rho(z)=z/\bar z\) into the rational norm-one torus, and split Gaussian
  primes supply an integer valuation-difference lattice. The displayed
  four-term product is exactly
  \(10{,}317{,}661{,}250(1+i)\), with every free phase valuation cancelled.
  The nearest-cotangent tangent update has an exact Euclidean numerator bound;
  strict Lehmer-height descent, LLL equivalence, and optimal compression are
  not claimed.
- On the recorded workstation, packed local `9 x 9` homogeneous slot blocks
  and the structured form have different CPU/CUDA crossover points. The
  reported winner is device- and length-dependent; there is no universal eager
  backend.
- For invertible value actions, the co-moving change of coordinates reduces
  the transported DeltaRule to an ordinary DeltaRule plus an action-prefix
  scan and frame changes. Exact and gradient checks pass under the stated
  arithmetic contracts.

## Open Claims

- Prior-art novelty of the combined representation-factored selective affine
  scan architecture remains provisional.
- Complete decomposition over every real algebraic scalar extension is not
  implemented. The promoted exact fields are \(\mathbb Q\) and the positive
  ordered quadratic field \(\mathbb Q(\sqrt2)\); generic number fields,
  towers, non-real embeddings, and robust detection from approximate or noisy
  floating-point generators remain open. The compiler still refuses when its
  exact idempotent search cannot certify a division leaf.
- Production competitiveness requires fused, cross-device comparisons; eager
  scan wins and one local GPU do not establish a general systems result.
- No general no-go theorem covers every nonlinear coordinate system for cyclic
  feedback.

## Dependencies

- Linear algebra, representation decompositions, and exact chronological
  composition are intrinsic dependencies.
- Programmes 02, 03, 04, and 06 may instantiate this algebra, but their
  identification, retrieval, exceptional-representation, and model-quality
  claims do not follow from scan closure.
- Programme 07 may reuse exact affine-composition tooling. That is a tooling
  dependency only, not evidence about Collatz dynamics.

## Non-claims

- Associativity in exact arithmetic is not bitwise floating-point equality.
- `O(N)` work is not by itself a fused-kernel speed claim.
- Schur or isotypic factorization is not intrinsically Spin(8), triality, or a
  memory-capacity mechanism.
- Scanner throughput is not retrieval quality or language-model quality.

## Canonical Evidence

- [Intertwiner SchurScan theorem](../../Spin-Space-Research/docs/experiments/INTERTWINER_SCHURSCAN_THEOREM.md)
- [Ordered work-efficient scan benchmark](../../Spin-Space-Research/docs/experiments/INTERTWINER_SCHURSCAN_BENCHMARK_RESULTS.md)
- [Spin(3) isotypic Schur-scan result](../../Spin-Space-Research/docs/experiments/SPIN3_ISOTYPIC_SCHUR_SCAN_RESULTS.md)
- [Complex- and quaternionic-type Schur blocks](../../Spin-Space-Research/docs/experiments/DIVISION_SCHUR_SCAN_RESULTS.md)
- [Exact Schur-type detection and basis extraction](../../Spin-Space-Research/docs/experiments/SCHUR_TYPE_DETECTION_RESULTS.md)
- [Exact reducible isotypic decomposition](../../Spin-Space-Research/docs/experiments/REDUCIBLE_ISOTYPIC_DECOMPOSITION_RESULTS.md)
- [Exact algebraic scalar-extension design](../../Spin-Space-Research/docs/ALGEBRAIC_EXTENSION_DESIGN.md)
- [Native \(\mathbb Q(\sqrt2)\) isotypic decomposition](../../Spin-Space-Research/docs/experiments/ALGEBRAIC_ISOTYPIC_DECOMPOSITION_RESULTS.md)
- [Concrete Spin(9) algebraic-to-isotypic bridge](../../Spin-Space-Research/docs/experiments/SPIN9_SLICE_ISOTYPIC_BRIDGE_RESULTS.md)
- [Local-algebra scanner optimization](../../Spin-Space-Research/docs/experiments/SCHURSCAN_MEMORY_SCANNER_OPTIMIZATION_RESULTS.md)
- [Co-moving compiler and parity gates](../../Spin-Space-Research/docs/experiments/SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md)
- [Exact Gaussian phase-lattice audit](GAUSSIAN_PHASE_LATTICE_AUDIT.md)

The final two reports are cross-program evidence. Their algebra/compiler claims
belong here; their retrieval and hardware conclusions belong to Programme 03.

## Reproduction and Horizon Expansion

Programme 07 now contains a bounded exact control that uses the same affine
composition species in a number-theoretic setting: inverse Collatz valuation
words compose as rational affine maps, while a finite residue/path-merging
certificate supplies a testable compiler target. This is a reusable algebraic
pattern, not a transfer of theorems. The control artifact and its explicit
limits are in
[`collatz_inverse_frontier_bounded_20260810.json`](../../research-programs/07-collatz-inverse-frontier-dynamics/artifacts/collatz_inverse_frontier_bounded_20260810.json).

The next mathematical extension is a finite-state weighted affine semigroup
whose states retain residue/admissibility data and whose path score retains
both slope and offset. A maximum-cycle calculation may identify candidates
such as `(1,4)`, but record occurrence and minimality still require separate
integer certificates.
