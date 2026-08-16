# Spin-Space Research

Exact and computational research in Spin geometry, Clifford/Dirac sensing,
representation dynamics, associative Schur scans, and noncommutative
state-space systems.

This directory is the root repository's provenance-preserving exact-research
and theorem-harness layer. It is an ordinary root-owned directory, not a
submodule or second repository. The single authoritative claim map is the
[seven-programme index](../research-programs/README.md); detailed Spin(8) and
Spin(9) evidence ledgers live directly beside Programme 05, while compiler,
identification, and memory claims route to Programmes 01--04.

The [documentation index](docs/README.md) is the entry point for the exact
mathematics and experiment reports. The [public release boundary](PUBLICATION_SCOPE.md)
states what is and is not committed. A result in one claim family is not
evidence for another without an explicit bridge. In particular, this
directory does not claim a production-ready language model.

The workspace directory and display title were broadened on 2026-08-11 from
`Spin8-Triality-Research` to `Spin-Space-Research`. The published Python
package slug and historical GitHub remote remain in `pyproject.toml` and
`CITATION.cff` as provenance identifiers; repository ownership is now unified
at the workspace root.

The current claim statuses, replay tiers, and failure boundaries are defined in
the [gate and boundary audit](docs/GATE_AND_BOUNDARY_AUDIT_2026-08-06.md).

Reviewers of the strongest compact theorem can bypass the chronological
archive and begin with the
[balanced Cayley-spectrum referee package](referee/cayley-information-spectrum/README.md).

Positive results, negative results, and partially passed gates are retained
together. The central open mathematical target is the unrestricted signed
Dirac--Gram inequality. The repository proves several constrained and boundary
families, reduces the present two-edge obstruction to explicit polynomial
positivity gates, certifies those gates on the complete frozen `h=0` family,
reconstructs the unrestricted seven-variable margin exactly, and records
broader numerical counterexample searches without promoting them to proof.

## Current status

### Proved

- The maintained Spin(8) gamma system extends exactly to the nine symmetric
  Clifford involutions of the real 16-dimensional Spin(9) spin module. Three
  generic spinor probes have trivial common stabilizer, whereas a generic pair
  retains \(\operatorname{SU}(3)\). The associated sensing problem factors
  through a rank-three frame operator. Its linear information map has an exact
  nine-dimensional vector-grade kernel, and the convex approximate-design
  relaxation has complete optimizer family
  \(3I_{16}/16+\sum_i v_iP_i\), \(\lVert v\rVert\leq3/16\). No exact
  three-probe frame attains that relaxed optimum; its global optimum remains
  open. The algebraic symmetric candidate is nevertheless now proved to be a
  strict local optimum on the complete \(44\)-dimensional rank-three frame
  stratum modulo Spin(9): its quotient Hessian decomposes as
  \(V_1\oplus(V_5\otimes\mathbb R^2)\), and the exact coupled \(V_5\)
  multiplicity block is negative definite. The concrete Cayley-null slice is
  now also connected to that abstract decomposition by an exact
  \(\mathbb Q(\sqrt2)\) intertwiner: it rationalizes
  \(V_1\oplus V_5\), maps the curve tangent to the canonical trivial
  coordinate, aligns the supported \(V_5\), and passes the complete
  \(V_1\oplus2V_5\) representation through the reducible compiler. At the
  Cayley-null plane, an exact
  invariant reconstruction and six-cell Bernstein atlas additionally prove
  that every pure \(V_5\) graph has determinant ratio below \(101/100\); the
  algebraic symmetric candidate lies strictly above that complete family. The
  next coupled \(V_1\oplus V_5\) numerator has 18,600 rational-invariant
  coefficients. Exact boundary-adapted blow-ups at both projective rank-loss
  points place the exceptional families below \(26/25\). The formerly open
  finite-radius interior is now covered by 312 strict compact Bernstein leaves,
  eight exact handoff boxes, and eight strict local core/low-\(q\) charts. A
  22-prime replay checks both embeddings of \(\sqrt2\), 18 graded rank gates,
  and 13,914,692 raw determinants; its 175-digit prime product exceeds twice
  the exact residual coefficient bound. Consequently every graph in the
  complete coupled slice has determinant ratio at most \(21/20\). Exact
  optimality of the algebraic candidate, the second supported \(V_5\), and the
  unrestricted quotient remain open.
- One shared 28-dimensional bivector action generates the vector and both
  chiral eight-dimensional triality representations. The implementation checks
  the full \(\mathfrak{so}(8)\) brackets, center signatures, triality
  equivariance, scan parity, and norm preservation.
- The seven discrete octonion left operators generate the plus-extraspecial
  group \(2_+^{1+6}\) of order 128. All signed Fano-basis automorphisms form
  the non-split group \(2^3.\operatorname{PSL}(2,7)\), and their combined
  associative matrix closure is the split perfect group
  \(2_+^{1+6}:\operatorname{PSL}(2,7)\) of order 21,504. This identifies a
  known abstract group in the repository's fixed representation; it is not a
  new finite-group claim.
- The common-carrier closure of the vector and two half-spin views of the
  exact binary-icosahedral embedding is the reducible perfect group
  \(
  ((2.A_5\times2.A_5)/C_{2,\mathrm{diag}})\times2.A_5
  \) of order 864,000. Its two four-dimensional blocks carry the 600-cell
  rotation group and one binary-icosahedral action, while its Klein-four
  center acts by independent block signs. This is an exact embedding result,
  not a new abstract group or a model-performance claim.
- Within the triality sensor model, every four-probe design has a
  positive-dimensional stabilizer. Every mixed five-probe allocation has an
  open dense free stratum, while a five-probe design confined to one
  representation retains a \(\operatorname{Spin}(3)\) stabilizer.
- The balanced five-query information operator has

  \[
  \det I=\frac{81}{1024},\qquad
  \operatorname{tr}I=35,\qquad
  \operatorname{tr}(I^{-1})=43.
  \]

  Its Cayley family splits into fixed invariant blocks of dimensions
  \(8+8+8+4\), explaining the determinant structurally. On the orthonormal
  balanced information family, the Cayley-null design also uniquely minimizes
  \(\operatorname{tr}(I^{-1})\) and \(\operatorname{tr}(I^{-2})\), while
  \(\operatorname{tr}I=35\) and \(\operatorname{tr}(I^2)=67\) remain constant.
- The strengthened Dirac--Gram inequality

  \[
  \det I(X)\leq
  \det(XX^{\mathsf T})^3\det I(Q)
  \]

  is proved on the signed-star, Cayley-null edge, and variable-Cayley one-edge
  families. On the signed-star family the inequality is strict in the open
  parameter box, and its orientation-sensitive sector has the exact asymmetry
  factor \((1-u)(v-w)(1-z)^3\). Its normalized equality set is completely
  classified: \(z=1\) or \((u,v,w)=(0,0,0)\).
- The second residual edge is locally stable along the orthonormal equality
  line. At finite edge size, its eight signed margins reduce exactly to four
  degree-six conditions and four degree-twelve polynomial conditions. A
  complete 34-leaf triangular Bernstein atlas now proves all eight margins
  nonnegative on the frozen `h=0` two-edge family; interval-indeterminate
  controls are replayed with exact integer arithmetic.
- On the unrestricted seven-circle chart, triality symmetry reduces all
  physical margins to sixteen exact seven-variable polynomial sectors. Two
  disjoint rational grids reconstruct identical coefficient maps from
  2,500,000 exact determinants, and 32 fresh rational points verify all 512
  sector identities. The full tangent cone along the orthonormal equality line
  is nonnegative; at its calibrated endpoint, every tangent-null cone lifts by
  the strictly positive quartic \(128(p^2+q^2)^2\). This is an exact structural
  and local theorem, not yet a global positivity proof. A separate exact
  boundary-supported Bernstein decomposition proves globally that the trivial
  Fourier amplitude dominates the Euclidean norm of all fifteen nontrivial
  modes. The proof isolates the only four native-basis obstructions onto two
  identical three-variable faces, certifies those faces in triangular charts,
  and leaves a 588,245-control nonnegative remainder. This controls the RMS
  orientation deviation on the complete seven-cube, but does not yet force
  every individual orientation margin to be nonnegative. A Walsh-convolution
  bound additionally proves the first four elementary-symmetric orientation
  invariants nonnegative, leaving the invariant hierarchy
  \(e_5,\ldots,e_{16}\) open. A complementary exact theorem closes the
  complete four-variable face \(u_a=u_h=0,\ c^2=1\): its three surviving
  nontrivial Walsh modes form a Klein-four block, and every principal minor
  of the associated group-circulant matrix is nonnegative. This is a boundary
  theorem, not the unrestricted seven-variable result.
- On the adjacent five-variable face \(u_a=0,\ c^2=1\), eight surviving
  sectors form \((\mathbb Z/2\mathbb Z)^3\). An exact subgroup-chain Schur
  reduction splits the problem into two commuting Klein-four blocks. The
  complete first block is now proved positive semidefinite, and the scalar
  minor of the second block follows exactly from the global Fourier-energy
  theorem. All three quadratic minors of the second block are now proved on
  the complete five-cube, as is the cubic minor. The 6,082,148-term
  determinant is proved on the complete coordinate boundary: nine faces
  collapse exactly to one-mode perfect squares, while \(y=1\) is
  \(Z=X^2\). On \([1/8,7/8]^5\), a 2,140-leaf adaptive exact rational
  Bernstein atlas proves the stronger inequality that the trivial Walsh
  amplitude strictly dominates the other seven in absolute sum. The remaining
  width-\(1/8\) boundary collars are open, so the full adjacent face is not yet
  a theorem.
- A triangular recurrence driven by an equivariant bilinear intertwiner has an
  exact finite lift, an associative staged scan, and fixed recurrent state.
  \(\operatorname{Spin}(8)\) triality is the exceptional instance studied here.

### Numerical and empirical evidence

- Before the exact atlas was constructed, a float64 CUDA campaign tested
  851,968 interior and boundary points of the finite two-edge polynomial gates
  without finding a violation. That historical screen is now supporting
  falsification evidence, not the theorem certificate.
- A separate ten-seed search tested 860,160 five-query designs and 1,680
  gradient starts without finding a global equal-five-query challenger.
- The historical SSM experiments verify streaming recurrence, scan/recurrent
  parity, and several mechanism-level learning gates. They do not establish a
  language-model advantage at competitive scale.
- A frozen 10-seed, 64-slot overlapping-semantic campaign supports shared
  cross-view routing and coarse-to-fine interference reduction. Three-process
  CUDA measurements then show that one fused gathered-state kernel beats both
  eager dense and eager gathered controls on the named RTX 2070 SUPER. This is
  a supplied-frame synthetic and hardware-specific result, not extra triality
  capacity.

These results are counterexample searches or finite experiments. They are not
substitutes for the open global proofs or matched large-scale benchmarks.

### Still open

- the unrestricted seven-invariant signed Dirac--Gram inequality;
- global equal-five-query D-optimality over all allocations and frames;
- classification of exceptional nonprincipal five-probe strata;
- a triality-specific advantage over direct-slot, delta-rule, fast-weight, and
  structured orthogonal sequence baselines;
- competitive language-model scale-up and measured production throughput.

## Main result families

### 1. Algebra and recurrence

The algebra layer constructs all three eight-dimensional triality actions from
one shared \(\mathfrak{so}(8)\) generator and verifies their common invariant
tensor. The recurrent layer separates parallel training from streaming
inference without changing the transition law. Triangular bilinear coupling is
the exact boundary at which a finite staged scan remains possible; generic
feedback causes polynomial degree to grow without bound.

Read:
[foundations](docs/FOUNDATIONS.md),
[triality algebra](docs/experiments/SPIN8_TRIALITY_ALGEBRA_RESULTS.md), and
[Intertwiner SchurScans](docs/experiments/INTERTWINER_SCHURSCAN_THEOREM.md).
The later
[division-algebra Schur scan extension](docs/experiments/DIVISION_SCHUR_SCAN_RESULTS.md)
adds canonical complex- and quaternionic-type multiplicity blocks with exact
commutant audits and explicit noncommutative order checks.
The subsequent
[exact Schur-type detector](docs/experiments/SCHUR_TYPE_DETECTION_RESULTS.md)
solves a supplied rational commutant, rejects split/repeated controls, and
extracts real, complex, or generalized quaternion multiplication bases under
an explicit complete-reducibility assumption.
The subsequent
[reducible isotypic decomposition layer](docs/experiments/REDUCIBLE_ISOTYPIC_DECOMPOSITION_RESULTS.md)
constructs certified rational commutant projectors, irreducible leaves,
inter-copy alignments, and complete real/complex/quaternionic multiplicity
blocks. It fails closed when a bounded rational splitting search is
insufficient.
The new
[exact algebraic scalar-extension layer](docs/experiments/ALGEBRAIC_ISOTYPIC_DECOMPOSITION_RESULTS.md)
promotes \(\mathbb Q(\sqrt2)\) as a declared ordered coefficient field and
propagates it through nullspaces, projectors, Schur classification, and aligned
isotypic reconstruction. A genuine rational obstruction splits into two real
lines only after extension, while the negative-square control remains complex
type. The native Spin(9) slice and full quotient compile directly, without the
earlier rationalizing bridge. The companion
[Clifford signature theorem](docs/manuscripts/CLIFFORD_SIGNATURE_EXTENSION.md)
certifies the Spin(8) triality-module controls and the exact inclusion
\(\mathrm{Cl}(3,0)\hookrightarrow\mathrm{Cl}^0(1,4)\subset
\mathrm{Cl}(1,4)\) inside the maintained Spin(9) matrices.
The [work-efficient scan benchmark](docs/experiments/INTERTWINER_SCHURSCAN_BENCHMARK_RESULTS.md)
separates algebraic work, dependency depth, memory, and eager CPU/CUDA timing.
The [local-algebra memory-scanner optimization](docs/experiments/SCHURSCAN_MEMORY_SCANNER_OPTIMIZATION_RESULTS.md)
then combines Schur block preservation with small homogeneous matrices. On the
named RTX 2070 SUPER it is the fastest tested eager prefix backend at every
tested length, with a reversed-order replication; CPU winners remain
length-dependent.
The [equivariant-identification gate](docs/experiments/INTERTWINER_SCHURSCAN_EQUIVARIANT_IDENTIFICATION_RESULTS.md)
then isolates a controlled empirical consequence: a known one-dimensional
intertwiner extrapolates from a proper coordinate subspace, while an
unrestricted tensor trained on the same endpoints requires explicit group
augmentation. The same result for SO(3) prevents a triality-specific reading.
The [matched learned-retrieval campaign](docs/experiments/MATCHED_LEARNED_RETRIEVAL_RESULTS.md)
then separates update capacity from address inference on 10 frozen seeds.
Hard/discretized slots are the local quality and eager-tensor-program leader;
oracle delta is exact; direct and triality slots tie under hard routing; and
triality's supported role is shared cross-view action completion rather than a
superior overwrite law. A later
[prospective paired-action replication](docs/experiments/TASK_B_PAIRED_ACTION_REPLICATION_RESULTS.md)
closes the last Task B row on untouched seeds `20`--`29`: shared actions remain
exact through direct and delta memory, independently fitted actions fail the
held-out view, and routing-matched controls localize the deficit to the action.
The preceding strict historical replay remains a recorded provenance failure
because its metric-only source artifact did not retain learned parameters. A
corrected three-process systems audit now includes
official Flash Linear Attention kernels: fused recurrent delta wins the
transport-free core through length 2,048 and fused chunk delta wins at 4,096.
That kernel result excludes the campaign's noncommuting value transport and
does not alter the quality verdict.
The subsequent
[hierarchical Spin(8)/Spin(9) memory result](docs/experiments/SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md)
closes that systems gap without changing the capacity verdict. A stable
invertible-action co-moving compiler includes the noncommuting prefix scan,
inverse-frame solve, and read frame around official FLA kernels; at length
4,096 it is `5.20x` faster forward and `5.30x` faster forward+backward than
direct slots, with 128 rather than 64 logical state scalars. The same work
shows that two-slot hierarchical routing suppresses long-stream interference,
while exact hard routing makes direct and DeltaRule memories identical on the
frozen separable alias world. Spin(9) Clifford binding is exact but remains a
same-width direct-memory gauge; its Hopf map is proposed only as a coarse
cross-chiral router.
The newer
[large-slot semantic hierarchy and fused gather result](docs/experiments/LARGE_SLOT_SEMANTIC_HIERARCHY_RESULTS.md)
replaces the separable eight-slot proxy with 64 overlapping keys. Shared
three-view completion and hierarchical retrieval pass `10/10` frozen seeds,
and a one-kernel gathered recurrence runs in about 50 microseconds across the
tested grid: `7.60x`--`8.12x` faster than eager dense direct memory and
`12.21x`--`13.00x` faster than eager dense delta memory. Direct and delta still
agree under a common hard route, and the supplied inverse frame prevents an
action-discovery or triality-capacity interpretation.
The publication-facing
[memory benchmark atlas](docs/experiments/MEMORY_BENCHMARK_ATLAS.md) renders
the frozen quality, systems, official-FLA, and co-moving-transport cohorts as
separate reproducible figures. It also states the concrete FLA hybrid-model
integration boundary: the selected-block mechanism is a candidate third mixer,
not yet a registered layer or trained language model.

### 2. Identifiability and shared-family learning

Jointly constraining several observed actions to arise from one shared
representation removes relational null directions that independent fitting
cannot see. The five-probe theorems then identify the sharp generic sensing
boundary inside the triality action model: four probes are insufficient, while
five mixed probes are generically free.

Read:
[five-probe results](docs/experiments/SPIN8_FIVE_PROBE_RESULTS.md),
[continuous orbit theorem](docs/experiments/SPIN8_CONTINUOUS_PROBE_ORBIT_THEOREM.md),
and [blind shared-action results](docs/experiments/SPIN8_BLIND_ALIAS_ACTION_RESULTS.md).

### 3. Active sensing and Cayley information geometry

Each unit query contributes a rank-seven projector. The balanced sensor is a
strict local optimum modulo its 28-dimensional group orbit, and its information
operator has an exact block decomposition. Approximate design is a separate
problem: the equal five-point design is not globally optimal when fractional
measurement weights are permitted, whereas an eight-probe isotropic design is.

Read:
[Cayley spectrum](docs/experiments/SPIN8_CAYLEY_SPECTRUM_RESULTS.md),
[Cayley block theorem](docs/experiments/SPIN8_CAYLEY_BLOCK_THEOREM.md), and
[active sensing](docs/experiments/SPIN8_ACTIVE_SENSING_RESULTS.md).

### 4. Signed Dirac--Gram program

The exact proof program uses common triality symmetry, rational reconstruction,
rank-predicted boundary factors, and Bernstein/Duffy positivity certificates.
The one-edge and frozen `h=0` two-edge families are complete. The latter uses a
34-leaf rational-circle triangular atlas, outward Bernstein enclosures, and
exact integer replay for every cancellation control. The final Cholesky
residual has now been reconstructed exactly in sixteen sectors; the remaining
gap is a domain-wide sign certificate, not an unknown polynomial identity.

Read:
[current synthesis](docs/DIRAC_GRAM_TWO_EDGE_STATUS_2026-08-06.md),
[one-edge theorem](docs/experiments/SPIN8_DIRAC_ONE_EDGE_RESULTS.md),
[two-edge local theorem](docs/experiments/SPIN8_TWO_EDGE_BOUNDARY_KERNEL_RESULTS.md),
and
[finite polynomial reduction](docs/experiments/SPIN8_TWO_EDGE_FINITE_REDUCTION_RESULTS.md),
and
[two-edge atlas theorem](docs/experiments/SPIN8_DIRAC_TWO_EDGE_ATLAS_RESULTS.md),
and
[unrestricted reconstruction and tangent theorem](docs/experiments/SPIN8_DIRAC_UNRESTRICTED_RECONSTRUCTION_RESULTS.md),
and
[complete global adjacent-octet quadratic gate](docs/experiments/SPIN8_DIRAC_OCTET_QUADRATIC_RESULTS.md).

### 5. Learning and compilation lineage

The earlier finite-group experiments separate continuous optimization from
exact algebraic compilation. They study when training finds approximate
noncommutative actions, when a compiler can recover a discrete action, and why
shared-family retraction succeeds where independent normalization leaves
unconstrained directions. These are mechanism studies, not evidence that
\(\operatorname{Spin}(8)\) has already improved general language modeling.

Read:
[research map](docs/RESEARCH_MAP.md),
[experiment index](docs/EXPERIMENT_INDEX.md), and
[research audit and next strategy](docs/RESEARCH_AUDIT_AND_NEXT_STRATEGY_2026-08-06.md).

## Reading paths

| Reader | Start here |
|---|---|
| Non-specialist | [The Mathematics in Plain Language](docs/MATHEMATICS_IN_PLAIN_LANGUAGE.md) |
| Mathematician | [Triality Information Geometry manuscript](docs/PAPER_DRAFT_TRIALITY_INFORMATION_GEOMETRY.md) |
| Publication reader | [Cayley Spectrum paper](docs/manuscripts/CAYLEY_INFORMATION_SPECTRUM.md) and [Signed-Star paper](docs/manuscripts/SIGNED_STAR_DIRAC_GRAM.md) |
| Sequence-model researcher | [Foundations](docs/FOUNDATIONS.md) and [Research Map](docs/RESEARCH_MAP.md) |
| Reproducer or reviewer | [Reproducibility](docs/REPRODUCIBILITY.md) and [Artifact Manifest](ARTIFACTS.sha256) |
| Future contributor | [Mathematical Writing Standard](docs/MATHEMATICAL_WRITING_STANDARD.md) |
| Manuscript reviewer | [Full Manuscript Audit](docs/MANUSCRIPT_AUDIT_2026-08-06.md) |

For a complete, status-labelled tour of the documentation, begin with the
[Documentation Guide](docs/README.md).

## Repository map

| Path | Contents |
|---|---|
| [src](src/README.md) | Algebra, recurrence, exact-certificate, and falsifier harnesses |
| [tests](tests/) | Foundational, theorem, streaming, and documentation contracts |
| [Documentation Guide](docs/README.md) | Reader paths, claim-status legend, and logically grouped manuscripts |
| [Manuscripts](docs/manuscripts/README.md) | Self-contained theorem papers separated from the chronological archive |
| [Research Map](docs/RESEARCH_MAP.md) | Detailed research lineage and interpretation boundaries |
| [Experiment Index](docs/EXPERIMENT_INDEX.md) | Every preregistration, result, correction, and negative finding |
| [Research Audit and Next Strategy](docs/RESEARCH_AUDIT_AND_NEXT_STRATEGY_2026-08-06.md) | Paper-scale contributions, correction ledger, and next strategy |
| [Literature Audit](docs/LITERATURE_AUDIT_2026-08-06.md) | Primary-source literature audit and baseline requirements |
| [Provenance and History](docs/PROVENANCE_AND_HISTORY.md) | Extraction snapshot, post-extraction amendments, and historical-reading policy |
| [artifacts](artifacts/README.md) | Raw outputs retained for reproducibility |
| [Artifact Manifest](ARTIFACTS.sha256) | SHA-256 manifest for published artifacts |
| [Provenance](PROVENANCE.json) | Original extraction boundary and source hashes |

## Installation

Python 3.11 or newer is recommended.

    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    python -m pip install --upgrade pip
    python -m pip install -e ".[full]"

The exact symbolic gates use the base dependencies. The full extra adds the
JAX/Flax, dataset, and tokenizer dependencies required by the historical SSM
lineage. On Windows, the optional fused gathered-memory benchmark additionally
uses:

    python -m pip install -e ".[cuda-windows]"

The fused tests skip when CUDA or that optional dependency is unavailable.

## Validation

    python -m unittest discover -s tests -p "test_*.py"
    python tools/audit_math_docs.py
    python tools/verify_artifact_manifest.py

Pytest currently collects 362 maintained tests. The complete suite includes
several expensive exact certificates and was not replayed wholesale during the
2026-08-11 scalar-extension audit; the result documents list the targeted
passing subsets. The earlier bounded full run used six CPU cores, took 375.8
seconds including supervision, peaked at 4.074 GiB of process-tree resident
memory, and completed without crossing the 15 GiB watchdog. Read
[Reproducibility](docs/REPRODUCIBILITY.md) before comparing a rerun with a
frozen artifact.

## Scientific scope

The archive maintains four interpretation boundaries:

1. an exact algebraic certificate is not an empirical training result;
2. a finite numerical search is not a continuous proof;
3. a theorem on a constrained family is not the unrestricted theorem;
4. a mechanism-level SSM result is not a competitive language-model result.

The next mathematical task is a covariance-orbit reduction for the final
Cholesky residual in the unrestricted Dirac--Gram inequality. The memory track
has completed the transport-free compact-WY comparison, the full transported-
value compiler, the prospective Task B action replay, and the frozen large-slot
semantic/fused-gather campaign. Its next falsifier is a matched three-branch
model combining a recent window, fused selected blocks, and a compressed global
summary under equal state, parameter, token, and measured-compute budgets.

## Provenance and license

The original extraction covers 463 scientific files through source commit
a367a80. Checkpoints, transient logs, caches, virtual environments, and the
unrelated 44.8 MB historical language-model checkpoint are excluded. Later
research is explicitly post-extraction and is covered by ARTIFACTS.sha256 and
Git history.

Licensed under the GNU Affero General Public License v3.0. See [LICENSE](LICENSE).
