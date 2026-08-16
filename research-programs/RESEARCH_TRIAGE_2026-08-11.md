# Local research-programme triage

Date: 2026-08-11

This is a point-in-time execution triage across the seven canonical research
programmes. It selects work by the smallest honest remaining proof or
implementation obligation, not by the most ambitious surrounding conjecture.
Existing dirty work in the parent and nested repositories was treated as the
baseline and left intact.

## Selection

Programme 01 contained the lowest-hanging local obligation: the repository had
an exact real-type Schur-scan construction, but no canonical implementation of
the complex- and quaternionic-type division-algebra blocks. That gap is now
closed for an explicitly supplied irreducible type and multiplicity space.

The new construction implements

```text
x -> (g x) M + b
```

for `C` and `H`, with left action by `g` and right multiplicity mixing by `M`.
The factor order is preserved under chronological affine composition, including
the noncommutative quaternionic case. Exact centralizer calculations recover
dimensions 2 and 4 respectively, and deterministic float64 audits check
composition, parallel/recurrent scan parity, left/right commutation, and
float64 gradient parity.

Canonical evidence:

- [implementation](../Spin-Space-Research/src/division_schur_scan.py)
- [result and claim boundary](../Spin-Space-Research/docs/experiments/DIVISION_SCHUR_SCAN_RESULTS.md)
- [replay tests](../Spin-Space-Research/tests/test_division_schur_scan.py)
- [audit artifact](../Spin-Space-Research/artifacts/division_schur_scan_20260811.json)

This first result is an exact algebra and implementation result. By itself it
did **not** provide automatic isotypic decomposition or Schur-type detection,
integration into the maintained rotor model, or evidence of sequence-model
superiority.

## Same-day continuation

The first listed next priority is now complete within an explicit domain. The
[exact Schur-type detector](../Spin-Space-Research/docs/experiments/SCHUR_TYPE_DETECTION_RESULTS.md)
solves the commutant of supplied rational generators and extracts real,
complex, or generalized quaternion multiplication bases under complete
reducibility. It rejects a split dimension-two algebra, a doubled complex
irrep, and a missing complete-reducibility assumption.

## Second same-day continuation

The next listed Programme 01 obligation is also complete within a fail-closed
rational-splitting domain. The
[reducible isotypic compiler](../Spin-Space-Research/docs/experiments/REDUCIBLE_ISOTYPIC_DECOMPOSITION_RESULTS.md)
constructs exact commutant idempotents, certifies every irreducible
real/complex/quaternionic leaf, groups equivalent copies by exact intertwiner
spaces, and reconstructs aligned isotypic coordinates. Its maintained controls
include repeated `R`, `C`, and `H` types, their mixed direct sum,
`Cl(3,0) = 2V0 + 2V1`, and the Spin(9) quotient model `V1 + 2V5`, including
non-orthogonal rational conjugacies.

At this stage the compiler explicitly refused a nondivision leaf when its
bounded rational idempotent search could not split it. The next continuation
below adds the first declared algebraic field; noisy numerical decomposition
remains open.

## Third same-day continuation

The lowest remaining exact obligation was the first honest coefficient-field
extension. It is now complete for the declared positive ordered quadratic
field \(\mathbb Q(\sqrt2)\). The field layer provides exact membership,
ordered sign, sparse nullspace, rank, and projective normalization; the Schur
detector and reducible compiler propagate that field through commutants,
centers, minimal-polynomial projectors, intertwiners, and block alignment.

The decisive control is not a matrix whose square roots cancel away. For

\[
A=\begin{pmatrix}0&2\\1&0\end{pmatrix},\qquad A^2=2I,
\]

the rational compiler correctly refuses the unresolved two-dimensional leaf,
whereas the quadratic compiler constructs the two rank-one projectors
\((I\pm A/\sqrt2)/2\). In contrast, the control with square \(-2I\) remains a
single complex-type real irreducible after adjoining \(\sqrt2\). Exact dense
algebraic conjugacies and a \(\sqrt3\)-outside-field refusal gate protect the
same boundary. Native algebraic Spin(9) matrices then reproduce the slice
\(V_1\oplus V_5\) and full quotient \(V_1\oplus2V_5\) directly.

Canonical evidence:

- [design and algebraic rationale](../Spin-Space-Research/docs/ALGEBRAIC_EXTENSION_DESIGN.md)
- [result and controls](../Spin-Space-Research/docs/experiments/ALGEBRAIC_ISOTYPIC_DECOMPOSITION_RESULTS.md)
- [implementation](../Spin-Space-Research/src/exact_real_scalar_field.py)
- [replay tests](../Spin-Space-Research/tests/test_algebraic_isotypic_decomposition.py)

Generic number fields, algebraic towers, non-real embeddings, and noisy
decomposition remain open.

## Fourth same-day continuation

The same exact machinery was applied to actual Spin and Clifford modules rather
than only compiler fixtures. It certifies the three inequivalent real-type
Spin(8) modules before and after independent dense
\(\mathbb Q(\sqrt2)\) conjugacies, the restriction of the real Spin(9) spin
module to the two chiral Spin(8) sectors, and the faithful signature chain

\[
\mathrm{Cl}(3,0)\hookrightarrow\mathrm{Cl}^0(1,4)
\subset\mathrm{Cl}(1,4).
\]

The full \(\mathrm{Cl}(1,4)\) module splits into two quaternionic sectors; the
embedded \(\mathrm{Cl}(3,0)\) action on all 16 real coordinates is four copies
of a four-dimensional complex-type irreducible. This is a representation
theorem, not a Lorentzian rewrite or quality result for the maintained rotor
SSM.

Canonical evidence:

- [theorem narrative](../Spin-Space-Research/docs/manuscripts/CLIFFORD_SIGNATURE_EXTENSION.md)
- [implementation](../Spin-Space-Research/src/clifford_signature_extension.py)
- [replay tests](../Spin-Space-Research/tests/test_clifford_signature_extension.py)

## Fifth same-day continuation

A supplied invariant-theoretic reformulation of a four-term Machin compression
was audited as a new, independent Programme 01 arithmetic claim. Its useful
core survives after replacing the non-algebraic normalization (z/|z|) by the
rational norm-one-torus quotient (z/\bar z). Exact Gaussian factorization
shows that all split-prime valuation differences cancel and verifies

\[
(239+i)(7+i)^2(4+i)^2(268+i)^2
=10{,}317{,}661{,}250(1+i).
\]

The same audit proves the tangent-subtraction formula and nearest-integer
numerator bound for one Alferov-style step. It does not promote the supplied
geodesic-height, strict Lehmer descent, LLL, optimality, or novelty language.

Canonical evidence:

- [corrected theorem and boundary](01-associative-scan-algebra-and-compilers/GAUSSIAN_PHASE_LATTICE_AUDIT.md)
- [exact implementation](01-associative-scan-algebra-and-compilers/src/gaussian_phase_lattice.py)
- [replay tests](01-associative-scan-algebra-and-compilers/tests/test_gaussian_phase_lattice.py)

## Programme comparison

| Programme | Nearest open obligation | Triage outcome |
|---|---|---|
| 01 — Associative scan algebra and compilers | Exact division blocks, irreducible type detection, reducible reconstruction, and first algebraic scalar extension | **Selected and completed** for certified splittings over \(\mathbb Q\) and \(\mathbb Q(\sqrt2)\) under complete reducibility; general number fields and noisy decomposition remain open |
| 02 — Equivariant identification | Identification from less structured or noisy observations | Requires a new frozen empirical protocol and matched controls |
| 03 — Structured memory and retrieval | End-to-end selected-block FLA and broader systems validation | Requires model integration, training, and systems experiments |
| 04 — Triality/Clifford dynamics | Exact Spin(8)/Spin(9) restriction and signature hierarchy, then matched representation-dynamics tasks | **Exact algebraic layer completed**; empirical benefit still requires new matched datasets and baselines |
| 05 — Information geometry and Dirac--Gram sensing | Exact Spin(9) candidate optimality, second \(V_5\), and the Spin(8) octet determinant finite-radius interior | **The former Spin(9) `21/20` bottleneck was closed on 2026-08-12** by the complete coupled finite-radius certificate; the remaining gates are narrower and independent |
| 06 — Rotor noncommutative SSMs | Rotor-specific quality/compute evidence and fused kernels | Maintained recurrence gates already pass; remaining claims need substantial baselines or kernel work |
| 07 — Collatz inverse frontier | Resumable compact large scans, literature comparison, or asymptotic proof | Finite engineering is separable; asymptotic claims are not locally low-hanging |

## Historical Spin(9) route diagnosis — superseded 2026-08-12

> The diagnostic below remains an accurate record of the failed dense route.
> It is no longer the current theorem boundary. The sparse projective atlas and
> characteristic-zero lift in
> [`SPIN9_V1_V5_RECONSTRUCTION.md`](../Spin-Space-Research/docs/manuscripts/SPIN9_V1_V5_RECONSTRUCTION.md)
> subsequently proved the complete (21/20) finite-radius coupled-slice bound.

Programme 05 initially appeared equally local because
`spin9_v1_v5_gap.py` already contains an exact degree-84 control tensor on an
`85^3 = 614,125` coefficient cube. A depth-18 exact residual-box attempt did
not finish within the interactive window and was stopped. A bounded depth-6
diagnostic finished in about 159 seconds but left 27 boxes unresolved across
the two certificate branches.

That observation is a tooling diagnosis only:

- it does not disprove the `21/20` bound;
- it does not establish any sign-changing point;
- it was not promoted to an artifact or theorem;
- it indicates that the next attempt should exploit sparsity, symmetry-adapted
  charts, or a faster exact polynomial backend instead of deeper subdivision of
  the dense coefficient cube.

The already documented Spin(9) boundary and blow-up certificates remain valid
within their stated domains. The unrestricted interior step remains open.

## Next honest priorities

1. **Completed:** exact real/complex/quaternionic Schur-type detection and
   aligned reducible decomposition over \(\mathbb Q\), followed by the first
   native ordered scalar extension over \(\mathbb Q(\sqrt2)\).
2. **Completed:** exact Spin(8) module-separation controls, Spin(9)-to-Spin(8)
   chirality restriction, and the faithful
   \(\mathrm{Cl}(3,0)\hookrightarrow\mathrm{Cl}^0(1,4)\) embedding, without
   transferring those identities into a model claim.
3. Extend the compiler to a field-agnostic real-algebraic interface only when
   a concrete repository obligation supplies acceptance fixtures for ordering,
   embeddings, and factorization; do not claim arbitrary-number-field support
   from the quadratic case.
4. Replace Programme 05's dense tensor subdivision with a sparse or local-chart
   certificate engine before retrying the global `21/20` inequality.
5. Treat Programme 07's compact resumable storage as engineering evidence only;
   keep it separate from literature and asymptotic proof obligations.

This ordering is operational, not a ranking of the programmes' mathematical
importance.
