"""Machine-readable evidence contracts for the maintained research gates.

This module does not prove any mathematical claim.  Its purpose is narrower:
it prevents one evidence layer from being reported as another.  In particular,
an artifact hash is not an algebraic replay, a numerical search is not a
positivity certificate, and a passing implementation test is not an empirical
architecture comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STATUSES = {
    "operational",
    "validated_implementation",
    "empirical",
    "proved_exact",
    "proved_hybrid",
    "exact_negative",
    "exact_reduction",
    "numerical_only",
    "open",
}

EVIDENCE_LAYERS = {
    "artifact_hash",
    "static_contract",
    "resource_contract",
    "exact_arithmetic",
    "symbolic_identity",
    "exact_reconstruction",
    "positivity_certificate",
    "exact_counterexample",
    "external_theorem",
    "floating_point_falsifier",
    "raw_artifact",
    "checkpoint_replay",
    "implementation_parity",
    "negative_control",
    "multi_seed",
}


@dataclass(frozen=True)
class GateContract:
    """The acceptance boundary for one current claim family."""

    gate_id: str
    claim: str
    status: str
    evidence_layers: tuple[str, ...]
    test_suites: tuple[str, ...]
    artifacts: tuple[str, ...]
    boundary_obligations: tuple[str, ...]
    limitations: tuple[str, ...]
    replay_tier: str
    external_inputs: tuple[str, ...] = ()


GATES: tuple[GateContract, ...] = (
    GateContract(
        gate_id="repository_integrity",
        claim="Published bytes, relative links, notation delimiters, and bounded-run settings agree with the maintained archive contract.",
        status="operational",
        evidence_layers=("artifact_hash", "static_contract", "resource_contract"),
        test_suites=(
            "tests/test_artifact_manifest.py",
            "tests/test_documentation_contract.py",
            "tests/test_gate_contracts.py",
            "tests/test_spin8_resource_limits.py",
        ),
        artifacts=(),
        boundary_obligations=(
            "A hash match checks bytes, not mathematical truth.",
            "The documentation audit is syntactic and link-level, not a semantic proof.",
            "Resource tests verify configured ceilings, not worst-case complexity.",
        ),
        limitations=("This gate cannot promote any scientific claim.",),
        replay_tier="unit",
    ),
    GateContract(
        gate_id="ga_ssm_streaming_contract",
        claim="The maintained rotor SSM implementations expose constant recurrent state and agree across full, chunked, and token-streaming execution.",
        status="validated_implementation",
        evidence_layers=("implementation_parity", "negative_control"),
        test_suites=("tests/test_ga_ssm.py", "tests/test_rotor_ssm_torch.py"),
        artifacts=(),
        boundary_obligations=(
            "Identity initialization must retain a nonzero tangent gradient.",
            "Long scans must agree within declared floating-point tolerance.",
            "CUDA coverage is conditional on CUDA availability and is reported separately.",
        ),
        limitations=(
            "Correct recurrence and gradients do not establish a sequence-model advantage.",
        ),
        replay_tier="unit",
    ),
    GateContract(
        gate_id="finite_group_recurrence_evidence",
        claim="The finite-group harness implements the stated fixed-action, compiler, holonomy, and streaming controls and preserves the archived cohort results.",
        status="empirical",
        evidence_layers=(
            "implementation_parity",
            "negative_control",
            "raw_artifact",
            "multi_seed",
        ),
        test_suites=("tests/test_recurrence_harness.py",),
        artifacts=(
            "artifacts/mechanistic_a5_ga_holonomy_multiscale_dense_seed0_1500.json",
            "artifacts/q8_spinor_quality_gate_validation_dense_seeds10_19.json",
        ),
        boundary_obligations=(
            "Functional accuracy, positive margin, and raw homomorphism gates remain distinct.",
            "Held-out words do not by themselves test changed-generator transfer.",
            "Pure fixed-token operators structurally compose unseen bigrams.",
        ),
        limitations=(
            "Most training checkpoints are not published, so the archive verifies harness semantics and raw reports but cannot retrain every historical cohort.",
        ),
        replay_tier="artifact_only_empirical",
    ),
    GateContract(
        gate_id="triangular_schurscan",
        claim="A triangular bilinear recurrence with an equivariant intertwiner admits two associative affine scans and a finite homogeneous lift.",
        status="proved_exact",
        evidence_layers=(
            "exact_arithmetic",
            "symbolic_identity",
            "implementation_parity",
        ),
        test_suites=("tests/test_intertwiner_schurscan.py",),
        artifacts=("artifacts/intertwiner_schurscan_20260806.json",),
        boundary_obligations=(
            "The staged parallel scan must equal the sequential recurrence.",
            "The SO(3) control must reduce to the cross product.",
            "State-dependent feedback is excluded by the degree obstruction.",
        ),
        limitations=("This theorem is not specific evidence of a Spin(8) advantage.",),
        replay_tier="unit",
    ),
    GateContract(
        gate_id="triality_algebra_and_memory",
        claim="The maintained Spin(8) triality tensor is equivariant, unit-key binding is exactly invertible, and the staged memory recurrences obey their algebraic contracts.",
        status="proved_exact",
        evidence_layers=(
            "exact_arithmetic",
            "symbolic_identity",
            "implementation_parity",
        ),
        test_suites=("tests/test_foundational_contracts.py",),
        artifacts=("artifacts/intertwiner_schurscan_20260806.json",),
        boundary_obligations=(
            "Single-pair inversion is separated from multi-pair superposition capacity.",
            "Multiplicity-slot capacity is rank-limited by the number of channels.",
            "Any nonlinear cleanup remains outside the associative recurrence.",
        ),
        limitations=(
            "Exact binding is not a high-capacity VSA result and does not establish a retrieval advantage.",
        ),
        replay_tier="bounded_full",
    ),
    GateContract(
        gate_id="shared_family_retraction",
        claim="Jointly constrained action and address families complete held-out relational structure that independently normalized controls leave underidentified.",
        status="empirical",
        evidence_layers=(
            "raw_artifact",
            "multi_seed",
            "negative_control",
            "implementation_parity",
        ),
        test_suites=("tests/test_foundational_contracts.py",),
        artifacts=(
            "artifacts/spin8_blind_shared_action_seeds0_9.json",
            "artifacts/spin8_joint_sensor_retraction_seeds20_29.json",
            "artifacts/spin8_learned_address_seeds0_9.json",
            "artifacts/spin8_continuous_alias_seeds0_9.json",
        ),
        boundary_obligations=(
            "Independent controls must fit the supplied observations before failure is informative.",
            "Direct transport and binding paths must be tested separately to exclude bypasses.",
            "Logical-ID and continuous-alias regimes must not be conflated.",
        ),
        limitations=(
            "The direct-memory control also succeeds in the latent-address task; that result supports shared-family retraction, not a triality-specific advantage.",
        ),
        replay_tier="artifact_only_empirical",
    ),
    GateContract(
        gate_id="four_vs_five_probe_identifiability",
        claim="Four shared triality probes retain a positive-dimensional principal stabilizer, whereas every mixed five-probe allocation has a nonempty open free stratum.",
        status="proved_hybrid",
        evidence_layers=("exact_arithmetic", "symbolic_identity", "external_theorem"),
        test_suites=(
            "tests/test_global_five_probe_certificate.py",
            "tests/test_spin8_continuous_probe_orbits.py",
            "tests/test_spin8_coordinate_geometry.py",
        ),
        artifacts=(
            "artifacts/spin8_global_five_probe_certificate_20260806.json",
            "artifacts/spin8_continuous_probe_orbits_20260806.json",
            "artifacts/spin8_coordinate_geometry_20260806.json",
        ),
        boundary_obligations=(
            "Single-view and mixed-view allocations have different four-probe stabilizers.",
            "Generic freeness is not the claim that every five-probe configuration is free.",
            "Independent per-view actions are outside the shared-action theorem.",
        ),
        limitations=(
            "The theorem is an identifiability result, not a conditioning bound.",
        ),
        replay_tier="bounded_full",
        external_inputs=(
            "The principal-orbit theorem for smooth compact-group actions: every isotropy contains a conjugate of the principal isotropy, and the principal stratum is open and dense.",
        ),
    ),
    GateContract(
        gate_id="balanced_cayley_information_family",
        claim="The orthonormal balanced information family has the exact 8+8+8+4 block spectrum and determinant (1-z)^3(9-z)^2/1024, with z=c^2.",
        status="proved_hybrid",
        evidence_layers=(
            "exact_arithmetic",
            "symbolic_identity",
            "external_theorem",
        ),
        test_suites=(
            "tests/test_foundational_contracts.py",
            "tests/test_cayley_referee_package.py",
            "tests/test_spin8_publication_theorems.py",
        ),
        artifacts=(
            "artifacts/spin8_cayley_blocks_20260806.json",
            "artifacts/spin8_cayley_criteria_20260806.json",
            "artifacts/spin8_cayley_flag_20260806.json",
        ),
        boundary_obligations=(
            "The endpoints z=1 have exact rank 25 and three equal first-order losses.",
            "The Cayley-null point z=0 is regular in the oriented cover but a boundary after c and -c are identified.",
            "The exact flag calculation excludes an internal continuous split invariant.",
        ),
        limitations=(
            "Local Lie-rank calculations do not prove the global orbit classification.",
            "The family theorem does not establish global five-query optimality.",
        ),
        replay_tier="external_plus_exact",
        external_inputs=(
            "Classical cohomogeneity-one classification of the Spin(7) action on the oriented Grassmannian of four-planes.",
        ),
    ),
    GateContract(
        gate_id="balanced_local_exact_design",
        claim="The balanced equal-five configuration is a strict local exact-design optimum modulo symmetry.",
        status="proved_exact",
        evidence_layers=("exact_arithmetic", "symbolic_identity"),
        test_suites=("tests/test_spin8_five_query_local_geometry.py",),
        artifacts=("artifacts/spin8_five_query_local_geometry_20260806.json",),
        boundary_obligations=(
            "The quotient Hessian removes symmetry zero modes.",
            "A finite circle atlas checks non-coordinate tangent directions.",
        ),
        limitations=(
            "Local optimality is not global optimality across all allocations.",
        ),
        replay_tier="unit",
    ),
    GateContract(
        gate_id="approximate_design_domain",
        claim="The eight-support isotropic design is D-optimal in the approximate-design domain, and exact five-query weights cannot realize that shortcut.",
        status="proved_hybrid",
        evidence_layers=(
            "exact_arithmetic",
            "symbolic_identity",
            "negative_control",
            "external_theorem",
        ),
        test_suites=("tests/test_spin8_approximate_design_audit.py",),
        artifacts=("artifacts/spin8_approximate_design_audit_20260806.json",),
        boundary_obligations=(
            "Exact-design and approximate-design feasible sets are kept separate.",
            "The Kiefer-Wolfowitz sensitivity equality is checked exactly.",
        ),
        limitations=(
            "This does not settle the equal-weight five-query exact-design problem.",
        ),
        replay_tier="unit",
        external_inputs=(
            "The Kiefer--Wolfowitz general equivalence theorem for D-optimal approximate designs.",
        ),
    ),
    GateContract(
        gate_id="d4_24cell_bridge",
        claim="The maintained D4/24-cell bridge identities hold exactly, together with the recorded non-equivalence certificate.",
        status="proved_exact",
        evidence_layers=("exact_arithmetic", "symbolic_identity"),
        test_suites=("tests/test_spin8_d4_24cell_bridge.py",),
        artifacts=("artifacts/spin8_d4_24cell_bridge_20260806.json",),
        boundary_obligations=(
            "The bridge identity and the non-equivalence statement are tested independently.",
        ),
        limitations=(
            "The bridge does not transfer optimality between inequivalent design domains.",
        ),
        replay_tier="unit",
    ),
    GateContract(
        gate_id="octonion_operator_finite_groups",
        claim=(
            "The fixed Fano-plane left operators generate 2_+^(1+6), the "
            "signed basis automorphisms form the nonsplit 2^3.PSL(2,7), and "
            "their combined matrix closure is the split perfect "
            "2_+^(1+6):PSL(2,7) of order 21,504."
        ),
        status="proved_hybrid",
        evidence_layers=(
            "exact_arithmetic",
            "symbolic_identity",
            "external_theorem",
        ),
        test_suites=("tests/test_octonion_operator_groups.py",),
        artifacts=("artifacts/octonion_operator_groups_20260817.json",),
        boundary_obligations=(
            "Raw unit octonions remain a nonassociative Moufang loop; the theorem concerns associative matrix composition.",
            "The extraspecial type is fixed by the exact square count, not order alone.",
            "Both split-status claims exhaust every lift of one fixed generating quotient pair.",
            "External nomenclature is separated from the exact repository-specific closure and invariants.",
        ),
        limitations=(
            "The abstract groups are known and this is not a new finite simple group or a classification of all finite Spin(8) subgroups.",
            "The group theorem establishes no ML-quality or kernel-speed advantage.",
        ),
        replay_tier="bounded_full",
        external_inputs=(
            "The automorphism group of the Fano plane is PSL(2,7) isomorphic to GL(3,2), and the standard extraspecial-group Arf-count classification.",
        ),
    ),
    GateContract(
        gate_id="signed_star_dirac_gram",
        claim="The strengthened Dirac--Gram inequality and its complete equality classification hold on the full four-parameter signed-star ansatz.",
        status="proved_exact",
        evidence_layers=(
            "exact_arithmetic",
            "exact_reconstruction",
            "symbolic_identity",
            "positivity_certificate",
        ),
        test_suites=(
            "tests/test_foundational_contracts.py",
            "tests/test_spin8_publication_theorems.py",
        ),
        artifacts=(
            "artifacts/spin8_dirac_star_20260804.json",
            "artifacts/spin8_dirac_star_foundations_20260806.json",
            "artifacts/spin8_dirac_star_structure_20260806.json",
        ),
        boundary_obligations=(
            "Circle-quotient divisibility is proved before interpolation.",
            "Both orientation signs and disjoint exact grids are checked.",
            "Equality is classified as z=1 or the orthonormal star centre.",
        ),
        limitations=("Three residual Cholesky correlations lie outside this ansatz.",),
        replay_tier="expensive_exact",
    ),
    GateContract(
        gate_id="conditional_decorrelation_map",
        claim="Monotone removal of the selected residual correlations at fixed star coordinates and normalized Cayley invariant is false.",
        status="exact_negative",
        evidence_layers=("exact_arithmetic", "exact_counterexample"),
        test_suites=("tests/test_foundational_contracts.py",),
        artifacts=("artifacts/spin8_conditional_counterexample_20260804.json",),
        boundary_obligations=(
            "The rational witness includes an exact positive-definite Gram certificate.",
            "Only the specified deformation is falsified, not every invariant-preserving path.",
        ),
        limitations=(
            "The unrestricted Dirac--Gram inequality is not falsified by this witness.",
        ),
        replay_tier="unit",
    ),
    GateContract(
        gate_id="variable_cayley_one_edge",
        claim="The strengthened Dirac--Gram inequality and equality set hold on the variable-Cayley one-residual-edge family.",
        status="proved_exact",
        evidence_layers=(
            "exact_arithmetic",
            "exact_reconstruction",
            "symbolic_identity",
            "positivity_certificate",
        ),
        test_suites=(
            "tests/test_foundational_contracts.py",
            "tests/test_spin8_publication_theorems.py",
        ),
        artifacts=(
            "artifacts/spin8_dirac_one_edge_determinant_cache_20260806.json",
            "artifacts/spin8_dirac_one_edge_duffy_20260806.json",
            "artifacts/spin8_dirac_one_edge_equality_20260806.json",
        ),
        boundary_obligations=(
            "Both Duffy charts and their common boundary are certified.",
            "Exact holdouts are distinct from reconstruction nodes.",
            "Equality is z=1 or the orthonormal one-edge centre.",
        ),
        limitations=(
            "The second and third residual Cholesky edges remain outside this family.",
        ),
        replay_tier="expensive_exact",
    ),
    GateContract(
        gate_id="multiplicity_gauge",
        claim="The multiplicity-space gauge reduction and its rank statement hold exactly on the declared representation.",
        status="proved_exact",
        evidence_layers=("exact_arithmetic", "symbolic_identity"),
        test_suites=("tests/test_spin8_multiplicity_gauge.py",),
        artifacts=("artifacts/spin8_multiplicity_gauge_20260806.json",),
        boundary_obligations=("Gauge and physical directions are counted separately.",),
        limitations=(
            "A gauge count is not an optimization or generalization theorem.",
        ),
        replay_tier="unit",
    ),
    GateContract(
        gate_id="two_edge_exact_reconstruction",
        claim="All eight symmetry-allowed two-edge sector polynomials have been reconstructed exactly and checked on disjoint exact holdouts.",
        status="exact_reduction",
        evidence_layers=(
            "exact_arithmetic",
            "exact_reconstruction",
            "symbolic_identity",
        ),
        test_suites=(
            "tests/test_spin8_dirac_two_edge.py",
            "tests/test_spin8_dirac_two_edge_amplitude.py",
            "tests/test_spin8_dirac_two_edge_reconstruct.py",
            "tests/test_spin8_dirac_two_edge_shared_reconstruct.py",
        ),
        artifacts=(
            "artifacts/spin8_dirac_two_edge_all_sectors_coefficients_20260806.json",
            "artifacts/spin8_dirac_two_edge_all_sectors_holdouts_20260806.json",
            "artifacts/spin8_dirac_two_edge_amplitude_20260806.json",
        ),
        boundary_obligations=(
            "Walsh support follows from exact common-triality symmetries.",
            "Every determinant boundary branch has the declared rank loss.",
            "Reconstruction identity is separated from sign certification.",
        ),
        limitations=(
            "Exact polynomial recovery does not prove the recovered margins nonnegative.",
        ),
        replay_tier="expensive_exact",
    ),
    GateContract(
        gate_id="two_edge_local_kernel",
        claim="The second residual edge is locally nonnegative at the orthonormal equality line, while the proposed global quadratic-Schur proof strategy has an exact counterexample.",
        status="exact_negative",
        evidence_layers=(
            "exact_arithmetic",
            "symbolic_identity",
            "exact_counterexample",
        ),
        test_suites=(
            "tests/test_spin8_dirac_two_edge_kernel.py",
            "tests/test_spin8_dirac_two_edge_kernel_flint.py",
        ),
        artifacts=(
            "artifacts/spin8_dirac_two_edge_orthonormal_transverse_20260806.json",
            "artifacts/spin8_two_edge_kernel_flint_20260806.json",
        ),
        boundary_obligations=(
            "Odd transverse derivatives vanish on the equality line.",
            "The exact counterexample rejects only the quadratic-Schur certificate strategy.",
            "FLINT independently replays the SymPy jet arithmetic.",
        ),
        limitations=("Local nonnegativity does not imply global two-edge positivity.",),
        replay_tier="bounded_full",
    ),
    GateContract(
        gate_id="two_edge_finite_polynomial_gate",
        claim="The two-edge problem reduces reversibly to four degree-six and four degree-twelve radical-free polynomial inequalities, with a proved endpoint-jet structure.",
        status="exact_reduction",
        evidence_layers=(
            "exact_arithmetic",
            "symbolic_identity",
            "floating_point_falsifier",
        ),
        test_suites=(
            "tests/test_spin8_dirac_two_edge_finite.py",
            "tests/test_spin8_publication_theorems.py",
        ),
        artifacts=(
            "artifacts/spin8_dirac_two_edge_endpoints_20260806.json",
            "artifacts/spin8_two_edge_finite_falsifier_20260806.json",
            "artifacts/spin8_dirac_two_edge_atlas_20260807.json",
        ),
        boundary_obligations=(
            "Both signs introduced by radical elimination remain explicit.",
            "The i2=1 endpoint core and first transverse jet are checked exactly in all eight sectors.",
            "Interior GPU search is a falsifier only.",
        ),
        limitations=(
            "This reduction is now joined to a separate global atlas certificate; it still does not cover the final h residual.",
        ),
        replay_tier="bounded_full",
    ),
    GateContract(
        gate_id="two_edge_global_positivity",
        claim="All eight finite two-edge polynomial margins are nonnegative on their complete feasible domain.",
        status="proved_exact",
        evidence_layers=(
            "exact_arithmetic",
            "exact_reconstruction",
            "positivity_certificate",
        ),
        test_suites=(
            "tests/test_spin8_dirac_two_edge_finite.py",
            "tests/test_spin8_publication_theorems.py",
        ),
        artifacts=(
            "artifacts/spin8_dirac_two_edge_all_sectors_coefficients_20260806.json",
            "artifacts/spin8_dirac_two_edge_atlas_20260807.json",
        ),
        boundary_obligations=(
            "Both children of every triangular split must be retained in the cover.",
            "Every interval-indeterminate control must receive exact integer replay.",
            "The certificate covers non-vertex interiors and all chart boundaries.",
        ),
        limitations=(
            "The theorem is confined to the frozen h=0 two-edge family and does not classify its complete equality set.",
        ),
        replay_tier="expensive_exact",
    ),
    GateContract(
        gate_id="endpoint_octet_quadratic_schur",
        claim=(
            "All three quadratic principal-minor families of the adjacent "
            "Cayley-endpoint octet are nonnegative on the complete "
            "five-dimensional endpoint domain."
        ),
        status="proved_exact",
        evidence_layers=(
            "exact_arithmetic",
            "symbolic_identity",
            "positivity_certificate",
        ),
        test_suites=("tests/test_spin8_endpoint_octet_quadratic.py",),
        artifacts=(
            "artifacts/spin8_dirac_endpoint_octet_quadratic_0_global_20260808.json",
            "artifacts/spin8_dirac_endpoint_octet_quadratic_1_global_20260808.json",
            "artifacts/spin8_dirac_endpoint_octet_quadratic_2_global_20260808.json",
        ),
        boundary_obligations=(
            "All three nontrivial Klein-four quadratic modes must be certified independently.",
            "Each assembled certificate must hash-bind and replay its eight source artifacts.",
            "The residual equality neighbourhood must be covered by all five max-coordinate blow-up charts.",
        ),
        limitations=(
            "This quadratic theorem alone does not address the cubic or determinant; later gates prove the cubic and both determinant endpoint faces, while the determinant interior remains open.",
        ),
        replay_tier="expensive_exact",
    ),
    GateContract(
        gate_id="endpoint_octet_cubic_global",
        claim=(
            "The adjacent-octet cubic principal-minor family is nonnegative "
            "on its complete five-dimensional endpoint domain."
        ),
        status="proved_exact",
        evidence_layers=(
            "exact_arithmetic",
            "symbolic_identity",
            "positivity_certificate",
        ),
        test_suites=(
            "tests/test_spin8_endpoint_octet_cubic.py",
            "tests/test_endpoint_octet_runtime_replay.py",
        ),
        artifacts=(
            "artifacts/spin8_dirac_endpoint_octet_cubic_yzero_20260808.json",
            "artifacts/spin8_dirac_endpoint_octet_cubic_tangent_20260810.json",
            "artifacts/spin8_dirac_endpoint_octet_cubic_corner_20260810.json",
            "artifacts/spin8_dirac_endpoint_octet_cubic_boundary_00010_atlas_20260810.json",
            "artifacts/spin8_dirac_endpoint_octet_cubic_atlas_nested_00001_complete_20260810.json",
            "artifacts/spin8_dirac_endpoint_octet_cubic_coarse_atlas_20260811.json",
            "artifacts/spin8_dirac_endpoint_octet_cubic_certificate_20260811.json",
        ),
        boundary_obligations=(
            "The y=0 face must be proved for every value of its four remaining variables.",
            "The tangent factor identities and radical-factor sign argument must replay exactly.",
            "All five max-coordinate blow-up charts and every selected-face atlas must form a complete hash-bound cover.",
            "All sixteen cells of the irreducible box-00010 selected face must certify independently.",
            "All thirty-two children of first-level box 00001 must certify, with only child 00001 delegated to the independently proved finite-radius corner.",
            "The thirty remaining first-level boxes must pass exact batched Bernstein audits, forming the complete ordered 32-box cover.",
            "The characteristic-zero endpoint decomposition must be recomputed with zero division remainder before the quotient atlas is assembled into the global cubic theorem.",
        ),
        limitations=(
            "This theorem proves the cubic principal minor, not the fourth-order second-block determinant, the complete endpoint octet, or unrestricted Dirac--Gram positivity.",
        ),
        replay_tier="expensive_exact",
    ),
    GateContract(
        gate_id="endpoint_octet_determinant_endpoints",
        claim=(
            "The adjacent-octet fourth-order Schur determinant has the exact "
            "radical-free five-variable reconstruction and is nonnegative on "
            "both endpoint faces y=0 and y=1."
        ),
        status="proved_exact",
        evidence_layers=(
            "exact_arithmetic",
            "exact_reconstruction",
            "symbolic_identity",
            "positivity_certificate",
        ),
        test_suites=("tests/test_spin8_endpoint_octet_determinant.py",),
        artifacts=("artifacts/spin8_dirac_endpoint_octet_determinant_20260812.json",),
        boundary_obligations=(
            "The radical-free formula must agree with both the generic four-by-four determinant and the product of the four Walsh eigenvalues.",
            "The y=0 face must be covered by all fifteen certifying coarse cells and all sixteen children of the sole rejected coarse cell.",
            "The y=1 face may be delegated only through the hash-bound exact identity Z=X^2 and the proved X-block theorem.",
        ),
        limitations=(
            "The open interior 0<y<1 still prevents promotion of the complete adjacent endpoint octet and unrestricted Dirac--Gram theorem.",
        ),
        replay_tier="expensive_exact",
    ),
    GateContract(
        gate_id="endpoint_octet_determinant_tangent",
        claim=(
            "After the exact descent t=y^2, the adjacent-octet determinant "
            "has an order-eight nonnegative tangent form at its remaining "
            "equality corner."
        ),
        status="proved_exact",
        evidence_layers=(
            "exact_arithmetic",
            "exact_reconstruction",
            "symbolic_identity",
            "positivity_certificate",
        ),
        test_suites=("tests/test_spin8_endpoint_octet_determinant.py",),
        artifacts=(
            "artifacts/spin8_dirac_endpoint_octet_determinant_tangent_20260812.json",
        ),
        boundary_obligations=(
            "Every power of y must be even before the exact substitution t=y^2 is made.",
            "The first nonzero Taylor form at ud=ue=ug=ui=0, t=1 must be extracted without floating-point truncation.",
            "The order-eight form must equal 2^48 times the square of the manifest radical-factor quartic.",
        ),
        limitations=(
            "This proves the exceptional divisor only; a finite-radius punctured-neighbourhood certificate and the rest of the determinant interior remain open.",
        ),
        replay_tier="expensive_exact",
    ),
    GateContract(
        gate_id="endpoint_octet_determinant_coordinate_boundary",
        claim=(
            "The adjacent-octet fourth-order Schur determinant is "
            "nonnegative on the complete ten-face coordinate boundary of "
            "its five-cube."
        ),
        status="proved_exact",
        evidence_layers=(
            "exact_arithmetic",
            "symbolic_identity",
            "artifact_hash",
        ),
        test_suites=("tests/test_spin8_endpoint_octet_determinant_boundary.py",),
        artifacts=(
            "artifacts/spin8_dirac_endpoint_octet_determinant_boundary_20260816.json",
        ),
        boundary_obligations=(
            "On each of the nine non-y=1 faces, tau must vanish and at most one nontrivial forced radical square may survive.",
            "The zero-mode and all three one-mode specializations of the generic determinant must be exact perfect squares.",
            "The y=1 face may be delegated only through the hash-bound Z=X^2 endpoint theorem.",
            "The artifact must retain an explicit false flag for strict-interior determinant positivity.",
        ),
        limitations=(
            "This proves the coordinate boundary only; all-five-coordinate strict-interior positivity, the complete adjacent octet, and unrestricted Dirac--Gram positivity remain open.",
        ),
        replay_tier="bounded_full",
    ),
    GateContract(
        gate_id="endpoint_octet_degree_matched_selector_route",
        claim=(
            "The degree-matched two-endpoint quotient Q24 in "
            "D=D0(1-t)^24+D1*t^24+t(1-t)Q24 is nonnegative."
        ),
        status="exact_negative",
        evidence_layers=(
            "exact_arithmetic",
            "symbolic_identity",
            "exact_counterexample",
        ),
        test_suites=("tests/test_spin8_endpoint_octet_determinant.py",),
        artifacts=(
            "artifacts/spin8_dirac_endpoint_octet_determinant_tangent_20260812.json",
        ),
        boundary_obligations=(
            "The selector identity and zero division remainder must be checked exactly.",
            "The rational witness must lie in the five-cube and evaluate Q24 to a strictly negative exact numerator.",
            "The negative quotient value must not be reported as a negative determinant value.",
        ),
        limitations=(
            "Only this over-strong sufficient decomposition is disproved; determinant nonnegativity remains open.",
        ),
        replay_tier="expensive_exact",
    ),
    GateContract(
        gate_id="endpoint_octet_central_core_dominance",
        claim=(
            "The trivial endpoint-octet Walsh amplitude strictly dominates "
            "the sum of the seven nontrivial absolute amplitudes on "
            "[1/4,3/4]^5, so every physical margin is strictly positive."
        ),
        status="proved_exact",
        evidence_layers=(
            "exact_arithmetic",
            "positivity_certificate",
            "artifact_hash",
        ),
        test_suites=("tests/test_spin8_endpoint_octet_core_dominance.py",),
        artifacts=(
            "artifacts/spin8_dirac_endpoint_octet_core_dominance_20260816.json",
        ),
        boundary_obligations=(
            "The 32 dyadic boxes must form the complete Cartesian partition of [1/4,3/4]^5.",
            "Every residual and forced-square bound must come from an exact Bernstein convex-hull transform.",
            "Every square-root upper bound must be an outward dyadic ceiling verified by exact squaring.",
            "The trivial-amplitude lower bound must exceed the sum of all seven nontrivial absolute upper bounds on every box.",
        ),
        limitations=(
            "This strict theorem covers only the central core; the boundary collars, complete adjacent octet, unrestricted Dirac--Gram inequality, and global exact-design problem remain open.",
        ),
        replay_tier="bounded_full",
    ),
    GateContract(
        gate_id="endpoint_octet_extended_core_dominance",
        claim=(
            "An adaptive 2,140-leaf exact atlas proves that the trivial "
            "endpoint-octet Walsh amplitude strictly dominates the seven "
            "nontrivial absolute amplitudes on [1/8,7/8]^5."
        ),
        status="proved_exact",
        evidence_layers=(
            "exact_arithmetic",
            "positivity_certificate",
            "artifact_hash",
        ),
        test_suites=("tests/test_spin8_endpoint_octet_core_dominance_atlas.py",),
        artifacts=(
            "artifacts/spin8_dirac_endpoint_octet_core_dominance_atlas_20260816.json",
        ),
        boundary_obligations=(
            "The 32 affine roots must cover [1/8,7/8]^5 and every rejected node must delegate to all 32 five-axis children.",
            "Every one of the 2,140 leaves must have a strictly positive exact rational dominance gap.",
            "Every Bernstein transform and outward radical bound must be recomputed by the full source harness; the compact test verifies the stored summaries and complete prefix tree.",
            "No box may remain unresolved at the frozen maximum refinement depth four.",
        ),
        limitations=(
            "This strict theorem does not cover the width-1/8 boundary collars, complete adjacent octet, unrestricted Dirac--Gram inequality, or global exact-design problem.",
        ),
        replay_tier="expensive_exact",
    ),
    GateContract(
        gate_id="global_five_query_exact_design",
        claim="The balanced equal-five allocation is globally D-optimal among all exact five-query allocations and nonorthogonal probes.",
        status="open",
        evidence_layers=("floating_point_falsifier", "raw_artifact", "multi_seed"),
        test_suites=("tests/test_spin8_gpu_design_audit.py",),
        artifacts=("artifacts/spin8_gpu_design_cohort_20260806.json",),
        boundary_obligations=(
            "Every allocation, nonorthogonal interior deformation, and rank-deficient boundary must be addressed.",
            "Kiefer-Wolfowitz approximate-design optimality cannot substitute for this exact-design gate.",
        ),
        limitations=(
            "Open: the current GPU campaign is a counterexample search only.",
        ),
        replay_tier="open",
    ),
    GateContract(
        gate_id="final_residual_exact_bridge",
        claim="The unrestricted chart has a 16-sector sign reduction of degree at most four in h^2, and the complete h-extension of the former equality slice satisfies the strengthened Dirac--Gram inequality.",
        status="proved_exact",
        evidence_layers=(
            "exact_arithmetic",
            "symbolic_identity",
            "positivity_certificate",
        ),
        test_suites=("tests/test_spin8_publication_theorems.py",),
        artifacts=("artifacts/spin8_dirac_final_residual_20260807.json",),
        boundary_obligations=(
            "The exact equality-slice determinant must be reduced in the two circle relations.",
            "Every floating-positive near-singular candidate must be rationalized and replayed exactly.",
            "The sign reduction must retain all 16 quotient characters and both h and H boundary factors.",
        ),
        limitations=(
            "The exact positivity theorem is confined to a=d=e=g=i=0; the 16-sector reduction does not certify global sign in the other six variables.",
        ),
        replay_tier="bounded_full",
    ),
    GateContract(
        gate_id="unrestricted_dirac_gram",
        claim="The strengthened Dirac--Gram inequality holds on the unrestricted feasible Gram--Cayley domain.",
        status="open",
        evidence_layers=(
            "exact_arithmetic",
            "symbolic_identity",
            "exact_reconstruction",
            "floating_point_falsifier",
        ),
        test_suites=(
            "tests/test_spin8_dirac_two_edge_finite.py",
            "tests/test_spin8_publication_theorems.py",
        ),
        artifacts=(
            "artifacts/spin8_dirac_two_edge_atlas_20260807.json",
            "artifacts/spin8_dirac_final_residual_20260807.json",
            "artifacts/spin8_dirac_unrestricted_structure_20260807.json",
            "artifacts/spin8_dirac_unrestricted_comparison_20260807.json",
            "artifacts/spin8_dirac_unrestricted_tangent_20260807.json",
            "artifacts/spin8_dirac_unrestricted_core_20260807.json",
            "artifacts/spin8_dirac_unrestricted_energy_20260807.json",
            "artifacts/spin8_dirac_endpoint_klein_face_20260807.json",
            "artifacts/spin8_two_edge_finite_falsifier_20260806.json",
        ),
        boundary_obligations=(
            "All 16 final-residual sectors, non-vertex interiors, and singular boundaries must receive a domain-wide sign certificate.",
        ),
        limitations=(
            "Open: the exact equality slice and final-axis degree reduction do not imply global positivity in the other six variables.",
        ),
        replay_tier="open",
    ),
    GateContract(
        gate_id="unrestricted_polynomial_identity",
        claim="The unrestricted Dirac--Gram margin has an exact sixteen-sector seven-variable polynomial representation, exact local endpoint control, and an exact global full-sector Fourier-energy bound.",
        status="exact_reduction",
        evidence_layers=(
            "exact_arithmetic",
            "symbolic_identity",
            "exact_reconstruction",
        ),
        test_suites=("tests/test_spin8_publication_theorems.py",),
        artifacts=(
            "artifacts/spin8_dirac_unrestricted_structure_20260807.json",
            "artifacts/spin8_dirac_unrestricted_coefficients_20260807/alpha_summary.json",
            "artifacts/spin8_dirac_unrestricted_coefficients_20260807/beta_summary.json",
            "artifacts/spin8_dirac_unrestricted_comparison_20260807.json",
            "artifacts/spin8_dirac_unrestricted_tangent_20260807.json",
            "artifacts/spin8_dirac_unrestricted_core_20260807.json",
            "artifacts/spin8_dirac_unrestricted_energy_20260807.json",
            "artifacts/spin8_dirac_endpoint_klein_face_20260807.json",
        ),
        boundary_obligations=(
            "The two complete coefficient maps must agree exactly on disjoint rational grids.",
            "Fresh rational holdouts must recompute all sixteen sectors from direct determinants.",
            "The calibrated endpoint requires a weighted fourth-order blow-up because the tangent cone is degenerate there.",
            "The coupled-core Bernstein theorem is restricted to c^2<=2/3 and does not absorb the other thirteen sectors.",
            "The full-sector Fourier-energy theorem controls the RMS Walsh deviation on the complete seven-cube; an L2 bound does not establish every physical margin.",
            "The first four elementary-symmetric orientation invariants are globally nonnegative; e5 through e16 remain open.",
            "The complete ua=uh=0, c^2=1 endpoint face is positive by an exact Klein-four group-circulant principal-minor certificate; other endpoint faces remain open.",
        ),
        limitations=(
            "Exact reconstruction and local positivity do not prove global nonnegativity on the seven-cube.",
        ),
        replay_tier="expensive_exact",
    ),
    GateContract(
        gate_id="triality_specific_ml_advantage",
        claim="Triality transport improves state efficiency, extrapolation, sample efficiency, or measured throughput over matched modern memory baselines.",
        status="open",
        evidence_layers=("raw_artifact", "negative_control"),
        test_suites=(
            "tests/test_recurrence_harness.py",
            "tests/test_foundational_contracts.py",
            "tests/test_matched_retrieval_campaign.py",
        ),
        artifacts=(
            "artifacts/spin8_learned_address_seeds0_9.json",
            "artifacts/matched_retrieval_campaign_synthesis_20260810.json",
            "artifacts/matched_retrieval_campaign_synthesis_task_b_closed_20260810.json",
        ),
        boundary_obligations=(
            "Direct slot memory, delta-rule memory, Householder transport, diagonal complex SSMs, and measured-throughput controls must be budget matched.",
            "A shared-family retraction win is not automatically a triality win.",
        ),
        limitations=(
            "Open: the frozen local campaign finds no triality-specific overwrite-law advantage; the positive shared-action result is a representation-prior result, not a storage-capacity result.",
            "Large-slot overlapping-semantic retrieval and matched model-level quality remain untested.",
        ),
        replay_tier="open",
    ),
    GateContract(
        gate_id="matched_learned_retrieval_campaign",
        claim="On the frozen synthetic alias task, hard/discretized 64-scalar slots are more robust than the matched learned-key delta row, while oracle delta remains exact and direct/triality hard slots remain gauge-equivalent.",
        status="empirical",
        evidence_layers=(
            "raw_artifact",
            "implementation_parity",
            "negative_control",
            "multi_seed",
        ),
        test_suites=(
            "tests/test_schurscan_delta_memory.py",
            "tests/test_matched_learned_retrieval.py",
            "tests/test_matched_retrieval_campaign.py",
            "tests/test_task_b_delta_action_replay.py",
            "tests/test_task_b_paired_action_replication.py",
        ),
        artifacts=(
            "artifacts/matched_learned_retrieval_task_a_seeds0_9.json",
            "artifacts/matched_memory_cores_cuda_rtx2070s_20260810.json",
            "artifacts/matched_retrieval_campaign_synthesis_20260810.json",
            "artifacts/task_b_delta_action_replay_seeds0_9.json",
            "artifacts/task_b_paired_action_replication_seeds20_29.json",
            "artifacts/matched_retrieval_campaign_synthesis_task_b_closed_20260810.json",
        ),
        boundary_obligations=(
            "Oracle delta must remain exact before learned-delta failure is assigned to address inference.",
            "Direct and triality rows must share identical routes, state size, events, and supplied actions.",
            "Hot, cold, overwrite, corruption, and sample-efficiency cohorts must remain separate.",
            "The eager CUDA tier must not be presented as a fused compact-WY comparison.",
        ),
        limitations=(
            "This is a controlled synthetic address task and does not establish production, language-model, or hardware-general superiority.",
            "The historical metric-only Task B replay failed its strict reproduction gate because learned parameters were not retained; the separate prospective seeds 20--29 cohort closes the action row without revising that failure.",
        ),
        replay_tier="artifact_only_empirical",
    ),
    GateContract(
        gate_id="task_b_shared_action_replication",
        claim="On the frozen partial-view synthetic task, a shared Spin(8) action family completes the held-out negative action and remains exact through direct and delta retrieval, while a parameter-richer independently fitted family does not.",
        status="empirical",
        evidence_layers=(
            "raw_artifact",
            "implementation_parity",
            "negative_control",
            "multi_seed",
        ),
        test_suites=(
            "tests/test_task_b_delta_action_replay.py",
            "tests/test_task_b_paired_action_replication.py",
            "tests/test_matched_retrieval_campaign.py",
        ),
        artifacts=(
            "artifacts/task_b_delta_action_replay_seeds0_9.json",
            "artifacts/task_b_paired_action_replication_seeds20_29.json",
            "artifacts/matched_retrieval_campaign_synthesis_task_b_closed_20260810.json",
        ),
        boundary_obligations=(
            "The failed historical replay and the prospective replication remain separate cohorts with separate verdicts.",
            "Direct and delta rows receive identical hard routes, actions, writes, queries, and event streams.",
            "Shared and independent actions fit the supplied observations before held-out completion is compared.",
            "Per-seed artifacts retain action coordinates, action matrices, router weights, training reports, and replay checks.",
        ),
        limitations=(
            "This supports a shared representation prior on a synthetic partial-action task, not extra triality storage capacity or a superior delta update law.",
            "The maintained SO(3) control prevents interpreting the mechanism as exceptional to triality without further matched equivariant baselines.",
        ),
        replay_tier="artifact_only_empirical",
    ),
    GateContract(
        gate_id="hierarchical_and_transported_memory_campaign",
        claim="The maintained same-router hierarchy, Spin(9) memory diagnostic, co-moving delta compiler, and independently aggregated CUDA measurements satisfy their declared implementation and frozen empirical contracts.",
        status="empirical",
        evidence_layers=(
            "raw_artifact",
            "implementation_parity",
            "negative_control",
            "multi_seed",
        ),
        test_suites=(
            "tests/test_aggregate_hierarchical_matched_retrieval.py",
            "tests/test_benchmark_matched_memory_cores.py",
            "tests/test_benchmark_replication_contracts.py",
            "tests/test_hierarchical_matched_retrieval.py",
            "tests/test_schurscan_comoving_delta.py",
            "tests/test_spin9_clifford_memory.py",
        ),
        artifacts=(
            "artifacts/hierarchical_matched_retrieval_seeds0_9.json",
            "artifacts/hierarchical_matched_retrieval_adversarial_seeds0_9.json",
            "artifacts/spin9_clifford_memory_boundary_20260810.json",
            "artifacts/matched_memory_cores_cuda_rtx2070s_frozen_aggregate_20260810.json",
            "artifacts/fla_delta_rule_cuda_rtx2070s_frozen_aggregate_20260810.json",
            "artifacts/comoving_fla_frozen_aggregate_20260810.json",
        ),
        boundary_obligations=(
            "Direct and delta rows share the same router before update-law comparisons are interpreted.",
            "The adversarial key-noise frontier remains distinct from the separable primary cohort.",
            "Spin(9) bind/unbind and Hopf diagnostics do not imply a same-width capacity advantage.",
            "Tuning, frozen selection, and independent measurement processes remain disjoint.",
            "The co-moving compiler's 128-scalar state is not presented as state-matched to 64-scalar direct slots.",
        ),
        limitations=(
            "This older campaign idealizes gathered-block bandwidth; the separately classified large-slot fused-gather gate measures a physical sparse-state implementation.",
            "The CUDA result is hardware- and implementation-specific and does not establish model-level quality or triality-specific capacity.",
        ),
        replay_tier="artifact_only_empirical",
    ),
    GateContract(
        gate_id="large_slot_semantic_fused_gather_memory",
        claim="On the frozen 64-slot overlapping-semantic task, a supplied-frame shared router completes missing view cells, block routing improves direct and delta retrieval, and a fused gathered-state recurrence beats matched eager controls on the named GPU.",
        status="empirical",
        evidence_layers=(
            "raw_artifact",
            "implementation_parity",
            "negative_control",
            "multi_seed",
        ),
        test_suites=(
            "tests/test_aggregate_fused_gathered_block_memory_benchmarks.py",
            "tests/test_aggregate_gathered_block_memory_benchmarks.py",
            "tests/test_aggregate_large_slot_semantic_hierarchy.py",
            "tests/test_benchmark_fused_gathered_block_memory.py",
            "tests/test_benchmark_gathered_block_memory.py",
            "tests/test_large_slot_semantic_hierarchy.py",
        ),
        artifacts=(
            "artifacts/large_slot_semantic_hierarchy_seeds30_39.json",
            "artifacts/gathered_block_memory_cuda_aggregate_20260810.json",
            "artifacts/fused_gathered_block_memory_cuda_aggregate_20260810.json",
            "artifacts/fused_gathered_block_memory_recurrent_diagnostic_20260810.json",
        ),
        boundary_obligations=(
            "The supplied inverse Spin(8) frame is not reported as learned action discovery.",
            "Shared and three-times-larger independent routers receive the same examples, optimization budget, and canonicalization.",
            "Direct and delta rows share routes, values, streams, and queries before update-law comparisons are interpreted.",
            "Eager masked/gathered parity and fused/eager parity pass before any timing is interpreted.",
            "Development, frozen quality, eager timing, fused timing, and post-registered recurrent diagnostic cohorts remain separately labelled.",
        ),
        limitations=(
            "The shared-router result uses a designed missing-label/view split and supplied action frames; it does not establish representation discovery or extra storage capacity.",
            "The fused kernel is an inference-only RTX 2070 SUPER result and does not establish gradients, language-model quality, cross-hardware thresholds, or triality-specific capacity.",
        ),
        replay_tier="artifact_only_empirical",
    ),
    GateContract(
        gate_id="independent_exact_backend_crosscheck",
        claim="Selected publication-critical rational identities agree between SymPy and python-flint implementations.",
        status="operational",
        evidence_layers=("exact_arithmetic", "negative_control"),
        test_suites=(
            "tests/test_spin8_flint_crosscheck.py",
            "tests/test_spin8_publication_theorems.py",
        ),
        artifacts=(
            "artifacts/spin8_flint_crosscheck_20260806.json",
            "artifacts/spin8_publication_flint_crosscheck_20260806.json",
        ),
        boundary_obligations=(
            "Crosschecks must reconstruct independently rather than deserialize one backend's coefficients into the other.",
        ),
        limitations=(
            "Backend agreement supports arithmetic reliability; it is not a proof of an unstated inference.",
        ),
        replay_tier="unit",
    ),
    GateContract(
        gate_id="schurscan_backend_benchmark",
        claim="The maintained eager SchurScan benchmark records work, parity, and bounded CPU/CUDA forward and training timings for the stated hardware and tensor programs.",
        status="empirical",
        evidence_layers=("raw_artifact", "implementation_parity", "negative_control"),
        test_suites=("tests/test_benchmark_intertwiner_schurscan.py",),
        artifacts=(
            "artifacts/intertwiner_schurscan_cpu_i7_9700k_20260807.json",
            "artifacts/intertwiner_schurscan_cuda_rtx2070s_20260807.json",
            "artifacts/intertwiner_schurscan_cpu_training_i7_9700k_20260807.json",
            "artifacts/intertwiner_schurscan_cuda_training_rtx2070s_20260807.json",
        ),
        boundary_obligations=(
            "Tree and Hillis--Steele implementations must agree with the sequential recurrence before timing is interpreted.",
            "Work counts, dependency depth, and observed latency remain separate quantities.",
            "Forward and full-gradient timing are reported separately on the named hardware.",
        ),
        limitations=(
            "These eager PyTorch measurements do not establish fused-kernel, production-throughput, or hardware-general superiority.",
        ),
        replay_tier="artifact_only_empirical",
    ),
    GateContract(
        gate_id="schurscan_equivariant_identification",
        claim="A known one-dimensional equivariant bilinear family extrapolates from a proper coordinate subspace in the frozen Spin(8) and SO(3) teacher tasks, while a generic tensor needs explicit group augmentation to identify the missing directions.",
        status="empirical",
        evidence_layers=(
            "raw_artifact",
            "implementation_parity",
            "negative_control",
            "multi_seed",
        ),
        test_suites=("tests/test_intertwiner_schurscan_equivariant_identification.py",),
        artifacts=(
            "artifacts/intertwiner_schurscan_equivariant_identification_20260810.json",
        ),
        boundary_obligations=(
            "The restricted and generic fits must receive identical unaugmented endpoints before their orbit errors are compared.",
            "The augmented generic row receives four times as many labelled endpoints and is not a sample-matched competitor.",
            "Spin(8) and SO(3) must both be reported so a generic intertwiner-prior result is not presented as triality-specific.",
        ),
        limitations=(
            "The group representations and complete intertwiner direction are supplied; this is teacher-aligned identification, not representation discovery or a matched retrieval result.",
        ),
        replay_tier="artifact_only_empirical",
    ),
    GateContract(
        gate_id="schurscan_memory_scanner_benchmark",
        claim="The maintained eager memory-scanner benchmark records correctness-qualified CPU/CUDA timings and allocations for structured, local-block homogeneous, and dense compilers of one identical 64-scalar recurrence.",
        status="empirical",
        evidence_layers=(
            "raw_artifact",
            "implementation_parity",
            "negative_control",
        ),
        test_suites=(
            "tests/test_benchmark_schurscan_memory_scanners.py",
            "tests/test_spin8_triality_direct_memory_equivalence.py",
        ),
        artifacts=(
            "artifacts/schurscan_memory_scanners_cpu_i7_9700k_20260810.json",
            "artifacts/schurscan_memory_scanners_cpu_long_replication_20260810.json",
            "artifacts/schurscan_memory_scanners_cuda_rtx2070s_20260810.json",
            "artifacts/schurscan_memory_scanners_cuda_long_replication_20260810.json",
            "artifacts/schurscan_memory_scanners_cpu_optimized_20260810.json",
            "artifacts/schurscan_memory_scanners_cpu_optimized_long_replication_20260810.json",
            "artifacts/schurscan_memory_scanners_cuda_optimized_20260810.json",
            "artifacts/schurscan_memory_scanners_cuda_optimized_long_replication_20260810.json",
        ),
        boundary_obligations=(
            "Every timed backend must agree with the sequential slot recurrence before its timing is interpreted.",
            "Prepacked scan-only and end-to-end materialization timings must remain separately labelled.",
            "Device, dtype, shape, tree schedule, work, and incremental allocation must remain visible rather than being collapsed into one winner claim.",
        ),
        limitations=(
            "These eager measurements do not establish fused-kernel, hardware-general, modern-delta, or triality-specific memory superiority.",
        ),
        replay_tier="artifact_only_empirical",
    ),
    GateContract(
        gate_id="spin9_clifford_sensing_core",
        claim="The maintained real Spin(9) Clifford system, frozen spinor-rank witnesses, symmetric conditioning curve, and isotropy branching identities satisfy their exact algebraic contracts.",
        status="proved_exact",
        evidence_layers=("exact_arithmetic", "symbolic_identity"),
        test_suites=(
            "tests/test_spin9_dirac_clifford.py",
            "tests/test_spin9_three_spinor_conditioning.py",
            "tests/test_spin9_three_spinor_symmetry.py",
        ),
        artifacts=(
            "artifacts/spin9_dirac_clifford_gate_20260807.json",
            "artifacts/spin9_three_spinor_conditioning_20260807.json",
            "artifacts/spin9_three_spinor_symmetry_20260807.json",
        ),
        boundary_obligations=(
            "The nine Clifford involutions and the Spin(8) restriction are checked coefficientwise.",
            "Frozen rank witnesses and generic stabilizer statements are not conflated.",
            "The spectral factorization is scoped to the symmetric one-parameter family.",
        ),
        limitations=(
            "These exact identities do not prove the unrestricted global three-spinor design optimum or any sequence-model advantage.",
        ),
        replay_tier="bounded_full",
    ),
    GateContract(
        gate_id="spin9_frame_and_local_design",
        claim="The Spin(9) information map admits the stated frame-operator reduction, and the symmetric candidate has an exact negative-definite quotient Hessian on the complete local rank-three stratum modulo Spin(9).",
        status="proved_exact",
        evidence_layers=("exact_arithmetic", "symbolic_identity"),
        test_suites=(
            "tests/test_spin9_frame_operator.py",
            "tests/test_spin9_grassmann_slice.py",
            "tests/test_spin9_local_hessian.py",
        ),
        artifacts=(
            "artifacts/spin9_frame_operator_20260807.json",
            "artifacts/spin9_grassmann_slice_20260807.json",
            "artifacts/spin9_local_hessian_exact.json",
        ),
        boundary_obligations=(
            "Frame rank and information-matrix rank are kept distinct.",
            "The quotient slice includes both V5 multiplicity copies and their coupling.",
            "The exact local Hessian is normalized by the intrinsic Grassmann tangent metric.",
        ),
        limitations=(
            "Strict local D-optimality does not imply a global optimum over all rank-three frames.",
        ),
        replay_tier="bounded_full",
    ),
    GateContract(
        gate_id="exact_division_schur_isotypic_compiler",
        claim=(
            "For supplied exact completely reducible real representations, the maintained "
            "compiler detects real, complex, and quaternionic Schur type, decomposes the "
            "covered reducible rational fixtures, and extends the same audited construction "
            "to Q(sqrt(2))."
        ),
        status="proved_exact",
        evidence_layers=(
            "exact_arithmetic",
            "symbolic_identity",
            "exact_reconstruction",
        ),
        test_suites=(
            "tests/test_division_schur_scan.py",
            "tests/test_schur_type_detector.py",
            "tests/test_reducible_isotypic_decomposition.py",
            "tests/test_algebraic_isotypic_decomposition.py",
        ),
        artifacts=(
            "artifacts/division_schur_scan_20260811.json",
            "artifacts/schur_type_detection_20260811.json",
            "artifacts/reducible_isotypic_decomposition_20260811.json",
            "artifacts/algebraic_isotypic_decomposition_20260811.json",
        ),
        boundary_obligations=(
            "Complete reducibility is an explicit logical input rather than an inferred numerical property.",
            "Commutants, centers, primitive idempotents, intertwiners, and reconstructed generator actions are checked exactly.",
            "The Q(sqrt(2)) path audits field membership and refuses unsupported algebraic extensions.",
        ),
        limitations=(
            "The rational decomposition search is deliberately bounded and may refuse valid inputs when its exact witnesses are not exposed.",
            "Automatic number-field discovery, arbitrary algebraic extensions, and noisy floating-point representations remain open.",
        ),
        replay_tier="bounded_full",
    ),
    GateContract(
        gate_id="clifford_signature_and_spin9_isotypic_bridge",
        claim=(
            "The maintained real matrices realize the stated Cl(1,4), embedded Cl(3,0), "
            "Spin(8)-triality, and Cayley-null Spin(9) slice branching identities, including "
            "the exact Q(sqrt(2)) isotypic change of basis."
        ),
        status="proved_exact",
        evidence_layers=(
            "exact_arithmetic",
            "symbolic_identity",
            "exact_reconstruction",
        ),
        test_suites=(
            "tests/test_clifford_signature_extension.py",
            "tests/test_spin9_slice_isotypic_bridge.py",
        ),
        artifacts=(
            "artifacts/clifford_signature_extension_20260811.json",
            "artifacts/spin9_slice_isotypic_bridge_20260811.json",
        ),
        boundary_obligations=(
            "Abstract Clifford dimensions, represented image ranks, even-algebra sectors, and embedded subalgebras remain distinct in the ledger.",
            "Spin(8) triality modules are checked by exact intertwiner dimensions rather than labels.",
            "The concrete Spin(9) slice basis is reconciled exactly with the local-Hessian metric convention.",
        ),
        limitations=(
            "Cl(3,0) is embedded in, not identified with, Cl(1,4), and the maintained Cl(3,0) model is not thereby the same state model.",
            "These branching identities imply neither a global Spin(9) design optimum nor sequence-model superiority.",
        ),
        replay_tier="bounded_full",
    ),
    GateContract(
        gate_id="spin9_v5_global_shape_certificate",
        claim=(
            "Every pure V5 graph over the Cayley-null Spin(9) plane has information determinant "
            "strictly below 101/100 times the symmetric candidate determinant."
        ),
        status="proved_exact",
        evidence_layers=(
            "exact_arithmetic",
            "symbolic_identity",
            "exact_reconstruction",
            "positivity_certificate",
        ),
        test_suites=(
            "tests/test_spin9_v5_ray_certificate.py",
            "tests/test_spin9_v5_cartan_certificate.py",
        ),
        artifacts=(
            "artifacts/spin9_v5_ray_certificate_20260811.json",
            "artifacts/spin9_v5_cartan_reconstruction_20260811.json",
            "artifacts/spin9_v5_cartan_certificate_20260811.json",
        ),
        boundary_obligations=(
            "The radial and cubic shape invariants cover the complete pure-V5 orbit quotient.",
            "Modular reconstruction is checked at an unused prime before the characteristic-zero positivity atlas is accepted.",
            "The zero-cubic and axisymmetric boundary rays replay independently from the Cartan certificate.",
        ),
        limitations=(
            "Pure V5 graphs are only one isotypic sector of the full V1 plus two-V5 Grassmann normal slice.",
            "The strict 101/100 comparison is not a proof of the unrestricted global rank-three optimum.",
        ),
        replay_tier="expensive_exact",
    ),
    GateContract(
        gate_id="spin9_coupled_v1_v5_finite_radius",
        claim=(
            "Every finite graph in the complete coupled Spin(9) V1+V5 normal slice "
            "has determinant ratio at most 21/20 relative to Cayley-null."
        ),
        status="proved_exact",
        evidence_layers=(
            "exact_arithmetic",
            "symbolic_identity",
            "exact_reconstruction",
            "positivity_certificate",
        ),
        test_suites=(
            "tests/test_spin9_v1_v5_reconstruction.py",
            "tests/test_spin9_v1_v5_boundary_char0.py",
            "tests/test_spin9_v1_v5_blowup.py",
            "tests/test_spin9_v1_v5_global.py",
            "tests/test_spin9_v1_v5_char0.py",
            "tests/test_spin9_v1_v5_theorem.py",
        ),
        artifacts=(
            "artifacts/spin9_v1_v5_screen_20260811.json",
            "artifacts/spin9_v1_v5_reconstruction_20260811.json",
            "artifacts/spin9_v1_v5_boundary_char0_20260811.json",
            "artifacts/spin9_v1_v5_blowup_20260811.json",
            "artifacts/spin9_v1_v5_global_20260812.json",
            "artifacts/spin9_v1_v5_char0_20260812.json",
            "artifacts/spin9_v1_v5_theorem_20260812.json",
        ),
        boundary_obligations=(
            "Every compact depth-limit box must have an exact core/low-q local-chart handoff.",
            "Both square-root embeddings and every graded direction-rank gate must pass for all 22 identity primes.",
            "The identity-prime product must exceed twice the explicit characteristic-zero residual coefficient bound.",
        ),
        limitations=(
            "The 21/20 theorem does not prove exact optimality of the algebraic symmetric candidate even on this slice.",
            "The second supported V5, the nonpolar Grassmann quotient, and unrestricted rank-three optimality remain open.",
            "The float64 screen remains counterexample-search evidence only.",
        ),
        replay_tier="expensive_exact",
    ),
    GateContract(
        gate_id="spin9_pure_v1_candidate_line",
        claim=(
            "The algebraic symmetric candidate is the global determinant "
            "maximum on the complete real pure-V1 graph line, with its four "
            "finite graph-coordinate preimages classified exactly."
        ),
        status="proved_exact",
        evidence_layers=(
            "exact_arithmetic",
            "symbolic_identity",
            "positivity_certificate",
        ),
        test_suites=("tests/test_spin9_v1_candidate_line.py",),
        artifacts=("artifacts/spin9_v1_candidate_line_20260812.json",),
        boundary_obligations=(
            "The pure-V1 determinant formula must be identified exactly as a rational composition with the symmetric equiangular curve.",
            "The rational coordinate map must send the complete real graph line into -1/2<=c<=1.",
            "Derivative signs must prove the unique curve maximum at c*=(-17+sqrt(241))/24.",
            "All real roots of the stationary octic must be isolated and identified as the four graph preimages of c*.",
        ),
        limitations=(
            "This closes p=0 only; exact candidate optimality in the genuinely mixed p>0 region remains open.",
            "The second supported V5, the nonpolar quotient, and unrestricted rank-three optimality remain open.",
        ),
        replay_tier="bounded_full",
    ),
    GateContract(
        gate_id="spin9_global_three_spinor_design",
        claim="A bounded multistart numerical search found no three-spinor frame with larger determinant than the symmetric candidate.",
        status="numerical_only",
        evidence_layers=("floating_point_falsifier", "raw_artifact", "multi_seed"),
        test_suites=("tests/test_spin9_three_spinor_global_screen.py",),
        artifacts=("artifacts/spin9_three_spinor_global_screen_20260807.json",),
        boundary_obligations=(
            "Every reported start count, seed, tolerance, and optimization domain is preserved in the artifact.",
            "Boundary-biased and non-symmetric starts remain distinguishable in analysis.",
        ),
        limitations=(
            "The screen is counterexample-search evidence only; unrestricted global D-optimality remains open.",
        ),
        replay_tier="artifact_only_numerical",
    ),
)


def validate_gate_contracts(root: Path = ROOT) -> list[str]:
    """Return all schema and evidence-category violations."""

    errors: list[str] = []
    ids = [gate.gate_id for gate in GATES]
    if len(ids) != len(set(ids)):
        errors.append("gate identifiers are not unique")

    discovered_tests = {
        path.relative_to(root).as_posix() for path in (root / "tests").glob("test_*.py")
    }
    covered_tests: set[str] = set()

    for gate in GATES:
        prefix = gate.gate_id
        for field_name in (
            "evidence_layers",
            "test_suites",
            "artifacts",
            "boundary_obligations",
            "limitations",
            "external_inputs",
        ):
            if not isinstance(getattr(gate, field_name), tuple):
                errors.append(f"{prefix}: {field_name} must be a tuple")
        if gate.status not in STATUSES:
            errors.append(f"{prefix}: unknown status {gate.status!r}")
        unknown_layers = set(gate.evidence_layers) - EVIDENCE_LAYERS
        if unknown_layers:
            errors.append(f"{prefix}: unknown evidence layers {sorted(unknown_layers)}")
        if not gate.boundary_obligations:
            errors.append(f"{prefix}: has no explicit boundary obligations")
        if not gate.limitations:
            errors.append(f"{prefix}: has no explicit limitations")

        for relative in (*gate.test_suites, *gate.artifacts):
            if not (root / relative).is_file():
                errors.append(f"{prefix}: missing evidence path {relative}")
        covered_tests.update(gate.test_suites)

        if gate.status == "proved_exact":
            if "exact_arithmetic" not in gate.evidence_layers:
                errors.append(f"{prefix}: exact theorem lacks exact arithmetic")
            forbidden = {"external_theorem", "floating_point_falsifier"}
            if forbidden & set(gate.evidence_layers):
                errors.append(f"{prefix}: exact theorem depends on a non-exact layer")
        if gate.status == "proved_hybrid":
            if (
                "external_theorem" not in gate.evidence_layers
                or not gate.external_inputs
            ):
                errors.append(
                    f"{prefix}: hybrid theorem does not name its external input"
                )
        elif gate.external_inputs:
            errors.append(
                f"{prefix}: external inputs are only valid for hybrid theorems"
            )
        if (
            gate.status == "exact_negative"
            and "exact_counterexample" not in gate.evidence_layers
        ):
            errors.append(
                f"{prefix}: exact negative result lacks an exact counterexample"
            )
        if gate.status == "empirical" and (
            "raw_artifact" not in gate.evidence_layers or not gate.artifacts
        ):
            errors.append(f"{prefix}: empirical result lacks raw artifacts")
        if gate.status == "open":
            if not any("Open:" in limitation for limitation in gate.limitations):
                errors.append(f"{prefix}: open gate is not explicitly labelled open")
            forbidden = {"positivity_certificate", "checkpoint_replay"}
            if forbidden & set(gate.evidence_layers):
                errors.append(f"{prefix}: open gate advertises completion evidence")
        if gate.status == "exact_reduction" and any(
            layer in gate.evidence_layers
            for layer in ("positivity_certificate", "exact_counterexample")
        ):
            errors.append(f"{prefix}: reduction is conflated with resolution")

    missing_tests = sorted(discovered_tests - covered_tests)
    stale_tests = sorted(covered_tests - discovered_tests)
    if missing_tests:
        errors.append(f"unclassified test suites: {missing_tests}")
    if stale_tests:
        errors.append(f"registry lists nonexistent test suites: {stale_tests}")
    return errors


def main() -> int:
    errors = validate_gate_contracts()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"PASS: {len(GATES)} gate contracts cover every maintained test suite")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
