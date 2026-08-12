"""Focused regression tests for the two publication theorem certificates."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np
import psutil
import sympy as sp

from spin8_cayley_criteria import exact_cayley_criteria_certificate
from spin8_cayley_flag import exact_principal_flag_certificate
from spin8_dirac_endpoint_klein_face import certificate as endpoint_klein_face
from spin8_dirac_endpoint_octet import certificate as endpoint_octet
from spin8_dirac_final_residual import (
    exact_chart_invariants_certificate,
    exact_final_residual_equality_slice_certificate,
    exact_final_residual_structure_certificate,
    exact_full_multidegree_certificate,
)
from spin8_dirac_one_edge_equality import run as run_one_edge_equality
from spin8_dirac_star_foundations import run as run_star_foundations
from spin8_dirac_star_structure import exact_star_structure_certificate
from spin8_dirac_two_edge_atlas import ATLAS_TREES, _leaf_paths, verify_report
from spin8_dirac_two_edge_endpoints import exact_endpoint_jet_certificate
from spin8_dirac_two_edge_rational import (
    float_bernstein_enclosure,
    selected_scaled_integer_bernstein,
    triangle_blowup_power_tensor,
)
from spin8_dirac_unrestricted_compare import compare as compare_unrestricted
from spin8_dirac_unrestricted_core import certificate as unrestricted_core
from spin8_dirac_unrestricted_energy import certificate as unrestricted_energy
from spin8_dirac_unrestricted_tangent import certificate as unrestricted_tangent
from spin8_publication_flint_crosscheck import FLINT_AVAILABLE
from spin8_publication_flint_crosscheck import run as run_flint

TEST_WORKERS = max(1, min(6, len(psutil.Process().cpu_affinity())))


class PublicationTheoremTests(unittest.TestCase):
    def test_two_edge_atlas_arithmetic_primitives(self) -> None:
        power = np.zeros((3, 3), dtype=np.int64)
        power[0, 0] = 3
        power[1, 0] = -2
        power[0, 1] = 1
        power[2, 2] = 4

        lower = triangle_blowup_power_tensor(power, 0, 1, upper=False)
        upper = triangle_blowup_power_tensor(power, 0, 1, upper=True)
        tau = sp.Rational(2, 5)
        sigma = sp.Rational(3, 7)

        def evaluate(tensor: np.ndarray, point: tuple[sp.Rational, ...]):
            return sum(
                int(tensor[index])
                * sp.prod(point[axis] ** index[axis] for axis in range(tensor.ndim))
                for index in np.ndindex(tensor.shape)
            )

        self.assertEqual(
            evaluate(lower, (tau, sigma)),
            evaluate(power, (tau, tau * sigma)),
        )
        self.assertEqual(
            evaluate(upper, (tau, sigma)),
            evaluate(power, (tau * sigma, sigma)),
        )

        centre, radius, possible = float_bernstein_enclosure(power)
        for target in np.ndindex(power.shape):
            exact = sp.Integer(0)
            for source in np.ndindex(power.shape):
                if all(source[axis] <= target[axis] for axis in range(power.ndim)):
                    exact += int(power[source]) * sp.prod(
                        sp.Rational(
                            math.comb(target[axis], source[axis]),
                            math.comb(power.shape[axis] - 1, source[axis]),
                        )
                        for axis in range(power.ndim)
                    )
            self.assertLessEqual(centre[target] - radius[target], float(exact))
            self.assertGreaterEqual(centre[target] + radius[target], float(exact))
            if not possible[target]:
                self.assertEqual(exact, 0)

        selected = ((0, 1, 2), (0, 1, 2))
        scaled, _metadata = selected_scaled_integer_bernstein(power, selected)
        for target in np.ndindex(power.shape):
            exact = sp.Integer(0)
            for source in np.ndindex(power.shape):
                if all(source[axis] <= target[axis] for axis in range(power.ndim)):
                    exact += int(power[source]) * sp.prod(
                        sp.Rational(
                            math.comb(target[axis], source[axis]),
                            math.comb(power.shape[axis] - 1, source[axis]),
                        )
                        for axis in range(power.ndim)
                    )
            self.assertEqual(int(np.sign(int(scaled[target]))), int(sp.sign(exact)))

    def test_two_edge_atlas_artifact_contract(self) -> None:
        root = Path(__file__).parents[1]
        coefficient_path = (
            root
            / "artifacts"
            / "spin8_dirac_two_edge_all_sectors_coefficients_20260806.json"
        )
        report_path = root / "artifacts" / "spin8_dirac_two_edge_atlas_20260807.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertTrue(verify_report(report, coefficient_path))
        self.assertEqual(
            report["leaf_count"],
            sum(len(list(_leaf_paths(tree))) for tree in ATLAS_TREES.values()),
        )

    def test_two_edge_endpoint_jet_flag_law(self) -> None:
        artifact = (
            Path(__file__).parents[1]
            / "artifacts"
            / "spin8_dirac_two_edge_all_sectors_coefficients_20260806.json"
        )
        report = exact_endpoint_jet_certificate(artifact, workers=TEST_WORKERS)
        self.assertTrue(report["passed"])
        self.assertEqual(report["sector_count"], 8)
        self.assertTrue(
            report["fully_active_endpoint_core_is_cayley_independent_in_every_sector"]
        )
        self.assertEqual(
            report["first_i2_derivative_has_universal_extra_flag_factor"],
            "(1-d2)(1-e2)(1-g2)",
        )

    def test_complete_one_edge_equality_classification(self) -> None:
        artifacts = Path(__file__).parents[1] / "artifacts"
        report = run_one_edge_equality(
            artifacts / "spin8_dirac_one_edge_exact_20260804.json",
            artifacts / "spin8_dirac_one_edge_determinant_cache_20260806.json",
            artifacts / "spin8_dirac_one_edge_duffy_20260806.json",
            workers=TEST_WORKERS,
        )
        self.assertTrue(report["passed"])
        self.assertTrue(report["prior_one_edge_theorem_contract_passed"])
        self.assertEqual(
            report["complete_equality_set"],
            "z=1 or (u,v,r,w)=(0,0,0,0)",
        )

    def test_exact_split_isotropy_and_hybrid_flag_contract(self) -> None:
        report = exact_principal_flag_certificate()
        self.assertTrue(report["passed"])
        self.assertEqual(report["principal_plane_stabilizer_dimension"], 6)
        self.assertEqual(report["effective_restriction_rank_in_so4"], 6)
        self.assertEqual(report["principal_flag_stabilizer_dimension"], 2)
        self.assertEqual(report["flag_quotient_dimension"], 1)
        self.assertEqual(report["local_flag_quotient_dimension"], 1)
        self.assertTrue(report["cross_split_regression"]["rotation_is_so4"])
        self.assertTrue(
            report["cross_split_regression"]["characteristic_polynomials_match_exactly"]
        )
        self.assertTrue(report["principal_cross_checks_passed"])
        self.assertEqual(len(report["principal_rational_cross_checks"]), 5)
        for row in report["principal_rational_cross_checks"]:
            self.assertEqual(row["plane_stabilizer_dimension"], 6)
            self.assertEqual(row["effective_restriction_rank_in_so4"], 6)
            self.assertEqual(row["flag_stabilizer_dimension"], 2)
        for endpoint in report["singular_endpoint_representatives"]:
            self.assertEqual(endpoint["plane_stabilizer_dimension"], 9)
            self.assertEqual(endpoint["effective_restriction_rank_in_so4"], 6)
            self.assertEqual(endpoint["flag_stabilizer_dimension"], 5)
        self.assertIn("classical_global_input", report["proof_layers"])
        self.assertIn("not_recomputed_here", report["proof_layers"])

    def test_exact_final_residual_equality_slice(self) -> None:
        report = exact_final_residual_equality_slice_certificate()
        self.assertTrue(report["passed"])
        self.assertTrue(report["determinant_identity_verified"])
        self.assertTrue(report["gap_identity_verified"])
        self.assertEqual(report["minimum_bernstein_coefficient"], "53/2")
        self.assertEqual(report["equality_set"], "h=0 or c^2=1")

    def test_exact_complete_cholesky_invariants(self) -> None:
        report = exact_chart_invariants_certificate()
        self.assertTrue(report["passed"])
        self.assertTrue(report["gram_identity_verified"])
        self.assertTrue(report["cayley_identity_verified"])

    def test_exact_final_residual_sign_and_degree_reduction(self) -> None:
        report = exact_final_residual_structure_certificate()
        self.assertTrue(report["passed"])
        self.assertEqual(report["sector_count"], 16)
        self.assertTrue(report["hadamard_identity_verified"])
        self.assertTrue(report["all_independent_holdouts_match"])
        for row in report["sector_rows"]:
            self.assertLessEqual(
                row["observed_anchor_degree"], row["degree_ceiling_in_h_squared"]
            )

    def test_exact_unrestricted_structural_reduction(self) -> None:
        report = exact_full_multidegree_certificate()
        self.assertTrue(report["passed"])
        self.assertEqual(report["full_chart_sign_certificate"]["annihilator_order"], 16)
        self.assertTrue(
            report["full_boundary_divisibility_certificate"][
                "all_fourteen_branches_rank_25"
            ]
        )
        self.assertEqual(len(report["sector_rows"]), 16)
        for row in report["sector_rows"]:
            self.assertEqual(len(row["residual_polynomial_multidegree_upper_bound"]), 7)
            self.assertTrue(
                all(
                    degree <= 4
                    for degree in row["residual_polynomial_multidegree_upper_bound"]
                )
            )

    def test_exact_unrestricted_disjoint_reconstruction_and_holdout(self) -> None:
        coefficient_dir = (
            Path(__file__).parents[1]
            / "artifacts"
            / "spin8_dirac_unrestricted_coefficients_20260807"
        )
        report = compare_unrestricted(
            coefficient_dir=coefficient_dir,
            holdout_count=1,
        )
        self.assertTrue(report["passed"])
        self.assertTrue(report["all_complete_maps_match"])
        self.assertTrue(report["all_holdouts_match"])
        self.assertEqual(len(report["sector_comparison_rows"]), 16)

    def test_exact_unrestricted_tangent_and_endpoint_null_cone(self) -> None:
        coefficient_dir = (
            Path(__file__).parents[1]
            / "artifacts"
            / "spin8_dirac_unrestricted_coefficients_20260807"
        )
        report = unrestricted_tangent(coefficient_dir)
        self.assertTrue(report["passed"])
        endpoint = report["endpoint_null_cone_quartic_certificate"]
        self.assertTrue(endpoint["passed"])
        self.assertEqual(len(endpoint["orientation_rows"]), 16)
        self.assertEqual(
            endpoint["common_quartic_on_all_sixteen_tangent_null_cones"],
            "128*(p**2 + q**2)**2",
        )
        weighted = report["weighted_endpoint_blowup_certificate"]
        self.assertTrue(weighted["passed"])
        self.assertEqual(len(weighted["sign_rows"]), 4)
        self.assertEqual(
            weighted["manifestly_nonnegative_decomposition"],
            "16[5 alpha^2 + 5 iota^2 + 8(p^2+q^2)^2 + "
            "4 w^2(p^2+q^2) + 2 x^2 + 2 y^2]",
        )

    def test_exact_unrestricted_coupled_core(self) -> None:
        coefficient_dir = (
            Path(__file__).parents[1]
            / "artifacts"
            / "spin8_dirac_unrestricted_coefficients_20260807"
        )
        report = unrestricted_core(coefficient_dir, flint_threads=TEST_WORKERS)
        self.assertTrue(report["passed"])
        self.assertEqual(report["core_multidegree"], [6, 6, 6, 6, 6, 6, 4])
        self.assertEqual(report["bernstein_coefficient_count"], 588245)
        self.assertEqual(report["negative_scaled_bernstein_coefficient_count"], 0)
        self.assertEqual(report["minimum_scaled_bernstein_coefficient"], "0")

    def test_exact_unrestricted_full_sector_energy(self) -> None:
        coefficient_dir = (
            Path(__file__).parents[1]
            / "artifacts"
            / "spin8_dirac_unrestricted_coefficients_20260807"
        )
        report = unrestricted_energy(coefficient_dir, flint_threads=TEST_WORKERS)
        self.assertTrue(report["passed"])
        self.assertEqual(report["nontrivial_sector_count"], 15)
        self.assertEqual(report["energy_multidegree"], [6, 6, 6, 6, 6, 6, 4])
        mean = report["nonnegative_mean_certificate"]
        self.assertEqual(mean["multidegree"], [3, 3, 3, 3, 3, 3, 2])
        self.assertEqual(mean["negative_scaled_coefficient_count"], 0)
        self.assertEqual(mean["zero_scaled_coefficient_count"], 3)
        low = report["low_cayley_native_certificate"]
        self.assertEqual(low["coefficient_count"], 588245)
        self.assertEqual(low["negative_scaled_coefficient_count"], 0)
        self.assertEqual(low["zero_scaled_coefficient_count"], 35)
        native = report["full_cube_native_basis_audit"]
        self.assertEqual(native["negative_scaled_coefficient_count"], 4)
        self.assertTrue(report["native_obstructions_are_exactly_two_coupled_faces"])
        self.assertTrue(report["coupled_face_pair_identity"])
        self.assertTrue(report["high_cayley_coupled_face_certificate"]["passed"])
        remainder = report["global_remainder_certificate"]
        self.assertEqual(remainder["negative_scaled_coefficient_count"], 0)
        self.assertEqual(remainder["zero_scaled_coefficient_count"], 495)
        third = report["third_elementary_symmetric_certificate"]
        self.assertTrue(third["passed"])
        self.assertTrue(third["rational_constant_check"])
        self.assertEqual(third["global_lower_bound"], "e3 >= 1280/3*A0^3 >= 0")
        fourth = report["fourth_elementary_symmetric_certificate"]
        self.assertTrue(fourth["passed"])
        self.assertEqual(fourth["endpoint_value_at_r_16"], "2348/3")
        self.assertEqual(fourth["global_lower_bound"], "e4 >= 2348/3*A0^4 >= 0")

    def test_exact_endpoint_klein_face_psd(self) -> None:
        report = endpoint_klein_face(
            Path("artifacts/spin8_dirac_unrestricted_coefficients_20260807"),
            flint_threads=TEST_WORKERS,
        )
        self.assertTrue(report["passed"])
        self.assertTrue(report["surviving_masks_form_klein_four"])
        self.assertEqual(report["orientation_multiplicity"], 4)
        self.assertTrue(report["group_circulant_eigenvalue_multiplicities_verified"])
        self.assertTrue(report["one_mode_boundary_certificate"]["passed"])
        cubic = report["three_by_three_minor"]
        self.assertTrue(cubic["corner_identity_verified"])
        determinant = report["four_by_four_determinant"]
        self.assertTrue(determinant["ud_zero_face_is_square"])
        self.assertTrue(determinant["nested_corner_factorization_verified"])
        self.assertEqual(
            determinant["final_interior_remainder"][
                "negative_scaled_coefficient_count"
            ],
            0,
        )

    def test_exact_adjacent_endpoint_octet_reduction(self) -> None:
        report = endpoint_octet(
            Path("artifacts/spin8_dirac_unrestricted_coefficients_20260807"),
            flint_threads=TEST_WORKERS,
        )
        self.assertTrue(report["passed"])
        self.assertTrue(report["survivors_form_z2_cubed"])
        self.assertTrue(report["h_zero_subgroup_is_klein_four"])
        self.assertTrue(report["h_one_family_is_its_coset"])
        self.assertTrue(report["eight_patterns_each_have_multiplicity_two"])
        self.assertTrue(report["x_block"]["passed"])
        determinant = report["x_block"]["principal_minor_certificates"][-1]
        self.assertTrue(determinant["corner"]["passed"])
        self.assertEqual(
            determinant["five_variable_interior_remainder"][
                "negative_scaled_coefficient_count"
            ],
            0,
        )
        self.assertTrue(report["z_block"]["center_identity_verified"])
        self.assertTrue(report["z_block"]["center_nonnegative"])
        quadratic_corner = report["z_block"]["common_quadratic_corner"]
        self.assertTrue(quadratic_corner["passed"])
        self.assertTrue(quadratic_corner["three_quadratic_faces_are_identical"])
        self.assertTrue(quadratic_corner["exact_factorization_verified"])
        self.assertEqual(quadratic_corner["square_root_multidegree"], [6, 6, 12])
        self.assertEqual(report["z_block"]["higher_principal_minors"], "open")

    def test_exact_cayley_design_criteria(self) -> None:
        report = exact_cayley_criteria_certificate()
        self.assertTrue(report["passed"])
        self.assertEqual(report["trace"], "35")
        self.assertEqual(report["trace_square"], "67")
        self.assertEqual(report["second_moment_participation_ratio"], "1225/67")
        self.assertEqual(report["balanced_trace_inverse"], "43")
        self.assertEqual(
            report["endpoint_small_eigenvalue_slopes_in_1_minus_z"],
            ["1/8", "1/8", "1/8"],
        )

    def test_reduced_signed_star_certificate(self) -> None:
        artifact = (
            Path(__file__).parents[1] / "artifacts" / "spin8_dirac_star_20260804.json"
        )
        report = exact_star_structure_certificate(artifact)
        self.assertTrue(report["passed"])
        self.assertTrue(report["discovery_confirmation_maps_equal"])
        self.assertTrue(report["strict_on_open_unit_box"])
        self.assertTrue(report["exact_equality_support"])
        self.assertEqual(
            report["complete_normalized_equality_set"],
            "z=1 or (u,v,w)=(0,0,0)",
        )
        self.assertTrue(
            report["boundary_equality_audit"][
                "gram_faces_have_nonzero_normalized_margin_generically"
            ]
        )
        self.assertTrue(
            report["boundary_equality_audit"][
                "gram_faces_annihilate_unnormalized_margin"
            ]
        )

    def test_signed_star_structural_foundations(self) -> None:
        report = run_star_foundations()
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["parity"]["invariant_parity_masks"],
            [[0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0]],
        )
        self.assertTrue(
            all(
                row["universal_rank_upper_bound"] == 25
                for row in report["boundary_and_degree"]["boundary_branches"]
            )
        )
        self.assertEqual(
            report["boundary_and_degree"][
                "exact_cubic_order_witnesses_for_1024_det_over_delta_cubed"
            ],
            {
                "u=1,v=w=z=0": "25/2",
                "v=1,u=w=z=0": "75/2",
                "w=1,u=v=z=0": "75/2",
            },
        )

    @unittest.skipUnless(FLINT_AVAILABLE, "python-flint is not installed")
    def test_independent_flint_publication_crosscheck(self) -> None:
        artifact = (
            Path(__file__).parents[1] / "artifacts" / "spin8_dirac_star_20260804.json"
        )
        report = run_flint(source_artifact=artifact, flint_threads=TEST_WORKERS)
        self.assertTrue(report["passed"])
        self.assertTrue(report["cayley_criteria"]["passed"])
        self.assertTrue(report["signed_star_structure"]["passed"])


if __name__ == "__main__":
    unittest.main()
