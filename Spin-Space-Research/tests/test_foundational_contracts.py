"""Regression tests for the foundational metric and isotypic contracts."""

from __future__ import annotations

import copy
import json
import unittest
from fractions import Fraction
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
import torch
from torch import nn

from compare_recurrences import (
    GROUPS,
    group_prefix_products,
)
from compare_recurrences import (
    evaluate as evaluate_recurrence,
)
from GALib import (
    Spin3IsotypicLinear as JaxSpin3IsotypicLinear,
)
from GALib import (
    pack_spin3_isotypic,
    unpack_spin3_isotypic,
)
from GALib import (
    rotor_from_bivector as jax_rotor_from_bivector,
)
from GALib import (
    rotor_sandwich as jax_rotor_sandwich,
)
from mechanistic_group_actions import evaluate as evaluate_group_action
from pdssm_group_actions import evaluate as evaluate_pdssm
from rotor_ssm_torch import GA_DIM, GradeLinear
from schur_scan import (
    SchurAffineTransition,
    Spin3IsotypicLinear,
    apply_schur_affine,
    associative_schur_scan,
    pack_cl3_isotypic,
    unpack_cl3_isotypic,
)
from spin8_active_sensing import (
    SensorDesign,
    action_independence_audit,
    fixed_sensor,
    information_metrics,
)
from spin8_blind_alias_action import (
    calibration_complement,
    combined_design_audit,
    negative_calibration_basis,
)
from spin8_blind_alias_action import (
    evaluate_sequences as evaluate_blind_alias_sequences,
)
from spin8_blind_shared_action import (
    action_design_audit,
    joint_shared_retraction,
    observed_action,
    sample_teacher,
)
from spin8_cayley_blocks import exact_cayley_block_certificate
from spin8_cayley_spectrum import (
    cayley_invariance_audit,
    exact_cayley_spectrum_certificate,
    exact_partition_representatives,
    exact_restricted_orthogonalization_certificate,
)
from spin8_conditional_counterexample import (
    verify_artifact as verify_conditional_counterexample_artifact,
)
from spin8_continuous_alias import (
    AliasWorld,
    FrozenKeyPolicy,
    FrozenSlotPolicy,
    alias_world_audit,
    key_scan_parity,
    slot_endpoint_loss,
    slot_scan_parity,
)
from spin8_dirac_edge import (
    exact_walsh_symmetry_certificate,
)
from spin8_dirac_edge import (
    verify_artifact as verify_dirac_edge_artifact,
)
from spin8_dirac_edge import (
    verify_report as verify_dirac_edge_report,
)
from spin8_dirac_gram import (
    exact_approximate_design_rejection,
    exact_dirac_graph_certificate,
    exact_projector_geometry_certificate,
    exact_strengthened_slice_certificate,
    exact_whitening_flow_invariant_certificate,
)
from spin8_dirac_one_edge_exact import (
    EXPECTED_DEGREES as ONE_EDGE_EXPECTED_DEGREES,
)
from spin8_dirac_one_edge_exact import (
    HOLDOUTS as ONE_EDGE_HOLDOUTS,
)
from spin8_dirac_one_edge_exact import (
    _determinant as one_edge_direct_determinant,
)
from spin8_dirac_one_edge_exact import (
    _symmetry_certificate as one_edge_symmetry_certificate,
)
from spin8_dirac_one_edge_holdouts import verify_holdout_report
from spin8_dirac_one_edge_positivity import (
    _complement_first_two,
    _integer_bernstein_tensor,
    _lower_duffy_power_tensor,
    _sparse_tetrahedral_determinant,
    verify_assembled_report,
)
from spin8_dirac_star import (
    rational_circle,
)
from spin8_dirac_star import (
    verify_artifact as verify_dirac_star_artifact,
)
from spin8_five_probe_identifiability import (
    FIVE_MIXED,
    FIVE_SINGLE,
    FOUR_MIXED,
    exact_four_probe_witness,
    independent_observation_jacobian,
    make_probe_family,
    numerical_rank,
    shared_observation_jacobian,
)
from spin8_joint_sensor_retraction import (
    SoftSensorBank,
    exact_characteristic_coefficients,
    joint_retract_sensor,
    single_query_projector_audit,
)
from spin8_learned_address import (
    evaluate_mixed_sequences,
    log_sinkhorn,
    route_statistics,
)
from spin8_learned_address import (
    scan_parity as learned_address_scan_parity,
)
from spin8_triality import spin8_actions, torch_triality_generators
from spin8_triality_identifiability import invariant_space_audit
from spin8_triality_lift import (
    diagnostics as triality_lift_diagnostics,
)
from spin8_triality_lift import (
    triality_bind,
    triality_tensor,
    triality_unbind_negative,
)
from spin8_triality_memory import run as triality_memory_diagnostics


class IsotypicLayerTests(unittest.TestCase):
    def test_jax_pack_round_trip_and_equivariance(self) -> None:
        inputs = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 3, GA_DIM))
        trivial, active = pack_spin3_isotypic(inputs)
        np.testing.assert_allclose(
            unpack_spin3_isotypic(trivial, active), inputs, rtol=0, atol=0
        )
        frame = jax_rotor_from_bivector(jnp.asarray([0.3, -0.2, 0.4]))
        layer = JaxSpin3IsotypicLinear(3, 4)
        parameters = layer.init(jax.random.PRNGKey(2), inputs)
        outputs = layer.apply(parameters, inputs)
        transformed = layer.apply(parameters, jax_rotor_sandwich(frame, inputs))
        np.testing.assert_allclose(
            transformed,
            jax_rotor_sandwich(frame, outputs),
            rtol=3e-5,
            atol=3e-5,
        )

    def test_torch_pack_round_trip_and_hodge_capacity_separation(self) -> None:
        torch.manual_seed(3)
        inputs = torch.randn(7, 1, GA_DIM, dtype=torch.float64)
        trivial, active = pack_cl3_isotypic(inputs)
        torch.testing.assert_close(
            unpack_cl3_isotypic(trivial, active), inputs, rtol=0, atol=0
        )

        isotypic = Spin3IsotypicLinear(1, 1, use_bias=False).double()
        with torch.no_grad():
            isotypic.trivial_kernel.zero_()
            isotypic.active_kernel.zero_()
            isotypic.trivial_kernel[0, 0, 0, 1] = 1.0
            isotypic.trivial_kernel[0, 1, 0, 0] = 1.0
            isotypic.active_kernel[0, 0, 0, 1] = 1.0
            isotypic.active_kernel[0, 1, 0, 0] = 1.0
        target_trivial = trivial.reshape(7, 1, 2).flip(-1).flatten(-2)
        target_active = active.reshape(7, 1, 2, 3).flip(-2).flatten(-3, -2)
        target = unpack_cl3_isotypic(target_trivial, target_active)
        torch.testing.assert_close(isotypic(inputs), target, rtol=0, atol=0)

        vector_only = torch.zeros(4, 1, GA_DIM)
        vector_only[..., 1:4] = torch.randn(4, 1, 3)
        grade = GradeLinear(1, 1, use_bias=False)
        # Every GradeLinear parameterization maps a vector-only input to zero
        # bivector output; the Hodge-copy target has a nonzero bivector.
        self.assertEqual(float(grade(vector_only)[..., 4:7].abs().max().detach()), 0.0)
        self.assertGreater(
            float(isotypic.float()(vector_only)[..., 4:7].abs().max().detach()), 0.0
        )


class SchurScanTests(unittest.TestCase):
    def test_parallel_scan_matches_recurrence(self) -> None:
        torch.manual_seed(4)
        dtype = torch.float64
        batch, length, multiplicity = 2, 13, 4
        eye_m = torch.eye(multiplicity, dtype=dtype).expand(batch, length, -1, -1)
        skew = torch.randn(batch, length, 3, 3, dtype=dtype)
        skew = 0.1 * (skew - skew.transpose(-1, -2))
        transition = SchurAffineTransition(
            trivial_action=0.9 * eye_m + 0.01 * torch.randn_like(eye_m),
            active_multiplicity=0.9 * eye_m + 0.01 * torch.randn_like(eye_m),
            rotation=torch.matrix_exp(skew),
            trivial_drive=0.01 * torch.randn(batch, length, multiplicity, dtype=dtype),
            active_drive=0.01
            * torch.randn(batch, length, multiplicity, 3, dtype=dtype),
        )
        initial = (
            torch.randn(batch, multiplicity, dtype=dtype),
            torch.randn(batch, multiplicity, 3, dtype=dtype),
        )
        prefixes = associative_schur_scan(transition)
        parallel = apply_schur_affine(
            prefixes, (initial[0][:, None], initial[1][:, None])
        )
        state = initial
        recurrent = [[], []]
        for position in range(length):
            step = SchurAffineTransition(
                *(value[:, position] for value in transition.__dict__.values())
            )
            state = apply_schur_affine(step, state)
            recurrent[0].append(state[0])
            recurrent[1].append(state[1])
        for expected, values in zip(parallel, recurrent):
            torch.testing.assert_close(
                expected, torch.stack(values, dim=1), rtol=1e-12, atol=1e-12
            )

    def test_triangular_triality_lift_and_staged_scan_pass(self) -> None:
        report = triality_lift_diagnostics()
        self.assertTrue(report["passed"])
        self.assertEqual(report["lift_dimension"], 81)
        self.assertEqual(report["streaming_cache_scalars"], 24)
        self.assertEqual(
            report["degree_growth"]["two_way_feedback_degree"],
            [2, 4, 8, 16, 32, 64, 128, 256],
        )

    def test_triality_binding_is_exactly_invertible_with_a_unit_key(self) -> None:
        torch.manual_seed(5)
        dtype = torch.float64
        positive = nn.functional.normalize(torch.randn(32, 8, dtype=dtype), dim=-1)
        negative = torch.randn(32, 8, dtype=dtype)
        rho = triality_tensor(dtype=dtype)
        vector = triality_bind(positive, negative, rho)
        recovered = triality_unbind_negative(positive, vector, rho)
        torch.testing.assert_close(recovered, negative, rtol=1e-12, atol=1e-12)

    def test_triality_is_the_unique_infinitesimal_equivariant_bilinear_map(
        self,
    ) -> None:
        report = invariant_space_audit()
        self.assertEqual(report["constraint_shape"], [14336, 512])
        self.assertEqual(report["nullity"], 1)
        self.assertAlmostEqual(
            report["null_vector_abs_cosine_with_triality"], 1.0, places=12
        )
        self.assertGreater(report["second_smallest_singular_value"], 3.0)

    def test_triality_coded_memory_and_dynamic_slot_gates_pass(self) -> None:
        report = triality_memory_diagnostics()
        self.assertTrue(report["passed"])
        self.assertLess(report["capacity"]["maximum_exact_relative_error"], 1e-10)
        self.assertTrue(
            report["capacity"]["tight_frame_beats_random_all_overcomplete_cells"]
        )
        self.assertLess(report["dynamic_slot"]["final_retrieval_max_error"], 1e-10)

    def test_blind_shared_action_mask_is_identifiable_and_retracts(self) -> None:
        dtype = torch.float64
        generators = torch_triality_generators(dtype=dtype)
        random = torch.Generator().manual_seed(9)
        hidden = 0.12 * torch.randn(4, 28, generator=random, dtype=dtype)
        oracle = spin8_actions(hidden, generators)
        design = action_design_audit(hidden[:1], generators)
        self.assertEqual(design["minimum_rank"], 28)

        recovered, coordinates, report = joint_shared_retraction(
            observed_action(oracle),
            seed=9,
            generators=generators,
            adam_steps=200,
            lbfgs_steps=50,
        )
        self.assertLess(report["final_observed_mse"], 1e-10)
        self.assertGreater(
            float(
                nn.functional.cosine_similarity(
                    coordinates.flatten(), hidden.flatten(), dim=0
                )
            ),
            1.0 - 1e-9,
        )
        torch.testing.assert_close(recovered, oracle, rtol=0, atol=3e-6)

    def test_joint_address_family_is_globally_not_independently_normalized(
        self,
    ) -> None:
        torch.manual_seed(10)
        routes = log_sinkhorn(
            torch.randn(8, 8, dtype=torch.float64), 0.2, iterations=256
        )
        self.assertLess(float((routes.sum(dim=-1) - 1.0).abs().max()), 1e-12)
        self.assertLess(float((routes.sum(dim=-2) - 1.0).abs().max()), 1e-12)

        collided = torch.eye(8, dtype=torch.float64)
        collided[1] = collided[0]
        statistics = route_statistics(collided)
        self.assertEqual(statistics["rounded_collisions"], 1)
        self.assertGreater(statistics["maximum_column_sum_residual"], 0.9)

    def test_exact_latent_addresses_retrieve_and_scan_in_both_memories(self) -> None:
        routes = torch.eye(8, dtype=torch.float64)
        for kind in ("triality", "direct"):
            evaluation = evaluate_mixed_sequences(
                routes, kind=kind, length=32, seed=11, batch_size=48
            )
            self.assertGreaterEqual(evaluation["queries"], 256)
            self.assertGreater(evaluation["minimum_query_cosine"], 1.0 - 1e-12)
            self.assertLess(evaluation["maximum_relative_squared_error"], 1e-20)
            parity = learned_address_scan_parity(routes, kind=kind, seed=11)
            self.assertEqual(parity["streaming_state_scalars"], 64)
            self.assertLess(parity["parallel_recurrent_max_error"], 1e-12)

    def test_continuous_alias_world_has_device_independent_exact_radius(self) -> None:
        report = alias_world_audit(12)
        self.assertLess(report["center_gram_max_error"], 1e-12)
        self.assertLess(report["radius_cosine_max_error"], 1e-12)
        self.assertLess(report["cross_device_center_max_error"], 1e-12)

    def test_alias_routes_and_delta_keys_remain_scan_compatible(self) -> None:
        slot_policy = FrozenSlotPolicy("oracle_both", None, None)
        for memory_kind in ("triality", "direct"):
            report = slot_scan_parity(
                slot_policy, memory_kind=memory_kind, seed=13, radius=0.35
            )
            self.assertEqual(report["streaming_state_scalars"], 64)
            self.assertLess(report["parallel_recurrent_max_error"], 1e-12)

        world = AliasWorld.create(13, dtype=torch.float64, device=torch.device("cpu"))
        key_policy = FrozenKeyPolicy(world.centers, world.centers)
        for update_kind in ("delta", "fast_weight"):
            report = key_scan_parity(
                key_policy, update_kind=update_kind, seed=13, radius=0.35
            )
            self.assertEqual(report["streaming_state_scalars"], 64)
            self.assertLess(report["parallel_recurrent_max_error"], 1e-12)

    def test_cross_encoder_endpoint_rules_out_independent_gauges(self) -> None:
        routes = torch.eye(8, dtype=torch.float64)
        aligned, _ = slot_endpoint_loss(routes, routes, joint=True)
        shifted, _ = slot_endpoint_loss(routes, routes.roll(1, dims=0), joint=True)
        self.assertEqual(float(aligned), 0.0)
        self.assertGreater(float(shifted), 0.9)


class EvaluationContractTests(unittest.TestCase):
    class DummyRecurrence(nn.Module):
        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            score = 4.0 * tokens.float()
            return torch.stack((torch.zeros_like(score), score), dim=-1)

    class DummyGroupAction(nn.Module):
        def forward(
            self, tokens: torch.Tensor, *, return_recurrent_state: bool = False
        ):
            score = 4.0 * tokens.float()
            logits = torch.stack((torch.zeros_like(score), score), dim=-1)
            state = torch.zeros(tokens.shape[0], 1, 8, device=tokens.device)
            return (logits, state) if return_recurrent_state else logits

        def initial_state(self, batch_size: int) -> torch.Tensor:
            return torch.zeros(batch_size, 1, 8)

    def test_evaluators_weight_unequal_batches_by_label_count(self) -> None:
        batches = [
            (torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 3, dtype=torch.long)),
            (torch.ones(1, 3, dtype=torch.long), torch.zeros(1, 3, dtype=torch.long)),
        ]
        all_logits = torch.cat(
            [self.DummyRecurrence()(tokens).flatten(0, 1) for tokens, _ in batches]
        )
        all_targets = torch.cat([target.flatten() for _, target in batches])
        expected = float(nn.functional.cross_entropy(all_logits, all_targets))
        self.assertAlmostEqual(
            evaluate_recurrence(self.DummyRecurrence(), batches, torch.device("cpu"))[
                0
            ],
            expected,
            places=6,
        )
        self.assertAlmostEqual(
            evaluate_group_action(
                self.DummyGroupAction(), batches, torch.device("cpu")
            )[0],
            expected,
            places=6,
        )
        self.assertAlmostEqual(
            evaluate_pdssm(self.DummyGroupAction(), batches, torch.device("cpu"))[
                "validation_loss"
            ],
            expected,
            places=6,
        )

    def test_group_targets_are_same_position_prefix_products(self) -> None:
        group = GROUPS["q8"]
        tokens = np.asarray([[2, 4, 3]], dtype=np.int64)
        targets = group_prefix_products(tokens, group)
        first = group.table[0, tokens[0, 0]]
        second = group.table[first, tokens[0, 1]]
        third = group.table[second, tokens[0, 2]]
        np.testing.assert_array_equal(targets[0], np.asarray([first, second, third]))


class BlindAliasActionDesignTests(unittest.TestCase):
    def test_rank_two_calibration_split_is_orthogonal_and_complete(self) -> None:
        basis = negative_calibration_basis(
            17, dtype=torch.float64, device=torch.device("cpu")
        )
        complement = calibration_complement(basis)
        torch.testing.assert_close(
            basis.T @ basis, torch.eye(2, dtype=torch.float64), rtol=0, atol=1e-12
        )
        torch.testing.assert_close(
            complement.T @ complement,
            torch.eye(6, dtype=torch.float64),
            rtol=0,
            atol=1e-12,
        )
        torch.testing.assert_close(
            basis @ basis.T + complement @ complement.T,
            torch.eye(8, dtype=torch.float64),
            rtol=0,
            atol=1e-12,
        )

    def test_shared_family_closes_the_independent_twenty_one_dimensional_slack(
        self,
    ) -> None:
        generators = torch_triality_generators(dtype=torch.float64)
        teacher = sample_teacher(seed=18, generators=generators)
        basis = negative_calibration_basis(
            18, dtype=torch.float64, device=torch.device("cpu")
        )
        report = combined_design_audit(teacher.coefficients, generators, basis)
        self.assertEqual(report["minimum_shared_rank"], 28)
        self.assertEqual(report["independent_rank_pattern"], [(25, 25, 13)])
        self.assertEqual(report["minimum_independent_slack"], 21)

    def test_binding_bypasses_negative_action_while_direct_memory_consumes_it(
        self,
    ) -> None:
        generators = torch_triality_generators(dtype=torch.float64)
        oracle = sample_teacher(seed=19, generators=generators).actions
        perturbed = oracle.clone()
        perturbed[:, 2] = torch.eye(8, dtype=torch.float64)
        policy = FrozenSlotPolicy("oracle_both", None, None)

        oracle_binding = evaluate_blind_alias_sequences(
            oracle,
            oracle,
            policy,
            mode="binding",
            seed=19,
            length=128,
            batch_size=32,
        )
        perturbed_binding = evaluate_blind_alias_sequences(
            perturbed,
            oracle,
            policy,
            mode="binding",
            seed=19,
            length=128,
            batch_size=32,
        )
        self.assertEqual(perturbed_binding, oracle_binding)

        perturbed_direct = evaluate_blind_alias_sequences(
            perturbed,
            oracle,
            policy,
            mode="direct",
            seed=19,
            length=128,
            batch_size=32,
        )
        self.assertLess(perturbed_direct["mean_query_cosine"], 0.95)


class FiveProbeTrialityIdentifiabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generators = torch_triality_generators(dtype=torch.float64)
        self.probes = make_probe_family(
            23, dtype=torch.float64, device=torch.device("cpu")
        )
        self.zero = torch.zeros(28, dtype=torch.float64)

    def test_five_mixed_probes_are_sharp_for_shared_triality(self) -> None:
        four = numerical_rank(
            shared_observation_jacobian(
                self.zero, self.generators, self.probes, FOUR_MIXED
            )
        )
        five = numerical_rank(
            shared_observation_jacobian(
                self.zero, self.generators, self.probes, FIVE_MIXED
            )
        )
        single = numerical_rank(
            shared_observation_jacobian(
                self.zero, self.generators, self.probes, FIVE_SINGLE
            )
        )
        self.assertEqual((four["rank"], four["nullity"]), (25, 3))
        self.assertEqual((five["rank"], five["nullity"]), (28, 0))
        self.assertEqual((single["rank"], single["nullity"]), (25, 3))

    def test_independent_five_probe_family_retains_fifty_five_slack_directions(
        self,
    ) -> None:
        independent_zero = torch.zeros(3, 28, dtype=torch.float64)
        report = numerical_rank(
            independent_observation_jacobian(
                independent_zero,
                self.generators,
                self.probes,
                FIVE_MIXED,
            )
        )
        self.assertEqual((report["rank"], report["nullity"]), (29, 55))

    def test_four_probe_stabilizer_is_an_exact_hidden_action_witness(self) -> None:
        rho = triality_tensor(dtype=torch.float64)
        teacher = sample_teacher(seed=23, generators=self.generators)
        witness = exact_four_probe_witness(
            teacher.actions,
            self.generators,
            self.probes,
            rho,
            seed=23,
        )
        self.assertEqual(witness["jacobian"]["nullity"], 3)
        self.assertLess(witness["visible_endpoint_max_error"], 1e-12)
        self.assertLess(witness["hidden_negative_mean_cosine"], 0.99)
        self.assertLess(witness["alternative_triality_max_error"], 1e-12)


class ActiveTrialitySensingTests(unittest.TestCase):
    def test_left_invariant_information_is_action_independent(self) -> None:
        generators = torch_triality_generators(dtype=torch.float64)
        design = fixed_sensor(31, torch.device("cpu"))
        actions = sample_teacher(seed=31, generators=generators).actions
        audit = action_independence_audit(design, generators, actions)
        self.assertLess(audit["information_max_absolute_error"], 1e-12)
        self.assertLess(audit["spectrum_max_absolute_error"], 1e-12)

    def test_five_queries_need_multiple_triality_views(self) -> None:
        generators = torch_triality_generators(dtype=torch.float64)
        mixed = fixed_sensor(32, torch.device("cpu"))
        single = SensorDesign(torch.zeros(5, dtype=torch.long), mixed.vectors, "single")
        mixed_report = information_metrics(mixed, generators)
        single_report = information_metrics(single, generators)
        self.assertEqual((mixed_report["rank"], mixed_report["nullity"]), (28, 0))
        self.assertEqual((single_report["rank"], single_report["nullity"]), (25, 3))


class JointSensorRetractionTests(unittest.TestCase):
    def test_single_query_information_is_a_rank_seven_projector(self) -> None:
        generators = torch_triality_generators(dtype=torch.float64)
        audit = single_query_projector_audit(
            generators, seed=33, probes_per_representation=10
        )
        self.assertTrue(audit["passed"])
        self.assertEqual((audit["minimum_rank"], audit["maximum_rank"]), (7, 7))

    def test_exact_spectral_polynomial_implies_the_three_invariants(self) -> None:
        coefficients = exact_characteristic_coefficients()
        self.assertEqual(coefficients[0], 1)
        self.assertEqual(-coefficients[1], 35)
        self.assertEqual(coefficients[-1], Fraction(81, 1024))
        self.assertEqual(-coefficients[-2] / coefficients[-1], 43)

    def test_joint_retraction_evaluates_the_complete_assignment_family(self) -> None:
        generators = torch_triality_generators(dtype=torch.float64)
        cpu_generator = torch.Generator(device="cpu").manual_seed(34)
        bank = SoftSensorBank(
            logits=torch.zeros(5, 3, dtype=torch.float64),
            vectors=nn.functional.normalize(
                torch.randn(5, 3, 8, generator=cpu_generator, dtype=torch.float64),
                dim=-1,
            ),
            training={},
        )
        design, report = joint_retract_sensor(bank, generators)
        self.assertEqual(report["assignment_count"], 243)
        self.assertEqual(report["selection_gap"], 0.0)
        self.assertEqual(information_metrics(design, generators)["rank"], 28)


class CayleySpectrumTheoremTests(unittest.TestCase):
    def test_cayley_form_is_exactly_spin7_invariant(self) -> None:
        audit = cayley_invariance_audit()
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["stabilizer_generator_count"], 21)
        self.assertEqual(audit["maximum_infinitesimal_invariance_error"], 0.0)

    def test_balanced_characteristic_law_is_exact(self) -> None:
        certificate = exact_cayley_spectrum_certificate()
        self.assertTrue(certificate["passed"])
        self.assertEqual(certificate["balanced_determinant"], "81/1024")
        self.assertEqual(certificate["balanced_rank"], 28)
        self.assertEqual(certificate["calibrated_rank"], 25)

    def test_balanced_characteristic_law_has_exact_invariant_blocks(self) -> None:
        certificate = exact_cayley_block_certificate()
        self.assertTrue(certificate["passed"])
        self.assertEqual(certificate["block_dimensions"], [8, 8, 8, 4])
        self.assertEqual(
            certificate["balanced_block_determinants"],
            ["1/4", "9/16", "9/16", "1"],
        )
        self.assertEqual(certificate["balanced_global_determinant"], "81/1024")
        self.assertEqual(
            certificate["same_view_pair_basis_invariance"],
            {"positive_chiral": True, "negative_chiral": True},
        )

    def test_two_orthogonalization_slices_are_exact(self) -> None:
        certificate = exact_restricted_orthogonalization_certificate()
        self.assertTrue(certificate["same_view_correlation_identity"])
        self.assertTrue(certificate["cross_view_correlation_identity"])

    def test_allocation_targets_have_exact_representatives(self) -> None:
        certificate = exact_partition_representatives()
        self.assertTrue(certificate["passed"])
        self.assertEqual(len(certificate["rows"]), 5)

    def test_dirac_projector_geometry_is_exact(self) -> None:
        certificate = exact_projector_geometry_certificate()
        self.assertTrue(certificate["passed"])
        self.assertEqual(certificate["single_query_rank"], 7)
        self.assertEqual(certificate["single_query_trace"], "7")

    def test_strengthened_gram_bound_has_two_exact_slices(self) -> None:
        certificate = exact_strengthened_slice_certificate()
        self.assertTrue(certificate["passed"])
        self.assertTrue(certificate["all_bernstein_coefficients_strictly_positive"])

    def test_approximate_design_shortcut_is_exactly_rejected(self) -> None:
        certificate = exact_approximate_design_rejection()
        self.assertTrue(certificate["passed"])
        self.assertTrue(certificate["certificate_violated"])
        self.assertEqual(certificate["exact_design_threshold"], "28/5")

    def test_dirac_graph_schur_reduction_is_exact(self) -> None:
        certificate = exact_dirac_graph_certificate()
        self.assertTrue(certificate["passed"])
        self.assertEqual(certificate["reference_split"], [7, 21])

    def test_whitening_flow_invariants_are_exact(self) -> None:
        certificate = exact_whitening_flow_invariant_certificate()
        self.assertTrue(certificate["passed"])
        self.assertTrue(certificate["normalized_cayley_is_invariant"])

    def test_signed_star_family_artifact_replays(self) -> None:
        artifact = (
            Path(__file__).parents[1] / "artifacts" / "spin8_dirac_star_20260804.json"
        )
        self.assertTrue(verify_dirac_star_artifact(artifact))

    def test_star_rational_circle_never_promotes_zero_to_float(self) -> None:
        self.assertEqual(rational_circle(0), (0, 1))

    def test_conditional_decorrelation_counterexample_is_exact(self) -> None:
        artifact = (
            Path(__file__).parents[1]
            / "artifacts"
            / "spin8_conditional_counterexample_20260804.json"
        )
        self.assertTrue(verify_conditional_counterexample_artifact(artifact))

    def test_edge_walsh_restriction_follows_from_exact_symmetry(self) -> None:
        certificate = exact_walsh_symmetry_certificate()
        self.assertTrue(certificate["passed"])
        self.assertTrue(certificate["common_adjoint_conjugacy_verified"])
        self.assertEqual(certificate["fixed_e0_diagonal_cayley_symmetry_count"], 8)
        self.assertEqual(certificate["walsh_annihilator"], [[0, 0, 0, 0], [1, 1, 1, 0]])

    def test_dirac_edge_artifact_integrity_and_holdouts_recompute(self) -> None:
        artifact = (
            Path(__file__).parents[1] / "artifacts" / "spin8_dirac_edge_20260804.json"
        )
        self.assertTrue(artifact.is_file())
        self.assertTrue(verify_dirac_edge_artifact(artifact))

    def test_dirac_edge_verifier_rejects_stored_evidence_tampering(self) -> None:
        artifact = (
            Path(__file__).parents[1] / "artifacts" / "spin8_dirac_edge_20260804.json"
        )
        report = json.loads(artifact.read_text(encoding="utf-8"))

        bad_maps = copy.deepcopy(report)
        bad_maps["coefficient_maps_match"] = False
        self.assertFalse(verify_dirac_edge_report(bad_maps))

        bad_symmetry = copy.deepcopy(report)
        bad_symmetry["exact_walsh_symmetry"][
            "common_adjoint_conjugacy_verified"
        ] = False
        self.assertFalse(verify_dirac_edge_report(bad_symmetry))

        bad_degree = copy.deepcopy(report)
        bad_degree["exact_degree_divisibility"]["degree_certificate"][
            "raw_coordinate_pair_degree_upper_bound"
        ] = 13
        self.assertFalse(verify_dirac_edge_report(bad_degree))

    def test_variable_cayley_one_edge_symmetry_is_exact(self) -> None:
        certificate = one_edge_symmetry_certificate()
        self.assertTrue(certificate["passed"])
        self.assertEqual(len(certificate["induced_sign_group"]), 8)
        self.assertEqual(len(certificate["walsh_annihilator"]), 4)

    def test_variable_cayley_disjoint_reconstructions_match(self) -> None:
        artifact = (
            Path(__file__).parents[1]
            / "artifacts"
            / "spin8_dirac_one_edge_exact_20260804.json"
        )
        report = json.loads(artifact.read_text(encoding="utf-8"))
        self.assertTrue(report["coefficient_maps_match"])
        self.assertTrue(report["degrees_match"])
        for name, degrees in ONE_EDGE_EXPECTED_DEGREES.items():
            self.assertEqual(
                report["discovery"][name]["coefficients"],
                report["confirmation"][name]["coefficients"],
            )
            self.assertEqual(report["confirmation"][name]["degrees"], list(degrees))

    def test_variable_cayley_exact_holdouts_replay(self) -> None:
        root = Path(__file__).parents[1]
        reconstruction = root / "artifacts" / "spin8_dirac_one_edge_exact_20260804.json"
        artifact = root / "artifacts" / "spin8_dirac_one_edge_holdouts_20260806.json"
        report = json.loads(artifact.read_text(encoding="utf-8"))
        self.assertTrue(verify_holdout_report(report, reconstruction))
        self.assertEqual(report["exact_comparisons"], 256)
        self.assertEqual(report["mismatch_count"], 0)

        first = report["rows"][0]
        direct = one_edge_direct_determinant(
            ONE_EDGE_HOLDOUTS[first["holdout_index"]], tuple(first["signs"])
        )
        self.assertEqual(str(direct), first["direct_determinant"])

    def test_variable_cayley_final_theorem_artifact_replays(self) -> None:
        root = Path(__file__).parents[1]
        artifacts = root / "artifacts"
        report = json.loads(
            (artifacts / "spin8_dirac_one_edge_duffy_20260806.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(
            verify_assembled_report(
                report,
                artifacts / "spin8_dirac_one_edge_exact_20260804.json",
                artifacts / "spin8_dirac_one_edge_determinant_cache_20260806.json",
                artifacts / "spin8_dirac_one_edge_positivity_20260804.json",
                artifacts / "spin8_dirac_one_edge_holdouts_20260806.json",
            )
        )
        self.assertTrue(report["theorem_proved"])

    def test_duffy_power_transform_and_integer_bernstein_are_exact(self) -> None:
        # p(u,v)=u+2v becomes t(2-y) under u=ty, v=t(1-y).
        power = np.empty((2, 2), dtype=object)
        power.fill(0)
        power[1, 0] = 1
        power[0, 1] = 2
        duffy = _lower_duffy_power_tensor(power)
        self.assertEqual(duffy.shape, (3, 3))
        self.assertEqual(duffy[1, 0], 2)
        self.assertEqual(duffy[1, 1], -1)
        self.assertEqual(sum(abs(value) for value in duffy.flat), 3)

        # p(1-u,1-v)=3-u-2v.
        complemented = _complement_first_two(power)
        self.assertEqual(complemented[0, 0], 3)
        self.assertEqual(complemented[1, 0], -1)
        self.assertEqual(complemented[0, 1], -2)

        univariate = np.array([1, 1], dtype=object)
        controls, scale = _integer_bernstein_tensor(univariate)
        self.assertEqual(scale, 1)
        self.assertEqual(controls.tolist(), [1, 2])

    def test_sparse_tetrahedral_determinant_matches_direct_formula(self) -> None:
        variables = sp.symbols("u v r w z")
        u, v, r, w, z = variables
        rows = [
            sp.Poly(1 + u + z, *variables),
            sp.Poly(v + r, *variables),
            sp.Poly(w + z, *variables),
            sp.Poly(u * v + 1, *variables),
            sp.Poly(r * w + z, *variables),
        ]
        X, P, Q, S, C = rows
        sparse = _sparse_tetrahedral_determinant(X, P, Q, S, C)
        direct = sp.Poly(
            sp.expand(
                X.as_expr() ** 4
                - 2 * X.as_expr() ** 2 * (P + Q + S).as_expr()
                - 8 * X.as_expr() * C.as_expr()
                + P.as_expr() ** 2
                + Q.as_expr() ** 2
                + S.as_expr() ** 2
                - 2 * (P * Q + P * S + Q * S).as_expr()
            ),
            *variables,
        )
        self.assertEqual(sparse, direct)

    def test_variable_cayley_positivity_status_remains_honest(self) -> None:
        artifact = (
            Path(__file__).parents[1]
            / "artifacts"
            / "spin8_dirac_one_edge_positivity_20260804.json"
        )
        report = json.loads(artifact.read_text(encoding="utf-8"))
        self.assertTrue(report["boundary_adapted_face"]["x2_minus_p2_proved"])
        self.assertTrue(report["cubic"]["proved_nonnegative"])
        self.assertFalse(report["determinant"]["proved_nonnegative"])
        self.assertFalse(report["theorem_proved"])


if __name__ == "__main__":
    unittest.main()
