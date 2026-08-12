"""Correctness, fairness, streaming, and CUDA tests for the family harness."""

from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from compare_recurrences import (
    GROUPS,
    Q8_TABLE,
    group_prefix_products,
    make_group_batches,
    pair_split_audit,
    parse_held_out_pairs,
    parse_input_elements,
    q8_prefix_products,
    q8_products,
    state_and_pair_coverage_audit,
    streaming_equivalence,
)
from changed_generator_transfer import (
    evaluate_macro_actions,
    generated_subgroup,
    select_changed_generators,
)
from action_congruence_lattice import (
    enumerate_action_congruences,
    exact_congruence_lattice_audit,
    is_transition_congruence,
    set_partitions,
)
from a5_anchor_representation_audit import align_representation, defect_lie_audit
from mechanistic_group_actions import (
    MECHANISM_FAMILIES,
    PureGroupActionModel,
    _element_inverses,
    a5_orthogonal_irrep,
    algebraic_objectives,
    canonical_group_words,
    cayley_relation_loss,
    initialize_from_a5_irrep,
    path_holonomy_objectives,
    representation_diagnostics,
    streaming_equivalence as mechanism_streaming_equivalence,
)
from pdssm_group_actions import (
    ExactRegularPD,
    LearnedHardPD,
    _hard_column_one_hot,
    _hard_projected_permutation,
)
from robust_channel_gating import gate_vector, robust_margin_objective
from joint_a5_rounding import anchor_mechanism_diagnostics, full_rotor_action_matrices
from latent_group_discovery import (
    TransitionEvidence,
    exact_inverse_tokens,
    inverse_cover_partial_evidence,
    random_partial_evidence,
)
from inverse_cover_adversarial_audit import (
    MATCHINGS as INVERSE_MATCHINGS,
    adversarial_solution,
    evidence_from_solution,
    reverse_pair_variables,
)
from endpoint_group_discovery import (
    GroupEndpointOracle,
    endpoint_transition_cover,
    infer_four_token_inverse_matching,
    passive_representatives,
    recover_from_endpoint_queries,
)
from endpoint_credit_assignment_audit import (
    information_metrics,
    token_endpoint_joint,
)
from endpoint_representation_discovery import recover_endpoint_manifold
from representation_retraction import (
    compile_nearest_representation,
    local_joint_conjugacy_retraction,
    regular_irrep_candidates,
)
from spinor_center_fidelity_audit import (
    QUATERNIONS,
    distinct_matrix_count,
    quaternion_conjugation_rotation,
    quaternion_left_matrix,
)
from q8_spinor_joint_retraction import exact_q8_targets
from spin8_triality import (
    SPIN8_BIVECTOR_DIM,
    TRIALITY_REPRESENTATIONS,
    Spin8TrialitySSM,
    algebra_diagnostics as spin8_algebra_diagnostics,
    build_spin8_triality_algebra,
    recurrent_diagnostics as spin8_recurrent_diagnostics,
    so8_chart_equivalence_diagnostics,
    spin8_actions,
    torch_triality_generators,
)
from spin8_q8_joint_retraction import (
    exact_ambient_actions as spin8_exact_q8_ambient_actions,
    polar_q8_frames,
    positive_spin8_parameters,
    q8_active_actions,
    real_orthogonal_logarithm,
)
from spin8_q8_path_section_compiler import minimum_change_observer
from spin8_q8_regular_orbit_retraction import (
    exact_regular_ambient_actions,
    q8_right_regular_actions,
    regular_ambient_actions,
    regular_orbit_projection,
    right_regular_actions,
)
from table_blind_family_compiler import (
    align_orthogonal_representation,
    matrix_to_householders,
)
from spin8_state_only_compiler import (
    abstract_group_isomorphic,
    deterministic_kmeans,
)
from spin8_finest_congruence_compiler import quotient_certificate
from spin8_so8_optimizer_equivariance import run_optimizer_pair
from recurrence_families_torch import (
    FAMILY_NAMES,
    ComplexUnitaryRecurrence,
    GARotorRecurrence,
    HybridComplexGARecurrence,
    QuaternionEvenRecurrence,
    RecurrenceSequenceModel,
    quaternion_product,
    unit_quaternion_from_bivector,
)


class AlgebraFamilyTests(unittest.TestCase):
    def test_generic_regular_actions_match_q8_compatibility_wrappers(self) -> None:
        group = GROUPS["q8"]
        regular = right_regular_actions(group)
        np.testing.assert_array_equal(regular, q8_right_regular_actions())
        conjugations = np.stack((np.eye(8), np.eye(8)))
        generic = regular_ambient_actions(
            conjugations, group, (1, 5, 2, 6)
        )
        np.testing.assert_allclose(
            generic, exact_regular_ambient_actions(conjugations), atol=0.0
        )

    def test_four_householders_reconstruct_arbitrary_so4_action(self) -> None:
        generator = np.random.default_rng(81)
        for _ in range(8):
            matrix, _ = np.linalg.qr(generator.normal(size=(4, 4)))
            if np.linalg.det(matrix) < 0:
                matrix[:, 0] *= -1
            vectors = matrix_to_householders(matrix)
            reconstructed = np.eye(4)
            for vector in vectors:
                if np.linalg.norm(vector) > 1e-12:
                    unit = vector / np.linalg.norm(vector)
                    reconstructed = (
                        np.eye(4) - 2.0 * np.outer(unit, unit)
                    ) @ reconstructed
            np.testing.assert_allclose(reconstructed, matrix, atol=2e-7)

    def test_four_dimensional_intertwiner_recovers_shared_conjugacy(self) -> None:
        candidate = regular_irrep_candidates(GROUPS["q8"], 4, seed=91_019)[0]
        tokens = candidate.actions[[1, 5, 2, 6]]
        generator = np.random.default_rng(82)
        change, _ = np.linalg.qr(generator.normal(size=(4, 4)))
        learned = change[None] @ tokens @ change.T[None]
        recovered, rms = align_orthogonal_representation(
            learned, tokens, seed=83
        )
        self.assertLess(rms, 1e-10)
        np.testing.assert_allclose(
            learned, recovered[None] @ tokens @ recovered.T[None], atol=1e-9
        )

    def test_spin8_q8_orbit_retraction_is_joint_and_exact(self) -> None:
        torch.manual_seed(108)
        frame, _ = torch.linalg.qr(torch.randn(8, 4, dtype=torch.float64))
        orbit = torch.cat((frame, -frame), dim=1).unsqueeze(0).numpy()
        recovered, singular_values, projection = polar_q8_frames(orbit)
        np.testing.assert_allclose(recovered[0], frame.numpy(), atol=1e-12)
        self.assertLess(float(projection[0]), 1e-12)
        self.assertEqual(singular_values.shape, (1, 8))

        active = q8_active_actions()
        np.testing.assert_allclose(
            active.transpose(0, 2, 1) @ active,
            np.broadcast_to(np.eye(4), (active.shape[0], 4, 4)),
            atol=1e-12,
        )
        ambient = spin8_exact_q8_ambient_actions(recovered)
        parameters, imaginary, tangent = positive_spin8_parameters(ambient)
        generators = torch_triality_generators(("positive",), dtype=torch.float64)
        reconstructed = spin8_actions(
            torch.from_numpy(parameters), generators
        ).squeeze(-3).numpy()
        np.testing.assert_allclose(reconstructed, ambient, atol=1e-12)
        self.assertLess(imaginary, 1e-12)
        self.assertLess(tangent, 1e-12)

    def test_real_spin8_logarithm_pairs_minus_one_eigenspace(self) -> None:
        action = np.diag((-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
        tangent = real_orthogonal_logarithm(action)
        self.assertLess(np.max(np.abs(tangent + tangent.T)), 1e-12)
        torch.testing.assert_close(
            torch.matrix_exp(torch.from_numpy(tangent)),
            torch.from_numpy(action),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_regular_orbit_projection_preserves_exact_q8_section(self) -> None:
        torch.manual_seed(109)
        change, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64))
        seed = torch.randn(8, dtype=torch.float64)
        seed /= seed.norm()
        regular = torch.from_numpy(q8_right_regular_actions())
        orbit = torch.stack([change @ action @ seed for action in regular], dim=1)
        targets, conjugations, projection, _, commutant = regular_orbit_projection(
            orbit.unsqueeze(0).numpy()
        )
        self.assertLess(float(projection[0]), 1e-12)
        self.assertLess(float(commutant[0]), 1e-12)
        actions = exact_regular_ambient_actions(conjugations)
        for token, element in enumerate((1, 5, 2, 6)):
            for group_element in range(8):
                product = int(GROUPS["q8"].table[group_element, element])
                np.testing.assert_allclose(
                    actions[token, 0] @ targets[0, :, group_element],
                    targets[0, :, product],
                    atol=1e-12,
                )

    def test_minimum_change_observer_preserves_teacher_logits(self) -> None:
        generator = np.random.default_rng(110)
        old_weight = generator.normal(size=(8, 32))
        teacher = generator.normal(size=(32, 8))
        exact = generator.normal(size=(32, 8))
        transported = minimum_change_observer(old_weight, teacher, exact)
        np.testing.assert_allclose(
            transported @ exact,
            old_weight @ teacher,
            atol=1e-10,
        )

    def test_spin8_triality_algebra_passes_frozen_gate(self) -> None:
        report = spin8_algebra_diagnostics()
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["chirality_multiplicities"], {"positive": 8, "negative": 8})
        for representation in TRIALITY_REPRESENTATIONS:
            self.assertEqual(
                report["representations"][representation]["linear_rank"], 28
            )
        self.assertEqual(report["triality_equivariance_max_abs"], 0.0)

    def test_spin8_center_is_visible_differently_to_triality_representations(self) -> None:
        report = spin8_algebra_diagnostics()
        self.assertTrue(report["checks"]["two_pi_center"])
        self.assertTrue(report["checks"]["full_center_signatures"])
        self.assertEqual(report["center_signature_max_abs"]["omega"], 0.0)
        self.assertEqual(report["center_signature_max_abs"]["minus_omega"], 0.0)

    def test_spin8_shared_bivector_generates_three_orthogonal_actions(self) -> None:
        generators = torch_triality_generators(dtype=torch.float64)
        coefficients = torch.linspace(
            -0.2, 0.3, SPIN8_BIVECTOR_DIM, dtype=torch.float64
        )
        actions = spin8_actions(coefficients, generators)
        identity = torch.eye(8, dtype=torch.float64).expand(3, -1, -1)
        torch.testing.assert_close(actions.transpose(-1, -2) @ actions, identity)
        self.assertFalse(torch.allclose(actions[0], actions[1]))
        self.assertFalse(torch.allclose(actions[1], actions[2]))

    def test_positive_spin8_and_generic_so8_charts_are_exactly_equivalent(self) -> None:
        report = so8_chart_equivalence_diagnostics()
        self.assertTrue(report["passed"], report)
        self.assertLess(report["generator_reconstruction_max_abs_error"], 1e-12)
        self.assertLess(report["random_action_equivalence_max_abs_error"], 1e-12)
        self.assertLess(report["basis_change_orthogonality_max_abs_error"], 1e-12)

        mapping = torch.tensor(report["coefficient_map"])
        torch.manual_seed(2718)
        positive = PureGroupActionModel(
            4, 8, family="pure_spin8_positive", channels=2
        ).eval()
        torch.manual_seed(2718)
        generic = PureGroupActionModel(
            4, 8, family="pure_so8_exponential", channels=2
        ).eval()
        coefficients = 0.2 * torch.randn_like(positive.action_parameters)
        with torch.no_grad():
            positive.action_parameters.copy_(coefficients)
            generic.action_parameters.copy_(coefficients @ mapping)
        torch.testing.assert_close(
            positive.action_matrices(), generic.action_matrices(), rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            positive.initial_orbit_state, generic.initial_orbit_state
        )
        torch.testing.assert_close(positive.output_head.weight, generic.output_head.weight)

    def test_sgd_preserves_exact_spin8_so8_chart_equivalence(self) -> None:
        result = run_optimizer_pair("sgd", steps=3, batch_size=16)
        self.assertLess(result["maxima"]["coefficient_map_max_abs_error"], 1e-10)
        self.assertLess(result["maxima"]["action_max_abs_error"], 1e-10)
        self.assertLess(result["maxima"]["postupdate_logit_max_abs_error"], 1e-10)


    def test_spin8_fixed_construction_needs_no_fitted_alignment(self) -> None:
        algebra = build_spin8_triality_algebra()
        self.assertEqual(algebra.rho.shape, (8, 8, 8))
        self.assertEqual(algebra.gamma.shape, (8, 16, 16))
        self.assertEqual(algebra.positive_generators.shape, (28, 8, 8))
        self.assertEqual(algebra.negative_generators.shape, (28, 8, 8))

    def test_latent_group_recovery_uses_transition_evidence_only(self) -> None:
        group = GROUPS["a5"]
        inputs = parse_input_elements(
            ["23145", "31245", "23451", "51234"], group
        )
        batches = make_group_batches(
            group,
            4,
            512,
            16,
            9_901,
            input_elements=inputs,
            held_out_pairs=((0, 2),),
        )
        evidence = TransitionEvidence(group.order, len(inputs))
        for tokens, targets in batches:
            evidence.observe(tokens, targets)
        self.assertTrue(evidence.complete)
        recovered = evidence.recover(base_state=17)
        self.assertEqual(recovered.group.order, group.order)
        self.assertEqual(len(set(recovered.input_elements)), len(inputs))
        self.assertGreater(int(recovered.evidence_counts.min()), 0)
        # The reconstructed table and label gauge reproduce every observed
        # transition without consulting the original Cayley table.
        for state in range(group.order):
            element = recovered.state_to_element[state]
            for token, token_element in enumerate(recovered.input_elements):
                next_element = recovered.group.table[element, token_element]
                self.assertEqual(
                    int(recovered.element_to_state[next_element]),
                    int(evidence.next_states[state, token]),
                )
        candidates = regular_irrep_candidates(recovered.group, 3)
        self.assertEqual(len(candidates), 2)

    def test_inverse_cover_recovers_from_exactly_half_the_edges(self) -> None:
        group = GROUPS["a5"]
        inputs = parse_input_elements(
            ["23145", "31245", "23451", "51234"], group
        )
        batches = make_group_batches(
            group, 4, 512, 16, 18_901, input_elements=inputs
        )
        full = TransitionEvidence(group.order, len(inputs))
        for tokens, targets in batches:
            full.observe(tokens, targets)
        self.assertTrue(full.complete)
        true_inverses = exact_inverse_tokens(full.next_states)

        partial, mask = inverse_cover_partial_evidence(
            full, calibration_fraction=0.10, seed=71
        )
        self.assertAlmostEqual(mask["observed_fraction"], 0.55)
        self.assertFalse(partial.complete)
        inferred = partial.infer_inverse_pairs_and_complete()
        self.assertEqual(inferred, true_inverses)
        np.testing.assert_array_equal(partial.next_states, full.next_states)
        recovered = partial.recover(base_state=0)
        self.assertEqual(recovered.group.order, group.order)

        globally_minimal, minimal_mask = inverse_cover_partial_evidence(
            full, calibration_pairs_total=1, seed=74
        )
        self.assertEqual(minimal_mask["observed_edges"], 121)
        minimal_inferred = globally_minimal.infer_inverse_pairs_and_complete()
        self.assertEqual(minimal_inferred, true_inverses)
        np.testing.assert_array_equal(globally_minimal.next_states, full.next_states)

        exact_half, exact_half_mask = inverse_cover_partial_evidence(
            full, calibration_fraction=0.0, seed=72
        )
        self.assertEqual(exact_half_mask["observed_edges"], 120)
        half_inferred = exact_half.infer_inverse_pairs_and_complete()
        self.assertEqual(half_inferred, true_inverses)
        np.testing.assert_array_equal(exact_half.next_states, full.next_states)

        undercovered, _ = inverse_cover_partial_evidence(
            full, calibration_fraction=0.0, seed=75
        )
        missing_source, missing_token = np.argwhere(
            undercovered.next_states >= 0
        )[0]
        undercovered.next_states[missing_source, missing_token] = -1
        undercovered.counts[missing_source, missing_token] = 0
        with self.assertRaisesRegex(ValueError, "no inverse-token matching"):
            undercovered.infer_inverse_pairs_and_complete()

        random_mask = random_partial_evidence(
            full, observed_edges=mask["observed_edges"], seed=73
        )
        with self.assertRaises(ValueError):
            random_mask.infer_inverse_pairs_and_complete()

        pairs, coordinate_literals = reverse_pair_variables(
            full.next_states, true_inverses
        )
        solution, wrong_identity_support = adversarial_solution(
            full.next_states,
            pairs,
            coordinate_literals,
            INVERSE_MATCHINGS[1],
        )
        self.assertEqual(wrong_identity_support, 0)
        self.assertIsNotNone(solution)
        adversarial = evidence_from_solution(full, pairs, solution)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            adversarial.infer_inverse_pairs_and_complete()
        for source, token in pairs[0]:
            adversarial.next_states[source, token] = full.next_states[source, token]
            adversarial.counts[source, token] = full.counts[source, token]
        inferred_after_calibration = adversarial.infer_inverse_pairs_and_complete()
        self.assertEqual(inferred_after_calibration, true_inverses)
        np.testing.assert_array_equal(adversarial.next_states, full.next_states)

    def test_regular_representation_compiler_and_joint_retraction(self) -> None:
        group = GROUPS["a5"]
        inputs = parse_input_elements(
            ["23145", "31245", "23451", "51234"], group
        )
        candidates = regular_irrep_candidates(group, 3)
        self.assertEqual(len(candidates), 2)
        for candidate in candidates:
            self.assertLess(candidate.invariance_rms, 1e-12)
            self.assertLess(candidate.homomorphism_rms, 1e-12)

        inverses = _element_inverses(group)
        change = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        compiled = None
        for index, candidate in enumerate(candidates):
            exact = np.stack(
                [candidate.actions[inverses[element]] for element in inputs]
            )
            learned = change[None] @ exact @ change.T[None]
            compiled = compile_nearest_representation(
                learned, group, inputs, seed=19 + index
            )
            self.assertLess(compiled.alignment_rms, 1e-7)
            self.assertGreater(compiled.runner_up_rms, 0.1)
            np.testing.assert_allclose(
                compiled.character, candidate.character, rtol=1e-7, atol=1e-7
            )
        self.assertIsNotNone(compiled)

        perturbations = np.asarray(
            ([0.01, 0.00, 0.00], [0.00, -0.02, 0.00],
             [0.00, 0.00, 0.015], [0.01, -0.01, 0.00])
        )
        ambient = np.stack(
            [
                Rotation.from_rotvec(delta).as_matrix() @ action
                for delta, action in zip(perturbations, compiled.token_actions)
            ]
        )
        retracted, projection_rms, tangent_norm = local_joint_conjugacy_retraction(
            ambient, compiled.token_actions
        )
        self.assertGreater(projection_rms, 0.0)
        self.assertGreater(tangent_norm, 0.0)
        # Retraction changes all tokens by one conjugation, so the complete
        # mixed relation table remains exact rather than merely normalizing
        # each token independently.
        recovered = compile_nearest_representation(retracted, group, inputs, seed=23)
        self.assertLess(recovered.alignment_rms, 1e-7)

    def test_endpoint_only_queries_recover_the_complete_action(self) -> None:
        group = GROUPS["a5"]
        inputs = parse_input_elements(
            ["23145", "31245", "23451", "51234"], group
        )
        oracle = GroupEndpointOracle(group, inputs)
        recovered, report = recover_from_endpoint_queries(
            oracle.query,
            state_count=group.order,
            token_count=len(inputs),
            passive_samples=1_024,
            passive_word_length=16,
            seed=44_001,
        )
        self.assertEqual(report.passive_unique_labels, group.order)
        self.assertEqual(report.inferred_inverse_tokens, (1, 0, 3, 2))
        self.assertEqual(report.observed_transition_edges, 120)
        self.assertEqual(report.completed_transition_edges, 120)
        self.assertEqual(report.total_active_queries, 124)
        self.assertEqual(oracle.query_count, 1_148)
        labels = recovered.element_to_state
        np.testing.assert_array_equal(
            labels[recovered.group.table],
            group.table[labels[:, None], labels[None, :]],
        )

        negative_oracle = GroupEndpointOracle(group, inputs)
        identity = negative_oracle.query(())
        representatives, _ = passive_representatives(
            negative_oracle.query,
            token_count=4,
            state_count=group.order,
            samples=1_024,
            word_length=16,
            seed=44_002,
        )
        inverse = infer_four_token_inverse_matching(
            negative_oracle.query, identity
        )
        with self.assertRaisesRegex(ValueError, "does not complete"):
            endpoint_transition_cover(
                negative_oracle.query,
                representatives,
                inverse,
                state_count=group.order,
                omit_extension=(0, 0),
            )

    def test_endpoint_information_collapses_after_group_walk_mixing(self) -> None:
        group = GROUPS["a5"]
        inputs = parse_input_elements(
            ["23145", "31245", "23451", "51234"], group
        )
        length_one = token_endpoint_joint(group.table, inputs, 1, 0)
        length_sixteen = token_endpoint_joint(group.table, inputs, 16, 8)
        self.assertAlmostEqual(float(length_one.sum()), 1.0, places=12)
        self.assertAlmostEqual(float(length_sixteen.sum()), 1.0, places=12)
        self.assertAlmostEqual(
            information_metrics(length_one)["mutual_information_bits"],
            2.0,
            places=12,
        )
        self.assertLess(
            information_metrics(length_sixteen)["mutual_information_bits"],
            1e-3,
        )

    def test_endpoint_clusters_recover_table_without_supplied_cayley_action(self) -> None:
        group = GROUPS["a5"]
        inputs = parse_input_elements(
            ["23145", "31245", "23451", "51234"], group
        )
        inverses = _element_inverses(group)
        representation = a5_orthogonal_irrep(group, branch=0)
        token_rotations = np.stack(
            [representation[inverses[element]] for element in inputs]
        )
        batches = make_group_batches(
            group,
            32,
            256,
            8,
            81_003,
            input_elements=inputs,
            held_out_pairs=((0, 2),),
        )
        words = np.concatenate([tokens.numpy() for tokens, _ in batches])
        labels = np.concatenate(
            [targets[:, -1].numpy() for _, targets in batches]
        )
        recovered = recover_endpoint_manifold(
            token_rotations,
            words,
            labels,
            state_count=group.order,
        )
        mapping = recovered.label_to_element
        np.testing.assert_array_equal(
            recovered.group.table[mapping[:, None], mapping[None, :]],
            mapping[group.table],
        )
        self.assertLess(recovered.multiplication_max, 1e-12)
        self.assertGreater(recovered.minimum_assignment_gap, 0.5)

        permutation = np.random.default_rng(81_004).permutation(group.order)
        permuted = recover_endpoint_manifold(
            token_rotations,
            words,
            permutation[labels],
            state_count=group.order,
        )
        permuted_mapping = permuted.label_to_element[permutation]
        np.testing.assert_array_equal(
            permuted.group.table[
                permuted_mapping[:, None], permuted_mapping[None, :]
            ],
            permuted_mapping[group.table],
        )

    def test_spinor_action_retains_q8_center_that_sandwich_loses(self) -> None:
        left = np.stack([quaternion_left_matrix(q) for q in QUATERNIONS])
        sandwich = np.stack(
            [quaternion_conjugation_rotation(q) for q in QUATERNIONS]
        )
        self.assertEqual(distinct_matrix_count(left), 8)
        self.assertEqual(distinct_matrix_count(sandwich), 4)
        np.testing.assert_array_equal(left[4], -left[0])
        np.testing.assert_array_equal(sandwich[4], sandwich[0])
        np.testing.assert_allclose(left[1] @ left[1], left[4], atol=0.0)
        np.testing.assert_allclose(
            sandwich[1] @ sandwich[1], sandwich[0], atol=0.0
        )
        self.assertEqual(np.linalg.matrix_rank(np.eye(4) - left[1]), 4)
        duplicated_left_i = np.kron(np.eye(2), left[1])
        self.assertEqual(
            np.linalg.matrix_rank(np.eye(8) - duplicated_left_i), 8
        )

        chart_radius = float(np.arctanh(0.5))
        parameters = torch.tensor(
            [
                [-chart_radius, 0.0, 0.0],
                [chart_radius, 0.0, 0.0],
                [0.0, -chart_radius, 0.0],
                [0.0, chart_radius, 0.0],
            ]
        )

        def generated_action_count(family: str) -> int:
            model = PureGroupActionModel(
                4,
                8,
                family=family,
                channels=1,
                max_rotor_angle=2 * math.pi,
            )
            with torch.no_grad():
                model.action_parameters[:, 0].copy_(parameters)
            actions = model.action_matrices().detach().numpy()[:, 0]
            representatives = [np.eye(8)]
            frontier = [np.eye(8)]
            while frontier:
                current = frontier.pop()
                for action in actions:
                    candidate = action @ current
                    if not any(
                        np.max(np.abs(candidate - existing)) < 1e-5
                        for existing in representatives
                    ):
                        representatives.append(candidate)
                        frontier.append(candidate)
            return len(representatives)

        self.assertEqual(generated_action_count("pure_quaternion_spinor"), 8)
        self.assertEqual(generated_action_count("pure_ga_rotor"), 4)

        noisy = QUATERNIONS[[1, 5, 2, 6], None].copy()
        noisy[0, 0, 0] = 0.03
        noisy[2, 0, 1] = 0.08
        targets = exact_q8_targets(noisy)
        np.testing.assert_allclose(targets[1], -targets[0], atol=0.0)
        np.testing.assert_allclose(targets[3], -targets[2], atol=0.0)
        np.testing.assert_allclose(
            np.sum(targets[0, 0, 1:] * targets[2, 0, 1:]), 0.0, atol=1e-15
        )
        np.testing.assert_allclose(
            np.linalg.norm(targets[:, 0, 1:], axis=-1), 1.0, atol=1e-15
        )

    def test_a5_irrep_branches_and_lie_closure_audit(self) -> None:
        group = GROUPS["a5"]
        inputs = parse_input_elements(
            ["23145", "31245", "23451", "51234"], group
        )
        inverses = _element_inverses(group)
        first = a5_orthogonal_irrep(group, branch=0)
        second = a5_orthogonal_irrep(group, branch=1)
        first_tokens = np.stack([first[inverses[element]] for element in inputs])
        second_tokens = np.stack([second[inverses[element]] for element in inputs])
        self.assertGreater(
            abs(np.trace(first_tokens[2]) - np.trace(second_tokens[2])), 1.0
        )
        change = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
        )
        learned = change[None] @ first_tokens @ change.T[None]
        _, matching_rms = align_representation(learned, first_tokens, seed=17)
        _, other_rms = align_representation(learned, second_tokens, seed=18)
        self.assertLess(matching_rms, 1e-7)
        self.assertGreater(other_rms, 0.1)

        oracle = np.repeat(np.eye(3)[None], 4, axis=0)
        defects = oracle.copy()
        x_angle, y_angle = 0.01, 0.02
        defects[0] = np.asarray(
            [[1, 0, 0], [0, math.cos(x_angle), -math.sin(x_angle)],
             [0, math.sin(x_angle), math.cos(x_angle)]]
        )
        defects[2] = np.asarray(
            [[math.cos(y_angle), 0, math.sin(y_angle)], [0, 1, 0],
             [-math.sin(y_angle), 0, math.cos(y_angle)]]
        )
        lie = defect_lie_audit(defects, oracle, np.eye(3))
        self.assertEqual(lie["lie_closure_rank"], 3)

        exact_actions = full_rotor_action_matrices(
            first_tokens,
            max_rotor_angle=2.2,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        diagnostics = anchor_mechanism_diagnostics(
            exact_actions, group, inputs
        )
        self.assertLess(diagnostics["vector_homomorphism_rms"], 1e-10)
        self.assertLess(diagnostics["mixed_relator_rms_max"], 1e-10)

    def test_robust_channel_gate_keeps_anchor_fixed_and_backpropagates(self) -> None:
        raw = torch.zeros(3, requires_grad=True)
        gates = gate_vector(raw, anchor=2, channels=4)
        torch.testing.assert_close(
            gates, torch.tensor([0.5, 0.5, 1.0, 0.5])
        )
        strata = [
            {
                "length": 2,
                "channel_logits": torch.randn(5, 4, 3),
                "bias": torch.randn(3),
                "targets": torch.tensor([0, 1, 2, 0, 1]),
            }
        ]
        loss = robust_margin_objective(strata, gates)
        loss.backward()
        self.assertTrue(torch.isfinite(raw.grad).all())

    def test_changed_generator_selection_is_disjoint_and_generates_a5(self) -> None:
        group = GROUPS["a5"]
        original = parse_input_elements(
            ["23145", "31245", "23451", "51234"], group
        )
        changed = select_changed_generators(group, original)
        self.assertEqual(len(set(changed)), 4)
        self.assertTrue(set(changed).isdisjoint(original))
        self.assertEqual(len(generated_subgroup(group, (changed[0], changed[2]))), 60)
        second = select_changed_generators(group, original, selection_index=1)
        first_class = {
            frozenset(changed[:2]),
            frozenset(changed[2:]),
        }
        second_class = {
            frozenset(second[:2]),
            frozenset(second[2:]),
        }
        self.assertNotEqual(first_class, second_class)
        self.assertEqual(len(generated_subgroup(group, (second[0], second[2]))), 60)

    def test_changed_generator_evaluator_covers_all_channel_subsets(self) -> None:
        group = GROUPS["a5"]
        model = PureGroupActionModel(
            4, group.order, family="pure_ga_rotor", channels=4
        )
        inputs = parse_input_elements(
            ["23145", "31245", "23451", "51234"], group
        )
        batches = make_group_batches(
            group, 1, 4, 2, 1234, input_elements=inputs
        )
        result = evaluate_macro_actions(
            model,
            model.action_matrices(),
            batches,
            strong_channel=0,
            device=torch.device("cpu"),
        )
        self.assertEqual(len(result["channel_subset_final_position_accuracy"]), 15)
        self.assertEqual(len(result["channel_subset_repairs_over_strong_rate"]), 15)
        self.assertEqual(len(result["channel_subset_damages_over_strong_rate"]), 15)
        self.assertEqual(len(result["channel_subset_both_correct_rate"]), 15)
        for subset in result["channel_subset_final_position_accuracy"]:
            partition = sum(
                result[key][subset]
                for key in (
                    "channel_subset_repairs_over_strong_rate",
                    "channel_subset_damages_over_strong_rate",
                    "channel_subset_both_correct_rate",
                    "channel_subset_both_wrong_same_prediction_rate",
                    "channel_subset_both_wrong_different_prediction_rate",
                )
            )
            self.assertAlmostEqual(partition, 1.0)

    def test_pd_hard_forward_is_column_one_hot_with_soft_gradient(self) -> None:
        scores = torch.randn(3, 7, 7, requires_grad=True)
        transition = _hard_column_one_hot(scores)
        torch.testing.assert_close(transition.sum(dim=-2), torch.ones(3, 7))
        torch.testing.assert_close(
            transition, transition.round(), atol=1e-7, rtol=0
        )
        (transition.square().sum()).backward()
        self.assertGreater(float(scores.grad.abs().sum()), 0.0)

    def test_exact_regular_pd_tracks_a5_prefixes_and_streams_exactly(self) -> None:
        group = GROUPS["a5"]
        inputs = parse_input_elements(
            ["23145", "31245", "23451", "51234"], group
        )
        model = ExactRegularPD(group, inputs)
        tokens, targets = make_group_batches(
            group, 1, 16, 32, 991, input_elements=inputs
        )[0]
        logits, full_state = model(tokens, return_recurrent_state=True)
        torch.testing.assert_close(logits.argmax(-1), targets)
        state = None
        pieces = []
        for position in range(tokens.shape[1]):
            step_logits, state = model(
                tokens[:, position : position + 1],
                recurrent_state=state,
                return_recurrent_state=True,
            )
            pieces.append(step_logits)
        torch.testing.assert_close(torch.cat(pieces, dim=1), logits)
        torch.testing.assert_close(state, full_state)

    def test_pd_projected_forward_is_a_bijection_with_soft_gradient(self) -> None:
        scores = torch.randn(2, 9, 9, requires_grad=True)
        transition = _hard_projected_permutation(scores)
        torch.testing.assert_close(transition.sum(dim=-2), torch.ones(2, 9))
        torch.testing.assert_close(transition.sum(dim=-1), torch.ones(2, 9))
        torch.testing.assert_close(
            transition, transition.round(), atol=1e-7, rtol=0
        )
        (transition.square().sum()).backward()
        self.assertGreater(float(scores.grad.abs().sum()), 0.0)

    def test_learned_pd_recurrence_preserves_one_hot_state_norm(self) -> None:
        torch.manual_seed(22)
        model = LearnedHardPD(vocab_size=4, output_size=60, state_size=60)
        tokens = torch.randint(0, 4, (8, 24))
        logits, state = model(tokens, return_recurrent_state=True)
        self.assertEqual(logits.shape, (8, 24, 60))
        torch.testing.assert_close(state.norm(dim=-1), torch.ones(8), atol=1e-6, rtol=0)

    def test_generator_alphabet_is_separate_from_group_state_labels(self) -> None:
        group = GROUPS["a5"]
        generators = parse_input_elements(["23145", "23451"], group)
        self.assertEqual(len(generators), 2)
        batches = make_group_batches(
            group, 2, 32, 8, 12, input_elements=generators
        )
        for tokens, targets in batches:
            self.assertTrue(bool(torch.all((tokens == 0) | (tokens == 1))))
            expected_first = torch.tensor(generators)[tokens[:, 0]]
            torch.testing.assert_close(targets[:, 0], expected_first)

    def test_held_out_pair_split_blocks_training_and_requires_evaluation_pair(self) -> None:
        group = GROUPS["s3"]
        pairs = parse_held_out_pairs(["132:213"], group)
        self.assertEqual(pairs, ((1, 2),))
        training = make_group_batches(
            group, 3, 64, 16, 10, held_out_pairs=pairs
        )
        evaluation = make_group_batches(
            group,
            3,
            64,
            16,
            11,
            held_out_pairs=pairs,
            require_held_out_pair=True,
        )
        training_audit = pair_split_audit(training, pairs)
        evaluation_audit = pair_split_audit(evaluation, pairs)
        self.assertEqual(training_audit["pair_occurrences"], 0)
        self.assertEqual(
            evaluation_audit["sequences_with_pair"],
            evaluation_audit["total_sequences"],
        )

    def test_inverse_augmented_a5_split_preserves_state_coverage(self) -> None:
        group = GROUPS["a5"]
        two = parse_input_elements(["23145", "23451"], group)
        four = parse_input_elements(
            ["23145", "31245", "23451", "51234"], group
        )
        collapsed = make_group_batches(
            group,
            8,
            256,
            16,
            21,
            input_elements=two,
            held_out_pairs=((0, 1),),
        )
        rich = make_group_batches(
            group,
            8,
            256,
            16,
            22,
            input_elements=four,
            held_out_pairs=((0, 2),),
        )
        collapsed_audit = state_and_pair_coverage_audit(
            collapsed, input_order=2, group_order=group.order
        )
        rich_audit = state_and_pair_coverage_audit(
            rich, input_order=4, group_order=group.order
        )
        self.assertLess(collapsed_audit["observed_group_states"], group.order)
        self.assertEqual(rich_audit["observed_group_states"], group.order)
        self.assertEqual(rich_audit["observed_input_pairs"], 15)

    def test_all_group_tables_are_associative_noncommutative_groups(self) -> None:
        for key, group in GROUPS.items():
            with self.subTest(group=key):
                np.testing.assert_array_equal(
                    group.table[0], np.arange(group.order)
                )
                np.testing.assert_array_equal(
                    group.table[:, 0], np.arange(group.order)
                )
                self.assertTrue(np.any(group.table != group.table.T))
                for left in range(group.order):
                    for middle in range(group.order):
                        for right in range(group.order):
                            self.assertEqual(
                                int(group.table[group.table[left, middle], right]),
                                int(group.table[left, group.table[middle, right]]),
                            )
                tokens = np.arange(group.order, dtype=np.int64)[None]
                prefixes = group_prefix_products(tokens, group)
                self.assertEqual(prefixes.shape, tokens.shape)

    def test_q8_table_is_noncommutative_and_targets_are_ordered(self) -> None:
        # i*j = k while j*i = -k.
        self.assertEqual(int(Q8_TABLE[1, 2]), 3)
        self.assertEqual(int(Q8_TABLE[2, 1]), 7)
        targets = q8_products(np.asarray([[1, 2], [2, 1]], dtype=np.int64))
        np.testing.assert_array_equal(targets, np.asarray([3, 7]))
        prefixes = q8_prefix_products(
            np.asarray([[1, 2, 3], [2, 1, 3]], dtype=np.int64)
        )
        np.testing.assert_array_equal(prefixes[:, :2], np.asarray([[1, 3], [2, 7]]))

    def test_complex_quaternion_and_ga_actions_preserve_norm(self) -> None:
        torch.manual_seed(0)
        state = torch.randn(3, 2, 8)

        complex_layer = ComplexUnitaryRecurrence(channels=2)
        phases = torch.randn(3, 2, 4)
        complex_rotated = complex_layer.apply_action(phases, state)
        torch.testing.assert_close(
            complex_rotated.norm(dim=-1), state.norm(dim=-1), rtol=1e-5, atol=1e-5
        )

        quaternion_layer = QuaternionEvenRecurrence(channels=2)
        quaternions = unit_quaternion_from_bivector(torch.randn(3, 2, 2, 3))
        quaternion_rotated = quaternion_layer.apply_action(quaternions, state)
        torch.testing.assert_close(
            quaternion_rotated.norm(dim=-1), state.norm(dim=-1), rtol=1e-5, atol=1e-5
        )

        ga_layer = GARotorRecurrence(channels=2, selective_rotation=True)
        _, rotors, _ = ga_layer.transition_parameters(torch.randn(3, 4, 2, 8))
        ga_rotated = ga_layer.apply_action(rotors[:, 0], state)
        torch.testing.assert_close(
            ga_rotated.norm(dim=-1), state.norm(dim=-1), rtol=1e-5, atol=1e-5
        )

        hybrid_layer = HybridComplexGARecurrence(channels=2)
        _, hybrid_action, _ = hybrid_layer.transition_parameters(
            torch.randn(3, 4, 2, 8)
        )
        hybrid_rotated = hybrid_layer.apply_action(hybrid_action[:, 0], state)
        torch.testing.assert_close(
            hybrid_rotated.norm(dim=-1), state.norm(dim=-1), rtol=1e-5, atol=1e-5
        )

    def test_quaternion_action_is_noncommutative(self) -> None:
        i = torch.tensor([0.0, 1.0, 0.0, 0.0])
        j = torch.tensor([0.0, 0.0, 1.0, 0.0])
        self.assertFalse(torch.allclose(quaternion_product(i, j), quaternion_product(j, i)))


class MechanisticGroupActionTests(unittest.TestCase):
    def test_a5_generators_reach_every_group_element(self) -> None:
        group = GROUPS["a5"]
        generators = parse_input_elements(["23145", "23451"], group)
        words = canonical_group_words(group, generators)
        self.assertEqual(len(words), 60)
        self.assertEqual(words[0], ())
        self.assertLessEqual(max(map(len, words)), 10)

    def test_pure_actions_preserve_norm_and_stream_exactly(self) -> None:
        torch.manual_seed(12)
        tokens = torch.arange(24).reshape(3, 8) % 2
        for family in MECHANISM_FAMILIES:
            with self.subTest(family=family):
                model = PureGroupActionModel(
                    2,
                    60,
                    family=family,
                    channels=2,
                    max_rotor_angle=(2.2 if family == "pure_ga_rotor" else math.pi),
                ).eval()
                initial = model.initial_state(tokens.shape[0])
                _, final = model(tokens, return_recurrent_state=True)
                torch.testing.assert_close(
                    final.norm(dim=-1),
                    initial.norm(dim=-1),
                    rtol=1e-5,
                    atol=1e-5,
                )
                errors = mechanism_streaming_equivalence(model, tokens)
                self.assertLess(max(errors.values()), 1e-5)
                matrices = model.action_matrices()
                identity = torch.eye(8)
                gram = matrices.transpose(-1, -2) @ matrices
                torch.testing.assert_close(
                    gram,
                    identity.expand_as(gram),
                    rtol=1e-5,
                    atol=1e-5,
                )

    def test_pure_models_have_no_write_decay_or_residual_parameters(self) -> None:
        for family in MECHANISM_FAMILIES:
            names = {
                name
                for name, _ in PureGroupActionModel(
                    2, 60, family=family, channels=2
                ).named_parameters()
            }
            self.assertEqual(
                names,
                {
                    "initial_orbit_state",
                    "action_parameters",
                    "logit_scale",
                    "output_head.weight",
                    "output_head.bias",
                },
            )

    def test_mechanism_metrics_distinguish_valid_trivial_from_faithful_action(self) -> None:
        group = GROUPS["a5"]
        generators = parse_input_elements(["23145", "23451"], group)
        model = PureGroupActionModel(
            2, 60, family="pure_ga_rotor", channels=2
        ).eval()
        metrics = representation_diagnostics(model, group, generators)
        self.assertLess(metrics["operator_orthogonality_max"], 1e-6)
        self.assertLess(metrics["linear_homomorphism_max"], 1e-6)
        self.assertLess(metrics["identity_word_state_drift_max"], 1e-6)
        self.assertEqual(metrics["prototype_minimum_margin"], 0.0)
        self.assertEqual(metrics["generator_commutator_separation"], 0.0)

    def test_rotor_token_actions_can_be_noncommutative(self) -> None:
        model = PureGroupActionModel(
            2, 8, family="pure_ga_rotor", channels=1
        ).eval()
        with torch.no_grad():
            model.action_parameters[0, 0, 0] = 0.7
            model.action_parameters[1, 0, 1] = 0.9
        actions = model.action_matrices()
        commutator = actions[1] @ actions[0] - actions[0] @ actions[1]
        self.assertGreater(float(commutator.detach().norm()), 0.1)

    def test_spin8_token_actions_are_noncommutative_and_receive_identity_gradients(self) -> None:
        model = PureGroupActionModel(
            2, 8, family="pure_spin8_positive", channels=1
        )
        tokens = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
        model(tokens).square().mean().backward()
        self.assertGreater(float(model.action_parameters.grad.norm()), 0.0)
        with torch.no_grad():
            model.action_parameters.zero_()
            model.action_parameters[0, 0, 0] = 0.7
            model.action_parameters[1, 0, 7] = 0.9
        actions = model.action_matrices()
        commutator = actions[1] @ actions[0] - actions[0] @ actions[1]
        self.assertGreater(float(commutator.detach().norm()), 0.1)

    def test_cayley_relation_loss_is_differentiable(self) -> None:
        group = GROUPS["a5"]
        generators = parse_input_elements(["23145", "23451"], group)
        model = PureGroupActionModel(
            2, 60, family="pure_ga_rotor", channels=1
        )
        identity_loss = cayley_relation_loss(model, group, generators)
        self.assertLess(float(identity_loss.detach()), 1e-10)
        with torch.no_grad():
            model.action_parameters[0, 0, 0] = 0.4
            model.action_parameters[1, 0, 1] = -0.6
        loss = cayley_relation_loss(model, group, generators)
        self.assertGreater(float(loss.detach()), 1e-3)
        tail_loss, _ = algebraic_objectives(
            model, group, generators, relation_loss_power=8.0
        )
        self.assertGreaterEqual(float(tail_loss.detach()), float(loss.detach()))
        tail_loss.backward()
        self.assertIsNotNone(model.action_parameters.grad)
        self.assertGreater(float(model.action_parameters.grad.norm()), 0.0)

    def test_holonomy_contract_detects_identity_collapse_and_accepts_exact_a5(self) -> None:
        group = GROUPS["a5"]
        generators = parse_input_elements(
            ["23145", "31245", "23451", "51234"], group
        )
        tokens, targets = make_group_batches(
            group,
            count=1,
            batch_size=32,
            sequence_length=16,
            seed=8181,
            input_elements=generators,
        )[0]
        identity_model = PureGroupActionModel(
            4,
            60,
            family="pure_ga_rotor",
            channels=2,
            max_rotor_angle=2.2,
        )
        holonomy, separation, _ = path_holonomy_objectives(
            identity_model,
            group,
            generators,
            tokens,
            targets,
            word_multiplier=2,
            batch_size=16,
        )
        self.assertLess(float(holonomy.detach()), 1e-10)
        self.assertGreater(float(separation.detach()), 0.1)
        logits = identity_model(tokens)
        task_loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), targets.flatten()
        )
        (task_loss + 0.01 * holonomy + 0.1 * separation).backward()
        self.assertGreater(float(identity_model.action_parameters.grad.norm()), 1e-5)

        exact_model = PureGroupActionModel(
            4,
            60,
            family="pure_ga_rotor",
            channels=2,
            max_rotor_angle=2.2,
        ).eval()
        initialize_from_a5_irrep(exact_model, group, generators)
        exact_holonomy, exact_separation, _ = path_holonomy_objectives(
            exact_model,
            group,
            generators,
            tokens,
            targets,
            word_multiplier=4,
            batch_size=16,
        )
        self.assertLess(float(exact_holonomy.detach()), 1e-5)
        self.assertLess(float(exact_separation.detach()), 1e-5)

    def test_character_projector_constructs_exact_faithful_a5_actions(self) -> None:
        group = GROUPS["a5"]
        generators = parse_input_elements(["23145", "23451"], group)
        for family in ("pure_ga_rotor", "pure_householder"):
            with self.subTest(family=family):
                model = PureGroupActionModel(
                    2, 60, family=family, channels=2
                ).eval()
                initialize_from_a5_irrep(model, group, generators)
                metrics = representation_diagnostics(model, group, generators)
                self.assertLess(metrics["cayley_edge_relation_rms"], 1e-5)
                self.assertLess(metrics["linear_homomorphism_rms"], 1e-5)
                self.assertLess(metrics["identity_word_state_drift_rms"], 1e-5)
                self.assertLess(metrics["orbit_cayley_edge_rms"], 1e-5)
                self.assertLess(metrics["orbit_homomorphism_rms"], 1e-5)
                self.assertLess(metrics["reachable_span_homomorphism_rms"], 1e-5)
                if metrics["orthogonal_complement_dimension"]:
                    self.assertLess(
                        metrics["orthogonal_complement_homomorphism_rms"], 1e-5
                    )
                if family == "pure_ga_rotor":
                    self.assertEqual(metrics["cl3_invariant_grade_dimension"], 4)
                    self.assertEqual(metrics["common_fixed_subspace_dimension"], 4)
                    self.assertLess(
                        metrics["cl3_invariant_grade_homomorphism_rms"], 1e-6
                    )
                else:
                    self.assertEqual(metrics["cl3_invariant_grade_dimension"], 0)
                    # The oracle embedding deliberately uses only three of the
                    # eight Householder coordinates per channel.
                    self.assertEqual(metrics["common_fixed_subspace_dimension"], 10)
                self.assertGreater(metrics["generator_commutator_separation"], 0.5)
                self.assertGreater(metrics["prototype_minimum_margin"], 0.1)


class StateOnlyCompilerTests(unittest.TestCase):
    def test_q8_congruence_lattice_is_exhaustive_and_not_metric_unique(self) -> None:
        group = GROUPS["q8"]
        next_states = group.table[:, (1, 5, 2, 6)]
        audit = exact_congruence_lattice_audit(next_states)
        self.assertEqual(audit["enumerated_set_partitions"], 4140)
        self.assertEqual(audit["transition_congruence_count"], 6)
        self.assertEqual(
            audit["congruence_count_by_block_count"],
            {"1": 1, "2": 3, "4": 1, "8": 1},
        )
        self.assertEqual(
            audit["regular_quotient_count_by_block_count"],
            {"1": 1, "2": 3, "4": 1, "8": 1},
        )
        self.assertFalse(
            audit["observation_free_unique_nontrivial_quotient_identifiable"]
        )

    def test_extreme_partitions_are_always_transition_congruences(self) -> None:
        action = np.asarray(((1, 2), (2, 0), (0, 1)), dtype=np.int64)
        self.assertTrue(is_transition_congruence(action, (0, 0, 0)))
        self.assertTrue(is_transition_congruence(action, (0, 1, 2)))
        self.assertEqual(sum(1 for _ in set_partitions(3)), 5)
        congruences = enumerate_action_congruences(action)
        self.assertGreaterEqual(len(congruences), 2)

    def test_deterministic_kmeans_recovers_separated_clouds(self) -> None:
        rng = np.random.default_rng(1203)
        expected_centers = np.asarray(((-4.0, 1.0), (0.5, -3.0), (5.0, 2.0)))
        points = np.concatenate(
            [center + 0.03 * rng.normal(size=(64, 2)) for center in expected_centers]
        )
        first = deterministic_kmeans(points, 3, seed=91, restarts=8)
        second = deterministic_kmeans(points, 3, seed=91, restarts=8)
        torch.testing.assert_close(
            torch.from_numpy(first.centers), torch.from_numpy(second.centers)
        )
        self.assertEqual(first.restart, second.restart)
        self.assertEqual(len(np.unique(first.labels)), 3)
        distances = np.sqrt(
            ((first.centers[:, None] - expected_centers[None, :]) ** 2).sum(-1)
        )
        self.assertLess(float(distances.min(axis=1).max()), 0.02)

    def test_abstract_group_isomorphism_ignores_element_names(self) -> None:
        table = GROUPS["q8"].table
        permutation = np.asarray((0, 5, 3, 6, 1, 7, 4, 2), dtype=np.int64)
        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(len(permutation))
        renamed = permutation[table[inverse[:, None], inverse[None, :]]]
        self.assertTrue(abstract_group_isomorphic(table, renamed))

        cyclic = np.fromfunction(
            lambda left, right: (left + right) % 8, (8, 8), dtype=int
        ).astype(np.int64)
        self.assertFalse(abstract_group_isomorphic(table, cyclic))

    def test_quotient_certificate_recognizes_q8_character(self) -> None:
        mapping = np.asarray((0, 0, 1, 1, 0, 0, 1, 1), dtype=np.int64)
        fine_centers = np.eye(8, dtype=np.float64)
        coarse_centers = np.stack(
            [fine_centers[mapping == value].mean(axis=0) for value in range(2)]
        )
        points = np.repeat(fine_centers, 16, axis=0)
        inputs = np.asarray((1, 5, 2, 6), dtype=np.int64)
        fine_next = Q8_TABLE[:, inputs]
        coarse_next = np.stack(
            [mapping[fine_next[np.flatnonzero(mapping == value)[0]]] for value in range(2)]
        )
        coarse = ({}, coarse_centers, coarse_next, 0)
        fine = ({}, fine_centers, fine_next, 0)
        certificate = quotient_certificate(points, coarse, fine)
        self.assertTrue(certificate["is_quotient"], certificate)
        self.assertEqual(certificate["fibre_sizes"], [4, 4])
        self.assertEqual(certificate["intertwining_fraction"], 1.0)


class HarnessContractTests(unittest.TestCase):
    def test_spin8_triality_recurrence_passes_frozen_gate(self) -> None:
        report = spin8_recurrent_diagnostics()
        self.assertTrue(report["passed"], report)
        self.assertGreater(report["metrics"]["identity_controller_gradient_norm"], 0.0)
        self.assertEqual(report["metrics"]["cache_scalars"], 48)

    def test_spin8_cache_shape_is_independent_of_context_length(self) -> None:
        model = Spin8TrialitySSM(
            4, channels=2, representations=TRIALITY_REPRESENTATIONS
        )
        short = torch.randn(3, 2, 4)
        long = torch.randn(3, 19, 4)
        _, short_state = model(short)
        _, long_state = model(long)
        self.assertEqual(short_state.shape, long_state.shape)
        self.assertEqual(short_state.shape, (3, 2, 3, 8))
        self.assertEqual(model.cache_scalars, 48)

    def make_model(self, family: str) -> RecurrenceSequenceModel:
        torch.manual_seed(7)
        return RecurrenceSequenceModel(
            vocab_size=8,
            output_size=8,
            family=family,
            channels=2,
            num_layers=2,
        ).eval()

    def test_all_families_have_identical_parameters_and_initial_function(self) -> None:
        tokens = torch.arange(18).reshape(2, 9) % 8
        models = [self.make_model(family) for family in FAMILY_NAMES]
        counts = {sum(parameter.numel() for parameter in model.parameters()) for model in models}
        self.assertEqual(len(counts), 1)
        reference = models[0](tokens)
        for model in models[1:]:
            torch.testing.assert_close(model(tokens), reference, rtol=1e-6, atol=1e-6)

    def test_every_family_exposes_exact_streaming_state(self) -> None:
        tokens = torch.arange(20).reshape(2, 10) % 8
        for family in FAMILY_NAMES:
            with self.subTest(family=family):
                model = self.make_model(family)
                errors = streaming_equivalence(model, tokens)
                self.assertLess(errors["chunked_logit_max_abs_error"], 1e-5)
                self.assertLess(errors["streaming_logit_max_abs_error"], 1e-5)
                self.assertLess(errors["streaming_state_max_abs_error"], 1e-5)
                states = model.initial_states(batch_size=2)
                self.assertEqual(len(states), model.num_layers)
                self.assertEqual(states[0].shape, (2, model.channels, 8))

    def test_static_rotor_is_token_independent_while_selective_rotor_is_not(self) -> None:
        torch.manual_seed(8)
        selective = GARotorRecurrence(channels=2, selective_rotation=True)
        static = GARotorRecurrence(channels=2, selective_rotation=False)
        with torch.no_grad():
            selective.control_projection.weight.normal_(std=0.1)
            selective.control_projection.bias.normal_(std=0.1)
        static.load_state_dict(selective.state_dict())
        inputs = torch.randn(2, 5, 2, 8)
        _, selective_rotors, _ = selective.transition_parameters(inputs)
        _, static_rotors, _ = static.transition_parameters(inputs)
        selective_change = (
            selective_rotors[:, 1:] - selective_rotors[:, :-1]
        ).detach().abs().max()
        self.assertGreater(float(selective_change), 1e-4)
        torch.testing.assert_close(
            static_rotors[:, 1:],
            static_rotors[:, :-1],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_grade_decay_is_shared_within_each_grade_but_selective_across_grades(self) -> None:
        torch.manual_seed(9)
        layer = GARotorRecurrence(
            channels=2, selective_rotation=True, grade_decay=True
        )
        with torch.no_grad():
            layer.control_projection.weight.normal_(std=0.1)
        decay, _, _ = layer.transition_parameters(torch.randn(2, 5, 2, 8))
        torch.testing.assert_close(decay[..., 1], decay[..., 2])
        torch.testing.assert_close(decay[..., 2], decay[..., 3])
        torch.testing.assert_close(decay[..., 4], decay[..., 5])
        torch.testing.assert_close(decay[..., 5], decay[..., 6])
        grade_means = torch.stack(
            (decay[..., 0], decay[..., 1], decay[..., 4], decay[..., 7]), dim=-1
        )
        self.assertGreater(float(grade_means.detach().var(dim=-1).mean()), 1e-7)

    def test_identity_rotations_receive_gradients(self) -> None:
        for family in (
            "complex_unitary",
            "quaternion_even",
            "ga_rotor_selective",
            "ga_rotor_grade_decay",
            "hybrid_complex_ga",
        ):
            with self.subTest(family=family):
                model = self.make_model(family).train()
                tokens = torch.arange(16).reshape(2, 8) % 8
                model(tokens).square().mean().backward()
                gradient = model.blocks[0].recurrence.control_projection.weight.grad
                self.assertIsNotNone(gradient)
                grouped = gradient.reshape(model.channels, 8, -1)
                if family == "complex_unitary":
                    rotation_gradient = grouped[:, 1::2]
                elif family == "quaternion_even":
                    rotation_gradient = grouped.reshape(
                        model.channels, 2, 4, -1
                    )[:, :, 1:4]
                elif family == "ga_rotor_selective":
                    rotation_gradient = grouped[:, 1:4]
                elif family == "ga_rotor_grade_decay":
                    rotation_gradient = grouped[:, 4:7]
                else:
                    rotation_gradient = torch.cat(
                        (
                            grouped[:1, 1::2].flatten(),
                            grouped[1:, 1:4].flatten(),
                        )
                    )
                self.assertGreater(float(rotation_gradient.abs().sum()), 0.0)

    def test_cuda_forward_backward_when_available(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        tokens = torch.arange(32, device="cuda").reshape(4, 8) % 8
        for family in FAMILY_NAMES:
            with self.subTest(family=family):
                model = self.make_model(family).cuda().train()
                loss = model(tokens).square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
