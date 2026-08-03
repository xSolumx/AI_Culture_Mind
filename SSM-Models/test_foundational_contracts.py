"""Regression tests for the foundational metric and isotypic contracts."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch import nn

from GALib import (
    Spin3IsotypicLinear as JaxSpin3IsotypicLinear,
    pack_spin3_isotypic,
    rotor_from_bivector as jax_rotor_from_bivector,
    rotor_sandwich as jax_rotor_sandwich,
    unpack_spin3_isotypic,
)
from compare_recurrences import GROUPS, evaluate as evaluate_recurrence, group_prefix_products
from mechanistic_group_actions import evaluate as evaluate_group_action
from rotor_ssm_torch import GA_DIM, GradeLinear
from schur_scan import (
    SchurAffineTransition,
    Spin3IsotypicLinear,
    apply_schur_affine,
    associative_schur_scan,
    pack_cl3_isotypic,
    unpack_cl3_isotypic,
)
from spin8_blind_shared_action import (
    action_design_audit,
    joint_shared_retraction,
    observed_action,
)
from spin8_learned_address import (
    evaluate_mixed_sequences,
    log_sinkhorn,
    route_statistics,
    scan_parity as learned_address_scan_parity,
)
from spin8_triality import spin8_actions, torch_triality_generators
from spin8_triality_identifiability import invariant_space_audit
from spin8_triality_lift import (
    diagnostics as triality_lift_diagnostics,
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
        self.assertEqual(
            float(grade(vector_only)[..., 4:7].abs().max().detach()), 0.0
        )
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
            active_drive=0.01 * torch.randn(batch, length, multiplicity, 3, dtype=dtype),
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

    def test_triality_is_the_unique_infinitesimal_equivariant_bilinear_map(self) -> None:
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
        self.assertLess(
            report["capacity"]["maximum_exact_relative_error"], 1e-10
        )
        self.assertTrue(
            report["capacity"]["tight_frame_beats_random_all_overcomplete_cells"]
        )
        self.assertLess(
            report["dynamic_slot"]["final_retrieval_max_error"], 1e-10
        )

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

    def test_joint_address_family_is_globally_not_independently_normalized(self) -> None:
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
            evaluate_recurrence(self.DummyRecurrence(), batches, torch.device("cpu"))[0],
            expected,
            places=6,
        )
        self.assertAlmostEqual(
            evaluate_group_action(self.DummyGroupAction(), batches, torch.device("cpu"))[0],
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


if __name__ == "__main__":
    unittest.main()
