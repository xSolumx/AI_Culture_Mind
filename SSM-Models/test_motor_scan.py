"""Algebra, geometry, scan, cache, and gradient tests for rigid Spin motors."""

from __future__ import annotations

import unittest

import torch
from pure_rotor_ssm.motor_scan import (
    DirectMotorPoseTracker,
    DirectProductPoseTracker,
    MotorCompositionClassifier,
    MotorTokenComposition,
    dual_quaternion_product,
    motor_from_rotation_translation,
    motor_inverse,
    motor_prefix_scan,
    motor_to_matrix,
    motor_transform_points,
    motor_translation,
    normalize_motor,
)
from pure_rotor_ssm.spin_scan import quaternion_prefix_scan, unit_quaternion


class MotorAlgebraTests(unittest.TestCase):
    def test_product_matches_homogeneous_matrix_composition(self) -> None:
        torch.manual_seed(140)
        left = motor_from_rotation_translation(
            torch.randn(3, 5, 4, dtype=torch.float64),
            torch.randn(3, 5, 3, dtype=torch.float64),
        )
        right = motor_from_rotation_translation(
            torch.randn(3, 5, 4, dtype=torch.float64),
            torch.randn(3, 5, 3, dtype=torch.float64),
        )
        product = dual_quaternion_product(left, right)
        torch.testing.assert_close(
            motor_to_matrix(product),
            motor_to_matrix(left) @ motor_to_matrix(right),
            rtol=2e-13,
            atol=2e-13,
        )

    def test_associativity_inverse_and_point_action(self) -> None:
        torch.manual_seed(141)
        motors = [
            motor_from_rotation_translation(
                torch.randn(4, 4, dtype=torch.float64),
                torch.randn(4, 3, dtype=torch.float64),
            )
            for _ in range(3)
        ]
        left = dual_quaternion_product(
            dual_quaternion_product(motors[0], motors[1]), motors[2]
        )
        right = dual_quaternion_product(
            motors[0], dual_quaternion_product(motors[1], motors[2])
        )
        torch.testing.assert_close(left, right, rtol=2e-13, atol=2e-13)
        identity = dual_quaternion_product(motors[0], motor_inverse(motors[0]))
        expected = torch.zeros_like(identity)
        expected[..., 0] = 1
        torch.testing.assert_close(identity, expected, rtol=2e-13, atol=2e-13)

        points = torch.randn(4, 3, dtype=torch.float64)
        actual = motor_transform_points(motors[0], points)
        homogeneous = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
        expected_points = torch.einsum(
            "...ij,...j->...i", motor_to_matrix(motors[0]), homogeneous
        )[..., :3]
        torch.testing.assert_close(actual, expected_points, rtol=2e-13, atol=2e-13)

    def test_study_projection_does_not_euclidean_normalize_translation(self) -> None:
        rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        translation = torch.tensor([[20.0, -30.0, 40.0]], dtype=torch.float64)
        motor = motor_from_rotation_translation(rotation, translation)
        projected = normalize_motor(7.0 * motor)
        torch.testing.assert_close(
            motor_translation(projected), translation, rtol=1e-13, atol=1e-13
        )
        real, dual = projected.split(4, dim=-1)
        torch.testing.assert_close(
            torch.linalg.vector_norm(real, dim=-1),
            torch.ones(1, dtype=torch.float64),
        )
        torch.testing.assert_close(
            (real * dual).sum(dim=-1), torch.zeros(1, dtype=torch.float64)
        )

    def test_center_sign_is_retained_but_rigid_action_is_blind(self) -> None:
        motor = motor_from_rotation_translation(
            torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float64),
            torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64),
        )
        self.assertFalse(torch.equal(motor, -motor))
        torch.testing.assert_close(motor_to_matrix(motor), motor_to_matrix(-motor))
        point = torch.tensor([[4.0, -2.0, 1.0]], dtype=torch.float64)
        torch.testing.assert_close(
            motor_transform_points(motor, point),
            motor_transform_points(-motor, point),
        )

    def test_zero_translation_subgroup_matches_spin_scan(self) -> None:
        torch.manual_seed(142)
        rotations = torch.randn(2, 33, 4, 4, dtype=torch.float64)
        translations = torch.zeros(2, 33, 4, 3, dtype=torch.float64)
        motors = motor_from_rotation_translation(rotations, translations)
        motor_states, _ = motor_prefix_scan(motors, mode="parallel")
        spin_states, _ = quaternion_prefix_scan(rotations, mode="parallel")
        torch.testing.assert_close(
            motor_states[..., :4], spin_states, rtol=2e-12, atol=2e-12
        )
        torch.testing.assert_close(
            motor_states[..., 4:], torch.zeros_like(motor_states[..., 4:])
        )


class MotorScanTests(unittest.TestCase):
    @staticmethod
    def _scan_value_and_gradient(raw: torch.Tensor, mode: str):
        current = raw.detach().clone().requires_grad_(True)
        states, final_state = motor_prefix_scan(current, mode=mode)
        weights = torch.linspace(
            0.1, 1.0, states.numel(), dtype=states.dtype
        ).reshape_as(states)
        objective = (states * weights).sum() + final_state.square().mean()
        (gradient,) = torch.autograd.grad(objective, (current,))
        return states, final_state, gradient

    def test_direct_pose_trackers_scan_cache_and_step_contracts(self) -> None:
        torch.manual_seed(73)
        token_ids = torch.randint(0, 7, (3, 11))
        for model in (DirectMotorPoseTracker(7), DirectProductPoseTracker(7)):
            parallel = model(token_ids, scan_mode="parallel")
            recurrent = model(token_ids, scan_mode="recurrent")
            torch.testing.assert_close(parallel, recurrent, rtol=2e-5, atol=2e-5)

            first, cache = model(
                token_ids[:, :5],
                return_recurrent_state=True,
                scan_mode="recurrent",
            )
            second = model(token_ids[:, 5:], cache, scan_mode="recurrent")
            torch.testing.assert_close(
                torch.cat((first, second), dim=1),
                recurrent,
                rtol=2e-5,
                atol=2e-5,
            )

            if hasattr(model, "initial_state"):
                state = model.initial_state(len(token_ids))
            else:
                state = model.composition.initial_state(len(token_ids))[:, 0]
            step_outputs = []
            for position in range(token_ids.shape[1]):
                output, state = model.step(token_ids[:, position], state)
                step_outputs.append(output)
            torch.testing.assert_close(
                torch.stack(step_outputs, dim=1),
                recurrent,
                rtol=2e-5,
                atol=2e-5,
            )

    def test_parallel_recurrent_forward_and_gradient_parity(self) -> None:
        torch.manual_seed(143)
        rotations = unit_quaternion(torch.randn(2, 31, 3, 4, dtype=torch.float64))
        translations = 0.1 * torch.randn(2, 31, 3, 3, dtype=torch.float64)
        raw = motor_from_rotation_translation(rotations, translations)
        parallel = self._scan_value_and_gradient(raw, "parallel")
        recurrent = self._scan_value_and_gradient(raw, "recurrent")
        for actual, expected in zip(parallel, recurrent):
            torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)

    def test_padding_cache_classifier_and_step_contracts(self) -> None:
        torch.manual_seed(144)
        layer = MotorTokenComposition(input_vocab_size=7, lanes=3).double()
        tokens = torch.randint(0, 7, (2, 19))
        full, full_final = layer(tokens, scan_mode="parallel")
        first, cache = layer(tokens[:, :8], scan_mode="parallel")
        second, cache = layer(tokens[:, 8:], cache, scan_mode="parallel")
        torch.testing.assert_close(
            full, torch.cat((first, second), dim=1), rtol=2e-11, atol=2e-11
        )
        torch.testing.assert_close(full_final, cache, rtol=2e-11, atol=2e-11)

        valid = torch.tensor(
            [[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]],
            dtype=torch.bool,
        )
        padded, padded_final = layer(
            tokens[:, :8], attention_mask=valid, scan_mode="parallel"
        )
        torch.testing.assert_close(padded[0, 5:], padded[0, 4].expand_as(padded[0, 5:]))
        torch.testing.assert_close(padded_final[0], padded[0, 4])

        model = MotorCompositionClassifier(
            input_vocab_size=7, output_size=11, lanes=3, decoder_hidden=19
        ).double()
        full_logits, full_state = model(
            tokens, return_recurrent_state=True, scan_mode="parallel"
        )
        state = model.composition.initial_state(2, dtype=torch.float64)
        pieces = []
        for position in range(tokens.shape[1]):
            logits, state = model.step(tokens[:, position], state)
            pieces.append(logits)
        torch.testing.assert_close(
            full_logits, torch.stack(pieces, dim=1), rtol=2e-11, atol=2e-11
        )
        torch.testing.assert_close(full_state, state, rtol=2e-11, atol=2e-11)

    def test_long_scan_preserves_motor_constraints(self) -> None:
        torch.manual_seed(145)
        rotations = torch.randn(2, 4096, 2, 4)
        translations = 0.01 * torch.randn(2, 4096, 2, 3)
        motors = motor_from_rotation_translation(rotations, translations)
        states, final_state = motor_prefix_scan(motors, mode="parallel")
        real, dual = states.split(4, dim=-1)
        torch.testing.assert_close(
            torch.linalg.vector_norm(real, dim=-1),
            torch.ones_like(real[..., 0]),
            rtol=3e-6,
            atol=3e-6,
        )
        torch.testing.assert_close(
            (real * dual).sum(dim=-1),
            torch.zeros_like(real[..., 0]),
            rtol=0,
            atol=3e-5,
        )
        self.assertTrue(bool(torch.isfinite(states).all()))
        torch.testing.assert_close(final_state, states[:, -1], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_parallel_backward(self) -> None:
        torch.manual_seed(146)
        model = MotorCompositionClassifier(
            input_vocab_size=11, output_size=17, lanes=4, decoder_hidden=31
        ).cuda()
        tokens = torch.randint(0, 11, (4, 257), device="cuda")
        parallel = model(tokens, scan_mode="parallel")
        recurrent = model(tokens, scan_mode="recurrent")
        torch.testing.assert_close(parallel, recurrent, rtol=3e-4, atol=3e-5)
        parallel.square().mean().backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))


if __name__ == "__main__":
    unittest.main()
