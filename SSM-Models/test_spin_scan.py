"""Contract tests for the experimental sign-sensitive Spin composition scan."""

from __future__ import annotations

import unittest

import torch
from pure_rotor_ssm.spin_scan import (
    SpinCompositionClassifier,
    SpinTokenComposition,
    quaternion_prefix_scan,
    quaternion_product,
    quaternion_to_rotor,
    rotor_to_quaternion,
    unit_quaternion,
)
from pure_rotor_ssm.torch_backend import (
    rotor_product,
    specialized_rotor_sandwich,
)


class SpinCompositionAlgebraTests(unittest.TestCase):
    def test_compact_product_matches_clifford_rotor_product(self) -> None:
        torch.manual_seed(120)
        left = unit_quaternion(torch.randn(3, 7, 4, dtype=torch.float64))
        right = unit_quaternion(torch.randn(3, 7, 4, dtype=torch.float64))
        expected = rotor_product(quaternion_to_rotor(left), quaternion_to_rotor(right))
        actual = quaternion_to_rotor(quaternion_product(left, right))
        torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(
            rotor_to_quaternion(actual),
            quaternion_product(left, right),
            rtol=1e-13,
            atol=1e-13,
        )

    def test_center_sign_is_retained_but_conjugation_cannot_see_it(self) -> None:
        quarter_turn = torch.tensor(
            [[[[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]],
            dtype=torch.float64,
        ).expand(1, 2, 2, 4)
        states, final_state = quaternion_prefix_scan(quarter_turn, mode="parallel")
        expected_center = torch.tensor(
            [[[-1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(final_state, expected_center)
        self.assertFalse(torch.equal(states[:, 0], states[:, 1]))

        rotor = quaternion_to_rotor(states[:, :1])
        negated = -rotor
        probe = torch.randn(1, 1, 2, 8, dtype=torch.float64)
        torch.testing.assert_close(
            specialized_rotor_sandwich(rotor, probe),
            specialized_rotor_sandwich(negated, probe),
            rtol=1e-13,
            atol=1e-13,
        )


class SpinCompositionScanTests(unittest.TestCase):
    @staticmethod
    def _scan_value_and_gradients(raw: torch.Tensor, mode: str):
        current = raw.detach().clone().requires_grad_(True)
        states, final_state = quaternion_prefix_scan(current, mode=mode)
        weights = torch.linspace(
            0.1, 1.0, states.numel(), dtype=states.dtype
        ).reshape_as(states)
        objective = (states * weights).sum() + final_state.square().sum()
        (gradient,) = torch.autograd.grad(objective, (current,))
        return states, final_state, gradient

    def test_parallel_recurrent_forward_and_gradient_parity(self) -> None:
        torch.manual_seed(121)
        raw = torch.randn(2, 31, 5, 4, dtype=torch.float64)
        parallel = self._scan_value_and_gradients(raw, "parallel")
        recurrent = self._scan_value_and_gradients(raw, "recurrent")
        for actual, expected in zip(parallel, recurrent):
            torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)

    def test_padding_is_identity_and_cache_matches_full_sequence(self) -> None:
        torch.manual_seed(122)
        layer = SpinTokenComposition(input_vocab_size=7, lanes=3).double()
        tokens = torch.randint(0, 7, (2, 19))
        full, full_final = layer(tokens, scan_mode="parallel")
        first, cache = layer(tokens[:, :8], scan_mode="parallel")
        second, cache = layer(tokens[:, 8:], cache, scan_mode="parallel")
        torch.testing.assert_close(
            full, torch.cat((first, second), dim=1), rtol=1e-12, atol=1e-12
        )
        torch.testing.assert_close(full_final, cache, rtol=1e-12, atol=1e-12)

        valid = torch.tensor(
            [[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]],
            dtype=torch.bool,
        )
        padded, padded_final = layer(
            tokens[:, :8], attention_mask=valid, scan_mode="parallel"
        )
        torch.testing.assert_close(
            padded[0, 5:],
            padded[0, 4].expand_as(padded[0, 5:]),
            rtol=1e-13,
            atol=1e-13,
        )
        torch.testing.assert_close(padded_final[0], padded[0, 4])

    def test_long_scan_remains_on_unit_sphere(self) -> None:
        torch.manual_seed(123)
        inputs = torch.randn(2, 4096, 4, 4, dtype=torch.float32)
        states, final_state = quaternion_prefix_scan(inputs, mode="parallel")
        torch.testing.assert_close(
            torch.linalg.vector_norm(states, dim=-1),
            torch.ones_like(states[..., 0]),
            rtol=2e-6,
            atol=2e-6,
        )
        torch.testing.assert_close(final_state, states[:, -1], rtol=0, atol=0)

    def test_classifier_streaming_and_recurrent_step(self) -> None:
        torch.manual_seed(124)
        model = SpinCompositionClassifier(
            input_vocab_size=9,
            output_size=13,
            lanes=4,
            decoder_hidden=17,
        ).double()
        tokens = torch.randint(0, 9, (2, 23))
        full, full_state = model(
            tokens,
            return_recurrent_state=True,
            scan_mode="parallel",
        )
        first, cache = model(
            tokens[:, :10],
            return_recurrent_state=True,
            scan_mode="parallel",
        )
        second, cache = model(
            tokens[:, 10:],
            cache,
            return_recurrent_state=True,
            scan_mode="parallel",
        )
        torch.testing.assert_close(
            full, torch.cat((first, second), dim=1), rtol=1e-12, atol=1e-12
        )
        torch.testing.assert_close(full_state, cache, rtol=1e-12, atol=1e-12)

        state = model.composition.initial_state(2, dtype=torch.float64)
        step_logits = []
        for position in range(tokens.shape[1]):
            logits, state = model.step(tokens[:, position], state)
            step_logits.append(logits)
        torch.testing.assert_close(
            full,
            torch.stack(step_logits, dim=1),
            rtol=1e-12,
            atol=1e-12,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_parallel_forward_backward(self) -> None:
        torch.manual_seed(125)
        model = SpinCompositionClassifier(
            input_vocab_size=11,
            output_size=17,
            lanes=8,
            decoder_hidden=31,
        ).cuda()
        tokens = torch.randint(0, 11, (4, 257), device="cuda")
        parallel = model(tokens, scan_mode="parallel")
        recurrent = model(tokens, scan_mode="recurrent")
        torch.testing.assert_close(parallel, recurrent, rtol=2e-4, atol=2e-5)
        parallel.square().mean().backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))


if __name__ == "__main__":
    unittest.main()
