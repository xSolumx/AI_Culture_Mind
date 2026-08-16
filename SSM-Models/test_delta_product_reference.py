"""Equation, scan, cache, and gradient tests for the DeltaProduct reference."""

from __future__ import annotations

import unittest

import torch
from delta_product_reference import (
    DeltaProductReferenceLayer,
    DeltaProductReferenceModel,
    compose_delta_updates_per_token,
    delta_product_scan,
    delta_update_affine,
)


class DeltaProductEquationTests(unittest.TestCase):
    def test_one_update_matches_official_delta_equation(self) -> None:
        torch.manual_seed(130)
        key = torch.nn.functional.normalize(
            torch.randn(2, 3, 5, dtype=torch.float64), dim=-1
        )
        value = torch.randn(2, 3, 5, dtype=torch.float64)
        beta = 2 * torch.rand(2, 3, dtype=torch.float64)
        state = torch.randn(2, 3, 5, 5, dtype=torch.float64)
        transition_a, transition_b = delta_update_affine(key, value, beta)
        affine = transition_a @ state + transition_b
        residual = value - torch.einsum("...d,...dv->...v", key, state)
        official = state + key.unsqueeze(-1) * (
            beta.unsqueeze(-1) * residual
        ).unsqueeze(-2)
        torch.testing.assert_close(affine, official, rtol=1e-13, atol=1e-13)

    def test_composed_updates_match_expanded_recurrence(self) -> None:
        torch.manual_seed(131)
        keys = torch.nn.functional.normalize(
            torch.randn(2, 7, 4, 3, 5, dtype=torch.float64), dim=-1
        )
        values = torch.randn_like(keys)
        betas = 2 * torch.rand(2, 7, 4, 3, dtype=torch.float64)
        transition_a, transition_b = compose_delta_updates_per_token(
            keys, values, betas
        )
        initial = torch.randn(2, 3, 5, 5, dtype=torch.float64)
        expected = []
        state = initial
        for position in range(keys.shape[1]):
            for update in range(keys.shape[2]):
                key = keys[:, position, update]
                value = values[:, position, update]
                beta = betas[:, position, update]
                residual = value - torch.einsum("...d,...dv->...v", key, state)
                state = state + key.unsqueeze(-1) * (
                    beta.unsqueeze(-1) * residual
                ).unsqueeze(-2)
            expected.append(state)
        expected = torch.stack(expected, dim=1)
        actual, actual_final = delta_product_scan(
            transition_a, transition_b, initial, mode="recurrent"
        )
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            actual_final, expected[:, -1], rtol=1e-12, atol=1e-12
        )


class DeltaProductModelTests(unittest.TestCase):
    @staticmethod
    def _value_and_gradients(
        layer: DeltaProductReferenceLayer, values: torch.Tensor, mode: str
    ):
        inputs = values.detach().clone().requires_grad_(True)
        sequence, final_state = layer(inputs, scan_mode=mode)
        weights = torch.linspace(
            0.1, 1.0, sequence.numel(), dtype=sequence.dtype
        ).reshape_as(sequence)
        objective = (sequence * weights).sum() + final_state.square().mean()
        gradients = torch.autograd.grad(objective, (inputs, *layer.parameters()))
        return sequence, final_state, gradients

    def test_parallel_recurrent_forward_and_gradient_parity(self) -> None:
        torch.manual_seed(132)
        layer = DeltaProductReferenceLayer(
            hidden_size=12, num_heads=3, num_householder=4
        ).double()
        inputs = torch.randn(2, 17, 12, dtype=torch.float64)
        parallel = self._value_and_gradients(layer, inputs, "parallel")
        recurrent = self._value_and_gradients(layer, inputs, "recurrent")
        torch.testing.assert_close(parallel[0], recurrent[0], rtol=2e-10, atol=2e-11)
        torch.testing.assert_close(parallel[1], recurrent[1], rtol=2e-10, atol=2e-11)
        for actual, expected in zip(parallel[2], recurrent[2]):
            torch.testing.assert_close(actual, expected, rtol=5e-9, atol=5e-10)

    def test_model_parameter_state_and_streaming_contracts(self) -> None:
        torch.manual_seed(133)
        model = DeltaProductReferenceModel(
            input_vocab_size=120,
            output_size=120,
            hidden_size=32,
            num_heads=4,
            num_householder=4,
            intermediate_size=112,
        ).double()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 29288)
        self.assertEqual(model.recurrent_state_scalars, 256)
        tokens = torch.randint(0, 4, (2, 21))
        full, full_state = model(
            tokens, return_recurrent_state=True, scan_mode="parallel"
        )
        first, cache = model(
            tokens[:, :9], return_recurrent_state=True, scan_mode="parallel"
        )
        second, cache = model(
            tokens[:, 9:],
            cache,
            return_recurrent_state=True,
            scan_mode="parallel",
        )
        torch.testing.assert_close(
            full, torch.cat((first, second), dim=1), rtol=2e-10, atol=2e-11
        )
        torch.testing.assert_close(full_state, cache, rtol=2e-10, atol=2e-11)

        state = model.initial_state(2, dtype=torch.float64)
        pieces = []
        for position in range(tokens.shape[1]):
            logits, state = model.step(tokens[:, position], state)
            pieces.append(logits)
        torch.testing.assert_close(
            full, torch.stack(pieces, dim=1), rtol=2e-10, atol=2e-11
        )

    def test_padding_is_an_identity_transition(self) -> None:
        torch.manual_seed(134)
        layer = DeltaProductReferenceLayer(
            hidden_size=8, num_heads=2, num_householder=4
        ).double()
        inputs = torch.randn(2, 9, 8, dtype=torch.float64)
        valid = torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            dtype=torch.bool,
        )
        transition_a, transition_b, _ = layer.transitions(inputs, valid)
        identity = torch.eye(4, dtype=torch.float64).expand_as(transition_a[0, 4:])
        torch.testing.assert_close(transition_a[0, 4:], identity)
        torch.testing.assert_close(
            transition_b[0, 4:], torch.zeros_like(transition_b[0, 4:])
        )
        _, final_state = delta_product_scan(transition_a, transition_b, mode="parallel")
        states, _ = delta_product_scan(transition_a, transition_b, mode="recurrent")
        torch.testing.assert_close(final_state[0], states[0, 3], rtol=1e-12, atol=1e-12)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_parallel_backward(self) -> None:
        torch.manual_seed(135)
        model = DeltaProductReferenceModel(input_vocab_size=120, output_size=120).cuda()
        tokens = torch.randint(0, 4, (4, 129), device="cuda")
        logits = model(tokens, scan_mode="parallel")
        self.assertTrue(bool(torch.isfinite(logits).all()))
        logits.square().mean().backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))


if __name__ == "__main__":
    unittest.main()
