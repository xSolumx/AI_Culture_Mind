"""WSL/Linux CUDA contracts for the optional fused octonion recurrence."""

from __future__ import annotations

import unittest

import torch
from pure_rotor_ssm.octonion_operator_scan import octonion_state_scan, unit_octonion
from pure_rotor_ssm.octonion_operator_triton import (
    fused_octonion_state_scan,
    triton_is_available,
)


@unittest.skipUnless(triton_is_available(), "Triton CUDA is unavailable")
class FusedOctonionRecurrenceTests(unittest.TestCase):
    def test_forward_matches_parenthesized_recurrence(self) -> None:
        torch.manual_seed(20260820)
        tokens = torch.randn(3, 37, 4, 8, device="cuda")
        initial = unit_octonion(torch.randn(3, 4, 8, device="cuda"))
        expected, expected_final = octonion_state_scan(
            tokens, initial, mode="recurrent"
        )
        actual, actual_final = fused_octonion_state_scan(tokens, initial)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(actual_final, expected_final, rtol=2e-5, atol=2e-6)

    def test_backward_matches_parenthesized_recurrence(self) -> None:
        torch.manual_seed(20260821)
        base_tokens = torch.randn(2, 19, 3, 8, device="cuda")
        base_initial = unit_octonion(torch.randn(2, 3, 8, device="cuda"))
        weights = torch.randn_like(base_tokens)

        def gradients(fused: bool) -> tuple[torch.Tensor, torch.Tensor]:
            tokens = base_tokens.clone().requires_grad_(True)
            initial = base_initial.clone().requires_grad_(True)
            if fused:
                states, _ = fused_octonion_state_scan(tokens, initial)
            else:
                states, _ = octonion_state_scan(tokens, initial, mode="recurrent")
            return torch.autograd.grad((states * weights).sum(), (tokens, initial))

        expected = gradients(False)
        actual = gradients(True)
        torch.testing.assert_close(actual[0], expected[0], rtol=5e-4, atol=2e-4)
        torch.testing.assert_close(actual[1], expected[1], rtol=5e-4, atol=2e-4)

    def test_long_forward_is_finite_and_keeps_unit_norm(self) -> None:
        torch.manual_seed(20260822)
        tokens = torch.randn(8, 4096, 4, 8, device="cuda")
        states, final = fused_octonion_state_scan(tokens)
        self.assertTrue(torch.isfinite(states).all())
        self.assertTrue(torch.isfinite(final).all())
        self.assertLess(
            float((torch.linalg.vector_norm(states, dim=-1) - 1).abs().max()),
            2e-4,
        )


if __name__ == "__main__":
    unittest.main()
