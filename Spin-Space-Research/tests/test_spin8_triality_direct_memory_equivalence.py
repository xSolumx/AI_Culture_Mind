from __future__ import annotations

import unittest

import torch

from spin8_triality_direct_memory_equivalence import (
    binding_operator,
    diagnostics,
)
from spin8_triality_lift import triality_bind, triality_tensor
from spin8_triality_memory import (
    SlotTransition,
    apply_slot,
    associative_slot_scan,
    pack_slot_homogeneous_matrices,
    packed_homogeneous_slot_scan,
    packed_homogeneous_slot_states,
    work_efficient_slot_scan,
)


class Spin8TrialityDirectMemoryEquivalenceTests(unittest.TestCase):
    @staticmethod
    def _random_transition(
        length: int, *, requires_grad: bool = False
    ) -> SlotTransition:
        generator = torch.Generator().manual_seed(100 + length)
        retention = (
            0.85
            + 0.14 * torch.rand(2, length, 3, generator=generator, dtype=torch.float64)
        ).requires_grad_(requires_grad)
        action = (
            torch.eye(4, dtype=torch.float64).reshape(1, 1, 4, 4)
            + 0.03
            * torch.randn(2, length, 4, 4, generator=generator, dtype=torch.float64)
        ).requires_grad_(requires_grad)
        drive = (
            0.05
            * torch.randn(2, length, 3, 4, generator=generator, dtype=torch.float64)
        ).requires_grad_(requires_grad)
        return SlotTransition(retention, action, drive)

    def test_binding_operator_matches_triality_bind(self) -> None:
        generator = torch.Generator().manual_seed(3)
        keys = torch.nn.functional.normalize(
            torch.randn(7, 8, generator=generator, dtype=torch.float64), dim=-1
        )
        values = torch.randn(7, 8, generator=generator, dtype=torch.float64)
        rho = triality_tensor(dtype=torch.float64)
        operator_value = torch.einsum("bij,bj->bi", binding_operator(keys, rho), values)
        torch.testing.assert_close(
            operator_value,
            triality_bind(keys, values, rho),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_dynamic_direct_and_triality_memories_are_gauge_equivalent(self) -> None:
        report = diagnostics()
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["streaming_state_scalars_each"], 64)
        self.assertFalse(
            report["claims"][
                "triality_specific_capacity_or_retrieval_advantage_established"
            ]
        )

    def test_work_efficient_slot_scan_matches_hillis_at_irregular_lengths(self) -> None:
        for length in (1, 2, 3, 5, 8, 17, 31, 64, 127):
            with self.subTest(length=length):
                transition = self._random_transition(length)
                expected = associative_slot_scan(transition)
                actual = work_efficient_slot_scan(transition)
                for left, right in zip(
                    (actual.retention, actual.action, actual.drive),
                    (expected.retention, expected.action, expected.drive),
                ):
                    torch.testing.assert_close(left, right, rtol=2e-11, atol=2e-11)

    def test_work_efficient_slot_scan_preserves_gradients(self) -> None:
        initial = torch.randn(2, 3, 4, dtype=torch.float64)
        weights = torch.randn(2, 7, 3, 4, dtype=torch.float64)
        losses = []
        gradients = []
        for scan in (work_efficient_slot_scan, associative_slot_scan):
            transition = self._random_transition(7, requires_grad=True)
            states = apply_slot(scan(transition), initial[:, None])
            loss = (states * weights).sum()
            losses.append(loss)
            gradients.append(
                torch.autograd.grad(
                    loss,
                    (transition.retention, transition.action, transition.drive),
                )
            )
        torch.testing.assert_close(losses[0], losses[1], rtol=1e-11, atol=1e-11)
        for left, right in zip(gradients[0], gradients[1]):
            torch.testing.assert_close(left, right, rtol=2e-10, atol=2e-10)

    def test_local_homogeneous_slot_backends_match_structured_scan(self) -> None:
        initial = torch.randn(2, 3, 4, dtype=torch.float64)
        for length in (1, 3, 8, 17, 31):
            with self.subTest(length=length):
                transition = self._random_transition(length)
                expected = apply_slot(
                    work_efficient_slot_scan(transition), initial[:, None]
                )
                packed = pack_slot_homogeneous_matrices(transition)
                for backend in ("hillis_steele", "work_efficient"):
                    actual = packed_homogeneous_slot_states(
                        packed, initial, backend=backend
                    )
                    end_to_end = packed_homogeneous_slot_scan(
                        transition, initial, backend=backend
                    )
                    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)
                    torch.testing.assert_close(
                        end_to_end, expected, rtol=2e-11, atol=2e-11
                    )

    def test_local_homogeneous_slot_scan_preserves_gradients(self) -> None:
        initial = torch.randn(2, 3, 4, dtype=torch.float64)
        weights = torch.randn(2, 7, 3, 4, dtype=torch.float64)
        losses = []
        gradients = []
        for kind in ("packed", "structured"):
            transition = self._random_transition(7, requires_grad=True)
            if kind == "packed":
                states = packed_homogeneous_slot_scan(
                    transition, initial, backend="work_efficient"
                )
            else:
                states = apply_slot(
                    work_efficient_slot_scan(transition), initial[:, None]
                )
            loss = (states * weights).sum()
            losses.append(loss)
            gradients.append(
                torch.autograd.grad(
                    loss,
                    (transition.retention, transition.action, transition.drive),
                )
            )
        torch.testing.assert_close(losses[0], losses[1], rtol=1e-11, atol=1e-11)
        for left, right in zip(gradients[0], gradients[1]):
            torch.testing.assert_close(left, right, rtol=3e-10, atol=3e-10)


if __name__ == "__main__":
    unittest.main()
