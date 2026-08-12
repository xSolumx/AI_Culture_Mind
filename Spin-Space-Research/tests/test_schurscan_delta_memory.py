from __future__ import annotations

import unittest

import torch

from schurscan_delta_memory import (
    DeltaTransition,
    apply_delta,
    chunkwise_delta_scan,
    compose_delta,
    delta_read,
    delta_write_transitions,
    recurrent_delta_states,
    scanned_delta_states,
    value_transport_transitions,
    work_efficient_delta_scan,
)


class SchurScanDeltaMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260810)
        self.dtype = torch.float64

    def random_transition(
        self,
        length: int,
        *,
        batch: int = 2,
        key_dimension: int = 4,
        value_dimension: int = 3,
        requires_grad: bool = False,
    ) -> DeltaTransition:
        key_identity = torch.eye(key_dimension, dtype=self.dtype).reshape(
            1, 1, key_dimension, key_dimension
        )
        value_identity = torch.eye(value_dimension, dtype=self.dtype).reshape(
            1, 1, value_dimension, value_dimension
        )
        key_action = (
            key_identity
            + 0.03
            * torch.randn(batch, length, key_dimension, key_dimension, dtype=self.dtype)
        ).requires_grad_(requires_grad)
        value_action = (
            value_identity
            + 0.03
            * torch.randn(
                batch, length, value_dimension, value_dimension, dtype=self.dtype
            )
        ).requires_grad_(requires_grad)
        drive = (
            0.05
            * torch.randn(
                batch, length, key_dimension, value_dimension, dtype=self.dtype
            )
        ).requires_grad_(requires_grad)
        return DeltaTransition(key_action, value_action, drive)

    def test_composition_is_associative_and_matches_application(self) -> None:
        transition = self.random_transition(3)
        steps = [
            DeltaTransition(
                transition.key_action[:, index],
                transition.value_action[:, index],
                transition.drive[:, index],
            )
            for index in range(3)
        ]
        left = compose_delta(steps[2], compose_delta(steps[1], steps[0]))
        right = compose_delta(compose_delta(steps[2], steps[1]), steps[0])
        state = torch.randn(2, 4, 3, dtype=self.dtype)
        torch.testing.assert_close(
            apply_delta(left, state),
            apply_delta(right, state),
            rtol=2e-13,
            atol=2e-13,
        )
        sequential = state
        for step in steps:
            sequential = apply_delta(step, sequential)
        torch.testing.assert_close(
            apply_delta(left, state), sequential, rtol=2e-13, atol=2e-13
        )

    def test_work_efficient_and_chunkwise_scans_match_recurrence(self) -> None:
        for length in (1, 2, 3, 7, 16, 17, 31, 65, 127):
            for chunk_size in (1, 4, 16, 32, 64):
                with self.subTest(length=length, chunk_size=chunk_size):
                    transition = self.random_transition(length)
                    initial = torch.randn(2, 4, 3, dtype=self.dtype)
                    expected = recurrent_delta_states(transition, initial)
                    work_efficient = scanned_delta_states(
                        transition, initial, backend="work_efficient"
                    )
                    chunked = scanned_delta_states(
                        transition,
                        initial,
                        backend="chunkwise",
                        chunk_size=chunk_size,
                    )
                    torch.testing.assert_close(
                        work_efficient, expected, rtol=5e-12, atol=5e-12
                    )
                    torch.testing.assert_close(
                        chunked, expected, rtol=5e-12, atol=5e-12
                    )

    def test_chunkwise_and_recurrent_gradients_match(self) -> None:
        initial_base = torch.randn(2, 4, 3, dtype=self.dtype)
        weights = torch.randn(2, 11, 4, 3, dtype=self.dtype)
        common = self.random_transition(11)
        losses = []
        gradients = []
        for backend in ("recurrent", "chunkwise"):
            transition = DeltaTransition(
                common.key_action.clone().requires_grad_(True),
                common.value_action.clone().requires_grad_(True),
                common.drive.clone().requires_grad_(True),
            )
            initial = initial_base.clone().requires_grad_(True)
            if backend == "recurrent":
                states = recurrent_delta_states(transition, initial)
            else:
                states = scanned_delta_states(
                    transition, initial, backend="chunkwise", chunk_size=4
                )
            loss = (states * weights).sum()
            losses.append(loss)
            gradients.append(
                torch.autograd.grad(
                    loss,
                    (
                        transition.key_action,
                        transition.value_action,
                        transition.drive,
                        initial,
                    ),
                )
            )
        torch.testing.assert_close(losses[0], losses[1], rtol=1e-11, atol=1e-11)
        for recurrent, chunked in zip(gradients[0], gradients[1]):
            torch.testing.assert_close(recurrent, chunked, rtol=2e-10, atol=2e-10)

    def test_one_hot_delta_keys_overwrite_exactly(self) -> None:
        keys = torch.eye(4, dtype=self.dtype)[torch.tensor([[0, 1, 2, 3, 1, 1]])]
        values = torch.randn(1, 6, 3, dtype=self.dtype)
        transition = delta_write_transitions(
            keys, values, torch.ones(1, 6, dtype=self.dtype)
        )
        states = scanned_delta_states(
            transition,
            torch.zeros(1, 4, 3, dtype=self.dtype),
            backend="chunkwise",
            chunk_size=4,
        )
        expected = torch.stack(
            (values[0, 0], values[0, 5], values[0, 2], values[0, 3]), dim=0
        )
        torch.testing.assert_close(states[0, -1], expected, rtol=0, atol=1e-14)
        query = torch.eye(4, dtype=self.dtype).reshape(1, 4, 4)
        prediction = delta_read(states[:, -1, None].expand(-1, 4, -1, -1), query)
        torch.testing.assert_close(prediction[0], expected, rtol=0, atol=1e-14)

    def test_value_transport_rotates_every_stored_value(self) -> None:
        keys = torch.eye(3, dtype=self.dtype).reshape(1, 3, 3)
        values = torch.randn(1, 3, 3, dtype=self.dtype)
        writes = delta_write_transitions(
            keys, values, torch.ones(1, 3, dtype=self.dtype)
        )
        skew = torch.randn(1, 1, 3, 3, dtype=self.dtype)
        skew = skew - skew.transpose(-1, -2)
        rotation = torch.matrix_exp(skew)
        transport = value_transport_transitions(rotation, key_dimension=3)
        transition = DeltaTransition(
            torch.cat((writes.key_action, transport.key_action), dim=1),
            torch.cat((writes.value_action, transport.value_action), dim=1),
            torch.cat((writes.drive, transport.drive), dim=1),
        )
        final = recurrent_delta_states(
            transition, torch.zeros(1, 3, 3, dtype=self.dtype)
        )[:, -1]
        expected = torch.einsum("bij,bkj->bki", rotation[:, 0], values)
        torch.testing.assert_close(final, expected, rtol=2e-13, atol=2e-13)

    def test_additive_write_retains_stale_value(self) -> None:
        key = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], dtype=self.dtype)
        value = torch.tensor([[[1.0], [-1.0]]], dtype=self.dtype)
        additive = torch.einsum("btk,btv->btkv", key, value).sum(dim=1)
        delta = delta_write_transitions(key, value, torch.ones(1, 2, dtype=self.dtype))
        corrected = recurrent_delta_states(
            delta, torch.zeros(1, 2, 1, dtype=self.dtype)
        )[:, -1]
        self.assertEqual(float(additive[0, 0, 0]), 0.0)
        self.assertEqual(float(corrected[0, 0, 0]), -1.0)

    def test_prefix_objects_match_across_scan_schedules(self) -> None:
        transition = self.random_transition(23)
        expected = work_efficient_delta_scan(transition)
        actual = chunkwise_delta_scan(transition, chunk_size=8)
        for left, right in zip(
            (expected.key_action, expected.value_action, expected.drive),
            (actual.key_action, actual.value_action, actual.drive),
        ):
            torch.testing.assert_close(left, right, rtol=5e-12, atol=5e-12)


if __name__ == "__main__":
    unittest.main()
