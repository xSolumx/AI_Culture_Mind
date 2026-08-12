from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from schurscan_comoving_delta import (
    comoving_delta_read_sequence,
    cumulative_actions,
    recurrent_state_scalars,
)
from schurscan_delta_memory import (
    compose_delta,
    delta_read,
    delta_write_transitions,
    scanned_delta_states,
    value_transport_transitions,
)


def random_orthogonal_actions(
    batch: int, length: int, dimension: int, *, generator: torch.Generator
) -> torch.Tensor:
    raw = torch.randn(
        batch, length, dimension, dimension, generator=generator, dtype=torch.float64
    )
    skew = raw - raw.transpose(-1, -2)
    return torch.matrix_exp(0.15 * skew)


def transported_reference(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    actions: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    write = delta_write_transitions(keys, values, beta)
    transport = value_transport_transitions(
        actions, key_dimension=keys.shape[-1]
    )
    transition = compose_delta(write, transport)
    initial = torch.zeros(
        values.shape[0],
        keys.shape[-1],
        values.shape[-1],
        dtype=values.dtype,
    )
    states = scanned_delta_states(
        transition, initial, backend="chunkwise", chunk_size=chunk_size
    )
    return delta_read(states, queries)


class ComovingDeltaTests(unittest.TestCase):
    def test_forward_matches_transported_recurrence_at_irregular_lengths(self) -> None:
        for length in (1, 3, 7, 17):
            with self.subTest(length=length):
                generator = torch.Generator().manual_seed(910_000 + length)
                keys = F.normalize(
                    torch.randn(2, length, 4, generator=generator, dtype=torch.float64),
                    dim=-1,
                )
                values = torch.randn(
                    2, length, 5, generator=generator, dtype=torch.float64
                )
                queries = F.normalize(
                    torch.randn(2, length, 4, generator=generator, dtype=torch.float64),
                    dim=-1,
                )
                actions = random_orthogonal_actions(
                    2, length, 5, generator=generator
                )
                beta = torch.sigmoid(
                    torch.randn(2, length, generator=generator, dtype=torch.float64)
                )
                expected = transported_reference(
                    keys, values, queries, actions, beta, chunk_size=4
                )
                actual = comoving_delta_read_sequence(
                    keys,
                    values,
                    queries,
                    actions,
                    beta=beta,
                    action_backend="work_efficient",
                    delta_chunk_size=4,
                )
                self.assertLess(float((actual - expected).abs().max()), 2e-12)

    def test_gradients_match_transported_recurrence(self) -> None:
        generator = torch.Generator().manual_seed(920_001)
        raw_actions = 0.15 * torch.randn(
            1, 9, 4, 4, generator=generator, dtype=torch.float64
        )
        base = (
            F.normalize(
                torch.randn(1, 9, 3, generator=generator, dtype=torch.float64), dim=-1
            ),
            torch.randn(1, 9, 4, generator=generator, dtype=torch.float64),
            F.normalize(
                torch.randn(1, 9, 3, generator=generator, dtype=torch.float64), dim=-1
            ),
            raw_actions,
            torch.sigmoid(
                torch.randn(1, 9, generator=generator, dtype=torch.float64)
            ),
        )
        gradients = []
        outputs = []
        for implementation in (
            lambda k, v, q, raw, b: transported_reference(
                k,
                v,
                q,
                torch.matrix_exp(raw - raw.transpose(-1, -2)),
                b,
                chunk_size=4,
            ),
            lambda k, v, q, raw, b: comoving_delta_read_sequence(
                k,
                v,
                q,
                torch.matrix_exp(raw - raw.transpose(-1, -2)),
                beta=b,
                delta_chunk_size=4,
            ),
        ):
            tensors = tuple(tensor.detach().clone().requires_grad_(True) for tensor in base)
            output = implementation(*tensors)
            weight = torch.linspace(
                0.2, 1.1, output.numel(), dtype=output.dtype
            ).reshape_as(output)
            gradients.append(torch.autograd.grad((output * weight).sum(), tensors))
            outputs.append(output)
        self.assertLess(
            float((outputs[0] - outputs[1]).detach().abs().max()), 2e-12
        )
        for expected, actual in zip(gradients[0], gradients[1], strict=True):
            self.assertLess(float((actual - expected).detach().abs().max()), 2e-11)

    def test_general_invertible_actions_match_in_ambient_coordinates(self) -> None:
        generator = torch.Generator().manual_seed(925_001)
        identity = torch.eye(4, dtype=torch.float64).reshape(1, 1, 4, 4)
        base = (
            torch.randn(1, 7, 3, generator=generator, dtype=torch.float64),
            torch.randn(1, 7, 4, generator=generator, dtype=torch.float64),
            torch.randn(1, 7, 3, generator=generator, dtype=torch.float64),
            identity
            + 0.04
            * torch.randn(1, 7, 4, 4, generator=generator, dtype=torch.float64),
            torch.sigmoid(
                torch.randn(1, 7, generator=generator, dtype=torch.float64)
            ),
        )
        gradients = []
        outputs = []
        for implementation in (
            lambda k, v, q, a, b: transported_reference(
                k, v, q, a, b, chunk_size=4
            ),
            lambda k, v, q, a, b: comoving_delta_read_sequence(
                k, v, q, a, beta=b, delta_chunk_size=4
            ),
        ):
            tensors = tuple(tensor.detach().clone().requires_grad_(True) for tensor in base)
            output = implementation(*tensors)
            weights = torch.linspace(
                0.3, 1.2, output.numel(), dtype=output.dtype
            ).reshape_as(output)
            outputs.append(output)
            gradients.append(torch.autograd.grad((output * weights).sum(), tensors))
        self.assertLess(
            float((outputs[0] - outputs[1]).detach().abs().max()), 3e-12
        )
        for expected, actual in zip(gradients[0], gradients[1], strict=True):
            self.assertLess(float((actual - expected).detach().abs().max()), 3e-11)

    def test_action_scan_order_and_state_count(self) -> None:
        generator = torch.Generator().manual_seed(930_002)
        actions = random_orthogonal_actions(1, 6, 3, generator=generator)
        prefixes = cumulative_actions(actions)
        recurrent = torch.eye(3, dtype=torch.float64)
        for position in range(actions.shape[1]):
            recurrent = actions[:, position] @ recurrent
            self.assertLess(
                float((prefixes[:, position] - recurrent).abs().max()), 1e-13
            )
        self.assertEqual(recurrent_state_scalars(8, 8), 128)


if __name__ == "__main__":
    unittest.main()
