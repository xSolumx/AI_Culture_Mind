"""CPU theorem gate for the isotypic Tensor-Core scheduling identity."""

import torch

from benchmark_spin8_isotypic_tensor_core import isotypic_action


def test_isotypic_action_equals_block_diagonal_representation() -> None:
    generator = torch.Generator().manual_seed(20_260_821)
    dimension = 8
    multiplicity = 5
    action = torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    state = torch.randn(multiplicity, dimension, generator=generator, dtype=torch.float64)

    factored = isotypic_action(action, state)
    dense = torch.kron(torch.eye(multiplicity, dtype=torch.float64), action)
    dense_result = (dense @ state.reshape(-1)).reshape(multiplicity, dimension)

    torch.testing.assert_close(factored, dense_result, rtol=0, atol=1e-12)


def test_isotypic_factorization_preserves_chronological_order() -> None:
    generator = torch.Generator().manual_seed(20_260_822)
    first = torch.randn(8, 8, generator=generator, dtype=torch.float64)
    second = torch.randn(8, 8, generator=generator, dtype=torch.float64)
    state = torch.randn(7, 8, generator=generator, dtype=torch.float64)

    recurrent = isotypic_action(second, isotypic_action(first, state))
    composed = isotypic_action(second @ first, state)

    torch.testing.assert_close(recurrent, composed, rtol=0, atol=1e-11)
