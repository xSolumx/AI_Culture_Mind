"""Semantic and gradient gates for the continuous affine chunk compiler."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from chunk_parallel_scan import (
    chunk_parallel_spin8_scan,
    factorized_triality_actions,
)
from pure_spin8_ssm.torch_backend import (
    Spin8AffineTransition,
    recurrent_spin8_scan,
)
from spin8_triality import torch_triality_generators


@pytest.mark.parametrize("length,chunk_size", [(1, 1), (5, 2), (8, 4), (9, 4)])
def test_chunk_parallel_matches_recurrent_with_all_gradients(
    length: int, chunk_size: int
) -> None:
    torch.manual_seed(20_260_821 + length)
    batch, channels, factors = 2, 2, 6
    generators = torch_triality_generators(dtype=torch.float64)[:, :factors].contiguous()
    base = (
        0.03 * torch.randn(batch, length, channels, factors, dtype=torch.float64),
        0.8 + 0.1 * torch.rand(batch, length, channels, dtype=torch.float64),
        0.01
        * torch.randn(batch, length, channels, 3, 8, dtype=torch.float64),
        torch.randn(batch, channels, 3, 8, dtype=torch.float64),
    )
    recurrent_inputs = tuple(value.clone().requires_grad_() for value in base)
    chunk_inputs = tuple(value.clone().requires_grad_() for value in base)

    def run(values, *, chunked: bool):
        coordinates, scale, drive, initial = values
        transition = Spin8AffineTransition(
            scale=scale,
            action=factorized_triality_actions(coordinates, generators),
            drive=drive,
        )
        if chunked:
            return chunk_parallel_spin8_scan(
                transition, initial, chunk_size=chunk_size
            )[0]
        return recurrent_spin8_scan(transition, initial)[0]

    expected = run(recurrent_inputs, chunked=False)
    actual = run(chunk_inputs, chunked=True)
    output_gradient = torch.randn_like(actual)
    expected_gradients = torch.autograd.grad(
        expected, recurrent_inputs, output_gradient
    )
    actual_gradients = torch.autograd.grad(actual, chunk_inputs, output_gradient)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=2e-11, atol=2e-11
        )


def test_chunk_parallel_matches_recurrent_with_isotypic_retention() -> None:
    torch.manual_seed(20_260_829)
    batch, length, channels, factors = 2, 7, 2, 6
    generators = torch_triality_generators(dtype=torch.float64)[:, :factors].contiguous()
    base = (
        0.03 * torch.randn(batch, length, channels, factors, dtype=torch.float64),
        0.8 + 0.1 * torch.rand(batch, length, channels, 3, dtype=torch.float64),
        0.01
        * torch.randn(batch, length, channels, 3, 8, dtype=torch.float64),
        torch.randn(batch, channels, 3, 8, dtype=torch.float64),
    )
    recurrent_inputs = tuple(value.clone().requires_grad_() for value in base)
    chunk_inputs = tuple(value.clone().requires_grad_() for value in base)

    def run(values, *, chunked: bool):
        coordinates, scale, drive, initial = values
        transition = Spin8AffineTransition(
            scale=scale,
            action=factorized_triality_actions(coordinates, generators),
            drive=drive,
        )
        if chunked:
            return chunk_parallel_spin8_scan(transition, initial, chunk_size=4)[0]
        return recurrent_spin8_scan(transition, initial)[0]

    expected = run(recurrent_inputs, chunked=False)
    actual = run(chunk_inputs, chunked=True)
    output_gradient = torch.randn_like(actual)
    expected_gradients = torch.autograd.grad(
        expected, recurrent_inputs, output_gradient
    )
    actual_gradients = torch.autograd.grad(actual, chunk_inputs, output_gradient)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=2e-11, atol=2e-11
        )
