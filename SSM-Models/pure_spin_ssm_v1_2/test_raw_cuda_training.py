"""Gradient and stream contracts for the raw CUDA Spin training backend."""

from __future__ import annotations

import copy
import sys
from itertools import combinations
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from model import PureSpinSSMV12, PureSpinV12Config
from pure_spin8_ssm.factorized_scan import triton_controller_factorized_scan
from raw_cuda import (
    raw_cuda_controller_factorized_scan,
    raw_cuda_coordinate_factorized_scan,
)
from spin8_triality import torch_triality_generators


def controller_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(20_260_821)
    batch, length, channels, input_size = 2, 5, 2, 16
    features = torch.randn(batch, length, input_size, device="cuda")
    weight = 0.03 * torch.randn(channels * 28, input_size, device="cuda")
    bias = 0.01 * torch.randn(channels * 28, device="cuda")
    generators = torch_triality_generators(device="cuda")
    scale = 0.8 + 0.1 * torch.rand(batch, length, channels, device="cuda")
    drive = 0.01 * torch.randn(batch, length, channels, 3, 8, device="cuda")
    initial = torch.randn(batch, channels, 3, 8, device="cuda")
    gate = torch.tensor(
        [[1.0, 1.0, 1.0, 0.0, 0.0], [1.0] * length], device="cuda"
    )
    return features, weight, bias, generators, scale, drive, initial, gate


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_raw_cuda_controller_full_gradient_parity() -> None:
    inputs = controller_inputs()
    triton_inputs = tuple(
        tensor.detach().clone().requires_grad_(index not in {3, 7})
        for index, tensor in enumerate(inputs)
    )
    raw_inputs = tuple(
        tensor.detach().clone().requires_grad_(index not in {3, 7})
        for index, tensor in enumerate(inputs)
    )
    triton_output = triton_controller_factorized_scan(*triton_inputs)
    raw_output = raw_cuda_controller_factorized_scan(*raw_inputs)
    output_gradient = torch.randn_like(raw_output)
    differentiable = (0, 1, 2, 4, 5, 6)
    triton_gradients = torch.autograd.grad(
        triton_output,
        tuple(triton_inputs[index] for index in differentiable),
        output_gradient,
    )
    raw_gradients = torch.autograd.grad(
        raw_output,
        tuple(raw_inputs[index] for index in differentiable),
        output_gradient,
    )
    torch.testing.assert_close(raw_output, triton_output, rtol=4e-5, atol=4e-5)
    for actual, expected in zip(raw_gradients, triton_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=9e-4, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_raw_cuda_controller_uses_current_stream() -> None:
    inputs = controller_inputs()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.no_grad():
        raw_output = raw_cuda_controller_factorized_scan(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.no_grad():
        expected = triton_controller_factorized_scan(*inputs)
    torch.testing.assert_close(raw_output, expected, rtol=4e-5, atol=4e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_raw_cuda_coordinate_full_gradient_parity() -> None:
    features, weight, bias, generators, scale, drive, initial, gate = (
        controller_inputs()
    )
    channels = scale.shape[-1]
    coordinates = torch.nn.functional.linear(features, weight, bias).reshape(
        *features.shape[:2], channels, 28
    )
    coordinates = coordinates * gate[..., None, None]
    triton_inputs = tuple(
        tensor.detach().clone().requires_grad_(index != 1)
        for index, tensor in enumerate(
            (coordinates, generators, scale, drive, initial)
        )
    )
    raw_inputs = tuple(
        tensor.detach().clone().requires_grad_(index != 1)
        for index, tensor in enumerate(
            (coordinates, generators, scale, drive, initial)
        )
    )
    from pure_spin8_ssm.factorized_scan import factorized_coordinate_spin8_scan

    triton_output = factorized_coordinate_spin8_scan(
        *triton_inputs, backend="triton"
    )
    raw_output = raw_cuda_coordinate_factorized_scan(*raw_inputs)
    output_gradient = torch.randn_like(raw_output)
    differentiable = (0, 2, 3, 4)
    triton_gradients = torch.autograd.grad(
        triton_output,
        tuple(triton_inputs[index] for index in differentiable),
        output_gradient,
    )
    raw_gradients = torch.autograd.grad(
        raw_output,
        tuple(raw_inputs[index] for index in differentiable),
        output_gradient,
    )
    torch.testing.assert_close(raw_output, triton_output, rtol=4e-5, atol=4e-5)
    for actual, expected in zip(raw_gradients, triton_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=9e-4, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_raw_cuda_coordinate_zero_scale_fallback_gradient_parity() -> None:
    features, weight, bias, generators, scale, drive, initial, gate = (
        controller_inputs()
    )
    channels = scale.shape[-1]
    coordinates = torch.nn.functional.linear(features, weight, bias).reshape(
        *features.shape[:2], channels, 28
    )
    coordinates = coordinates * gate[..., None, None]
    scale[:, 1, 0] = 0.0
    scale[:, 3, 1] = 1.0e-9
    base = (coordinates, generators, scale, drive, initial)
    triton_inputs = tuple(
        tensor.detach().clone().requires_grad_(index != 1)
        for index, tensor in enumerate(base)
    )
    raw_inputs = tuple(
        tensor.detach().clone().requires_grad_(index != 1)
        for index, tensor in enumerate(base)
    )
    from pure_spin8_ssm.factorized_scan import factorized_coordinate_spin8_scan

    triton_output = factorized_coordinate_spin8_scan(
        *triton_inputs, backend="triton"
    )
    raw_output = raw_cuda_coordinate_factorized_scan(*raw_inputs)
    output_gradient = torch.randn_like(raw_output)
    differentiable = (0, 2, 3, 4)
    triton_gradients = torch.autograd.grad(
        triton_output,
        tuple(triton_inputs[index] for index in differentiable),
        output_gradient,
    )
    raw_gradients = torch.autograd.grad(
        raw_output,
        tuple(raw_inputs[index] for index in differentiable),
        output_gradient,
    )
    torch.testing.assert_close(raw_output, triton_output, rtol=4e-5, atol=4e-5)
    for actual, expected in zip(raw_gradients, triton_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=9e-4, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_nested_spin3_subgroup_matches_zero_padded_spin8() -> None:
    _, _, _, generators, scale, drive, initial, _ = controller_inputs()
    indices = tuple(
        index
        for index, (left, right) in enumerate(combinations(range(8), 2))
        if left < 3 and right < 3
    )
    index = torch.tensor(indices, device="cuda")
    subset_coordinates = torch.randn(
        *scale.shape, len(indices), device="cuda", requires_grad=True
    )
    subset_generators = generators.index_select(1, index).contiguous()
    subset_inputs = (
        subset_coordinates,
        subset_generators,
        scale.detach().clone().requires_grad_(),
        drive.detach().clone().requires_grad_(),
        initial.detach().clone().requires_grad_(),
    )
    full_coordinates = torch.zeros(*scale.shape, 28, device="cuda")
    full_coordinates[..., index] = subset_coordinates.detach()
    full_inputs = (
        full_coordinates.requires_grad_(),
        generators,
        scale.detach().clone().requires_grad_(),
        drive.detach().clone().requires_grad_(),
        initial.detach().clone().requires_grad_(),
    )
    subset_output = raw_cuda_coordinate_factorized_scan(*subset_inputs)
    full_output = raw_cuda_coordinate_factorized_scan(*full_inputs)
    output_gradient = torch.randn_like(full_output)
    subset_gradients = torch.autograd.grad(
        subset_output, (subset_inputs[0], *subset_inputs[2:]), output_gradient
    )
    full_gradients = torch.autograd.grad(
        full_output, (full_inputs[0], *full_inputs[2:]), output_gradient
    )
    torch.testing.assert_close(subset_output, full_output, rtol=4e-5, atol=4e-5)
    torch.testing.assert_close(
        subset_gradients[0], full_gradients[0].index_select(-1, index),
        rtol=9e-4, atol=2e-3,
    )
    for actual, expected in zip(
        subset_gradients[1:], full_gradients[1:], strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=9e-4, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_raw_cuda_full_model_gradient_parity() -> None:
    torch.manual_seed(20_260_822)
    triton_model = PureSpinSSMV12(
        PureSpinV12Config(d_model=16, num_layers=1, spin_channels=1, d_conv=3)
    ).cuda()
    raw_model = copy.deepcopy(triton_model)
    tokens = torch.randint(0, 256, (2, 5), device="cuda")
    triton_output = triton_model(tokens, scan_mode="compiled_controller")["logits"]
    raw_output = raw_model(tokens, scan_mode="raw_cuda_controller")["logits"]
    output_gradient = torch.randn_like(raw_output)
    triton_gradients = torch.autograd.grad(
        triton_output, tuple(triton_model.parameters()), output_gradient
    )
    raw_gradients = torch.autograd.grad(
        raw_output, tuple(raw_model.parameters()), output_gradient
    )
    torch.testing.assert_close(raw_output, triton_output, rtol=5e-5, atol=5e-5)
    for actual, expected in zip(raw_gradients, triton_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_raw_cuda_factorized_full_model_gradient_parity() -> None:
    torch.manual_seed(20_260_823)
    triton_model = PureSpinSSMV12(
        PureSpinV12Config(d_model=16, num_layers=1, spin_channels=1, d_conv=3)
    ).cuda()
    raw_model = copy.deepcopy(triton_model)
    tokens = torch.randint(0, 256, (2, 5), device="cuda")
    triton_output = triton_model(tokens, scan_mode="compiled_factorized")["logits"]
    raw_output = raw_model(tokens, scan_mode="raw_cuda_factorized")["logits"]
    output_gradient = torch.randn_like(raw_output)
    triton_gradients = torch.autograd.grad(
        triton_output, tuple(triton_model.parameters()), output_gradient
    )
    raw_gradients = torch.autograd.grad(
        raw_output, tuple(raw_model.parameters()), output_gradient
    )
    torch.testing.assert_close(raw_output, triton_output, rtol=5e-5, atol=5e-5)
    for actual, expected in zip(raw_gradients, triton_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_nested_group_schedule_full_model_backward() -> None:
    model = PureSpinSSMV12(
        PureSpinV12Config(
            d_model=16,
            num_layers=4,
            spin_channels=1,
            d_conv=3,
            group_schedule=(3, 4, 6, 8),
        )
    ).cuda()
    assert [len(block.subgroup_indices) for block in model.blocks] == [3, 6, 15, 28]
    tokens = torch.randint(0, 256, (2, 5), device="cuda")
    logits = model(tokens, scan_mode="raw_cuda_factorized")["logits"]
    logits.square().mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
