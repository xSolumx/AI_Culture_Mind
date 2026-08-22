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

from pure_spin8_ssm.factorized_scan import triton_controller_factorized_scan
from pure_spin8_ssm.torch_backend import Spin8AffineTransition, recurrent_spin8_scan
from spin8_triality import torch_triality_generators

from chunk_parallel_scan import factorized_triality_actions
from coupled_isotypic_scan import (
    CoupledIsotypicTransition,
    recurrent_coupled_scan,
)
from independent_block_scan import recurrent_independent_block_scan
from model import PureSpinSSMV12, PureSpinV12Config
from raw_cuda import (
    raw_cuda_controller_factorized_scan,
    raw_cuda_coordinate_factorized_scan,
    raw_cuda_coupled_coordinate_scan,
    raw_cuda_hybrid_coordinate_scan,
    raw_cuda_independent_block_scan,
    raw_cuda_isotypic_coordinate_scan,
    raw_cuda_spin_delta_scan,
)
from spin_delta_scan import SpinDeltaTransition, recurrent_spin_delta_scan


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


def coupled_inputs(factor_count: int = 6) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(20_260_824)
    batch, length = 2, 5
    coordinates = 0.03 * torch.randn(
        batch, length, factor_count, device="cuda"
    )
    generators = torch_triality_generators(device="cuda")[:, :factor_count]
    angles = 0.1 * torch.randn(batch, length, device="cuda")
    cosine, sine = torch.cos(angles), torch.sin(angles)
    scale = 0.75 + 0.15 * torch.rand(batch, length, 2, device="cuda")
    rotation = torch.stack(
        (cosine, -sine, sine, cosine), dim=-1
    ).reshape(batch, length, 2, 2)
    left = scale[..., :, None] * rotation
    drive = 0.01 * torch.randn(batch, length, 2, 3, 8, device="cuda")
    initial = torch.randn(batch, 2, 3, 8, device="cuda")
    return coordinates, generators.contiguous(), left, drive, initial


def independent_block_inputs(factor_count: int = 6) -> tuple[torch.Tensor, ...]:
    coordinates, generators, left, drive, initial = coupled_inputs(factor_count)
    torch.manual_seed(20_260_826)
    independent_coordinates = 0.03 * torch.randn(
        *coordinates.shape[:2], 2, factor_count, device="cuda"
    )
    return independent_coordinates, generators, left, drive, initial


def spin_delta_inputs(factor_count: int = 6) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(20_260_842 + factor_count)
    batch, length, heads = 2, 5, 2
    coordinates = 0.03 * torch.randn(
        batch, length, heads, factor_count, device="cuda"
    )
    generators = torch_triality_generators(device="cuda")[:, :factor_count]
    raw_left = 0.08 * torch.randn(
        batch, length, heads, 2, 2, device="cuda"
    )
    left = 0.8 * torch.eye(2, device="cuda") + raw_left
    drive = 0.01 * torch.randn(
        batch, length, heads, 2, 3, 8, device="cuda"
    )
    initial = torch.randn(batch, heads, 2, 3, 8, device="cuda")
    return coordinates, generators.contiguous(), left, drive, initial


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("factor_count", [3, 6, 15, 28])
def test_raw_cuda_coupled_full_gradient_parity(factor_count: int) -> None:
    base = coupled_inputs(factor_count)
    semantic_inputs = tuple(
        tensor.detach().clone().requires_grad_(index != 1)
        for index, tensor in enumerate(base)
    )
    raw_inputs = tuple(
        tensor.detach().clone().requires_grad_(index != 1)
        for index, tensor in enumerate(base)
    )
    actions = factorized_triality_actions(
        semantic_inputs[0].unsqueeze(-2), semantic_inputs[1]
    ).squeeze(-4)
    transition = CoupledIsotypicTransition(
        semantic_inputs[2], actions, semantic_inputs[3]
    )
    expected, _ = recurrent_coupled_scan(transition, semantic_inputs[4])
    actual = raw_cuda_coupled_coordinate_scan(*raw_inputs)
    output_gradient = torch.randn_like(actual)
    differentiable = (0, 2, 3, 4)
    expected_gradients = torch.autograd.grad(
        expected,
        tuple(semantic_inputs[index] for index in differentiable),
        output_gradient,
    )
    actual_gradients = torch.autograd.grad(
        actual,
        tuple(raw_inputs[index] for index in differentiable),
        output_gradient,
    )
    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-5)
    for raw_gradient, semantic_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            raw_gradient, semantic_gradient, rtol=1e-3, atol=3e-3
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("factor_count", [3, 6, 15, 28])
def test_raw_cuda_spin_delta_full_gradient_parity(factor_count: int) -> None:
    base = spin_delta_inputs(factor_count)
    semantic_inputs = tuple(
        tensor.detach().clone().requires_grad_(index != 1)
        for index, tensor in enumerate(base)
    )
    raw_inputs = tuple(
        tensor.detach().clone().requires_grad_(index != 1)
        for index, tensor in enumerate(base)
    )
    transition = SpinDeltaTransition(
        semantic_inputs[2],
        factorized_triality_actions(semantic_inputs[0], semantic_inputs[1]),
        semantic_inputs[3],
    )
    expected, _ = recurrent_spin_delta_scan(transition, semantic_inputs[4])
    actual = raw_cuda_spin_delta_scan(*raw_inputs)
    output_gradient = torch.randn_like(actual)
    differentiable = (0, 2, 3, 4)
    expected_gradients = torch.autograd.grad(
        expected,
        tuple(semantic_inputs[index] for index in differentiable),
        output_gradient,
    )
    actual_gradients = torch.autograd.grad(
        actual,
        tuple(raw_inputs[index] for index in differentiable),
        output_gradient,
    )
    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-5)
    for raw_gradient, semantic_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            raw_gradient, semantic_gradient, rtol=1e-3, atol=3e-3
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("factor_count", [3, 6, 15, 28])
def test_raw_cuda_independent_block_full_gradient_parity(
    factor_count: int,
) -> None:
    base = independent_block_inputs(factor_count)
    semantic_inputs = tuple(
        tensor.detach().clone().requires_grad_(index != 1)
        for index, tensor in enumerate(base)
    )
    raw_inputs = tuple(
        tensor.detach().clone().requires_grad_(index != 1)
        for index, tensor in enumerate(base)
    )
    actions = factorized_triality_actions(semantic_inputs[0], semantic_inputs[1])
    expected, _ = recurrent_independent_block_scan(
        semantic_inputs[2], actions, semantic_inputs[3], semantic_inputs[4]
    )
    actual = raw_cuda_independent_block_scan(*raw_inputs)
    output_gradient = torch.randn_like(actual)
    differentiable = (0, 2, 3, 4)
    expected_gradients = torch.autograd.grad(
        expected,
        tuple(semantic_inputs[index] for index in differentiable),
        output_gradient,
    )
    actual_gradients = torch.autograd.grad(
        actual,
        tuple(raw_inputs[index] for index in differentiable),
        output_gradient,
    )
    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-5)
    for raw_gradient, semantic_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            raw_gradient, semantic_gradient, rtol=1e-3, atol=3e-3
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("factor_count", [3, 6, 15, 28])
def test_raw_cuda_isotypic_retention_full_gradient_parity(
    factor_count: int,
) -> None:
    torch.manual_seed(20_260_829 + factor_count)
    batch, length, channels = 2, 5, 2
    base = (
        0.03
        * torch.randn(batch, length, channels, factor_count, device="cuda"),
        torch_triality_generators(device="cuda")[:, :factor_count].contiguous(),
        0.75 + 0.15 * torch.rand(batch, length, channels, 3, device="cuda"),
        0.01 * torch.randn(batch, length, channels, 3, 8, device="cuda"),
        torch.randn(batch, channels, 3, 8, device="cuda"),
    )
    semantic_inputs = tuple(
        tensor.detach().clone().requires_grad_(index != 1)
        for index, tensor in enumerate(base)
    )
    raw_inputs = tuple(
        tensor.detach().clone().requires_grad_(index != 1)
        for index, tensor in enumerate(base)
    )
    transition = Spin8AffineTransition(
        semantic_inputs[2],
        factorized_triality_actions(semantic_inputs[0], semantic_inputs[1]),
        semantic_inputs[3],
    )
    expected, _ = recurrent_spin8_scan(transition, semantic_inputs[4])
    actual = raw_cuda_hybrid_coordinate_scan(*raw_inputs)
    output_gradient = torch.randn_like(actual)
    differentiable = (0, 2, 3, 4)
    expected_gradients = torch.autograd.grad(
        expected,
        tuple(semantic_inputs[index] for index in differentiable),
        output_gradient,
    )
    actual_gradients = torch.autograd.grad(
        actual,
        tuple(raw_inputs[index] for index in differentiable),
        output_gradient,
    )
    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-5)
    for raw_gradient, semantic_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            raw_gradient, semantic_gradient, rtol=1e-3, atol=3e-3
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_raw_cuda_independent_block_singular_left_replay_parity() -> None:
    base = list(independent_block_inputs(6))
    base[2] = base[2].clone()
    base[2][:, 1::2, 1] = 0.0
    semantic_inputs = tuple(
        tensor.detach().clone().requires_grad_(index != 1)
        for index, tensor in enumerate(base)
    )
    raw_inputs = tuple(
        tensor.detach().clone().requires_grad_(index != 1)
        for index, tensor in enumerate(base)
    )
    actions = factorized_triality_actions(semantic_inputs[0], semantic_inputs[1])
    expected, _ = recurrent_independent_block_scan(
        semantic_inputs[2], actions, semantic_inputs[3], semantic_inputs[4]
    )
    actual = raw_cuda_independent_block_scan(*raw_inputs)
    output_gradient = torch.randn_like(actual)
    differentiable = (0, 2, 3, 4)
    expected_gradients = torch.autograd.grad(
        expected,
        tuple(semantic_inputs[index] for index in differentiable),
        output_gradient,
    )
    actual_gradients = torch.autograd.grad(
        actual,
        tuple(raw_inputs[index] for index in differentiable),
        output_gradient,
    )
    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=4e-5)
    for raw_gradient, semantic_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            raw_gradient, semantic_gradient, rtol=1e-3, atol=3e-3
        )


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
@pytest.mark.parametrize("factor_count", [3, 6, 15, 28])
def test_raw_cuda_isotypic_full_gradient_parity(factor_count: int) -> None:
    _, _, _, generators, scale, drive, initial, _ = controller_inputs()
    coordinates = 0.03 * torch.randn(
        *scale.shape, factor_count, device="cuda"
    )
    base = (
        coordinates,
        generators[:, :factor_count].contiguous(),
        scale,
        drive,
        initial,
    )
    packed_inputs = tuple(
        tensor.detach().clone().requires_grad_(index != 1)
        for index, tensor in enumerate(base)
    )
    split_inputs = tuple(
        tensor.detach().clone().requires_grad_(index != 1)
        for index, tensor in enumerate(base)
    )
    packed_output = raw_cuda_coordinate_factorized_scan(*packed_inputs)
    split_output = raw_cuda_isotypic_coordinate_scan(*split_inputs)
    hybrid_inputs = tuple(
        tensor.detach().clone().requires_grad_(index != 1)
        for index, tensor in enumerate(base)
    )
    hybrid_output = raw_cuda_hybrid_coordinate_scan(*hybrid_inputs)
    output_gradient = torch.randn_like(split_output)
    differentiable = (0, 2, 3, 4)
    packed_gradients = torch.autograd.grad(
        packed_output,
        tuple(packed_inputs[index] for index in differentiable),
        output_gradient,
    )
    split_gradients = torch.autograd.grad(
        split_output,
        tuple(split_inputs[index] for index in differentiable),
        output_gradient,
    )
    hybrid_gradients = torch.autograd.grad(
        hybrid_output,
        tuple(hybrid_inputs[index] for index in differentiable),
        output_gradient,
    )
    torch.testing.assert_close(split_output, packed_output, rtol=4e-5, atol=4e-5)
    torch.testing.assert_close(hybrid_output, packed_output, rtol=4e-5, atol=4e-5)
    for actual, expected in zip(split_gradients, packed_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=9e-4, atol=2e-3)
    for actual, expected in zip(hybrid_gradients, packed_gradients, strict=True):
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
@pytest.mark.parametrize("retention_mode", ["isotypic", "isotypic_spectrum"])
def test_raw_cuda_isotypic_retention_full_model_gradient_parity(
    retention_mode: str,
) -> None:
    torch.manual_seed(20_260_829)
    config = PureSpinV12Config(
        d_model=16,
        num_layers=1,
        spin_channels=2,
        d_conv=3,
        retention_mode=retention_mode,
    )
    semantic_model = PureSpinSSMV12(config).cuda()
    raw_model = copy.deepcopy(semantic_model)
    with torch.no_grad():
        if retention_mode == "isotypic":
            offset = semantic_model.blocks[0].spin.retention_offset_controller
            assert offset is not None
            offset.weight.normal_(std=0.02)
        else:
            log_rates = semantic_model.blocks[0].spin.retention_log_rates
            assert log_rates is not None
            log_rates.normal_(std=0.02)
        raw_model.load_state_dict(semantic_model.state_dict())
    tokens = torch.randint(0, 256, (2, 5), device="cuda")
    expected = semantic_model(tokens, scan_mode="chunk_parallel")["logits"]
    actual = raw_model(tokens, scan_mode="raw_cuda_hybrid")["logits"]
    output_gradient = torch.randn_like(actual)
    semantic_gradients = torch.autograd.grad(
        expected, tuple(semantic_model.parameters()), output_gradient
    )
    raw_gradients = torch.autograd.grad(
        actual, tuple(raw_model.parameters()), output_gradient
    )
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-5)
    for raw_gradient, semantic_gradient in zip(
        raw_gradients, semantic_gradients, strict=True
    ):
        torch.testing.assert_close(
            raw_gradient, semantic_gradient, rtol=1e-3, atol=3e-3
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_raw_cuda_coupled_full_model_gradient_parity() -> None:
    torch.manual_seed(20_260_825)
    config = PureSpinV12Config(
        d_model=16,
        num_layers=1,
        spin_channels=2,
        d_conv=3,
        recurrence="coupled_isotypic",
        recurrent_multiplicity="orthogonal",
    )
    semantic_model = PureSpinSSMV12(config).cuda()
    raw_model = copy.deepcopy(semantic_model)
    tokens = torch.randint(0, 256, (2, 5), device="cuda")
    expected = semantic_model(tokens, scan_mode="coupled_recurrent")["logits"]
    actual = raw_model(tokens, scan_mode="raw_cuda_coupled")["logits"]
    output_gradient = torch.randn_like(actual)
    semantic_gradients = torch.autograd.grad(
        expected, tuple(semantic_model.parameters()), output_gradient
    )
    raw_gradients = torch.autograd.grad(
        actual, tuple(raw_model.parameters()), output_gradient
    )
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-5)
    for raw_gradient, semantic_gradient in zip(
        raw_gradients, semantic_gradients, strict=True
    ):
        torch.testing.assert_close(
            raw_gradient, semantic_gradient, rtol=1e-3, atol=3e-3
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("coupling_scale", ["unit", "retention_step"])
def test_raw_cuda_independent_block_full_model_gradient_parity(
    coupling_scale: str,
) -> None:
    torch.manual_seed(20_260_827)
    config = PureSpinV12Config(
        d_model=16,
        num_layers=1,
        spin_channels=2,
        d_conv=3,
        recurrence="independent_block",
        recurrent_multiplicity="orthogonal",
        recurrent_coupling_scale=coupling_scale,
    )
    semantic_model = PureSpinSSMV12(config).cuda()
    raw_model = copy.deepcopy(semantic_model)
    with torch.no_grad():
        semantic_model.blocks[0].recurrent_multiplicity_controller.weight.normal_(
            std=0.02
        )
        raw_model.load_state_dict(semantic_model.state_dict())
    tokens = torch.randint(0, 256, (2, 5), device="cuda")
    expected = semantic_model(tokens, scan_mode="block_recurrent")["logits"]
    actual = raw_model(tokens, scan_mode="raw_cuda_block")["logits"]
    output_gradient = torch.randn_like(actual)
    semantic_gradients = torch.autograd.grad(
        expected, tuple(semantic_model.parameters()), output_gradient
    )
    raw_gradients = torch.autograd.grad(
        actual, tuple(raw_model.parameters()), output_gradient
    )
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-5)
    for raw_gradient, semantic_gradient in zip(
        raw_gradients, semantic_gradients, strict=True
    ):
        torch.testing.assert_close(
            raw_gradient, semantic_gradient, rtol=1e-3, atol=3e-3
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("group_dimension", [3, 4, 6, 8])
def test_raw_cuda_spin_delta_full_model_gradient_parity(
    group_dimension: int,
) -> None:
    torch.manual_seed(20_260_849 + group_dimension)
    config = PureSpinV12Config(
        d_model=16,
        num_layers=1,
        spin_channels=2,
        d_conv=3,
        recurrence="spin_delta",
        group_schedule=(group_dimension,),
    )
    semantic_model = PureSpinSSMV12(config).cuda()
    raw_model = copy.deepcopy(semantic_model)
    with torch.no_grad():
        for controller in (
            semantic_model.blocks[0].delta_write_controller,
            semantic_model.blocks[0].delta_erase_controller,
            semantic_model.blocks[0].delta_erase_strength_controller,
            semantic_model.blocks[0].delta_query_controller,
        ):
            assert controller is not None
            controller.weight.normal_(std=0.02)
        raw_model.load_state_dict(semantic_model.state_dict())
    tokens = torch.randint(0, 256, (2, 5), device="cuda")
    expected = semantic_model(tokens, scan_mode="delta_recurrent")["logits"]
    actual = raw_model(tokens, scan_mode="raw_cuda_delta")["logits"]
    output_gradient = torch.randn_like(actual)
    semantic_gradients = torch.autograd.grad(
        expected, tuple(semantic_model.parameters()), output_gradient
    )
    raw_gradients = torch.autograd.grad(
        actual, tuple(raw_model.parameters()), output_gradient
    )
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-5)
    for raw_gradient, semantic_gradient in zip(
        raw_gradients, semantic_gradients, strict=True
    ):
        torch.testing.assert_close(
            raw_gradient, semantic_gradient, rtol=1e-3, atol=3e-3
        )


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
