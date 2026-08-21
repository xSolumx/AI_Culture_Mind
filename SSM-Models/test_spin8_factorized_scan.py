"""Gates for the trainable factorized-coordinate Spin(8) scan."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from pure_spin8_ssm.compiler import (
    HardwareTarget,
    RuntimeShape,
    compile_spin8_training_plan,
)
from pure_spin8_ssm.factorized_scan import (
    eager_factorized_coordinate_scan,
    triton_controller_factorized_scan,
    triton_factorized_coordinate_scan,
)
from pure_spin8_ssm.torch_backend import PureSpin8SSMLayer, spin8_factorized_actions
from spin8_triality import TRIALITY_REPRESENTATIONS, torch_triality_generators


def _inputs(
    device: torch.device | str,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(20_260_822)
    batch, length, channels = 2, 7, 3
    coordinates = 0.08 * torch.randn(
        batch,
        length,
        channels,
        28,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    generators = torch_triality_generators(dtype=dtype, device=device)
    scale = 0.8 + 0.1 * torch.rand(
        batch,
        length,
        channels,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    drive = 0.01 * torch.randn(
        batch,
        length,
        channels,
        3,
        8,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    initial = torch.randn(
        batch,
        channels,
        3,
        8,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return coordinates, generators, scale, drive, initial


def _materialized_reference(
    coordinates: torch.Tensor,
    generators: torch.Tensor,
    scale: torch.Tensor,
    drive: torch.Tensor,
    initial: torch.Tensor,
) -> torch.Tensor:
    actions = spin8_factorized_actions(
        coordinates, generators, TRIALITY_REPRESENTATIONS
    )
    state = initial
    rows = []
    for position in range(coordinates.shape[1]):
        state = (
            scale[:, position, :, None, None]
            * torch.einsum(
                "bcrij,bcrj->bcri", actions[:, position], state
            )
            + drive[:, position]
        )
        rows.append(state)
    return torch.stack(rows, dim=1)


def test_eager_direct_factors_equal_materialized_actions() -> None:
    inputs = _inputs("cpu", dtype=torch.float64)
    direct = eager_factorized_coordinate_scan(*inputs)
    materialized = _materialized_reference(*inputs)
    torch.testing.assert_close(direct, materialized, rtol=0, atol=2e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_triton_factorized_full_gradient_parity() -> None:
    inputs = _inputs("cuda")
    eager_inputs = tuple(
        tensor.detach().clone().requires_grad_(index != 1)
        for index, tensor in enumerate(inputs)
    )
    compiled_inputs = tuple(
        tensor.detach().clone().requires_grad_(index != 1)
        for index, tensor in enumerate(inputs)
    )
    eager = eager_factorized_coordinate_scan(*eager_inputs)
    compiled = triton_factorized_coordinate_scan(*compiled_inputs)
    gradient = torch.randn_like(eager)
    eager_gradients = torch.autograd.grad(
        eager,
        (eager_inputs[0], *eager_inputs[2:]),
        gradient,
    )
    compiled_gradients = torch.autograd.grad(
        compiled,
        (compiled_inputs[0], *compiled_inputs[2:]),
        gradient,
    )
    torch.testing.assert_close(compiled, eager, rtol=2e-5, atol=2e-5)
    for actual, expected in zip(compiled_gradients, eager_gradients):
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=7e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_linear_controller_receives_compiled_coordinate_gradient() -> None:
    coordinates, generators, scale, drive, initial = _inputs("cuda")
    batch, length, channels = coordinates.shape[:3]
    controller = torch.nn.Linear(6, channels * 28, device="cuda")
    features = torch.randn(batch, length, 6, device="cuda")
    controlled_coordinates = 0.05 * controller(features).reshape(
        batch, length, channels, 28
    )
    output = triton_factorized_coordinate_scan(
        controlled_coordinates,
        generators,
        scale,
        drive,
        initial,
    )
    weight_gradient, bias_gradient = torch.autograd.grad(
        output.square().mean(),
        (controller.weight, controller.bias),
    )
    assert bool(torch.isfinite(weight_gradient).all())
    assert bool(torch.isfinite(bias_gradient).all())
    assert float(weight_gradient.norm()) > 0.0
    assert float(bias_gradient.norm()) > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_fused_linear_controller_matches_staged_controller_gradients() -> None:
    torch.manual_seed(20_260_831)
    batch, length, channels, input_size = 2, 7, 3, 6
    features_a = torch.randn(
        batch, length, input_size, device="cuda", requires_grad=True
    )
    weight_a = (
        0.03
        * torch.randn(channels * 28, input_size, device="cuda")
    ).requires_grad_()
    bias_a = (
        0.01 * torch.randn(channels * 28, device="cuda")
    ).requires_grad_()
    _, generators, scale, drive, initial = _inputs("cuda")
    scale_a = scale.detach().clone().requires_grad_()
    drive_a = drive.detach().clone().requires_grad_()
    initial_a = initial.detach().clone().requires_grad_()
    gate = torch.tensor(
        [[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0] * length],
        device="cuda",
    )
    features_b = features_a.detach().clone().requires_grad_()
    weight_b = weight_a.detach().clone().requires_grad_()
    bias_b = bias_a.detach().clone().requires_grad_()
    scale_b = scale_a.detach().clone().requires_grad_()
    drive_b = drive_a.detach().clone().requires_grad_()
    initial_b = initial_a.detach().clone().requires_grad_()

    coordinates = torch.nn.functional.linear(
        features_a, weight_a, bias_a
    ).reshape(batch, length, channels, 28)
    coordinates = coordinates * gate[..., None, None]
    staged = triton_factorized_coordinate_scan(
        coordinates, generators, scale_a, drive_a, initial_a
    )
    fused = triton_controller_factorized_scan(
        features_b,
        weight_b,
        bias_b,
        generators,
        scale_b,
        drive_b,
        initial_b,
        gate,
    )
    gradient = torch.randn_like(staged)
    staged_gradients = torch.autograd.grad(
        staged,
        (features_a, weight_a, bias_a, scale_a, drive_a, initial_a),
        gradient,
    )
    fused_gradients = torch.autograd.grad(
        fused,
        (features_b, weight_b, bias_b, scale_b, drive_b, initial_b),
        gradient,
    )
    torch.testing.assert_close(fused, staged, rtol=3e-5, atol=3e-5)
    for actual, expected in zip(fused_gradients, staged_gradients):
        torch.testing.assert_close(actual, expected, rtol=6e-4, atol=1e-3)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "compiled_mode",
    ["compiled_factorized", "compiled_controller", "compiled_auto"],
)
def test_maintained_layer_compiled_factorized_full_gradient_parity(
    device: str,
    compiled_mode: str,
) -> None:
    torch.manual_seed(20_260_828)
    eager_layer = PureSpin8SSMLayer(
        6, channels=2, action_mode="factorized"
    ).to(device)
    compiled_layer = copy.deepcopy(eager_layer)
    eager_input = torch.randn(2, 9, 6, device=device, requires_grad=True)
    compiled_input = eager_input.detach().clone().requires_grad_()
    valid_mask = torch.tensor(
        [[True, True, True, True, False, False, False, False, False], [True] * 9],
        device=device,
    )
    eager, _ = eager_layer(
        eager_input,
        valid_mask=valid_mask,
        scan_mode="recurrent",
        return_raw_states=True,
    )
    compiled, _ = compiled_layer(
        compiled_input,
        valid_mask=valid_mask,
        scan_mode=compiled_mode,
        return_raw_states=True,
    )
    eager_parameters = tuple(
        parameter
        for name, parameter in eager_layer.named_parameters()
        if name != "coupling_logits"
    )
    compiled_parameters = tuple(
        parameter
        for name, parameter in compiled_layer.named_parameters()
        if name != "coupling_logits"
    )
    gradient = torch.randn_like(eager)
    eager_gradients = torch.autograd.grad(
        eager, (eager_input, *eager_parameters), gradient
    )
    compiled_gradients = torch.autograd.grad(
        compiled, (compiled_input, *compiled_parameters), gradient
    )
    tolerance = 2e-12 if device == "cpu" else 8e-4
    torch.testing.assert_close(compiled, eager, rtol=tolerance, atol=tolerance)
    for actual, expected in zip(compiled_gradients, eager_gradients):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=tolerance,
            atol=tolerance,
        )


def test_training_compiler_preserves_controller_reuse_on_profiled_shape() -> None:
    profile = (
        Path(__file__).resolve().parent
        / "experiments"
        / "artifacts"
        / "spin8_factorized_training_rtx2070s_20260821.json"
    )
    plan = compile_spin8_training_plan(
        RuntimeShape(4, 128, "float32", "cuda", training=True),
        HardwareTarget(
            "NVIDIA GeForce RTX 2070 SUPER", "cuda", (7, 5), True, False
        ),
        channels=4,
        input_size=16,
        hardware_profile_path=profile,
    )
    assert plan.lowering == "direct_factor_compiled_scan"
    assert plan.profile_median_microseconds is not None
    assert "reuse" in plan.reason
