"""Acceptance gates for isotypic-to-silicon compiler v2.1.1."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from pure_spin8_ssm.compiler import (
    COMPILER_VERSION,
    HardwareTarget,
    IsotypicBlock,
    RuntimeShape,
    blocks_from_exact_certificate,
    compile_isotypic_plan,
    spin8_triality_blocks,
)
from pure_spin8_ssm.continuous_scan import (
    eager_continuous_spin8_scan,
    triton_continuous_is_available,
    triton_scalar_continuous_spin8_scan,
    triton_tensor_core_continuous_spin8_scan,
)
from pure_spin8_ssm.self_calibrating_ssm import SelfCalibratingSpin8SSMLayer
from pure_spin8_ssm.self_calibration import (
    complete_oriented_so8_frame,
    spin8_actions_from_seven_probes,
)
from pure_spin8_ssm.torch_backend import PureSpin8SSMLayer, spin8_factorized_actions
from spin8_triality import TRIALITY_REPRESENTATIONS, torch_triality_generators

ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = (
    ROOT
    / "Spin-Space-Research"
    / "artifacts"
    / "reducible_isotypic_decomposition_20260811.json"
)
PROFILE = (
    Path(__file__).resolve().parent
    / "experiments"
    / "artifacts"
    / "spin8_compiler_v211_rtx2070s_20260821.json"
)


def _random_triality(batch: int, *, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20_260_821)
    coordinates = 0.2 * torch.randn(batch, 28, generator=generator, dtype=dtype)
    generators = torch_triality_generators(dtype=dtype)
    return spin8_factorized_actions(
        coordinates, generators, TRIALITY_REPRESENTATIONS
    )


def _vector_probes(actions: torch.Tensor) -> torch.Tensor:
    return actions[..., 0, :, :7].transpose(-1, -2)


def test_seven_hodge_probes_reconstruct_oriented_action() -> None:
    actions = _random_triality(5, dtype=torch.float64)
    reconstructed = complete_oriented_so8_frame(
        _vector_probes(actions), project=False
    )
    torch.testing.assert_close(reconstructed, actions[:, 0], rtol=0, atol=1e-12)
    identity = torch.eye(8, dtype=torch.float64)
    torch.testing.assert_close(
        reconstructed.transpose(-1, -2) @ reconstructed,
        identity.expand(5, 8, 8),
        rtol=0,
        atol=2e-12,
    )
    torch.testing.assert_close(
        torch.linalg.det(reconstructed),
        torch.ones(5, dtype=torch.float64),
        rtol=0,
        atol=2e-12,
    )


def test_givens_lift_recovers_triality_after_one_kernel_bit() -> None:
    true_actions = _random_triality(4, dtype=torch.float64)
    probes = _vector_probes(true_actions)
    generators = torch_triality_generators(dtype=torch.float64)
    canonical, _ = spin8_actions_from_seven_probes(
        probes, generators, project=False
    )
    plus_error = (canonical[..., 1, :, :] - true_actions[..., 1, :, :]).abs().amax(
        dim=(-1, -2)
    )
    minus_error = (canonical[..., 1, :, :] + true_actions[..., 1, :, :]).abs().amax(
        dim=(-1, -2)
    )
    lift_sign = torch.where(plus_error <= minus_error, 1.0, -1.0)
    recovered, triangular = spin8_actions_from_seven_probes(
        probes,
        generators,
        lift_sign=lift_sign,
        project=False,
    )
    torch.testing.assert_close(recovered, true_actions, rtol=0, atol=3e-12)
    torch.testing.assert_close(
        triangular,
        torch.eye(8, dtype=torch.float64).expand(4, 8, 8),
        rtol=0,
        atol=2e-12,
    )


def test_lift_bit_changes_both_spinors_and_not_vector() -> None:
    true_actions = _random_triality(2, dtype=torch.float64)
    probes = _vector_probes(true_actions)
    generators = torch_triality_generators(dtype=torch.float64)
    positive, _ = spin8_actions_from_seven_probes(
        probes, generators, lift_sign=torch.ones(2, dtype=torch.float64)
    )
    negative, _ = spin8_actions_from_seven_probes(
        probes, generators, lift_sign=-torch.ones(2, dtype=torch.float64)
    )
    torch.testing.assert_close(positive[:, 0], negative[:, 0], rtol=0, atol=1e-12)
    torch.testing.assert_close(positive[:, 1:], -negative[:, 1:], rtol=0, atol=1e-12)


def test_qr_projection_is_oriented_and_differentiable() -> None:
    actions = _random_triality(3, dtype=torch.float64)
    probes = (_vector_probes(actions) + 0.01 * torch.randn(3, 7, 8)).requires_grad_()
    projected = complete_oriented_so8_frame(probes, project="qr")
    loss = projected.square().sum() + 0.01 * projected.sum()
    (gradient,) = torch.autograd.grad(loss, (probes,))
    identity = torch.eye(8, dtype=torch.float64)
    torch.testing.assert_close(
        projected.transpose(-1, -2) @ projected,
        identity.expand(3, 8, 8),
        rtol=0,
        atol=2e-12,
    )
    assert bool(torch.isfinite(gradient).all())


def test_exact_certificate_compiles_only_verified_signatures() -> None:
    blocks = blocks_from_exact_certificate(
        CERTIFICATE, "spin9_v1_plus_two_v5", shared_action=True
    )
    assert [(b.schur_type, b.multiplicity, b.irreducible_real_dimension) for b in blocks] == [
        ("real", 1, 1),
        ("real", 2, 5),
    ]
    assert [block.expected_commutant_dimension for block in blocks] == [1, 4]


def test_compiler_refuses_false_tensor_multiplicity() -> None:
    block = IsotypicBlock("8v", "real", 8, 16, shared_action=False)
    plan = compile_isotypic_plan(
        (block,),
        RuntimeShape(32, 128, "float16", "cuda", training=False),
        HardwareTarget(
            "NVIDIA GeForce RTX 2070 SUPER", "cuda", (7, 5), True, False
        ),
        hardware_profile_path=PROFILE,
    )
    assert plan.compiler_version == COMPILER_VERSION == "2.1.1"
    assert plan.schedules[0].backend != "triton_tensor_core"
    assert "actions differ" in plan.schedules[0].reason


def test_exact_profile_selects_only_recorded_winning_cell() -> None:
    hardware = HardwareTarget(
        "NVIDIA GeForce RTX 2070 SUPER", "cuda", (7, 5), True, False
    )
    blocks = spin8_triality_blocks(16, shared_action=True)
    winning = compile_isotypic_plan(
        blocks,
        RuntimeShape(32, 128, "float16", "cuda", training=False),
        hardware,
        hardware_profile_path=PROFILE,
    )
    losing = compile_isotypic_plan(
        blocks,
        RuntimeShape(8, 1024, "float16", "cuda", training=False),
        hardware,
        hardware_profile_path=PROFILE,
    )
    assert {row.backend for row in winning.schedules} == {"triton_tensor_core"}
    assert {row.backend for row in losing.schedules} == {"triton_scalar"}


def test_failed_hardware_profile_cannot_select_tensor_cores(
    tmp_path: Path,
) -> None:
    payload = json.loads(PROFILE.read_text(encoding="utf-8"))
    payload["passed"] = False
    failed_profile = tmp_path / "failed-profile.json"
    failed_profile.write_text(json.dumps(payload), encoding="utf-8")
    hardware = HardwareTarget(
        "NVIDIA GeForce RTX 2070 SUPER", "cuda", (7, 5), True, False
    )
    plan = compile_isotypic_plan(
        spin8_triality_blocks(16, shared_action=True),
        RuntimeShape(32, 128, "float16", "cuda", training=False),
        hardware,
        hardware_profile_path=failed_profile,
    )
    assert {row.backend for row in plan.schedules} == {"triton_scalar"}
    assert all(
        "no exact device/shape profile cell" in row.reason
        for row in plan.schedules
    )


def test_self_calibrating_layer_cpu_matches_explicit_actions() -> None:
    batch, length, channels = 2, 5, 3
    true_actions = _random_triality(batch * length, dtype=torch.float64).reshape(
        batch, length, 3, 8, 8
    )
    probes = _vector_probes(true_actions)
    generators = torch_triality_generators(dtype=torch.float64)
    canonical, _ = spin8_actions_from_seven_probes(
        probes, generators, project=False
    )
    plus_error = (canonical[..., 1, :, :] - true_actions[..., 1, :, :]).abs().amax(
        dim=(-1, -2)
    )
    minus_error = (canonical[..., 1, :, :] + true_actions[..., 1, :, :]).abs().amax(
        dim=(-1, -2)
    )
    lift_sign = torch.where(plus_error <= minus_error, 1.0, -1.0)
    layer = SelfCalibratingSpin8SSMLayer(
        channels=channels, projection="none", hardware_profile=None
    ).double()
    actual, _ = layer(probes, lift_sign, backend="eager")
    state = layer.initial_cache(batch, probes)
    rows = []
    for position in range(length):
        state = torch.einsum(
            "brij,bcrj->bcri", true_actions[:, position], state
        )
        rows.append(state)
    expected = torch.stack(rows, dim=1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=6e-12)


@pytest.mark.skipif(
    not triton_continuous_is_available(), reason="CUDA Triton is unavailable"
)
@pytest.mark.parametrize("shared", [True, False])
def test_scalar_triton_full_gradient_parity(shared: bool) -> None:
    torch.manual_seed(20_260_823)
    batch, length, channels, representations = 2, 9, 3, 3
    coordinates = 0.08 * torch.randn(
        batch * length * (1 if shared else channels), 28, device="cuda"
    )
    generated = spin8_factorized_actions(
        coordinates,
        torch_triality_generators(device="cuda"),
        TRIALITY_REPRESENTATIONS,
    )
    action_shape = (
        (batch, length, representations, 8, 8)
        if shared
        else (batch, length, channels, representations, 8, 8)
    )
    action_a = generated.reshape(action_shape).detach().requires_grad_()
    scale_a = (0.8 + 0.1 * torch.rand(batch, length, channels, device="cuda")).requires_grad_()
    drive_a = (0.01 * torch.randn(batch, length, channels, representations, 8, device="cuda")).requires_grad_()
    initial_a = torch.randn(batch, channels, representations, 8, device="cuda", requires_grad=True)
    action_b = action_a.detach().clone().requires_grad_()
    scale_b = scale_a.detach().clone().requires_grad_()
    drive_b = drive_a.detach().clone().requires_grad_()
    initial_b = initial_a.detach().clone().requires_grad_()

    eager = eager_continuous_spin8_scan(action_a, scale_a, drive_a, initial_a)
    compiled = triton_scalar_continuous_spin8_scan(
        action_b, scale_b, drive_b, initial_b
    )
    gradient = torch.randn_like(eager)
    eager_gradients = torch.autograd.grad(
        eager, (action_a, scale_a, drive_a, initial_a), gradient
    )
    compiled_gradients = torch.autograd.grad(
        compiled, (action_b, scale_b, drive_b, initial_b), gradient
    )
    torch.testing.assert_close(compiled, eager, rtol=2e-5, atol=2e-5)
    for actual, expected in zip(compiled_gradients, eager_gradients):
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=5e-4)


@pytest.mark.skipif(
    not triton_continuous_is_available(), reason="CUDA Triton is unavailable"
)
def test_tensor_core_shared_action_forward_parity() -> None:
    torch.manual_seed(20_260_824)
    batch, length, channels = 4, 17, 16
    coordinates = 0.03 * torch.randn(batch * length, 28, device="cuda")
    action = spin8_factorized_actions(
        coordinates,
        torch_triality_generators(device="cuda"),
        TRIALITY_REPRESENTATIONS,
    ).reshape(batch, length, 3, 8, 8).half()
    scale = torch.full((batch, length, channels), 0.95, device="cuda", dtype=torch.float16)
    drive = 0.005 * torch.randn(batch, length, channels, 3, 8, device="cuda", dtype=torch.float16)
    initial = torch.randn(batch, channels, 3, 8, device="cuda", dtype=torch.float16)
    eager = eager_continuous_spin8_scan(action, scale, drive, initial)
    compiled = triton_tensor_core_continuous_spin8_scan(
        action, scale, drive, initial
    )
    torch.testing.assert_close(compiled, eager, rtol=0.02, atol=0.02)


@pytest.mark.skipif(
    not triton_continuous_is_available(), reason="CUDA Triton is unavailable"
)
def test_maintained_layer_compiled_recurrent_full_gradient_integration() -> None:
    torch.manual_seed(20_260_827)
    layer = PureSpin8SSMLayer(6, channels=2).cuda()
    first = torch.randn(2, 17, 6, device="cuda", requires_grad=True)
    second = first.detach().clone().requires_grad_()
    eager, _ = layer(first, scan_mode="recurrent", return_raw_states=True)
    compiled, _ = layer(
        second, scan_mode="compiled_recurrent", return_raw_states=True
    )
    eager_gradient = torch.autograd.grad(eager.square().mean(), first)[0]
    compiled_gradient = torch.autograd.grad(compiled.square().mean(), second)[0]
    torch.testing.assert_close(compiled, eager, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        compiled_gradient, eager_gradient, rtol=2e-4, atol=2e-5
    )
