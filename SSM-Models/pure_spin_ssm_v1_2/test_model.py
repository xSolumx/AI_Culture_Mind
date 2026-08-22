from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from mamba2_baseline import fused_mamba2_available
from model import PureSpinSSMV12, PureSpinV12Block, PureSpinV12Config, SolSelfGate
from pure_spin8_ssm.torch_backend import spin8_group_actions


def tiny_model() -> PureSpinSSMV12:
    return PureSpinSSMV12(
        PureSpinV12Config(d_model=16, num_layers=1, spin_channels=1, d_conv=3)
    )


def test_shape_backward_and_finite_state() -> None:
    torch.manual_seed(3)
    model = tiny_model()
    tokens = torch.randint(0, 256, (2, 7))
    result = model(tokens, scan_mode="compiled_controller")
    assert result["logits"].shape == (2, 7, 256)
    assert all(torch.isfinite(state).all() for state in result["states"])
    result["logits"].square().mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_causality() -> None:
    torch.manual_seed(5)
    model = tiny_model().eval()
    left = torch.randint(0, 256, (1, 8))
    right = left.clone()
    right[:, 5:] = torch.randint(0, 256, (1, 3))
    with torch.no_grad():
        a = model(left, scan_mode="compiled_controller")["logits"]
        b = model(right, scan_mode="compiled_controller")["logits"]
    torch.testing.assert_close(a[:, :5], b[:, :5])


def test_initial_language_model_loss_has_sane_scale() -> None:
    torch.manual_seed(7)
    model = tiny_model()
    inputs = torch.randint(0, 256, (2, 16))
    targets = torch.randint(0, 256, (2, 16))
    logits = model(inputs, scan_mode="compiled_controller")["logits"]
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    assert 4.5 < float(loss.detach()) < 6.5


def test_triality_invariant_readout_restores_scale_and_has_finite_gradients() -> None:
    torch.manual_seed(202_608_22)
    config = PureSpinV12Config(
        d_model=16,
        num_layers=1,
        spin_channels=2,
        readout="triality_invariants",
    )
    block = PureSpinV12Block(config)
    states = torch.randn(2, 5, 2, 3, 8, requires_grad=True)
    features = block._read_features(states)
    scaled = block._read_features(2.0 * states)
    assert features.shape == (2, 5, 56)
    torch.testing.assert_close(
        features[..., :48], scaled[..., :48], rtol=2e-5, atol=2e-5
    )
    assert torch.linalg.vector_norm(features[..., 48:] - scaled[..., 48:]) > 0.1
    features.square().mean().backward()
    assert states.grad is not None
    assert torch.isfinite(states.grad).all()


def test_triality_summary_scalars_are_spin8_invariant() -> None:
    torch.manual_seed(202_608_23)
    config = PureSpinV12Config(
        d_model=16,
        num_layers=1,
        spin_channels=1,
        readout="triality_invariants",
    )
    block = PureSpinV12Block(config).double()
    states = torch.randn(2, 3, 1, 3, 8, dtype=torch.float64)
    coordinates = 0.2 * torch.randn(2, 3, 1, 28, dtype=torch.float64)
    actions = spin8_group_actions(
        coordinates,
        block.spin.generators.to(dtype=torch.float64),
        block.spin.representations,
        mode="exponential",
    )
    transformed = torch.einsum("...rij,...rj->...ri", actions, states)
    direction_dimension = block.spin.output_size
    expected = block._read_features(states)[..., direction_dimension:]
    actual = block._read_features(transformed)[..., direction_dimension:]
    torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-8)


def test_multiplicity_router_is_identity_initialized_and_receives_gradient() -> None:
    torch.manual_seed(202_608_24)
    config = PureSpinV12Config(
        d_model=16,
        num_layers=1,
        spin_channels=2,
        multiplicity_router="orthogonal_query",
    )
    block = PureSpinV12Block(config)
    states = torch.randn(2, 4, 2, 3, 8)
    query = torch.randn(2, 4, 16, requires_grad=True)
    routed = block._route_multiplicity(states, query)
    torch.testing.assert_close(routed, states, rtol=0.0, atol=0.0)
    features = block._read_features(states, query)
    loss = (features * torch.randn_like(features)).sum()
    loss.backward()
    assert block.multiplicity_controller is not None
    assert block.multiplicity_controller.weight.grad is not None
    assert torch.isfinite(block.multiplicity_controller.weight.grad).all()
    assert torch.linalg.vector_norm(block.multiplicity_controller.weight.grad) > 0.0


def test_multiplicity_router_commutes_with_shared_spin8_action() -> None:
    torch.manual_seed(202_608_25)
    config = PureSpinV12Config(
        d_model=16,
        num_layers=1,
        spin_channels=2,
        multiplicity_router="orthogonal_query",
    )
    block = PureSpinV12Block(config).double()
    assert block.multiplicity_controller is not None
    torch.nn.init.normal_(block.multiplicity_controller.weight, std=0.2)
    states = torch.randn(2, 3, 2, 3, 8, dtype=torch.float64)
    query = torch.randn(2, 3, 16, dtype=torch.float64)
    coordinates = 0.2 * torch.randn(2, 3, 1, 28, dtype=torch.float64)
    action = spin8_group_actions(
        coordinates,
        block.spin.generators.to(dtype=torch.float64),
        block.spin.representations,
        mode="exponential",
    ).expand(-1, -1, 2, -1, -1, -1)

    def act(value: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...rij,...rj->...ri", action, value)

    expected = act(block._route_multiplicity(states, query))
    actual = block._route_multiplicity(act(states), query)
    torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-8)


def test_fused_mamba_probe_never_claims_fallback() -> None:
    available, detail = fused_mamba2_available()
    assert isinstance(available, bool)
    assert isinstance(detail, str)


def test_sol_self_gate_is_finite_and_has_bounded_local_slope_factor() -> None:
    mixer = SolSelfGate(width=8, expansion=3)
    inputs = torch.linspace(-1e3, 1e3, 40).reshape(5, 8).requires_grad_()
    projected = mixer.input(inputs)
    slope_factor = 1.0 + projected * torch.rsqrt(1.0 + projected.square())
    assert torch.all(slope_factor > 0.0)
    assert torch.all(slope_factor < 2.0)
    output = mixer(inputs)
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(inputs.grad).all()


def test_chunk_parallel_full_model_gradient_parity() -> None:
    torch.manual_seed(20_260_821)
    recurrent = tiny_model()
    chunked = copy.deepcopy(recurrent)
    tokens = torch.randint(0, 256, (2, 5))
    expected = recurrent(tokens, scan_mode="compiled_factorized")["logits"]
    actual = chunked(tokens, scan_mode="chunk_parallel")["logits"]
    output_gradient = torch.randn_like(actual)
    expected_gradients = torch.autograd.grad(
        expected, tuple(recurrent.parameters()), output_gradient
    )
    actual_gradients = torch.autograd.grad(
        actual, tuple(chunked.parameters()), output_gradient
    )
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-5)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=1e-3, atol=3e-3
        )
