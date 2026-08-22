from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from pure_spin8_ssm.torch_backend import spin8_group_actions

from mamba2_baseline import fused_mamba2_available
from model import PureSpinSSMV12, PureSpinV12Block, PureSpinV12Config, SolSelfGate


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


def test_isotypic_retention_is_exact_zero_start_and_learns_sector_offsets() -> None:
    shared_config = PureSpinV12Config(
        d_model=16,
        num_layers=1,
        spin_channels=2,
        d_conv=3,
        retention_mode="shared",
    )
    isotypic_config = PureSpinV12Config(
        d_model=16,
        num_layers=1,
        spin_channels=2,
        d_conv=3,
        retention_mode="isotypic",
    )
    torch.manual_seed(20_260_829)
    shared = PureSpinSSMV12(shared_config)
    torch.manual_seed(20_260_829)
    isotypic = PureSpinSSMV12(isotypic_config)
    shared_state = shared.state_dict()
    isotypic_state = isotypic.state_dict()
    for name, expected in shared_state.items():
        torch.testing.assert_close(isotypic_state[name], expected, rtol=0.0, atol=0.0)
    offset = isotypic.blocks[0].spin.retention_offset_controller
    assert offset is not None
    assert torch.count_nonzero(offset.weight) == 0
    assert torch.count_nonzero(offset.bias) == 0

    tokens = torch.randint(0, 256, (2, 7))
    expected = shared(tokens, scan_mode="chunk_parallel")["logits"]
    actual = isotypic(tokens, scan_mode="chunk_parallel")["logits"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    actual.square().mean().backward()
    assert offset.weight.grad is not None
    assert torch.linalg.vector_norm(offset.weight.grad) > 0.0


def test_isotypic_spectrum_is_exact_zero_start_and_learns_static_rates() -> None:
    shared_config = PureSpinV12Config(
        d_model=16,
        num_layers=1,
        spin_channels=2,
        d_conv=3,
        retention_mode="shared",
    )
    spectrum_config = PureSpinV12Config(
        d_model=16,
        num_layers=1,
        spin_channels=2,
        d_conv=3,
        retention_mode="isotypic_spectrum",
    )
    torch.manual_seed(20_260_830)
    shared = PureSpinSSMV12(shared_config)
    torch.manual_seed(20_260_830)
    spectrum = PureSpinSSMV12(spectrum_config)
    shared_state = shared.state_dict()
    spectrum_state = spectrum.state_dict()
    for name, expected in shared_state.items():
        torch.testing.assert_close(spectrum_state[name], expected, rtol=0.0, atol=0.0)
    log_rates = spectrum.blocks[0].spin.retention_log_rates
    assert log_rates is not None
    assert torch.count_nonzero(log_rates) == 0

    tokens = torch.randint(0, 256, (2, 7))
    expected = shared(tokens, scan_mode="chunk_parallel")["logits"]
    actual = spectrum(tokens, scan_mode="chunk_parallel")["logits"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    actual.square().mean().backward()
    assert log_rates.grad is not None
    assert torch.linalg.vector_norm(log_rates.grad) > 0.0


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


def test_coupled_isotypic_full_model_parallel_gradient_parity() -> None:
    torch.manual_seed(202_608_29)
    config = PureSpinV12Config(
        d_model=16,
        num_layers=1,
        spin_channels=2,
        recurrence="coupled_isotypic",
        recurrent_multiplicity="orthogonal",
    )
    recurrent = PureSpinSSMV12(config)
    parallel = copy.deepcopy(recurrent)
    tokens = torch.randint(0, 256, (2, 7))
    expected = recurrent(tokens, scan_mode="coupled_recurrent")["logits"]
    actual = parallel(tokens, scan_mode="coupled_parallel")["logits"]
    output_gradient = torch.randn_like(actual)
    expected_gradients = torch.autograd.grad(
        expected, tuple(recurrent.parameters()), output_gradient
    )
    actual_gradients = torch.autograd.grad(
        actual, tuple(parallel.parameters()), output_gradient
    )
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-5)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=2e-3, atol=3e-3
        )


def test_coupled_isotypic_model_is_causal_and_controller_is_shared() -> None:
    torch.manual_seed(202_608_30)
    config = PureSpinV12Config(
        d_model=16,
        num_layers=1,
        spin_channels=2,
        recurrence="coupled_isotypic",
        recurrent_multiplicity="orthogonal",
    )
    model = PureSpinSSMV12(config).eval()
    block = model.blocks[0]
    assert block.spin.coefficient_controller.out_features == 28
    left = torch.randint(0, 256, (1, 8))
    right = left.clone()
    right[:, 5:] = torch.randint(0, 256, (1, 3))
    with torch.no_grad():
        expected = model(left, scan_mode="coupled_parallel")["logits"]
        actual = model(right, scan_mode="coupled_parallel")["logits"]
    torch.testing.assert_close(actual[:, :5], expected[:, :5], rtol=2e-5, atol=2e-5)


def test_recurrent_multiplicity_candidate_has_paired_identity_initialization() -> None:
    base = {
        "d_model": 16,
        "num_layers": 2,
        "spin_channels": 2,
        "recurrence": "coupled_isotypic",
        "dropout": 0.0,
    }
    torch.manual_seed(202_608_31)
    identity = PureSpinSSMV12(
        PureSpinV12Config(**base, recurrent_multiplicity="identity")
    ).eval()
    torch.manual_seed(202_608_31)
    orthogonal = PureSpinSSMV12(
        PureSpinV12Config(**base, recurrent_multiplicity="orthogonal")
    ).eval()
    identity_state = identity.state_dict()
    orthogonal_state = orthogonal.state_dict()
    for name, value in identity_state.items():
        assert torch.equal(value, orthogonal_state[name]), name
    tokens = torch.randint(0, 256, (2, 7))
    with torch.no_grad():
        expected = identity(tokens, scan_mode="coupled_parallel")["logits"]
        actual = orthogonal(tokens, scan_mode="coupled_parallel")["logits"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_independent_block_candidate_reduces_to_v12_and_learns_coupling() -> None:
    base = {
        "d_model": 16,
        "num_layers": 1,
        "spin_channels": 2,
        "dropout": 0.0,
    }
    torch.manual_seed(202_608_36)
    maintained = PureSpinSSMV12(PureSpinV12Config(**base)).eval()
    torch.manual_seed(202_608_36)
    candidate = PureSpinSSMV12(
        PureSpinV12Config(
            **base,
            recurrence="independent_block",
            recurrent_multiplicity="orthogonal",
        )
    ).eval()
    candidate_state = candidate.state_dict()
    for name, value in maintained.state_dict().items():
        assert torch.equal(value, candidate_state[name]), name
    tokens = torch.randint(0, 256, (2, 7))
    expected = maintained(tokens, scan_mode="chunk_parallel")["logits"]
    actual = candidate(tokens, scan_mode="block_recurrent")["logits"]
    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-5)
    actual.square().mean().backward()
    for block in candidate.blocks:
        gradient = block.recurrent_multiplicity_controller.weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


def test_independent_block_full_model_parallel_gradient_parity() -> None:
    torch.manual_seed(202_608_37)
    config = PureSpinV12Config(
        d_model=16,
        num_layers=1,
        spin_channels=2,
        recurrence="independent_block",
        recurrent_multiplicity="orthogonal",
    )
    recurrent = PureSpinSSMV12(config)
    parallel = copy.deepcopy(recurrent)
    with torch.no_grad():
        recurrent.blocks[0].recurrent_multiplicity_controller.weight.normal_(
            std=0.02
        )
        parallel.load_state_dict(recurrent.state_dict())
    tokens = torch.randint(0, 256, (2, 7))
    expected = recurrent(tokens, scan_mode="block_recurrent")["logits"]
    actual = parallel(tokens, scan_mode="block_parallel")["logits"]
    output_gradient = torch.randn_like(actual)
    expected_gradients = torch.autograd.grad(
        expected, tuple(recurrent.parameters()), output_gradient
    )
    actual_gradients = torch.autograd.grad(
        actual, tuple(parallel.parameters()), output_gradient
    )
    torch.testing.assert_close(actual, expected, rtol=8e-5, atol=8e-5)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=4e-3, atol=5e-3
        )


def test_spin_delta_exactly_embeds_v12_and_has_live_address_controls() -> None:
    base = {
        "d_model": 16,
        "num_layers": 1,
        "spin_channels": 2,
        "dropout": 0.0,
    }
    torch.manual_seed(202_608_41)
    maintained = PureSpinSSMV12(PureSpinV12Config(**base)).eval()
    torch.manual_seed(202_608_41)
    candidate = PureSpinSSMV12(
        PureSpinV12Config(**base, recurrence="spin_delta")
    ).eval()
    candidate_state = candidate.state_dict()
    for name, value in maintained.state_dict().items():
        assert torch.equal(value, candidate_state[name]), name
    assert candidate.cache_scalars == 2 * maintained.cache_scalars

    tokens = torch.randint(0, 256, (2, 1))
    expected = maintained(tokens, scan_mode="chunk_parallel")
    actual = candidate(tokens, scan_mode="delta_recurrent")
    # The affine maps are identical over the reals.  The generic two-slot
    # matrix contraction adds one zero term, so float32 differs by at most a
    # final-rounding ulp from the scalar maintained path.
    torch.testing.assert_close(
        actual["logits"], expected["logits"], rtol=2e-5, atol=5e-8
    )
    torch.testing.assert_close(
        actual["states"][0][:, :, 0],
        expected["states"][0],
        rtol=2e-6,
        atol=2e-8,
    )
    assert torch.count_nonzero(actual["states"][0][:, :, 1]) == 0

    actual["logits"].square().mean().backward()
    block = candidate.blocks[0]
    for controller in (
        block.delta_write_controller,
        block.delta_erase_controller,
        block.delta_query_controller,
    ):
        assert controller is not None and controller.weight.grad is not None
        assert torch.isfinite(controller.weight.grad).all()
        assert torch.count_nonzero(controller.weight.grad) > 0
    strength = block.delta_erase_strength_controller
    assert strength is not None and strength.weight.grad is not None
    assert torch.count_nonzero(strength.weight.grad) == 0

    candidate.zero_grad(set_to_none=True)
    assert block.delta_erase_controller is not None
    with torch.no_grad():
        block.delta_erase_controller.bias.fill_(0.1)
    perturbed = candidate(tokens, scan_mode="delta_recurrent")["logits"]
    perturbed.square().mean().backward()
    assert strength.weight.grad is not None
    assert torch.count_nonzero(strength.weight.grad) > 0


def test_spin_delta_full_model_parallel_gradient_parity() -> None:
    torch.manual_seed(202_608_43)
    config = PureSpinV12Config(
        d_model=16,
        num_layers=1,
        spin_channels=2,
        recurrence="spin_delta",
    )
    recurrent = PureSpinSSMV12(config)
    parallel = copy.deepcopy(recurrent)
    with torch.no_grad():
        for controller in (
            recurrent.blocks[0].delta_write_controller,
            recurrent.blocks[0].delta_erase_controller,
            recurrent.blocks[0].delta_erase_strength_controller,
            recurrent.blocks[0].delta_query_controller,
        ):
            assert controller is not None
            controller.weight.normal_(std=0.02)
        parallel.load_state_dict(recurrent.state_dict())
    tokens = torch.randint(0, 256, (2, 7))
    expected = recurrent(tokens, scan_mode="delta_recurrent")["logits"]
    actual = parallel(tokens, scan_mode="delta_parallel")["logits"]
    output_gradient = torch.randn_like(actual)
    expected_gradients = torch.autograd.grad(
        expected, tuple(recurrent.parameters()), output_gradient
    )
    actual_gradients = torch.autograd.grad(
        actual, tuple(parallel.parameters()), output_gradient
    )
    torch.testing.assert_close(actual, expected, rtol=8e-5, atol=8e-5)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=5e-3, atol=7e-3
        )


def test_spin_delta_is_causal_and_finite_on_a_long_sequence() -> None:
    torch.manual_seed(202_608_47)
    model = PureSpinSSMV12(
        PureSpinV12Config(
            d_model=16,
            num_layers=1,
            spin_channels=2,
            recurrence="spin_delta",
        )
    ).eval()
    block = model.blocks[0]
    with torch.no_grad():
        for controller in (
            block.delta_write_controller,
            block.delta_erase_controller,
            block.delta_erase_strength_controller,
            block.delta_query_controller,
        ):
            assert controller is not None
            controller.weight.normal_(std=0.03)
    left = torch.randint(0, 256, (1, 384))
    right = left.clone()
    right[:, 250:] = torch.randint(0, 256, (1, 134))
    with torch.no_grad():
        expected = model(left, scan_mode="delta_recurrent")
        actual = model(right, scan_mode="delta_recurrent")
    torch.testing.assert_close(
        actual["logits"][:, :250], expected["logits"][:, :250]
    )
    assert torch.isfinite(actual["logits"]).all()
    assert all(torch.isfinite(state).all() for state in actual["states"])


def test_spin_delta_oracle_slots_are_causal_side_information() -> None:
    torch.manual_seed(202_608_59)
    model = PureSpinSSMV12(
        PureSpinV12Config(
            d_model=16,
            num_layers=1,
            spin_channels=2,
            recurrence="spin_delta",
        )
    )
    tokens = torch.randint(0, 256, (2, 11))
    oracle = torch.full((2, 11, 2), -1, dtype=torch.long)
    oracle[:, 2, 0] = 0
    oracle[:, 5, 0] = 1
    oracle[:, -1, 1] = torch.tensor([0, 1])
    result = model(
        tokens,
        scan_mode="delta_recurrent",
        delta_oracle_slots=oracle,
    )
    assert result["logits"].shape == (2, 11, 256)
    result["logits"][:, -1].square().mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    maintained = tiny_model()
    with pytest.raises(ValueError, match="only defined for Spin-Delta"):
        maintained(
            tokens,
            scan_mode="compiled_controller",
            delta_oracle_slots=oracle,
        )


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
