"""Acceptance gates for the bounded recurrent structured Spin(8) mixer."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pure_spin8_ssm.torch_backend import (
    recurrent_spin8_scan,
    spin8_factorized_actions,
)
from spin8_triality import TRIALITY_REPRESENTATIONS

from hybrid_memory_v1_4.selected_block import LowRankLinear
from hybrid_memory_v1_4.structured_memory import (
    StructuredMemoryConfig,
    StructuredSpin8Memory,
)
from hybrid_memory_v1_4.structured_tier import generator_mask_for_rung

DTYPE = torch.float64


def _config(**overrides: object) -> StructuredMemoryConfig:
    values = {
        "model_dim": 6,
        "channels": 2,
        "rungs": (3, 4, 6, 8),
        "retention_min": 0.1,
        "retention_max": 0.9,
        "hard_eval": True,
    }
    values.update(overrides)
    return StructuredMemoryConfig(**values)  # type: ignore[arg-type]


def test_faithful_state_accounting_and_bounded_control_fields() -> None:
    torch.manual_seed(30)
    config = _config()
    model = StructuredSpin8Memory(config).double()
    initial = model.initial_state(3)
    assert initial.shape == (3, 2, 3, 8)
    assert torch.count_nonzero(initial) == 0
    assert config.state_shape == (2, 3, 8)
    assert model.state_scalars == model.cache_scalars == 48
    assert model.state_bytes(DTYPE, batch_size=3) == 3 * 48 * 8

    inputs = torch.randn(3, 5, 6, dtype=DTYPE)
    transition, diagnostics = model.compile_transition(inputs)
    assert transition.scale.shape == (3, 5, 2)
    assert transition.action.shape == (3, 5, 2, 3, 8, 8)
    assert transition.drive.shape == (3, 5, 2, 3, 8)
    retention = diagnostics["retention"]
    write_gate = diagnostics["write_gate"]
    bounded_drive = diagnostics["bounded_drive"]
    assert isinstance(retention, torch.Tensor)
    assert isinstance(write_gate, torch.Tensor)
    assert isinstance(bounded_drive, torch.Tensor)
    assert bool((retention >= config.retention_min).all())
    assert bool((retention <= config.retention_max).all())
    assert bool(((write_gate > 0.0) & (write_gate < 1.0)).all())
    assert bool((torch.linalg.vector_norm(bounded_drive, dim=-1) < 1.0).all())
    drive_limit = (1.0 - retention) * write_gate
    assert bool(
        (
            torch.linalg.vector_norm(transition.drive, dim=-1)
            <= drive_limit[..., None] + 1e-15
        ).all()
    )
    assert diagnostics["rung_probabilities"].shape == (3, 5, 2, 4)  # type: ignore[union-attr]
    assert diagnostics["expected_factors"].shape == (3, 5, 2)  # type: ignore[union-attr]


def test_compiled_affine_transition_is_the_forward_recurrence() -> None:
    torch.manual_seed(31)
    model = StructuredSpin8Memory(_config(channels=1)).double()
    inputs = torch.randn(2, 7, 6, dtype=DTYPE)
    initial = 0.2 * torch.randn(2, 1, 3, 8, dtype=DTYPE)
    transition, _ = model.compile_transition(inputs)
    raw_states, expected_state = recurrent_spin8_scan(transition, initial)
    expected_output = model.readout(raw_states)
    output, state = model(inputs, initial, scan_mode="recurrent")
    torch.testing.assert_close(output, expected_output, rtol=0.0, atol=0.0)
    torch.testing.assert_close(state, expected_state, rtol=0.0, atol=0.0)


def test_recurrent_parallel_output_state_and_full_gradient_parity_float64() -> None:
    torch.manual_seed(32)
    recurrent_model = StructuredSpin8Memory(_config(model_dim=5, channels=1)).double()
    parallel_model = copy.deepcopy(recurrent_model)
    recurrent_inputs = torch.randn(2, 9, 5, dtype=DTYPE, requires_grad=True)
    parallel_inputs = recurrent_inputs.detach().clone().requires_grad_()
    recurrent_initial = (0.2 * torch.randn(2, 1, 3, 8, dtype=DTYPE)).requires_grad_()
    parallel_initial = recurrent_initial.detach().clone().requires_grad_()

    recurrent_output, recurrent_state, recurrent_diagnostics = recurrent_model(
        recurrent_inputs,
        recurrent_initial,
        scan_mode="recurrent",
        return_diagnostics=True,
    )
    parallel_output, parallel_state, parallel_diagnostics = parallel_model(
        parallel_inputs,
        parallel_initial,
        scan_mode="parallel",
        return_diagnostics=True,
    )
    torch.testing.assert_close(
        parallel_output, recurrent_output, rtol=2e-11, atol=2e-11
    )
    torch.testing.assert_close(parallel_state, recurrent_state, rtol=2e-11, atol=2e-11)

    output_probe = torch.randn_like(recurrent_output)
    state_probe = torch.randn_like(recurrent_state)

    def loss(
        output: torch.Tensor,
        state: torch.Tensor,
        diagnostics: dict[str, torch.Tensor | str | bool],
    ) -> torch.Tensor:
        expected_factors = diagnostics["expected_factors"]
        assert isinstance(expected_factors, torch.Tensor)
        return (
            (output * output_probe).sum()
            + (state * state_probe).sum()
            + 0.01 * expected_factors.sum()
        )

    loss(recurrent_output, recurrent_state, recurrent_diagnostics).backward()
    loss(parallel_output, parallel_state, parallel_diagnostics).backward()
    torch.testing.assert_close(
        parallel_inputs.grad, recurrent_inputs.grad, rtol=8e-9, atol=8e-10
    )
    torch.testing.assert_close(
        parallel_initial.grad, recurrent_initial.grad, rtol=8e-9, atol=8e-10
    )
    recurrent_parameters = dict(recurrent_model.named_parameters())
    parallel_parameters = dict(parallel_model.named_parameters())
    assert recurrent_parameters.keys() == parallel_parameters.keys()
    for name, expected_parameter in recurrent_parameters.items():
        actual = parallel_parameters[name].grad
        expected = expected_parameter.grad
        assert actual is not None, name
        assert expected is not None, name
        assert torch.isfinite(actual).all(), name
        assert torch.isfinite(expected).all(), name
        torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-9)


def test_hard_eval_transition_uses_one_exact_rung_action() -> None:
    model = StructuredSpin8Memory(_config(model_dim=4, channels=1)).double()
    with torch.no_grad():
        model.tier.rung_controller.weight.zero_()
        model.tier.rung_controller.bias.copy_(
            torch.tensor([0.0, 1.0, 6.0, 2.0], dtype=DTYPE)
        )
        assert isinstance(model.tier.coefficient_controller, torch.nn.Linear)
        model.tier.coefficient_controller.weight.zero_()
        model.tier.coefficient_controller.bias.fill_(0.25)
    model.eval()
    transition, diagnostics = model.compile_transition(
        torch.zeros(2, 3, 4, dtype=DTYPE)
    )
    mask = generator_mask_for_rung(6).to(dtype=DTYPE)
    expected_coordinates = 0.25 * mask.expand(2, 3, 1, 28)
    torch.testing.assert_close(
        diagnostics["transport_coordinates"],  # type: ignore[arg-type]
        expected_coordinates,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(
        diagnostics["rung_probabilities"],  # type: ignore[arg-type]
        torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=DTYPE).expand(2, 3, 1, 4),
    )
    assert torch.equal(
        diagnostics["expected_factors"],  # type: ignore[arg-type]
        torch.full((2, 3, 1), 15.0, dtype=DTYPE),
    )
    expected_actions = spin8_factorized_actions(
        expected_coordinates,
        model.tier.generators.to(dtype=DTYPE),
        TRIALITY_REPRESENTATIONS,
    )
    torch.testing.assert_close(transition.action, expected_actions, rtol=0.0, atol=0.0)
    assert diagnostics["hard_selection"] is True


@pytest.mark.parametrize("mode", ["recurrent", "parallel"])
def test_all_invalid_mask_preserves_state_but_readout_remains_live(mode: str) -> None:
    torch.manual_seed(33)
    model = StructuredSpin8Memory(_config(model_dim=4, channels=1)).double()
    inputs = torch.randn(2, 6, 4, dtype=DTYPE)
    initial = torch.randn(2, 1, 3, 8, dtype=DTYPE)
    outputs, final = model(
        inputs,
        initial,
        valid_mask=torch.zeros(2, 6, dtype=torch.bool),
        scan_mode=mode,  # type: ignore[arg-type]
    )
    torch.testing.assert_close(final, initial, rtol=0.0, atol=0.0)
    expected = model.readout(initial)
    torch.testing.assert_close(
        outputs, expected[:, None].expand_as(outputs), rtol=2e-15, atol=2e-15
    )
    assert torch.isfinite(outputs).all()
    assert bool((outputs.abs().sum(dim=-1) > 0.0).any())


def test_bounded_finite_state_for_4096_recurrent_steps() -> None:
    torch.manual_seed(34)
    config = _config(
        model_dim=3,
        channels=1,
        retention_min=0.05,
        retention_max=0.95,
    )
    model = StructuredSpin8Memory(config).double().eval()
    state = model.initial_state(1)
    maximum_norm = 0.0
    with torch.no_grad():
        for _ in range(32):
            inputs = 10.0 * torch.empty(1, 128, 3, dtype=DTYPE).uniform_(-1.0, 1.0)
            transition, _ = model.compile_transition(inputs)
            states, state = recurrent_spin8_scan(transition, state)
            norms = torch.linalg.vector_norm(states, dim=-1)
            maximum_norm = max(maximum_norm, float(norms.max()))
            assert torch.isfinite(model.readout(states)).all()
            assert torch.isfinite(state).all()
    assert maximum_norm <= 1.0 + 2e-12


def test_readout_has_bounded_zero_state_jacobian() -> None:
    torch.manual_seed(130)
    model = StructuredSpin8Memory(_config()).double()
    states = torch.zeros(
        1, *model.config.state_shape, dtype=torch.float64, requires_grad=True
    )
    output = model.readout(states)
    jacobian = torch.autograd.grad(output.sum(), states)[0]
    assert torch.isfinite(jacobian).all()
    assert float(jacobian.norm()) < 20.0


def test_low_rank_controllers_step_replay_and_diagnostics() -> None:
    torch.manual_seed(35)
    config = _config(model_dim=5, channels=1, controller_rank=2)
    model = StructuredSpin8Memory(config).double()
    assert isinstance(model.controller, LowRankLinear)
    tier_controller = model.tier.coefficient_controller
    assert type(tier_controller).__name__ == "LowRankLinear"
    assert hasattr(tier_controller, "effective_weight")
    assert torch.linalg.matrix_rank(tier_controller.effective_weight()).item() <= 2
    expected_parameters = (
        2 * (config.model_dim + config.controller_width) + config.controller_width
    )
    assert sum(parameter.numel() for parameter in model.controller.parameters()) == (
        expected_parameters
    )
    assert torch.linalg.matrix_rank(model.controller.effective_weight()).item() <= 2

    inputs = torch.randn(2, 7, 5, dtype=DTYPE)
    full_output, full_state = model(inputs, scan_mode="recurrent")
    state = None
    pieces = []
    final_diagnostics: dict[str, torch.Tensor | str | bool] | None = None
    for position in range(inputs.shape[1]):
        output, state, final_diagnostics = model.step(
            inputs[:, position], state, return_diagnostics=True
        )
        pieces.append(output)
    assert state is not None
    assert final_diagnostics is not None
    torch.testing.assert_close(
        torch.stack(pieces, dim=1), full_output, rtol=2e-12, atol=2e-12
    )
    torch.testing.assert_close(state, full_state, rtol=2e-12, atol=2e-12)
    assert final_diagnostics["rung_probabilities"].shape == (2, 1, 4)  # type: ignore[union-attr]
    assert final_diagnostics["expected_factors"].shape == (2, 1)  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"model_dim": 0}, ValueError),
        ({"channels": 0}, ValueError),
        ({"rungs": (3, 3, 8)}, ValueError),
        ({"controller_rank": 0}, ValueError),
        ({"controller_rank": 5, "model_dim": 4}, ValueError),
        ({"retention_min": -0.1}, ValueError),
        ({"retention_max": 1.0}, ValueError),
        ({"hard_eval": 1}, TypeError),
    ],
)
def test_config_guards(overrides: dict[str, object], error: type[Exception]) -> None:
    with pytest.raises(error):
        _config(**overrides)


def test_input_state_mask_mode_and_finite_guards() -> None:
    model = StructuredSpin8Memory(_config()).double()
    good = torch.randn(2, 4, 6, dtype=DTYPE)
    state = model.initial_state(2)
    with pytest.raises(ValueError, match="inputs"):
        model(good[..., :-1])
    with pytest.raises(TypeError, match="floating"):
        model(torch.ones(2, 4, 6, dtype=torch.long))
    with pytest.raises(ValueError, match="match the module"):
        model(good.float())
    with pytest.raises(ValueError, match="finite"):
        invalid = good.clone()
        invalid[0, 0, 0] = torch.nan
        model(invalid)
    with pytest.raises(ValueError, match="state"):
        model(good, state[..., :-1])
    with pytest.raises(ValueError, match="finite"):
        invalid_state = state.clone()
        invalid_state[0, 0, 0, 0] = torch.inf
        model(good, invalid_state)
    with pytest.raises(ValueError, match="valid_mask"):
        model(good, valid_mask=torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(TypeError, match="torch.bool"):
        model(good, valid_mask=torch.ones(2, 4, dtype=DTYPE))
    with pytest.raises(ValueError, match="scan_mode"):
        model(good, scan_mode="work_efficient")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="step inputs"):
        model.step(good)
