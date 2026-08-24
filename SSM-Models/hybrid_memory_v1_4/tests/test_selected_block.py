"""Correctness gates for the semantic selected-block memory reference."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selected_block import (
    FullStateAffineTransition,
    LowRankLinear,
    SelectedBlockConfig,
    SelectedBlockMemory,
    apply_full_state_transition,
    compose_full_state_transitions,
    parallel_full_state_scan,
    recurrent_full_state_scan,
)

DTYPE = torch.float64


def _config(**overrides: object) -> SelectedBlockConfig:
    values = {
        "model_dim": 6,
        "heads": 2,
        "blocks": 3,
        "slots_per_block": 3,
        "value_dim": 4,
        "update_rank": 2,
        "retention_min": 0.2,
        "retention_max": 0.9,
    }
    values.update(overrides)
    return SelectedBlockConfig(**values)  # type: ignore[arg-type]


def _random_transition(batch: int = 2, length: int = 7) -> FullStateAffineTransition:
    linear = 0.1 * torch.randn(batch, length, 2, 4, 4, dtype=DTYPE)
    linear = linear + 0.7 * torch.eye(4, dtype=DTYPE)
    drive = 0.1 * torch.randn(batch, length, 2, 4, 3, dtype=DTYPE)
    return FullStateAffineTransition(linear, drive)


def test_composition_is_associative_and_matches_direct_application() -> None:
    torch.manual_seed(10)
    transition = _random_transition(length=3)
    tokens = [
        FullStateAffineTransition(
            transition.linear[:, index], transition.drive[:, index]
        )
        for index in range(3)
    ]
    first, second, third = tokens
    left = compose_full_state_transitions(
        third, compose_full_state_transitions(second, first)
    )
    right = compose_full_state_transitions(
        compose_full_state_transitions(third, second), first
    )
    state = torch.randn(2, 2, 4, 3, dtype=DTYPE)
    direct = apply_full_state_transition(
        third,
        apply_full_state_transition(second, apply_full_state_transition(first, state)),
    )
    torch.testing.assert_close(left.linear, right.linear, rtol=2e-15, atol=2e-15)
    torch.testing.assert_close(left.drive, right.drive, rtol=2e-15, atol=2e-15)
    torch.testing.assert_close(
        apply_full_state_transition(left, state), direct, rtol=2e-15, atol=2e-15
    )


def test_affine_parallel_and_recurrent_states_and_gradients_match() -> None:
    torch.manual_seed(11)
    recurrent_transition = _random_transition()
    recurrent_inputs = [
        recurrent_transition.linear.requires_grad_(),
        recurrent_transition.drive.requires_grad_(),
        torch.randn(2, 2, 4, 3, dtype=DTYPE, requires_grad=True),
    ]
    parallel_inputs = [
        value.detach().clone().requires_grad_() for value in recurrent_inputs
    ]
    recurrent_states, recurrent_final = recurrent_full_state_scan(
        FullStateAffineTransition(*recurrent_inputs[:2]), recurrent_inputs[2]
    )
    parallel_states, parallel_final = parallel_full_state_scan(
        FullStateAffineTransition(*parallel_inputs[:2]), parallel_inputs[2]
    )
    torch.testing.assert_close(
        parallel_states, recurrent_states, rtol=2e-12, atol=2e-12
    )
    torch.testing.assert_close(parallel_final, recurrent_final, rtol=2e-12, atol=2e-12)
    state_gradient = torch.randn_like(recurrent_states)
    recurrent_gradients = torch.autograd.grad(
        recurrent_states, recurrent_inputs, state_gradient
    )
    parallel_gradients = torch.autograd.grad(
        parallel_states, parallel_inputs, state_gradient
    )
    for actual, expected in zip(parallel_gradients, recurrent_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=3e-11, atol=3e-11)


def _run_model(
    model: SelectedBlockMemory,
    inputs: torch.Tensor,
    initial: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | bool]]:
    result = model(
        inputs,
        initial,
        scan_mode=mode,  # type: ignore[arg-type]
        return_diagnostics=True,
    )
    assert len(result) == 3
    return result


def test_dense_recurrent_parallel_output_state_and_full_gradient_parity() -> None:
    torch.manual_seed(12)
    recurrent_model = SelectedBlockMemory(_config()).double()
    parallel_model = SelectedBlockMemory(_config()).double()
    parallel_model.load_state_dict(recurrent_model.state_dict())
    recurrent_inputs = torch.randn(2, 9, 6, dtype=DTYPE, requires_grad=True)
    parallel_inputs = recurrent_inputs.detach().clone().requires_grad_()
    recurrent_initial = torch.randn(2, 2, 3, 3, 4, dtype=DTYPE, requires_grad=True)
    parallel_initial = recurrent_initial.detach().clone().requires_grad_()

    recurrent_output, recurrent_state, recurrent_diagnostics = _run_model(
        recurrent_model, recurrent_inputs, recurrent_initial, "dense_recurrent"
    )
    parallel_output, parallel_state, parallel_diagnostics = _run_model(
        parallel_model, parallel_inputs, parallel_initial, "dense_parallel"
    )
    torch.testing.assert_close(
        parallel_output, recurrent_output, rtol=2e-11, atol=2e-11
    )
    torch.testing.assert_close(parallel_state, recurrent_state, rtol=2e-11, atol=2e-11)

    def loss(
        output: torch.Tensor,
        state: torch.Tensor,
        diagnostics: dict[str, torch.Tensor | bool],
    ) -> torch.Tensor:
        router_loss = sum(
            diagnostics[name].square().mean()  # type: ignore[union-attr]
            for name in (
                "write_block_logits",
                "erase_block_logits",
                "read_block_logits",
            )
        )
        return output.square().mean() + 0.1 * state.square().mean() + 0.01 * router_loss

    loss(recurrent_output, recurrent_state, recurrent_diagnostics).backward()
    loss(parallel_output, parallel_state, parallel_diagnostics).backward()
    torch.testing.assert_close(
        parallel_inputs.grad, recurrent_inputs.grad, rtol=5e-10, atol=5e-11
    )
    torch.testing.assert_close(
        parallel_initial.grad, recurrent_initial.grad, rtol=5e-10, atol=5e-11
    )
    recurrent_parameters = dict(recurrent_model.named_parameters())
    parallel_parameters = dict(parallel_model.named_parameters())
    assert recurrent_parameters.keys() == parallel_parameters.keys()
    for name, parameter in recurrent_parameters.items():
        expected = parameter.grad
        actual = parallel_parameters[name].grad
        assert (actual is None) == (expected is None), name
        if actual is not None and expected is not None:
            torch.testing.assert_close(actual, expected, rtol=8e-10, atol=8e-11)


def test_physical_gather_matches_dense_recurrence() -> None:
    torch.manual_seed(13)
    model = SelectedBlockMemory(_config()).double()
    inputs = torch.randn(2, 8, 6, dtype=DTYPE)
    initial = torch.randn(2, 2, 3, 3, 4, dtype=DTYPE)
    gathered_output, gathered_state = model(
        inputs, initial.clone(), scan_mode="physical_gather"
    )
    dense_output, dense_state = model(
        inputs, initial.clone(), scan_mode="dense_recurrent"
    )
    torch.testing.assert_close(gathered_output, dense_output, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(gathered_state, dense_state, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("route_mode", ["soft", "straight_through"])
def test_differentiable_dense_routes_reach_all_coarse_controllers(
    route_mode: str,
) -> None:
    torch.manual_seed(130)
    model = SelectedBlockMemory(_config()).double()
    inputs = torch.randn(2, 6, 6, dtype=DTYPE, requires_grad=True)
    initial = torch.randn(2, 2, 3, 3, 4, dtype=DTYPE)
    output, final, diagnostics = model(
        inputs,
        initial,
        scan_mode="dense_recurrent",
        route_mode=route_mode,  # type: ignore[arg-type]
        return_diagnostics=True,
    )
    loss = output.square().mean() + final.square().mean()
    gradients = torch.autograd.grad(
        loss,
        tuple(
            diagnostics[name]
            for name in (
                "write_block_logits",
                "erase_block_logits",
                "read_block_logits",
            )
        ),
    )
    assert all(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)
    assert diagnostics["route_mode"] == route_mode
    assert diagnostics["hard_block_selection_differentiable"] is (
        route_mode == "straight_through"
    )


def test_straight_through_forward_is_exactly_hard_and_physical_fails_closed() -> None:
    torch.manual_seed(131)
    model = SelectedBlockMemory(_config()).double()
    inputs = torch.randn(2, 7, 6, dtype=DTYPE)
    initial = torch.randn(2, 2, 3, 3, 4, dtype=DTYPE)
    hard_output, hard_state = model(
        inputs, initial, scan_mode="dense_recurrent", route_mode="hard"
    )
    straight_output, straight_state = model(
        inputs,
        initial,
        scan_mode="dense_recurrent",
        route_mode="straight_through",
    )
    assert torch.equal(straight_output, hard_output)
    assert torch.equal(straight_state, hard_state)
    with pytest.raises(ValueError, match="physical_gather requires"):
        model(inputs, initial, route_mode="straight_through")


def test_readout_has_bounded_zero_state_jacobian() -> None:
    torch.manual_seed(132)
    model = SelectedBlockMemory(_config()).double()
    raw = torch.zeros(1, 1, 2, 4, dtype=DTYPE, requires_grad=True)
    projected = model._project_reads(raw)
    jacobian = torch.autograd.grad(projected.sum(), raw)[0]
    assert torch.isfinite(jacobian).all()
    assert float(jacobian.norm()) < 10.0


def test_hard_routing_changes_only_selected_mutation_blocks() -> None:
    torch.manual_seed(14)
    model = SelectedBlockMemory(_config(blocks=5)).double()
    inputs = torch.randn(2, 1, 6, dtype=DTYPE)
    initial = torch.randn(2, 2, 5, 3, 4, dtype=DTYPE)
    _, final, diagnostics = model(
        inputs,
        initial,
        scan_mode="physical_gather",
        return_diagnostics=True,
    )
    write = diagnostics["write_block_indices"][:, 0]  # type: ignore[index,union-attr]
    erase = diagnostics["erase_block_indices"][:, 0]  # type: ignore[index,union-attr]
    for batch in range(2):
        for head in range(2):
            touched = {int(write[batch, head]), int(erase[batch, head])}
            for block in range(5):
                if block not in touched:
                    assert torch.equal(
                        final[batch, head, block], initial[batch, head, block]
                    )
    assert diagnostics["hard_block_selection_differentiable"] is False
    for name in ("write_block_logits", "erase_block_logits", "read_block_logits"):
        assert diagnostics[name].shape == (2, 1, 2, 5)  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "mode", ["physical_gather", "dense_recurrent", "dense_parallel"]
)
def test_all_invalid_mask_preserves_state_but_read_remains_live(mode: str) -> None:
    torch.manual_seed(15)
    model = SelectedBlockMemory(_config()).double()
    inputs = torch.randn(2, 5, 6, dtype=DTYPE)
    initial = torch.randn(2, 2, 3, 3, 4, dtype=DTYPE)
    output, final = model(
        inputs,
        initial,
        valid_mask=torch.zeros(2, 5, dtype=torch.bool),
        scan_mode=mode,  # type: ignore[arg-type]
    )
    assert torch.equal(final, initial)
    assert torch.isfinite(output).all()
    assert bool((output.abs().sum(dim=-1) > 0).any())


def test_bounded_finite_state_for_4096_streaming_steps() -> None:
    torch.manual_seed(16)
    config = _config(
        model_dim=4,
        heads=1,
        blocks=2,
        slots_per_block=2,
        value_dim=2,
        update_rank=1,
        retention_min=0.1,
        retention_max=0.95,
    )
    model = SelectedBlockMemory(config).double()
    state = 0.25 * torch.randn(1, 1, 2, 2, 2, dtype=DTYPE)
    maximum = float(state.abs().max())
    bound = max(maximum, 1.0 / (1.0 - config.retention_max) + 1e-3)
    with torch.no_grad():
        for _ in range(4096):
            inputs = 10.0 * torch.empty(1, 4, dtype=DTYPE).uniform_(-1.0, 1.0)
            output, state = model.step(inputs, state)
            maximum = max(maximum, float(state.abs().max()))
            assert torch.isfinite(output).all()
            assert torch.isfinite(state).all()
    assert maximum <= bound


def test_true_low_rank_controller_factorization_and_accounting() -> None:
    torch.manual_seed(17)
    layer = LowRankLinear(7, 11, rank=3, bias=True).double()
    assert (
        sum(parameter.numel() for parameter in layer.parameters()) == 3 * (7 + 11) + 11
    )
    assert "weight" not in dict(layer.named_parameters())
    assert torch.linalg.matrix_rank(layer.effective_weight()).item() <= 3

    config = _config(controller_rank=2)
    model = SelectedBlockMemory(config).double()
    assert isinstance(model.controller, LowRankLinear)
    expected = (
        2 * (config.model_dim + model._controller_width) + model._controller_width
    )
    assert (
        sum(parameter.numel() for parameter in model.controller.parameters())
        == expected
    )
    assert model.state_scalars == 2 * 3 * 3 * 4
    assert model.state_bytes(torch.float64, batch_size=5) == 5 * model.state_scalars * 8


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"blocks": 0}, ValueError),
        ({"update_rank": 4}, ValueError),
        ({"controller_rank": 0}, ValueError),
        ({"retention_min": -0.1}, ValueError),
        ({"retention_max": 1.0}, ValueError),
    ],
)
def test_config_guards(overrides: dict[str, object], error: type[Exception]) -> None:
    with pytest.raises(error):
        _config(**overrides)


def test_input_state_mask_and_mode_shape_guards() -> None:
    model = SelectedBlockMemory(_config()).double()
    good = torch.randn(2, 4, 6, dtype=DTYPE)
    state = model.initial_state(2)
    with pytest.raises(ValueError, match="inputs"):
        model(good[..., :-1])
    with pytest.raises(TypeError, match="floating"):
        model(torch.ones(2, 4, 6, dtype=torch.long))
    with pytest.raises(ValueError, match="state"):
        model(good, state[..., :-1])
    with pytest.raises(ValueError, match="valid_mask"):
        model(good, valid_mask=torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(TypeError, match="torch.bool"):
        model(good, valid_mask=torch.ones(2, 4, dtype=DTYPE))
    with pytest.raises(ValueError, match="scan_mode"):
        model(good, scan_mode="recurrent")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="route_mode"):
        model(good, scan_mode="dense_recurrent", route_mode="relaxed")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="step inputs"):
        model.step(good)
