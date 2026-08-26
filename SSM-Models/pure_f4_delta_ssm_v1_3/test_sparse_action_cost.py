from __future__ import annotations

import torch

from .benchmark_sparse_action_cost import _verdict
from .benchmark_train import TrainingConfig, _build_model
from .model import parameter_count


def _row(milliseconds: float, peak: int, dense: bool = False) -> dict[str, object]:
    return {
        "timing": {"median_ms": milliseconds},
        "maximum_peak_allocated_bytes": peak,
        "any_dense_sequence_action": dense,
    }


def test_cost_verdict_keeps_marginal_and_mamba_gates_separate() -> None:
    summary = {
        "e6_primitive_dead": _row(10.0, 100),
        "e6_primitive_event": _row(12.0, 120),
        "e6_safe": _row(20.0, 240),
        "mamba2_official": _row(5.0, 100),
    }
    verdict = _verdict(summary)
    assert verdict["cheap_action_path_pass"] is True
    assert verdict["mamba_competitive_pass"] is False


def test_cost_verdict_fails_closed_on_dense_candidate_or_missing_arm() -> None:
    summary = {
        "e6_primitive_dead": _row(10.0, 100),
        "e6_primitive_event": _row(10.0, 100, dense=True),
        "e6_safe": _row(20.0, 240),
        "mamba2_official": _row(9.0, 90),
    }
    verdict = _verdict(summary)
    assert verdict["cheap_action_path_pass"] is False
    assert verdict["mamba_competitive_pass"] is False
    assert _verdict({})["cheap_action_path_pass"] is False


def test_dead_budget_and_active_primitive_have_identical_parameter_counts() -> None:
    config = TrainingConfig()
    dead = _build_model("e6_primitive_dead", config)
    active = _build_model("e6_primitive_event", config)
    assert parameter_count(dead) == parameter_count(active)
    assert parameter_count(active) == 40_858


def test_dead_budget_retains_event_controller_backward_and_optimizer_budget() -> None:
    model = _build_model("e6_primitive_dead", TrainingConfig())
    tokens = torch.randint(0, 256, (2, 64))
    model(tokens)["logits"].square().mean().backward()
    for block in model.blocks:
        assert block.action_controller is not None
        gradient = block.action_controller.weight.grad
        assert gradient is not None
        assert torch.count_nonzero(gradient) == 0
