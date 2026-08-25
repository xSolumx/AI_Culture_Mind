from __future__ import annotations

import pytest
import torch

from hybrid_memory_v1_4 import g15b_interleaved_cohort as cohort
from hybrid_memory_v1_4.g15b_interleaved_cohort import (
    ARM_SPECS,
    QUALITY_PHASES,
    build_model,
    commissioned_losses,
    complete_control_forward,
    frozen_config,
    scored_training_decisions,
)
from hybrid_memory_v1_4.g15b_interleaved_tasks import generate_interleaved_batch


def _batch(task: str = "overwrite"):
    return generate_interleaved_batch(
        task,
        2,
        64,
        2,
        4,
        2,
        seed=2309,
        needle_distance=32 if task == "needle" else None,
    )


def test_frozen_quality_schedule_has_exact_decision_count() -> None:
    config = frozen_config("quality")
    assert config.seeds == (2309, 2311, 2333)
    assert config.phases == QUALITY_PHASES
    assert config.evaluation_batch_cap == 16
    assert sum(phase.updates for phase in config.phases) == 4200
    assert scored_training_decisions(config.phases) == 375_360


def test_primary_arms_are_shape_and_initial_parameter_matched() -> None:
    models = {arm: build_model(arm, 23, torch.device("cpu")) for arm in ARM_SPECS}
    shapes = [
        {name: tuple(parameter.shape) for name, parameter in model.named_parameters()}
        for model in models.values()
    ]
    values = [dict(model.named_parameters()) for model in models.values()]
    assert shapes[1:] == shapes[:-1]
    for name in values[0]:
        assert torch.equal(values[0][name], values[1][name])
        assert torch.equal(values[0][name], values[2][name])
    assert models["I"].config.spin_dirac_readout_mode == "identity"
    assert models["C"].config.spin_dirac_readout_mode == "clifford"
    assert models["S"].config.spin_dirac_readout_mode == "clifford"


@pytest.mark.parametrize("arm", ARM_SPECS)
def test_commissioned_loss_is_finite_and_reaches_all_active_controls(arm: str) -> None:
    model = build_model(arm, 23, torch.device("cpu"))
    batch = _batch()
    output = model(batch.token_ids, return_diagnostics=True)
    diagnostics = output["diagnostics"][0]
    names = (
        "query_vector",
        "key_vector",
        "value_positive",
        "erase_strength",
        "write_strength",
        "retention",
        "transport_coordinates",
    )
    for name in names:
        if diagnostics[name].requires_grad:
            diagnostics[name].retain_grad()
    loss, components = commissioned_losses(output, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(components) == {
        "retrieval",
        "reverse_binding",
        "address",
        "write",
        "erase",
        "retention",
    }
    for name in names[:-1]:
        gradient = diagnostics[name].grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert float(torch.linalg.vector_norm(gradient)) > 0.0
    coordinate_gradient = diagnostics["transport_coordinates"].grad
    if arm == "I":
        assert coordinate_gradient is None
    else:
        assert coordinate_gradient is not None
        assert float(torch.linalg.vector_norm(coordinate_gradient)) > 0.0


@pytest.mark.parametrize("arm", ARM_SPECS)
def test_complete_learned_control_path_matches_ordinary_model(arm: str) -> None:
    model = build_model(arm, 23, torch.device("cpu")).eval()
    batch = _batch("selective")
    with torch.no_grad():
        ordinary = model(batch.token_ids, delta_scan_mode="parallel")["logits"]
        complete = complete_control_forward(model, batch, "learned")["logits"]
    assert torch.allclose(ordinary, complete, atol=2e-5, rtol=2e-5)


def test_complete_control_interventions_replace_full_controls_fail_closed() -> None:
    model = build_model("S", 23, torch.device("cpu")).eval()
    batch = _batch()
    with torch.no_grad():
        learned = complete_control_forward(model, batch, "learned")
        no_write = complete_control_forward(model, batch, "no_write")
        no_erase = complete_control_forward(model, batch, "no_erase")
        wrong = complete_control_forward(model, batch, "wrong_query")
        oracle = complete_control_forward(model, batch, "oracle")
    assert bool(no_write["diagnostics"]["write_strength"].eq(0).all())
    assert bool(no_erase["diagnostics"]["erase_strength"].eq(0).all())
    assert not torch.equal(
        learned["diagnostics"]["query_vector"], wrong["diagnostics"]["query_vector"]
    )
    assert bool(wrong["wrong_query_eligible"].any())
    assert torch.equal(
        oracle["diagnostics"]["write_strength"].squeeze(-1).gt(0).any(dim=-1),
        batch.write_event_mask,
    )
    assert torch.equal(
        oracle["diagnostics"]["erase_strength"].squeeze(-1).gt(0).any(dim=-1),
        batch.erase_event_mask,
    )


def test_unknown_complete_control_intervention_fails_closed() -> None:
    model = build_model("S", 23, torch.device("cpu"))
    with pytest.raises(ValueError, match="unknown G15B intervention"):
        complete_control_forward(model, _batch(), "partial_hook")  # type: ignore[arg-type]


def test_evaluation_requires_complete_batches() -> None:
    model = build_model("I", 23, torch.device("cpu"))
    with pytest.raises(ValueError, match="complete-batch decision count"):
        cohort._evaluate_cell(
            model,
            "mqar",
            128,
            seed=23,
            decisions=15,
            batch_cap=2,
        )


def test_evaluation_uses_available_no_grad_batch_capacity() -> None:
    assert cohort._evaluation_batch_size("mqar", decisions=16, cap=32) == 2
    assert cohort._evaluation_batch_size("needle", decisions=16, cap=32) == 16
    assert cohort._evaluation_batch_size("mqar", decisions=4096, cap=16) == 16
    assert cohort._evaluation_batch_size("needle", decisions=4096, cap=16) == 16
