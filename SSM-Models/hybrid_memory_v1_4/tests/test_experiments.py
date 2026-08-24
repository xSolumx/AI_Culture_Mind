from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from hybrid_memory_v1_4.experiments import (
    HybridVariant,
    StateSize,
    TrainingProtocol,
    delta_only_control,
    generate_task_batch,
    result_json,
    routing_auxiliary_loss,
    run_matched_experiment,
    run_state_size_sweep,
)
from hybrid_memory_v1_4.long_context_screen import (
    run_mechanical_screen,
    write_json_atomic,
)
from hybrid_memory_v1_4.model import HybridMemoryConfig, HybridMemoryLM
from hybrid_memory_v1_4.model import parameter_count as model_parameter_count
from hybrid_memory_v1_4.retrieval_screen import (
    build_quality_cohort,
    build_smoke_cohort,
)
from hybrid_memory_v1_4.tasks import DEFAULT_VOCABULARY


def _config(**overrides: object) -> HybridMemoryConfig:
    values: dict[str, object] = {
        "vocab_size": DEFAULT_VOCABULARY.vocab_size,
        "model_dim": 8,
        "layer_plan": ("selected_block",),
        "attention_heads": 1,
        "attention_window_size": 8,
        "delta_heads": 1,
        "delta_num_householder": 1,
        "selected_heads": 1,
        "selected_blocks": 2,
        "selected_slots_per_block": 2,
        "selected_value_dim": 4,
        "selected_update_rank": 1,
        "use_local_conv": False,
        "expansion": 1,
        "dropout": 0.0,
    }
    values.update(overrides)
    return HybridMemoryConfig(**values)


def _protocol(**overrides: object) -> TrainingProtocol:
    values: dict[str, object] = {
        "task": "mqar",
        "train_length": 16,
        "eval_lengths": (16, 24),
        "updates": 2,
        "train_batch_size": 2,
        "eval_batch_size": 2,
        "eval_batches": 1,
        "seeds": (7,),
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "chunk_size": 5,
        "parameter_gap_threshold": 10.0,
        "routing_auxiliary_coefficient": 0.1,
        "selected_training_route_mode": "straight_through",
    }
    values.update(overrides)
    return TrainingProtocol(**values)


def _paired_identical_result() -> object:
    config = _config()
    variants = (
        HybridVariant("selected_a", config),
        HybridVariant("selected_b", config),
    )
    return run_matched_experiment(variants, _protocol())


def test_fresh_training_episodes_and_paired_cohorts() -> None:
    result = _paired_identical_result()
    assert len(result.runs) == 2
    left, right = result.runs
    assert len(set(left.training_data_seeds)) == left.steps == 2
    assert len(set(left.training_batch_fingerprints)) == left.steps
    assert left.training_data_seeds == right.training_data_seeds
    assert left.training_batch_fingerprints == right.training_batch_fingerprints
    assert tuple(item.data_seeds for item in left.evaluations) == tuple(
        item.data_seeds for item in right.evaluations
    )
    assert all(item.finite for run in result.runs for item in run.evaluations)
    assert all(run.chunk_replay_confirmed for run in result.runs)


def test_matched_experiment_replays_deterministically() -> None:
    first = _paired_identical_result()
    second = _paired_identical_result()
    for left, right in zip(first.runs, second.runs, strict=True):
        assert left.training_data_seeds == right.training_data_seeds
        assert left.training_batch_fingerprints == right.training_batch_fingerprints
        assert left.mean_retrieval_loss == right.mean_retrieval_loss
        assert left.mean_routing_auxiliary_loss == right.mean_routing_auxiliary_loss
        assert left.untrained_evaluations == right.untrained_evaluations
        assert left.evaluations == right.evaluations


@pytest.mark.parametrize(
    "task,length",
    (
        ("mqar", 24),
        ("overwrite", 24),
        ("exact_distance_needle", 24),
        ("selective_copy", 24),
    ),
)
def test_routing_auxiliary_is_supervised_and_reaches_controller(
    task: str, length: int
) -> None:
    torch.manual_seed(11)
    model = HybridMemoryLM(_config())
    batch = generate_task_batch(task, batch_size=2, length=length, seed=13)
    output = model(batch.inputs, return_diagnostics=True)
    loss = routing_auxiliary_loss(output, batch)
    assert loss.isfinite() and loss.item() > 0.0
    loss.backward()
    controller = model.blocks[0].mixer.controller
    assert controller.weight.grad is not None
    assert torch.count_nonzero(controller.weight.grad).item() > 0


def test_parameter_gate_fails_closed_before_training() -> None:
    selected = HybridVariant("selected", _config())
    delta = delta_only_control("delta", selected.config)
    with pytest.raises(ValueError, match="parameter gap threshold exceeded"):
        run_matched_experiment(
            (selected, delta),
            _protocol(updates=1, parameter_gap_threshold=0.0),
        )


def test_result_schema_is_strict_json_and_long_eval_is_chunked() -> None:
    result = _paired_identical_result()
    payload = json.loads(result_json(result))
    assert payload["schema_version"] == 1
    assert payload["evidentiary"] is False
    assert payload["runs"][0]["evaluations"][-1]["length"] == 24
    assert payload["runs"][0]["untrained_evaluations"][-1]["length"] == 24
    assert payload["runs"][0]["chunk_replay_confirmed"] is True
    assert payload["source_files"]


def test_state_sweep_requires_control_and_records_exactly_one_control() -> None:
    base = _config()
    protocol = _protocol(updates=1, routing_auxiliary_coefficient=0.0)
    with pytest.raises(ValueError, match="explicit paired control"):
        run_state_size_sweep(
            protocol,
            control=None,
            base_config=base,
            sizes=(StateSize(2, 2, 4),),
        )

    control = delta_only_control("delta_control", base)
    result = run_state_size_sweep(
        protocol,
        control=control,
        base_config=base,
        sizes=(StateSize(2, 2, 4),),
    )
    assert len(result.rows) == 2
    assert sum(row.paired_control for row in result.rows) == 1
    assert all(row.quality for row in result.rows)
    assert all(
        quality.actual_cache_bytes <= quality.capacity_cache_bytes
        for row in result.rows
        for quality in row.quality
    )


def test_mechanical_screen_and_atomic_json(tmp_path: Path) -> None:
    result = run_mechanical_screen(
        lengths=(8, 17),
        chunk_size=5,
        batch_size=1,
        seed=19,
        model_dim=8,
        layer_plan=("selected_block", "attention"),
        attention_heads=1,
        attention_window_size=5,
        selected_blocks=2,
        slots_per_block=2,
        value_dim=4,
    )
    assert result.passed
    assert result.evidentiary is False
    assert result.layer_plan == ("selected_block", "attention")
    assert [row.chunks for row in result.rows] == [2, 4]
    assert all(
        row.actual_cache_bytes <= row.capacity_cache_bytes for row in result.rows
    )

    output = tmp_path / "mechanical.json"
    write_json_atomic(output, result)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["rows"][-1]["length"] == 17
    assert not list(tmp_path.glob("*.tmp"))


def test_protocol_and_official_control_helpers_fail_closed() -> None:
    with pytest.raises(ValueError, match="include train_length"):
        replace(_protocol(), eval_lengths=(24,))
    with pytest.raises(ValueError, match="at least two variants"):
        run_matched_experiment((HybridVariant("only", _config()),), _protocol())


def test_frozen_quality_builder_matches_documented_budget() -> None:
    variants, protocol = build_quality_cohort(routing_auxiliary_coefficient=0.0)
    counts = tuple(
        model_parameter_count(HybridMemoryLM(variant.config)) for variant in variants
    )
    assert counts == (107_552, 112_290)
    assert abs(counts[1] - counts[0]) / counts[0] < protocol.parameter_gap_threshold
    assert protocol.train_length == 512
    assert protocol.eval_lengths == (512, 2048, 8192)
    assert protocol.updates == 600
    assert protocol.seeds == (41, 43, 47)
    assert protocol.selected_training_route_mode == "straight_through"
    with pytest.raises(ValueError, match="must be 0.0 or 0.1"):
        build_quality_cohort(routing_auxiliary_coefficient=0.05)


def test_smoke_builder_is_explicitly_short_and_parameter_gated() -> None:
    variants, protocol = build_smoke_cohort()
    counts = tuple(
        model_parameter_count(HybridMemoryLM(variant.config)) for variant in variants
    )
    assert protocol.updates == 2
    assert protocol.seeds == (41,)
    assert abs(counts[1] - counts[0]) / counts[0] < protocol.parameter_gap_threshold
