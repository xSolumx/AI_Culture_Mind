"""Correctness gates for reusable numerical and streaming audits."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

HYBRID_ROOT = Path(__file__).resolve().parents[1]
SSM_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SSM_ROOT))
sys.path.insert(0, str(HYBRID_ROOT))

from delta_product_reference import DeltaProductReferenceLayer

import hybrid_memory_v1_4.audits as AUDITS
from hybrid_memory_v1_4.model import (
    HybridMemoryConfig,
    HybridMemoryLM,
)
from hybrid_memory_v1_4.precision_screen import run_precision_screen
from hybrid_memory_v1_4.selected_block import (
    SelectedBlockConfig,
    SelectedBlockMemory,
)
from hybrid_memory_v1_4.temporal_observability_screen import (
    run_temporal_observability_screen,
)

StructuredSpin8Tier = AUDITS.StructuredSpin8Tier
StructuredTierConfig = AUDITS.StructuredTierConfig


def _json_round_trip(report: dict[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(report, allow_nan=False))


def _small_model() -> HybridMemoryLM:
    torch.manual_seed(40)
    config = HybridMemoryConfig(
        vocab_size=29,
        model_dim=8,
        layer_plan=("delta_product", "attention", "selected_block"),
        attention_heads=2,
        attention_window_size=5,
        delta_heads=2,
        delta_num_householder=2,
        selected_heads=1,
        selected_blocks=2,
        selected_slots_per_block=2,
        selected_value_dim=3,
        selected_update_rank=1,
        conv_kernel=3,
        expansion=2,
        dropout=0.0,
    )
    return HybridMemoryLM(config).double().eval()


def test_precision_horizon_uses_direct_high_retention_float64_source() -> None:
    report = AUDITS.precision_horizon_audit(
        horizon=31,
        retention=0.9999,
        state_size=3,
        value_size=2,
        checkpoints=(1, 3, 11, 31),
        parallel_chunk_size=7,
        seed=41,
    )
    case = report["cases"][0]
    assert case["source_dtype"] == "float64"
    assert case["source_created_directly_in_float64"] is True
    assert case["retention_min"] == pytest.approx(0.9999, abs=0.0)
    assert case["retention_max"] == pytest.approx(0.9999, abs=0.0)
    assert case["transition_representation"] == "diagonal"
    assert case["reference"]["nonfinite_count"] == 0
    assert {(path["dtype"], path["mode"]) for path in case["paths"]} == {
        ("float16", "parallel"),
        ("float16", "recurrent"),
        ("float32", "parallel"),
        ("float32", "recurrent"),
    }
    assert all(path["nonfinite_count"] == 0 for path in case["paths"])
    assert all(path["final_error"] is not None for path in case["paths"])
    _json_round_trip(report)


def test_affine_precision_rejects_cast_or_low_precision_sources() -> None:
    linear = torch.full((1, 5, 1, 2), 0.9999, dtype=torch.float32)
    drive = torch.zeros(1, 5, 1, 2, 1, dtype=torch.float64)
    initial = torch.zeros(1, 1, 2, 1, dtype=torch.float64)
    with pytest.raises(TypeError, match="created in float64"):
        AUDITS.audit_affine_precision(
            linear, drive, initial, diagonal=True, checkpoints=(5,)
        )


def test_precision_includes_real_selected_block_and_delta_product_paths() -> None:
    torch.manual_seed(42)
    selected = (
        SelectedBlockMemory(
            SelectedBlockConfig(
                model_dim=4,
                heads=1,
                blocks=2,
                slots_per_block=2,
                value_dim=2,
                update_rank=1,
                retention_min=0.99,
                retention_max=0.9999,
            )
        )
        .double()
        .eval()
    )
    delta = (
        DeltaProductReferenceLayer(hidden_size=4, num_heads=1, num_householder=2)
        .double()
        .eval()
    )
    selected_inputs = torch.randn(1, 7, 4, dtype=torch.float64)
    delta_inputs = torch.randn(1, 7, 4, dtype=torch.float64)

    report = AUDITS.precision_horizon_audit(
        horizon=9,
        retention=0.9999,
        state_size=2,
        value_size=2,
        parallel_chunk_size=4,
        selected_block=selected,
        selected_inputs=selected_inputs,
        delta_product=delta,
        delta_inputs=delta_inputs,
    )
    cases = {case["name"]: case for case in report["cases"]}
    selected_case = cases["selected_block_diagonal_affine"]
    delta_case = cases["maintained_delta_product_matrix_affine"]
    assert selected_case["source_kind"] == "compiled_selected_block_transition"
    assert selected_case["off_diagonal_residual"] == 0.0
    assert delta_case["source_kind"] == ("compiled_maintained_delta_product_transition")
    assert delta_case["transition_representation"] == "matrix"
    assert all(
        path["nonfinite_count"] == 0
        for case in cases.values()
        for path in case["paths"]
    )
    _json_round_trip(report)


def test_hybrid_chunk_replay_checks_outputs_states_and_token_steps() -> None:
    model = _small_model()
    tokens = torch.randint(0, model.config.vocab_size, (2, 9))
    report = AUDITS.hybrid_chunk_replay_audit(
        model,
        tokens,
        chunk_partitions=((2, 1, 6), (1, 4, 9)),
    )
    assert report["full"]["output_nonfinite_count"] == 0
    for partition in report["partitions"]:
        assert partition["output_max_abs_error"] < 1e-10
        assert partition["state"]["max_abs_error"] < 1e-10
        assert partition["state"]["metadata_mismatch_count"] == 0
        assert partition["actual_bytes_match_full"] is True
    assert report["token_step"]["output_max_abs_error"] < 1e-10
    assert report["token_step"]["state"]["max_abs_error"] < 1e-10
    assert report["token_step"]["actual_bytes_match_full"] is True
    _json_round_trip(report)


def _routed_tier() -> tuple[StructuredSpin8Tier, torch.Tensor]:
    tier = (
        StructuredSpin8Tier(
            StructuredTierConfig(model_dim=3, channels=1, hard_eval=True)
        )
        .double()
        .eval()
    )
    with torch.no_grad():
        tier.rung_controller.weight.zero_()
        tier.rung_controller.bias.fill_(-10.0)
        tier.rung_controller.weight[0, 0] = 2.0
        tier.rung_controller.bias[0] = 0.0
        tier.rung_controller.weight[3, 0] = -2.0
        tier.rung_controller.bias[3] = 0.0
        assert isinstance(tier.coefficient_controller, torch.nn.Linear)
        tier.coefficient_controller.weight.zero_()
        tier.coefficient_controller.bias.zero_()
    recurrent_states = torch.zeros(1, 5, 3, dtype=torch.float64)
    recurrent_states[0, :, 0] = torch.tensor(
        [2.0, -2.0, 2.0, -2.0, 2.0], dtype=torch.float64
    )
    return tier, recurrent_states


def test_structured_audit_measures_supplied_state_routing_and_action_drift() -> None:
    tier, recurrent_states = _routed_tier()
    report = AUDITS.structured_rung_gauge_audit(
        tier, recurrent_states, composition_horizon=5
    )
    guarantee = report["parameterization_guarantee"]
    behavior = report["learned_behavior"]
    numerical = report["numerical_behavior"]
    assert guarantee["single_actions_are_orthogonal_in_exact_arithmetic"] is True
    assert guarantee["guarantees_learned_rung_use"] is False
    assert behavior["chart_switch_count"] == 4
    assert behavior["chart_switch_frequency"] == 1.0
    occupancy = {row["rung"]: row["count"] for row in behavior["rung_occupancy"]}
    assert occupancy == {3: 3, 4: 0, 6: 0, 8: 2}
    assert numerical["action_nonfinite_count"] == 0
    assert numerical["max_per_action_orthogonality_residual"] < 1e-14
    assert numerical["final_composed_orthogonality_residual"] < 1e-14
    _json_round_trip(report)


def test_structured_audit_accepts_loaded_checkpoint_but_never_fabricates() -> None:
    tier, recurrent_states = _routed_tier()
    checkpoint = {
        "config": asdict(tier.config),
        "state_dict": tier.state_dict(),
    }
    report = AUDITS.structured_rung_gauge_audit(
        recurrent_states=recurrent_states,
        checkpoint=checkpoint,
    )
    assert report["source"] == "supplied_checkpoint"

    with pytest.raises(ValueError, match="tier/mixer or checkpoint"):
        AUDITS.structured_rung_gauge_audit(recurrent_states=recurrent_states)
    with pytest.raises(TypeError, match="state_dict"):
        AUDITS.structured_rung_gauge_audit(
            recurrent_states=recurrent_states,
            checkpoint={"config": asdict(tier.config)},
        )


def test_cache_state_drift_uses_actual_states_and_model_byte_reports() -> None:
    model = _small_model()
    tokens = torch.randint(0, model.config.vocab_size, (2, 17))
    report = AUDITS.cache_state_drift_audit(model, tokens, chunk_size=4)
    assert report["processed_length"] == 17
    assert report["terminated_early"] is False
    assert report["output_nonfinite_count"] == 0
    assert report["state_nonfinite_count"] == 0
    assert report["final_actual_bytes"] > 0
    assert report["final_actual_bytes"] <= report["final_capacity_bytes"]
    assert report["byte_report_samples"][-1]["position"] == 17
    assert [layer["kind"] for layer in report["layers"]] == list(model.layer_plan)
    assert all(layer["sample_count"] == 5 for layer in report["layers"])
    assert all(layer["max_l2_norm"] is not None for layer in report["layers"])
    final_bytes = report["byte_report_samples"][-1]
    assert (
        sum(layer["actual_bytes"] for layer in final_bytes["layers"])
        == report["final_actual_bytes"]
    )
    _json_round_trip(report)


def _observability_model(layer_plan: tuple[str, ...]) -> HybridMemoryLM:
    torch.manual_seed(42)
    return HybridMemoryLM(
        HybridMemoryConfig(
            vocab_size=29,
            model_dim=8,
            layer_plan=layer_plan,  # type: ignore[arg-type]
            attention_heads=1,
            attention_window_size=8,
            delta_heads=1,
            delta_num_householder=1,
            selected_heads=1,
            selected_blocks=2,
            selected_slots_per_block=2,
            selected_value_dim=4,
            selected_update_rank=1,
            use_local_conv=False,
            expansion=1,
            dropout=0.0,
        )
    ).double()


def test_temporal_observability_separates_route_and_depth_failures() -> None:
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    hard = AUDITS.temporal_query_observability_audit(
        _observability_model(("selected_block",)),
        tokens,
        selected_layer_index=0,
        route_mode="hard",
    )
    assert hard["coarse_routes_connected"] is False
    assert hard["nonfinal_read_path_present"] is False
    assert hard["gradients"]["read_block_logits"]["connected"] is False
    assert hard["gradients"]["read_fine_logits"]["nonfinal_max_norm"] == 0.0
    assert hard["gradients"]["read_fine_logits"]["final_norm"] > 0.0

    one_block_st = AUDITS.temporal_query_observability_audit(
        _observability_model(("selected_block",)),
        tokens,
        selected_layer_index=0,
        route_mode="straight_through",
    )
    assert one_block_st["coarse_routes_connected"] is True
    assert one_block_st["nonfinal_read_path_present"] is False

    hybrid_st = AUDITS.temporal_query_observability_audit(
        _observability_model(("selected_block", "attention")),
        tokens,
        selected_layer_index=0,
        route_mode="straight_through",
    )
    assert hybrid_st["coarse_routes_connected"] is True
    assert hybrid_st["nonfinal_read_path_present"] is True
    assert hybrid_st["gradients"]["read_block_logits"]["nonfinal_max_norm"] > 0.0
    assert hybrid_st["gradients"]["read_fine_logits"]["nonfinal_max_norm"] > 0.0
    _json_round_trip(hard)
    _json_round_trip(one_block_st)
    _json_round_trip(hybrid_st)


def test_temporal_screen_records_failed_original_and_accepted_successor() -> None:
    report = run_temporal_observability_screen(seed=42)
    assert report["original_g3_passed"] is False
    assert report["successor_topology_acceptance_passed"] is True
    assert report["evidentiary"] is True
    assert report["source_files"]
    assert report["git_status"]
    _json_round_trip(report)


def test_precision_screen_wraps_all_three_real_paths() -> None:
    report = run_precision_screen(horizon=17, seed=43)
    assert report["horizon"] == 17
    assert report["completed_without_nonfinite"] is True
    cases = report["audit"]["cases"]
    assert {case["name"] for case in cases} == {
        "high_retention_diagonal_affine",
        "selected_block_diagonal_affine",
        "maintained_delta_product_matrix_affine",
    }
    assert all(case["horizon"] == 17 for case in cases)
    _json_round_trip(report)
