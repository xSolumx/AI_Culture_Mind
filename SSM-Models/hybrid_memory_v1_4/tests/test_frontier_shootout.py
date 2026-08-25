from __future__ import annotations

import pytest
import torch
from torch import nn

from hybrid_memory_v1_4.frontier_shootout import (
    ARMS,
    EXPECTED_PARAMETERS,
    MAX_PARAMETER_RESIDUAL,
    TARGET_PARAMETERS,
    _forward_logits,
    _hybrid_config,
    _mamba2_kwargs,
    _olmo_kwargs,
    build_model,
)
from hybrid_memory_v1_4.model import HybridMemoryLM


def test_g16_arm_roster_and_parameter_contract_are_frozen() -> None:
    assert ARMS == (
        "hybrid_v1_4_5",
        "hybrid_gdn2",
        "mamba2",
        "olmo_hybrid",
    )
    assert EXPECTED_PARAMETERS == {
        "hybrid_v1_4_5": 124_534,
        "hybrid_gdn2": 124_414,
        "mamba2": 124_172,
        "olmo_hybrid": 124_376,
    }
    assert all(
        abs(count - TARGET_PARAMETERS) / TARGET_PARAMETERS <= MAX_PARAMETER_RESIDUAL
        for count in EXPECTED_PARAMETERS.values()
    )


def test_g16_hybrids_preserve_the_g12_shell_and_isolate_the_edit_law() -> None:
    control = _hybrid_config("hybrid_v1_4_5")
    candidate = _hybrid_config("hybrid_gdn2")

    assert control.layer_plan == ("gated_delta", "attention")
    assert candidate.layer_plan == ("gated_delta_v2", "attention")
    for field in (
        "vocab_size",
        "model_dim",
        "attention_heads",
        "attention_window_size",
        "gated_delta_heads",
        "gated_delta_value_dim",
        "gated_delta_normalize_values",
        "gated_delta_identity_value_path",
        "gated_delta_identity_output_gate",
        "gated_delta_tie_query_key",
        "gated_delta_residual_scale_init",
        "gated_delta_minimum_retention",
        "gated_delta_initial_retention",
        "gated_delta_initial_erase_strength",
        "gated_delta_initial_write_strength",
        "use_local_conv",
        "conv_kernel",
        "tie_embeddings",
    ):
        assert getattr(candidate, field) == getattr(control, field), field
    assert control.gated_delta_key_dim == 24
    assert control.expansion == 5
    assert candidate.gated_delta_key_dim == 28
    assert candidate.expansion == 4
    assert sum(p.numel() for p in HybridMemoryLM(control).parameters()) == 124_534
    assert sum(p.numel() for p in HybridMemoryLM(candidate).parameters()) == 124_414


def test_g16_external_shapes_are_exact_and_kernel_safe() -> None:
    assert _mamba2_kwargs() == {
        "vocab_size": 512,
        "d_model": 56,
        "num_layers": 4,
        "d_state": 32,
        "expand": 2,
        "headdim": 16,
    }
    olmo = _olmo_kwargs()
    assert olmo["intermediate_size"] == 138
    assert olmo["layer_types"] == ["linear_attention", "full_attention"]
    assert olmo["max_position_embeddings"] == 4096


def test_g16_olmo_is_actual_transformers_and_parameter_matched() -> None:
    pytest.importorskip("transformers")
    model, metadata = build_model("olmo_hybrid", torch.device("cpu"))
    assert type(model).__module__.startswith("transformers.models.olmo_hybrid")
    assert metadata["runtime_class"].endswith(".OlmoHybridForCausalLM")
    assert metadata["parameter_count"] == 124_376
    assert abs(metadata["parameter_residual"]) <= MAX_PARAMETER_RESIDUAL
    tokens = torch.randint(0, 512, (2, 16))
    logits = _forward_logits("olmo_hybrid", model, tokens)
    assert logits.shape == (2, 16, 512)


class _DictionaryLogitModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": inputs.unsqueeze(-1).float()}


def test_g16_mamba2_forward_uses_repository_wrapper_contract() -> None:
    inputs = torch.ones(2, 8, dtype=torch.long)
    logits = _forward_logits("mamba2", _DictionaryLogitModel(), inputs)
    assert logits.shape == (2, 8, 1)
