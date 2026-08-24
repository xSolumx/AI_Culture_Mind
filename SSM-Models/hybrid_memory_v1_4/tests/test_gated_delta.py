"""Content-addressed Gated DeltaNet recurrence and integration contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hybrid_memory_v1_4.experiments import (
    gated_delta_association_auxiliary_loss,
    intermediate_retrieval_auxiliary_loss,
)
from hybrid_memory_v1_4.gated_delta import GatedDeltaConfig, GatedDeltaMemory
from hybrid_memory_v1_4.model import (
    GatedDeltaState,
    HybridMemoryConfig,
    HybridMemoryLM,
)
from hybrid_memory_v1_4.tasks import generate_mqar_batch


def test_config_rejects_invalid_dimensions_and_stability_bounds() -> None:
    with pytest.raises(ValueError, match="divisible"):
        GatedDeltaConfig(10, heads=4)
    with pytest.raises(ValueError, match="minimum_retention"):
        GatedDeltaConfig(8, minimum_retention=1.0)
    with pytest.raises(ValueError, match="initial_retention"):
        GatedDeltaConfig(8, minimum_retention=0.9, initial_retention=0.9)
    with pytest.raises(ValueError, match="initial_write_strength"):
        GatedDeltaConfig(8, initial_write_strength=1.0)
    with pytest.raises(ValueError, match="identity_value_path"):
        GatedDeltaConfig(16, heads=2, value_dim=3, identity_value_path=True)


def test_parallel_recurrent_and_arbitrary_chunk_replay_match() -> None:
    torch.manual_seed(1401)
    layer = GatedDeltaMemory(GatedDeltaConfig(16, heads=2)).double()
    inputs = torch.randn(2, 17, 16, dtype=torch.float64)
    recurrent, recurrent_state = layer(inputs, scan_mode="recurrent")
    parallel, parallel_state = layer(inputs, scan_mode="parallel")
    chunks = []
    state = None
    for start, stop in ((0, 3), (3, 11), (11, 13), (13, 17)):
        output, state = layer(inputs[:, start:stop], state, scan_mode="parallel")
        chunks.append(output)
    torch.testing.assert_close(recurrent, parallel, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(recurrent_state, parallel_state, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(
        recurrent, torch.cat(chunks, dim=1), rtol=1e-11, atol=1e-11
    )
    torch.testing.assert_close(recurrent_state, state, rtol=1e-11, atol=1e-11)


def test_valid_mask_freezes_state_and_zeroes_output() -> None:
    torch.manual_seed(1402)
    layer = GatedDeltaMemory(GatedDeltaConfig(12, heads=3)).double()
    inputs = torch.randn(2, 5, 12, dtype=torch.float64)
    valid = torch.tensor([[True, True, False, False, True], [False] * 5])
    output, state = layer(inputs, valid_mask=valid, scan_mode="parallel")
    first, first_state = layer(inputs[:1, :2], scan_mode="recurrent")
    last, expected = layer(inputs[:1, 4:5], first_state, scan_mode="recurrent")
    torch.testing.assert_close(output[0:1, :2], first)
    torch.testing.assert_close(output[0:1, 4:5], last)
    torch.testing.assert_close(state[0:1], expected)
    assert torch.count_nonzero(output[0, 2:4]) == 0
    assert torch.count_nonzero(output[1]) == 0
    assert torch.count_nonzero(state[1]) == 0


def test_gradients_reach_address_write_decay_value_and_read_paths() -> None:
    torch.manual_seed(1403)
    layer = GatedDeltaMemory(GatedDeltaConfig(16, heads=2)).double()
    inputs = torch.randn(2, 9, 16, dtype=torch.float64, requires_grad=True)
    output, state = layer(inputs, scan_mode="parallel")
    (output.square().mean() + state.square().mean()).backward()
    assert inputs.grad is not None and torch.count_nonzero(inputs.grad) > 0
    for name in (
        "query_projection.weight",
        "key_projection.weight",
        "value_projection.weight",
        "write_projection.weight",
        "decay_projection.weight",
        "output_gate.weight",
        "output_projection.weight",
    ):
        parameter = dict(layer.named_parameters())[name]
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name


def test_optional_value_normalization_fixes_each_head_norm() -> None:
    torch.manual_seed(1406)
    layer = GatedDeltaMemory(
        GatedDeltaConfig(16, heads=2, value_dim=6, normalize_values=True)
    ).double()
    inputs = torch.randn(2, 7, 16, dtype=torch.float64)
    *_, diagnostics = layer(inputs, return_diagnostics=True)
    value = diagnostics["value"]
    assert isinstance(value, torch.Tensor)
    expected = torch.full_like(value[..., 0], 6**0.5)
    torch.testing.assert_close(value.norm(dim=-1), expected)


def test_identity_value_path_and_gate_start_as_lossless_readout() -> None:
    layer = GatedDeltaMemory(
        GatedDeltaConfig(
            16,
            heads=2,
            value_dim=8,
            identity_value_path=True,
            identity_output_gate=True,
        )
    )
    torch.testing.assert_close(layer.value_projection.weight, torch.eye(16))
    torch.testing.assert_close(layer.output_projection.weight, torch.eye(16))
    inputs = torch.zeros(2, 3, 16)
    gate = 1.0 + torch.tanh(layer.output_gate(inputs))
    torch.testing.assert_close(gate, torch.ones_like(gate))


def test_model_default_pivots_to_content_addressed_memory_plus_attention() -> None:
    config = HybridMemoryConfig(vocab_size=197, model_dim=32, attention_heads=4)
    assert config.layer_plan == ("gated_delta", "attention")
    model = HybridMemoryLM(config)
    tokens = torch.randint(0, 197, (2, 13))
    output = model(tokens, delta_scan_mode="parallel")
    assert output["logits"].shape == (2, 13, 197)
    assert isinstance(output["states"][0], GatedDeltaState)
    report = model.state_byte_report(output["states"])
    assert report["layers"][0]["kind"] == "gated_delta"
    assert report["layers"][0]["capacity_components"]["memory"] > 0


def test_gated_delta_residual_initialization_is_explicit() -> None:
    model = HybridMemoryLM(
        HybridMemoryConfig(
            vocab_size=197,
            model_dim=32,
            layer_plan=("gated_delta", "attention"),
            gated_delta_residual_scale_init=0.0,
        )
    )
    assert float(model.blocks[0].residual_scale.detach()) == 0.0
    assert float(model.blocks[1].residual_scale.detach()) == -2.0


def test_association_auxiliary_aligns_matching_writes_and_supervises_events() -> None:
    torch.manual_seed(1404)
    batch = generate_mqar_batch(3, 4, 3, 24, seed=1405)
    model = HybridMemoryLM(
        HybridMemoryConfig(
            vocab_size=197,
            model_dim=32,
            layer_plan=("gated_delta",),
            gated_delta_heads=4,
        )
    )
    output = model(batch.inputs, return_diagnostics=True)
    loss = gated_delta_association_auxiliary_loss(output, batch)
    assert loss.ndim == 0 and torch.isfinite(loss) and loss > 0
    loss.backward()
    mixer = model.blocks[0].mixer
    assert isinstance(mixer, GatedDeltaMemory)
    for parameter in (
        mixer.query_projection.weight,
        mixer.key_projection.weight,
        mixer.write_projection.weight,
    ):
        assert parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0

    model.zero_grad(set_to_none=True)
    output = model(batch.inputs, return_diagnostics=True)
    intermediate = intermediate_retrieval_auxiliary_loss(output, batch)
    assert torch.isfinite(intermediate) and intermediate > 0
    intermediate.backward()
    assert mixer.value_projection.weight.grad is not None
    assert torch.count_nonzero(mixer.value_projection.weight.grad) > 0
