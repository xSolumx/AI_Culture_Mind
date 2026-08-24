"""End-to-end contracts for the complete hybrid memory language model."""

from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from delta_product_reference import DeltaProductReferenceLayer, GatedMLP
from hybrid_memory_v1_4.attention import (
    AttentionState,
    CausalSelfAttention,
)
from hybrid_memory_v1_4.model import (
    CausalDepthwiseConv1d,
    DeltaProductState,
    HybridMemoryConfig,
    HybridMemoryLM,
    SelectedBlockState,
    StructuredSpin8State,
    parameter_count,
)
from hybrid_memory_v1_4.selected_block import SelectedBlockMemory
from hybrid_memory_v1_4.structured_memory import StructuredSpin8Memory


def _small_config(**overrides: object) -> HybridMemoryConfig:
    values = {
        "vocab_size": 31,
        "model_dim": 8,
        "layer_plan": (
            "delta_product",
            "attention",
            "selected_block",
            "structured_spin8",
        ),
        "attention_heads": 2,
        "attention_window_size": 5,
        "delta_heads": 2,
        "delta_num_householder": 2,
        "selected_heads": 1,
        "selected_blocks": 2,
        "selected_slots_per_block": 2,
        "selected_value_dim": 4,
        "selected_update_rank": 1,
        "structured_channels": 1,
        "structured_rungs": (3, 4, 6, 8),
        "structured_controller_rank": 2,
        "structured_retention_min": 0.1,
        "structured_retention_max": 0.9,
        "structured_hard_eval": True,
        "use_local_conv": True,
        "conv_kernel": 3,
        "expansion": 2,
        "dropout": 0.0,
    }
    values.update(overrides)
    return HybridMemoryConfig(**values)  # type: ignore[arg-type]


def _assert_states_close(
    actual: tuple[object, ...], expected: tuple[object, ...]
) -> None:
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        assert type(left) is type(right)
        if isinstance(left, AttentionState) and isinstance(right, AttentionState):
            assert left.position == right.position
            torch.testing.assert_close(left.key_cache, right.key_cache)
            torch.testing.assert_close(left.value_cache, right.value_cache)
        elif isinstance(
            left, (DeltaProductState, SelectedBlockState, StructuredSpin8State)
        ) and isinstance(
            right, (DeltaProductState, SelectedBlockState, StructuredSpin8State)
        ):
            torch.testing.assert_close(left.memory, right.memory)
            torch.testing.assert_close(left.convolution, right.convolution)
        else:
            pytest.fail("unexpected state type")


def test_explicit_schedule_builds_all_four_uniform_shells() -> None:
    config = _small_config()
    model = HybridMemoryLM(config)
    assert config.num_layers == 4
    assert model.layer_plan == config.layer_plan
    assert [block.kind for block in model.blocks] == list(config.layer_plan)
    assert isinstance(model.blocks[0].mixer, DeltaProductReferenceLayer)
    assert isinstance(model.blocks[1].mixer, CausalSelfAttention)
    assert isinstance(model.blocks[2].mixer, SelectedBlockMemory)
    assert isinstance(model.blocks[3].mixer, StructuredSpin8Memory)
    assert model.blocks[1].local_conv is None
    assert model.blocks[0].local_conv is not None
    assert model.blocks[2].local_conv is not None
    assert model.blocks[3].local_conv is not None
    for block in model.blocks:
        assert isinstance(block.ffn, GatedMLP)
        assert hasattr(block, "mixer_norm")
        assert hasattr(block, "input_projection")
        assert hasattr(block, "residual_scale")
        assert hasattr(block, "ffn_norm")
        assert hasattr(block, "dropout")


def test_layer_diagnostics_are_explicit_and_opt_in() -> None:
    torch.manual_seed(28)
    model = HybridMemoryLM(_small_config()).double().eval()
    tokens = torch.randint(0, model.config.vocab_size, (2, 5))

    default = model(tokens)
    assert "diagnostics" not in default
    detailed = model(tokens, return_diagnostics=True)
    assert set(detailed) == {"logits", "states", "diagnostics"}
    torch.testing.assert_close(detailed["logits"], default["logits"])

    diagnostics = detailed["diagnostics"]
    assert isinstance(diagnostics, tuple)
    assert len(diagnostics) == len(model.layer_plan)
    delta, attention, selected, structured = diagnostics
    assert delta == {"kind": "delta_product", "scan_mode": "parallel"}
    assert attention == {"kind": "attention", "position": 5, "cache_length": 4}

    assert isinstance(selected, dict)
    assert selected["write_block_logits"].shape == (2, 5, 1, 2)
    assert selected["erase_block_logits"].shape == (2, 5, 1, 2)
    assert selected["read_block_logits"].shape == (2, 5, 1, 2)
    assert selected["write_gate"].shape == (2, 5, 1)
    assert selected["retention"].shape == (2, 5, 1)

    assert isinstance(structured, dict)
    assert structured["rung_probabilities"].shape == (2, 5, 1, 4)
    assert structured["selected_rung"].shape == (2, 5, 1)
    assert structured["transport_coordinates"].shape == (2, 5, 1, 28)
    assert structured["transition_drive"].shape == (2, 5, 1, 3, 8)
    assert structured["hard_selection"] is True

    with pytest.raises(TypeError, match="return_diagnostics"):
        model(tokens, return_diagnostics=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="return_diagnostics"):
        model.blocks[0](model.embedding(tokens), return_diagnostics=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"layer_plan": ()}, ValueError),
        ({"layer_plan": ["attention"]}, TypeError),
        ({"layer_plan": ("unknown",)}, ValueError),
        ({"model_dim": 0}, ValueError),
        ({"delta_heads": 3}, ValueError),
        ({"attention_window_size": 0}, ValueError),
        ({"conv_kernel": 0}, ValueError),
        ({"dropout": 1.0}, ValueError),
        ({"use_local_conv": 1}, TypeError),
        ({"selected_update_rank": 3}, ValueError),
        ({"structured_channels": 0}, ValueError),
        ({"structured_rungs": (3, 3, 8)}, ValueError),
        ({"structured_controller_rank": 9}, ValueError),
        ({"structured_retention_min": -0.1}, ValueError),
        ({"structured_retention_max": 1.0}, ValueError),
        ({"structured_hard_eval": 1}, TypeError),
    ],
)
def test_config_fails_closed(
    overrides: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        _small_config(**overrides)


def test_config_is_frozen() -> None:
    config = _small_config()
    with pytest.raises(FrozenInstanceError):
        config.dropout = 0.2  # type: ignore[misc]


def test_full_and_arbitrary_chunk_replay_match() -> None:
    torch.manual_seed(20)
    model = HybridMemoryLM(_small_config()).double().eval()
    tokens = torch.randint(0, model.config.vocab_size, (2, 13))
    with torch.no_grad():
        full = model(
            tokens,
            delta_scan_mode="recurrent",
            selected_scan_mode="physical_gather",
            structured_scan_mode="recurrent",
        )
        pieces = []
        states = None
        start = 0
        for stop in (1, 4, 5, 10, 13):
            chunk = model(
                tokens[:, start:stop],
                states,
                delta_scan_mode="recurrent",
                selected_scan_mode="physical_gather",
                structured_scan_mode="recurrent",
            )
            pieces.append(chunk["logits"])
            states = chunk["states"]
            start = stop
    torch.testing.assert_close(
        torch.cat(pieces, dim=1), full["logits"], rtol=2e-9, atol=2e-10
    )
    assert states is not None
    _assert_states_close(states, full["states"])


def test_token_step_replay_matches_full_forward() -> None:
    torch.manual_seed(21)
    model = HybridMemoryLM(_small_config()).double().eval()
    tokens = torch.randint(0, model.config.vocab_size, (2, 11))
    with torch.no_grad():
        full = model(
            tokens,
            delta_scan_mode="recurrent",
            selected_scan_mode="physical_gather",
            structured_scan_mode="recurrent",
        )
        states = None
        pieces = []
        for position in range(tokens.shape[1]):
            logits, states = model.step(tokens[:, position], states)
            pieces.append(logits)
    torch.testing.assert_close(
        torch.stack(pieces, dim=1), full["logits"], rtol=2e-9, atol=2e-10
    )
    assert states is not None
    _assert_states_close(states, full["states"])


def test_delta_parallel_and_recurrent_parity() -> None:
    torch.manual_seed(22)
    model = (
        HybridMemoryLM(
            _small_config(layer_plan=("delta_product",), use_local_conv=False)
        )
        .double()
        .eval()
    )
    tokens = torch.randint(0, model.config.vocab_size, (2, 9))
    recurrent = model(tokens, delta_scan_mode="recurrent")
    parallel = model(tokens, delta_scan_mode="parallel")
    torch.testing.assert_close(
        parallel["logits"], recurrent["logits"], rtol=3e-10, atol=3e-11
    )
    _assert_states_close(parallel["states"], recurrent["states"])


def test_selected_physical_and_dense_scan_parity() -> None:
    torch.manual_seed(23)
    model = (
        HybridMemoryLM(
            _small_config(layer_plan=("selected_block",), use_local_conv=False)
        )
        .double()
        .eval()
    )
    tokens = torch.randint(0, model.config.vocab_size, (2, 7))
    physical = model(tokens, selected_scan_mode="physical_gather")
    recurrent = model(tokens, selected_scan_mode="dense_recurrent")
    parallel = model(tokens, selected_scan_mode="dense_parallel")
    for other in (recurrent, parallel):
        torch.testing.assert_close(
            other["logits"], physical["logits"], rtol=2e-10, atol=2e-11
        )
        _assert_states_close(other["states"], physical["states"])


def test_structured_recurrent_and_parallel_integration_parity() -> None:
    torch.manual_seed(27)
    model = (
        HybridMemoryLM(
            _small_config(layer_plan=("structured_spin8",), use_local_conv=False)
        )
        .double()
        .eval()
    )
    tokens = torch.randint(0, model.config.vocab_size, (2, 8))
    recurrent = model(tokens, structured_scan_mode="recurrent")
    parallel = model(tokens, structured_scan_mode="parallel")
    torch.testing.assert_close(
        parallel["logits"], recurrent["logits"], rtol=3e-10, atol=3e-11
    )
    _assert_states_close(parallel["states"], recurrent["states"])


def test_cached_convolution_full_chunk_and_mask_semantics() -> None:
    torch.manual_seed(24)
    conv = CausalDepthwiseConv1d(4, 3).double().eval()
    inputs = torch.randn(2, 8, 4, dtype=torch.float64)
    full, full_cache = conv(inputs)
    first, cache = conv(inputs[:, :3])
    second, cache = conv(inputs[:, 3:], cache)
    torch.testing.assert_close(torch.cat((first, second), dim=1), full)
    torch.testing.assert_close(cache, full_cache)
    assert cache.shape == (2, 4, 2)

    mask = torch.tensor(
        [[True, True, False, True], [True, False, True, True]], dtype=torch.bool
    )
    _, masked_cache = conv(inputs[:, :4], valid_mask=mask)
    expected = []
    for batch in range(2):
        expected.append(inputs[batch, :4][mask[batch]][-2:].T)
    torch.testing.assert_close(masked_cache, torch.stack(expected))


def test_cache_shape_type_and_completeness_guards() -> None:
    model = HybridMemoryLM(_small_config()).double().eval()
    tokens = torch.randint(0, model.config.vocab_size, (2, 3))
    output = model(tokens)
    delta, attention, selected, structured = output["states"]
    assert isinstance(delta, DeltaProductState)
    assert isinstance(attention, AttentionState)
    assert isinstance(selected, SelectedBlockState)
    assert isinstance(structured, StructuredSpin8State)

    with pytest.raises(TypeError, match="omit"):
        model(tokens, (delta, None, selected, structured))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="convolution cache"):
        model(
            tokens,
            (
                replace(delta, convolution=delta.convolution[..., :1]),
                attention,
                selected,
                structured,
            ),
        )
    with pytest.raises(TypeError, match="AttentionState"):
        model(tokens, (delta, delta, selected, structured))  # type: ignore[arg-type]
    bad_attention = replace(attention, key_cache=attention.key_cache.float())
    with pytest.raises(ValueError, match="dtype"):
        model(tokens, (delta, bad_attention, selected, structured))
    with pytest.raises(ValueError, match="memory"):
        model(
            tokens,
            (
                delta,
                attention,
                replace(selected, memory=selected.memory[:, :, :1]),
                structured,
            ),
        )
    with pytest.raises(ValueError, match="structured_spin8 memory"):
        model(
            tokens,
            (
                delta,
                attention,
                selected,
                replace(structured, memory=structured.memory[..., :7]),
            ),
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_actual_and_capacity_byte_report(dtype: torch.dtype) -> None:
    model = HybridMemoryLM(_small_config()).to(dtype=dtype).eval()
    batch = 2
    tokens = torch.randint(0, model.config.vocab_size, (batch, 2))
    states = model(tokens)["states"]
    report = model.state_byte_report(states)
    element_size = torch.empty((), dtype=dtype).element_size()

    delta_memory = batch * 2 * 4 * 4 * element_size
    selected_memory = batch * 1 * 2 * 2 * 4 * element_size
    structured_memory = batch * 1 * 3 * 8 * element_size
    convolution = batch * 8 * 2 * element_size
    attention_actual = 2 * batch * 8 * 2 * element_size + 8
    attention_capacity = 2 * batch * 8 * 4 * element_size + 8
    expected_actual = (
        delta_memory
        + convolution
        + attention_actual
        + selected_memory
        + convolution
        + structured_memory
        + convolution
    )
    expected_capacity = (
        delta_memory
        + convolution
        + attention_capacity
        + selected_memory
        + convolution
        + structured_memory
        + convolution
    )

    assert [layer["kind"] for layer in report["layers"]] == list(model.layer_plan)
    assert report["actual_bytes"] == expected_actual
    assert report["capacity_bytes"] == expected_capacity
    assert report["totals"] == {
        "actual_bytes": expected_actual,
        "capacity_bytes": expected_capacity,
    }
    assert model.state_actual_bytes(states) == expected_actual
    assert model.state_capacity_bytes(batch, dtype) == expected_capacity
    empty = model.state_byte_report(batch_size=batch, dtype=dtype)
    assert empty["actual_bytes"] == 0
    assert empty["capacity_bytes"] == expected_capacity


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(25)
    model = HybridMemoryLM(_small_config()).double().eval()
    path = tmp_path / "hybrid.pt"
    model.save_checkpoint(path, {"note": "roundtrip"})
    loaded, metadata = HybridMemoryLM.load_checkpoint(path)
    loaded = loaded.double().eval()
    assert metadata == {"note": "roundtrip"}
    assert loaded.config == model.config
    assert loaded.config.layer_plan == model.config.layer_plan
    assert loaded.config.structured_channels == 1
    assert loaded.config.structured_rungs == (3, 4, 6, 8)
    assert loaded.config.structured_controller_rank == 2
    assert loaded.config.structured_retention_min == 0.1
    assert loaded.config.structured_retention_max == 0.9
    assert loaded.config.structured_hard_eval is True
    tokens = torch.randint(0, model.config.vocab_size, (1, 6))
    with torch.no_grad():
        torch.testing.assert_close(loaded(tokens)["logits"], model(tokens)["logits"])


def test_all_model_gradients_are_finite() -> None:
    torch.manual_seed(26)
    model = HybridMemoryLM(_small_config()).double().train()
    tokens = torch.randint(0, model.config.vocab_size, (2, 6))
    logits = model(
        tokens,
        delta_scan_mode="recurrent",
        selected_scan_mode="physical_gather",
        structured_scan_mode="recurrent",
    )["logits"]
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_parameter_count_matches_manual_sum() -> None:
    model = HybridMemoryLM(_small_config())
    assert parameter_count(model) == sum(
        parameter.numel() for parameter in model.parameters()
    )
