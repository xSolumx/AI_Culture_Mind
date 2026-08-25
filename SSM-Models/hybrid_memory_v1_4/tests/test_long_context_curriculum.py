from __future__ import annotations

import pytest
import torch

from hybrid_memory_v1_4.long_context_curriculum import (
    CONTEXT_LENGTHS,
    EXPECTED_EXPANSION,
    EXPECTED_MODEL_DIM,
    EXPECTED_PARAMETER_COUNT,
    TARGETS_PER_UPDATE,
    _forward_logits,
    _frozen_config,
    _macro_batch,
    _mixer_mode,
    _ordinary_evaluation,
)
from hybrid_memory_v1_4.long_context_diagnostic import _positionwise_evaluation
from hybrid_memory_v1_4.model import HybridMemoryConfig, HybridMemoryLM
from hybrid_memory_v1_4.tokenization import RawByteTokenizer


def test_macro_partitions_score_exactly_the_same_targets() -> None:
    stream = RawByteTokenizer().encode("0123456789abcdef" * 1024)
    reference = None
    starts = []
    for length in CONTEXT_LENGTHS:
        inputs, targets, byte_lengths, start = _macro_batch(
            stream,
            namespace="paired-test",
            macro_index=7,
            sequence_length=length,
            device=torch.device("cpu"),
        )
        assert inputs.shape == targets.shape == byte_lengths.shape
        assert targets.numel() == TARGETS_PER_UPDATE
        assert int(byte_lengths.sum()) == TARGETS_PER_UPDATE
        starts.append(start)
        flattened = targets.flatten()
        if reference is None:
            reference = flattened
        else:
            assert torch.equal(flattened, reference)
    assert len(set(starts)) == 1


def test_live_state_chunking_preserves_logits_and_gradients() -> None:
    torch.manual_seed(17)
    config = HybridMemoryConfig(
        vocab_size=32,
        model_dim=16,
        layer_plan=("gated_delta", "attention"),
        attention_heads=2,
        attention_window_size=8,
        gated_delta_key_dim=8,
        gated_delta_value_dim=4,
        expansion=1,
        use_local_conv=False,
    )
    full_model = HybridMemoryLM(config).double()
    chunked_model = HybridMemoryLM(config).double()
    chunked_model.load_state_dict(full_model.state_dict())
    tokens = torch.randint(0, config.vocab_size, (2, 16))

    full_logits = _forward_logits(full_model, tokens, chunk_size=64)
    chunked_logits = _forward_logits(chunked_model, tokens, chunk_size=4)
    torch.testing.assert_close(chunked_logits, full_logits, rtol=2e-10, atol=2e-10)

    full_logits.square().mean().backward()
    chunked_logits.square().mean().backward()
    for (full_name, full_parameter), (chunk_name, chunk_parameter) in zip(
        full_model.named_parameters(), chunked_model.named_parameters(), strict=True
    ):
        assert full_name == chunk_name
        assert full_parameter.grad is not None
        assert chunk_parameter.grad is not None
        torch.testing.assert_close(
            chunk_parameter.grad,
            full_parameter.grad,
            rtol=2e-9,
            atol=2e-10,
            msg=lambda message, name=full_name: f"{name}: {message}",
        )


def test_mixer_suppression_restores_residual_scale() -> None:
    model = HybridMemoryLM(
        HybridMemoryConfig(
            vocab_size=32,
            model_dim=16,
            layer_plan=("gated_delta", "attention"),
            attention_heads=2,
            gated_delta_key_dim=8,
            gated_delta_value_dim=4,
            expansion=1,
            use_local_conv=False,
        )
    )
    gated_delta = next(block for block in model.blocks if block.kind == "gated_delta")
    original = gated_delta.residual_scale.detach().clone()
    with _mixer_mode(model, "gated_delta_off"):
        torch.testing.assert_close(
            gated_delta.residual_scale,
            torch.full_like(gated_delta.residual_scale, -30.0),
        )
    torch.testing.assert_close(gated_delta.residual_scale, original)


def test_frozen_shape_matches_preregistration() -> None:
    config = _frozen_config(512)
    model = HybridMemoryLM(config)
    assert config.model_dim == EXPECTED_MODEL_DIM
    assert config.expansion == EXPECTED_EXPANSION
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        EXPECTED_PARAMETER_COUNT
    )


def test_position_bins_reconstruct_ordinary_byte_loss() -> None:
    torch.manual_seed(23)
    stream = RawByteTokenizer().encode("The small bird crossed the river. " * 400)
    model = HybridMemoryLM(
        HybridMemoryConfig(
            vocab_size=256,
            model_dim=16,
            layer_plan=("attention",),
            attention_heads=2,
            attention_window_size=128,
            expansion=1,
            use_local_conv=False,
        )
    )
    ordinary = _ordinary_evaluation(
        model,
        stream,
        sequence_length=256,
        macro_batches=1,
        device=torch.device("cpu"),
    )
    bins = _positionwise_evaluation(
        model,
        stream,
        sequence_length=256,
        bin_ends=(64, 128, 256),
        macro_batches=1,
        device=torch.device("cpu"),
    )
    weighted_bprb = sum(
        row["bits_per_raw_byte"] * row["scored_raw_bytes"] for row in bins
    )
    total_bytes = sum(row["scored_raw_bytes"] for row in bins)
    reconstructed_bprb = weighted_bprb / total_bytes
    assert reconstructed_bprb == pytest.approx(ordinary["bits_per_raw_byte"], rel=1e-6)
