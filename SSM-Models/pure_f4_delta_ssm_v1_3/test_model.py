from __future__ import annotations

import copy

import pytest
import torch

from .model import AlbertInvariantReadout, ExceptionalDeltaConfig, ExceptionalDeltaLM


def _tiny(**changes) -> ExceptionalDeltaLM:
    values = {
        "d_model": 16,
        "num_layers": 1,
        "memory_width": 3,
        "update_rank": 2,
        "d_conv": 3,
        "action_algebra": "e6",
        "channel_mixer": "jordan",
    }
    values.update(changes)
    return ExceptionalDeltaLM(ExceptionalDeltaConfig(**values))


def test_model_shape_causality_backward_and_finite_state() -> None:
    torch.manual_seed(5)
    model = _tiny()
    tokens = torch.randint(0, 256, (2, 6))
    result = model(tokens)
    assert result["logits"].shape == (2, 6, 256)
    assert result["states"][0].memory.shape == (2, 3, 27)
    assert result["states"][0].convolution.shape == (2, 16, 2)
    assert torch.isfinite(result["states"][0].memory).all()
    result["logits"].square().mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    model.eval()
    left = torch.randint(0, 256, (1, 7))
    right = left.clone()
    right[:, 4:] = torch.randint(0, 256, (1, 3))
    with torch.no_grad():
        left_logits = model(left)["logits"]
        right_logits = model(right)["logits"]
    torch.testing.assert_close(left_logits[:, :4], right_logits[:, :4])


def test_full_model_parallel_recurrent_output_and_gradient_parity() -> None:
    torch.manual_seed(7)
    recurrent = _tiny(channel_mixer="none").double()
    parallel = copy.deepcopy(recurrent)
    tokens = torch.randint(0, 256, (1, 4))
    expected = recurrent(tokens, scan_mode="recurrent")["logits"]
    actual = parallel(tokens, scan_mode="parallel")["logits"]
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-10)
    gradient = torch.randn_like(actual)
    expected_gradients = torch.autograd.grad(expected, tuple(recurrent.parameters()), gradient)
    actual_gradients = torch.autograd.grad(actual, tuple(parallel.parameters()), gradient)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=2e-8, atol=2e-8
        )


def test_every_action_tier_and_update_parameterization_executes() -> None:
    tokens = torch.randint(0, 256, (1, 2))
    for algebra in ("identity", "spin8", "spin9", "f4", "e6"):
        model = _tiny(
            action_algebra=algebra,
            channel_mixer="none",
            local_mixer="none",
            key_parameterization="unconstrained",
            retention_parameterization="unconstrained",
            retention_bias=0.9,
        )
        result = model(tokens)
        assert torch.isfinite(result["logits"]).all()


def test_subgroup_schedule_is_explicit_but_not_forced_monotone() -> None:
    config = ExceptionalDeltaConfig(
        d_model=8,
        num_layers=4,
        memory_width=2,
        action_schedule=("spin8", "spin9", "f4", "e6"),
    )
    model = ExceptionalDeltaLM(config)
    assert [block.action.coordinate_dim for block in model.blocks] == [28, 36, 52, 78]
    reverse = ExceptionalDeltaLM(
        ExceptionalDeltaConfig(
            d_model=8,
            num_layers=4,
            memory_width=2,
            action_schedule=("e6", "f4", "spin9", "spin8"),
        )
    )
    assert [block.action.coordinate_dim for block in reverse.blocks] == [78, 52, 36, 28]


def test_full_streaming_state_makes_chunked_model_exact() -> None:
    torch.manual_seed(23)
    model = _tiny(channel_mixer="none").double().eval()
    tokens = torch.randint(0, 256, (1, 7))
    with torch.no_grad():
        full = model(tokens)["logits"]
        first = model(tokens[:, :3])
        second = model(tokens[:, 3:], states=first["states"])
    chunked = torch.cat((first["logits"], second["logits"]), dim=1)
    torch.testing.assert_close(chunked, full, rtol=2e-11, atol=2e-11)


def test_custom_generator_bank_removes_exceptional_action_ceiling() -> None:
    generator = torch.tensor([[[0.0, -1.0], [1.0, 0.0]]], dtype=torch.float64)
    config = ExceptionalDeltaConfig(
        d_model=8,
        num_layers=1,
        memory_width=2,
        update_rank=1,
        local_mixer="none",
        channel_mixer="none",
    )
    model = ExceptionalDeltaLM(config, generator_banks=[generator]).double()
    result = model(torch.randint(0, 256, (1, 3)))
    assert result["states"][0].memory.shape == (1, 2, 2)


def test_padding_mask_preserves_complete_recurrent_state() -> None:
    torch.manual_seed(31)
    model = _tiny(channel_mixer="none").double().eval()
    prefix = torch.randint(0, 256, (1, 4))
    padded = torch.cat((prefix, torch.randint(0, 256, (1, 3))), dim=1)
    mask = torch.tensor([[True, True, True, True, False, False, False]])
    with torch.no_grad():
        expected = model(prefix)["states"][0]
        actual = model(padded, valid_mask=mask)["states"][0]
    torch.testing.assert_close(actual.memory, expected.memory, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(
        actual.convolution, expected.convolution, rtol=2e-11, atol=2e-11
    )


def test_all_e6_action_geometries_execute_end_to_end() -> None:
    tokens = torch.randint(0, 256, (1, 3))
    coordinate_dimensions = {}
    for geometry in ("direct", "polar", "cartan"):
        model = _tiny(action_geometry=geometry, channel_mixer="none")
        result = model(tokens)
        assert torch.isfinite(result["logits"]).all()
        coordinate_dimensions[geometry] = model.blocks[0].action.coordinate_dim
    assert coordinate_dimensions == {"direct": 78, "polar": 78, "cartan": 106}


def test_auto_scan_selects_parallel_for_sequences_and_recurrent_for_tokens() -> None:
    torch.manual_seed(41)
    model = _tiny(channel_mixer="none").double().eval()
    sequence = torch.randint(0, 256, (1, 5))
    with torch.no_grad():
        automatic = model(sequence, scan_mode="auto")["logits"]
        parallel = model(sequence, scan_mode="parallel")["logits"]
        one_auto = model(sequence[:, :1], scan_mode="auto")["logits"]
        one_recurrent = model(sequence[:, :1], scan_mode="recurrent")["logits"]
    torch.testing.assert_close(automatic, parallel, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(one_auto, one_recurrent, rtol=2e-11, atol=2e-11)


def test_albert_readout_preserves_scale_information_and_gradients() -> None:
    torch.manual_seed(43)
    readout = AlbertInvariantReadout().double()
    value = torch.randn(2, 3, 27, dtype=torch.float64, requires_grad=True)
    first = readout(value)
    second = readout(2.0 * value)
    assert first.shape[-1] == 30
    # RMS-normalized directions agree, but trace/energy/determinant features do not.
    torch.testing.assert_close(first[..., :27], second[..., :27], rtol=2e-6, atol=2e-6)
    assert float((first[..., 27:] - second[..., 27:]).abs().max().detach()) > 1e-3
    second.square().mean().backward()
    assert torch.isfinite(value.grad).all()


def test_custom_bank_auto_readout_is_vector_and_forced_albert_refuses() -> None:
    generator = torch.tensor([[[0.0, -1.0], [1.0, 0.0]]], dtype=torch.float64)
    config = ExceptionalDeltaConfig(
        d_model=8,
        num_layers=1,
        memory_width=2,
        update_rank=1,
        local_mixer="none",
        channel_mixer="none",
        readout_mode="auto",
    )
    model = ExceptionalDeltaLM(config, generator_banks=[generator])
    assert model.blocks[0].output_projection.in_features == 2
    with pytest.raises(ValueError, match="requires a 27D action"):
        ExceptionalDeltaLM(
            ExceptionalDeltaConfig(
                d_model=8,
                num_layers=1,
                memory_width=2,
                update_rank=1,
                readout_mode="albert_invariants",
            ),
            generator_banks=[generator],
        )
