"""Gated DeltaNet-2 semantic, scan, and common-shell contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hybrid_memory_v1_4.experiments import (
    gated_delta_association_auxiliary_loss,
    intermediate_retrieval_auxiliary_loss,
)
from hybrid_memory_v1_4.gated_delta import GatedDeltaConfig, GatedDeltaMemory
from hybrid_memory_v1_4.gated_delta_v2 import (
    GatedDeltaV2Config,
    GatedDeltaV2Memory,
)
from hybrid_memory_v1_4.model import (
    GatedDeltaState,
    HybridMemoryConfig,
    HybridMemoryLM,
)
from hybrid_memory_v1_4.tasks import generate_mqar_batch, generate_selective_copy_batch


def test_config_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="divisible"):
        GatedDeltaV2Config(10, heads=4)
    with pytest.raises(ValueError, match="initial_erase_strength"):
        GatedDeltaV2Config(8, initial_erase_strength=1.0)
    with pytest.raises(ValueError, match="initial_write_strength"):
        GatedDeltaV2Config(8, initial_write_strength=1.0)


def test_parallel_recurrent_and_arbitrary_chunks_match() -> None:
    torch.manual_seed(1451)
    layer = GatedDeltaV2Memory(
        GatedDeltaV2Config(16, heads=2, key_dim=6, value_dim=5)
    ).double()
    inputs = torch.randn(2, 19, 16, dtype=torch.float64)
    recurrent, recurrent_state = layer(inputs, scan_mode="recurrent")
    parallel, parallel_state = layer(inputs, scan_mode="parallel")
    chunks = []
    state = None
    for start, stop in ((0, 2), (2, 9), (9, 14), (14, 19)):
        output, state = layer(inputs[:, start:stop], state, scan_mode="parallel")
        chunks.append(output)
    torch.testing.assert_close(recurrent, parallel, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(recurrent_state, parallel_state, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(
        recurrent, torch.cat(chunks, dim=1), rtol=1e-11, atol=1e-11
    )
    torch.testing.assert_close(recurrent_state, state, rtol=1e-11, atol=1e-11)


def test_scalar_tied_gates_reduce_exactly_to_gated_deltanet_v1() -> None:
    torch.manual_seed(1452)
    config = GatedDeltaV2Config(
        12,
        heads=3,
        key_dim=4,
        value_dim=3,
        minimum_retention=0.0,
        initial_retention=0.9,
    )
    v2 = GatedDeltaV2Memory(config).double()
    v1 = GatedDeltaMemory(
        GatedDeltaConfig(
            12,
            heads=3,
            key_dim=4,
            value_dim=3,
            minimum_retention=0.0,
            initial_retention=0.9,
        )
    ).double()
    key = F.normalize(torch.randn(2, 7, 3, 4, dtype=torch.float64), dim=-1)
    value = torch.randn(2, 7, 3, 3, dtype=torch.float64)
    beta = torch.rand(2, 7, 3, dtype=torch.float64)
    alpha = torch.rand(2, 7, 3, dtype=torch.float64)
    v1_transition, v1_injection = v1._transitions(key, value, beta, alpha, None)
    v2_transition, v2_injection = v2._transitions(
        key,
        value,
        beta.unsqueeze(-1).expand_as(key),
        beta.unsqueeze(-1).expand_as(value),
        alpha.unsqueeze(-1).expand_as(key),
        None,
    )
    torch.testing.assert_close(v2_transition, v1_transition)
    torch.testing.assert_close(v2_injection, v1_injection)


def test_erase_and_write_can_change_independently() -> None:
    layer = GatedDeltaV2Memory(
        GatedDeltaV2Config(4, heads=1, key_dim=2, value_dim=2)
    ).double()
    key = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float64)
    value = torch.tensor([[[[2.0, -3.0]]]], dtype=torch.float64)
    ones_key = torch.ones_like(key)
    ones_value = torch.ones_like(value)
    zero_key = torch.zeros_like(key)
    zero_value = torch.zeros_like(value)

    no_erase, full_write = layer._transitions(
        key, value, zero_key, ones_value, ones_key, None
    )
    full_erase, no_write = layer._transitions(
        key, value, ones_key, zero_value, ones_key, None
    )
    torch.testing.assert_close(no_erase[0, 0, 0], torch.eye(2, dtype=torch.float64))
    torch.testing.assert_close(
        full_write[0, 0, 0],
        torch.tensor([[2.0, -3.0], [0.0, 0.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        full_erase[0, 0, 0],
        torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
    )
    assert torch.count_nonzero(no_write) == 0


def test_gradients_reach_all_read_write_erase_and_decay_paths() -> None:
    torch.manual_seed(1453)
    layer = GatedDeltaV2Memory(
        GatedDeltaV2Config(16, heads=2, key_dim=7, value_dim=6)
    ).double()
    inputs = torch.randn(2, 11, 16, dtype=torch.float64, requires_grad=True)
    output, state = layer(inputs, scan_mode="parallel")
    (output.square().mean() + state.square().mean()).backward()
    assert inputs.grad is not None and torch.count_nonzero(inputs.grad) > 0
    parameters = dict(layer.named_parameters())
    for name in (
        "query_projection.weight",
        "key_projection.weight",
        "value_projection.weight",
        "erase_projection.weight",
        "write_projection.weight",
        "decay_projection.weight",
        "output_gate.weight",
        "output_projection.weight",
    ):
        gradient = parameters[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert torch.count_nonzero(gradient) > 0, name


def test_common_shell_streaming_and_state_accounting() -> None:
    torch.manual_seed(1454)
    model = HybridMemoryLM(
        HybridMemoryConfig(
            vocab_size=101,
            model_dim=24,
            layer_plan=("gated_delta_v2",),
            gated_delta_heads=3,
            gated_delta_key_dim=6,
            gated_delta_value_dim=5,
        )
    ).double()
    tokens = torch.randint(0, 101, (2, 13))
    full = model(tokens, delta_scan_mode="recurrent")
    state = None
    logits = []
    for start, stop in ((0, 4), (4, 5), (5, 13)):
        chunk = model(tokens[:, start:stop], state, delta_scan_mode="parallel")
        logits.append(chunk["logits"])
        state = chunk["states"]
    torch.testing.assert_close(full["logits"], torch.cat(logits, dim=1))
    assert isinstance(full["states"][0], GatedDeltaState)
    report = model.state_byte_report(full["states"])
    assert report["layers"][0]["kind"] == "gated_delta_v2"
    assert report["layers"][0]["capacity_components"]["memory"] > 0


def test_integrated_auxiliary_losses_dispatch_to_gdn2_with_rectangular_state() -> None:
    torch.manual_seed(1455)
    batch = generate_mqar_batch(3, 4, 3, 24, seed=1456)
    model = HybridMemoryLM(
        HybridMemoryConfig(
            vocab_size=197,
            model_dim=24,
            layer_plan=("gated_delta_v2",),
            gated_delta_heads=3,
            gated_delta_key_dim=6,
            gated_delta_value_dim=5,
            use_local_conv=False,
        )
    )
    output = model(batch.inputs, return_diagnostics=True)
    association = gated_delta_association_auxiliary_loss(output, batch)
    assert association.ndim == 0 and torch.isfinite(association) and association > 0
    association.backward()
    mixer = model.blocks[0].mixer
    assert isinstance(mixer, GatedDeltaV2Memory)
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
    assert mixer.erase_projection.weight.grad is not None
    assert torch.count_nonzero(mixer.erase_projection.weight.grad) > 0


@pytest.mark.parametrize("kind", ("gated_delta_v2", "spin_dirac"))
def test_association_auxiliary_maps_selective_copy_queries_to_items(kind: str) -> None:
    torch.manual_seed(1460)
    batch = generate_selective_copy_batch(3, 8, 3, 32, seed=1461)
    model = HybridMemoryLM(
        HybridMemoryConfig(
            vocab_size=197,
            model_dim=16,
            layer_plan=(kind,),  # type: ignore[arg-type]
            gated_delta_heads=2,
            spin_dirac_heads=2,
            use_local_conv=False,
        )
    )
    output = model(batch.inputs, return_diagnostics=True)
    loss = gated_delta_association_auxiliary_loss(output, batch)
    assert loss.ndim == 0 and torch.isfinite(loss) and loss > 0
    loss.backward()
