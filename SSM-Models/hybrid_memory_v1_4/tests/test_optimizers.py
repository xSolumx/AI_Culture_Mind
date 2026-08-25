from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from hybrid_memory_v1_4.model import HybridMemoryConfig, HybridMemoryLM
from hybrid_memory_v1_4.optimizers import (
    BlockScalarSecondMomentAdamW,
    HarmonicMuonAdamW,
    ScalarSecondMomentAdamW,
    build_optimizer,
    partition_optimizer_parameters,
)
from hybrid_memory_v1_4.successor_screen import _retention_safe_config


def _model() -> HybridMemoryLM:
    return HybridMemoryLM(replace(_retention_safe_config(), vocab_size=256))


def test_optimizer_partition_is_complete_disjoint_and_semantic() -> None:
    model = _model()
    partition = partition_optimizer_parameters(model)
    grouped = [parameter for group in partition.named_groups for _, parameter in group]
    expected = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    assert len(grouped) == len(expected)
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in expected
    }
    assert len({id(parameter) for parameter in grouped}) == len(grouped)
    assert all(parameter.ndim == 2 for _, parameter in partition.muon)
    assert any(name == "embedding.weight" for name, _ in partition.adamw_no_decay)
    assert any("write_projection" in name for name, _ in partition.scalar_adamw)
    assert any("decay_projection" in name for name, _ in partition.scalar_adamw)
    assert any("local_conv.conv.weight" in name for name, _ in partition.adamw_decay)


def test_decoupled_erase_and_spin_transport_are_control_parameters() -> None:
    model = HybridMemoryLM(
        HybridMemoryConfig(
            vocab_size=64,
            model_dim=16,
            layer_plan=("gated_delta_v2", "spin_dirac"),
            gated_delta_heads=2,
            spin_dirac_heads=2,
            use_local_conv=False,
        )
    )
    partition = partition_optimizer_parameters(model)
    control_names = {name for name, _ in partition.scalar_adamw}
    erase_names = {
        name for name, _ in model.named_parameters() if "erase_projection" in name
    }
    coordinate_names = {
        name for name, _ in model.named_parameters() if "coordinate_projection" in name
    }
    assert erase_names and erase_names <= control_names
    assert coordinate_names and coordinate_names <= control_names


def test_scalar_second_moment_is_orthogonally_covariant() -> None:
    torch.manual_seed(7)
    parameter = torch.nn.Parameter(torch.randn(11, dtype=torch.float64))
    matrix, _ = torch.linalg.qr(torch.randn(11, 11, dtype=torch.float64))
    mapped = torch.nn.Parameter(matrix @ parameter.detach())
    optimizer = ScalarSecondMomentAdamW([parameter], lr=2e-3, weight_decay=0.01)
    mapped_optimizer = ScalarSecondMomentAdamW([mapped], lr=2e-3, weight_decay=0.01)
    for _ in range(8):
        gradient = torch.randn_like(parameter)
        parameter.grad = gradient
        mapped.grad = matrix @ gradient
        optimizer.step()
        mapped_optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        mapped_optimizer.zero_grad(set_to_none=True)
        torch.testing.assert_close(mapped, matrix @ parameter, atol=2e-14, rtol=2e-14)


def test_scalar_second_moment_rejects_zero_epsilon_and_handles_zero_gradient() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    with pytest.raises(ValueError, match="epsilon"):
        ScalarSecondMomentAdamW([parameter], eps=0.0)
    optimizer = ScalarSecondMomentAdamW([parameter])
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    torch.testing.assert_close(parameter, torch.tensor([1.0]))


def test_block_scalar_second_moment_is_rowwise_orthogonally_covariant() -> None:
    torch.manual_seed(9)
    parameter = torch.nn.Parameter(torch.randn(5, 7, dtype=torch.float64))
    matrix, _ = torch.linalg.qr(torch.randn(7, 7, dtype=torch.float64))
    mapped = torch.nn.Parameter(parameter.detach() @ matrix)
    optimizer = BlockScalarSecondMomentAdamW([parameter], lr=2e-3, weight_decay=0.01)
    mapped_optimizer = BlockScalarSecondMomentAdamW(
        [mapped], lr=2e-3, weight_decay=0.01
    )
    for _ in range(8):
        gradient = torch.randn_like(parameter)
        parameter.grad = gradient
        mapped.grad = gradient @ matrix
        optimizer.step()
        mapped_optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        mapped_optimizer.zero_grad(set_to_none=True)
        torch.testing.assert_close(mapped, parameter @ matrix, atol=2e-14, rtol=2e-14)
    state = optimizer.state[parameter]
    assert state["exp_avg_sq"].shape == (5, 1)


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="PyTorch Muon unavailable")
def test_harmonic_optimizer_steps_and_round_trips_state() -> None:
    torch.manual_seed(11)
    model = _model()
    optimizer = HarmonicMuonAdamW(model)
    tokens = torch.randint(0, 256, (2, 12))
    loss = model(tokens, delta_scan_mode="parallel")["logits"].square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    state = optimizer.state_dict()
    assert state["optimizer"] == "HarmonicMuonAdamW"
    assert sum(row["parameters"] for row in state["partition"]) == sum(
        parameter.numel() for parameter in model.parameters()
    )

    restored_model = _model()
    restored_model.load_state_dict(copy.deepcopy(model.state_dict()))
    restored = HarmonicMuonAdamW(restored_model)
    restored.load_state_dict(copy.deepcopy(state))
    assert len(restored.param_groups) == len(optimizer.param_groups)
    torch.manual_seed(12)
    for original, replica in zip(
        model.parameters(), restored_model.parameters(), strict=True
    ):
        gradient = torch.randn_like(original)
        original.grad = gradient
        replica.grad = gradient.clone()
    optimizer.step()
    restored.step()
    for original, replica in zip(
        model.parameters(), restored_model.parameters(), strict=True
    ):
        torch.testing.assert_close(original, replica)


def test_optimizer_factory_rejects_unknown_name() -> None:
    model = _model()
    assert isinstance(build_optimizer(model, "adamw"), torch.optim.AdamW)
    with pytest.raises(ValueError, match="unknown optimizer"):
        build_optimizer(model, "magic")
