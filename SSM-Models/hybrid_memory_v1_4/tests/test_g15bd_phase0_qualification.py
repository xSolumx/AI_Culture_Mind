"""Contracts for the G15B-D coupled residual-delta qualification."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hybrid_memory_v1_4.g15bd_phase0_qualification import qualify
from hybrid_memory_v1_4.model import HybridMemoryConfig, HybridMemoryLM, parameter_count


def _model(mode: str) -> HybridMemoryLM:
    torch.manual_seed(2581)
    return HybridMemoryLM(
        HybridMemoryConfig(
            vocab_size=97,
            model_dim=32,
            layer_plan=("transactional_delta",),
            gated_delta_heads=4,
            gated_delta_key_dim=8,
            gated_delta_value_dim=8,
            transactional_controller_mode="full",
            transactional_effective_edit_gate_mode=mode,  # type: ignore[arg-type]
            use_local_conv=True,
            conv_kernel=4,
            expansion=2,
            dropout=0.0,
            tie_embeddings=False,
        )
    )


def test_g15bd_arms_are_parameter_state_and_initialization_matched() -> None:
    product = _model("product")
    residual = _model("residual_delta")
    assert parameter_count(product) == parameter_count(residual)
    assert product.state_capacity_bytes(1, torch.float32) == residual.state_capacity_bytes(
        1, torch.float32
    )
    assert product.state_dict().keys() == residual.state_dict().keys()
    for name, tensor in product.state_dict().items():
        assert torch.equal(tensor, residual.state_dict()[name]), name


def test_g15bd_phase0_qualification_passes_on_semantic_cpu_path() -> None:
    report = qualify(torch.device("cpu"))
    assert report["passed"] is True
    assert report["matched_initialization"]["passed"] is True
    assert report["bounded_and_direct_law"]["passed"] is True
    assert report["bounded_and_direct_law"]["direct_residual_delta_law"][
        "passed"
    ] is True
    assert all(
        row["passed"]
        for arm in report["execution"].values()
        for row in arm.values()
    )
    assert all(row["passed"] for row in report["gradient_reach"].values())
