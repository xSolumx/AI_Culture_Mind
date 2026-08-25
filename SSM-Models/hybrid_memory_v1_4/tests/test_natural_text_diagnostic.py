from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hybrid_memory_v1_4.natural_text_diagnostic as diagnostic
from hybrid_memory_v1_4.natural_text_screen import _build_model


def test_ablation_restores_residual_scales(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _build_model("hybrid_v1_4_5", torch.device("cpu"))
    original = [block.residual_scale.detach().clone() for block in model.blocks]

    def fake_evaluate(*args: object) -> dict[str, float | int | bool]:
        return {"bits_per_byte": 1.0, "finite": True}

    monkeypatch.setattr(diagnostic, "_evaluate", fake_evaluate)
    validation = torch.arange(512, dtype=torch.long) % 256
    diagnostic._evaluate_ablation(model, validation, torch.device("cpu"), 0)
    for block, expected in zip(model.blocks, original, strict=True):
        torch.testing.assert_close(block.residual_scale, expected)
