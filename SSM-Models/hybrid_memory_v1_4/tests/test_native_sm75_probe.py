"""Fail-closed unit contracts for native runtime qualification."""

from __future__ import annotations

import torch
from torch import nn

from hybrid_memory_v1_4 import native_sm75_probe


def _mock_cuda_accounting(monkeypatch: object) -> None:
    monkeypatch.setattr(native_sm75_probe, "_require_sm75", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _seed: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device=None: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device=None: 0)


def test_finite_layer_probe_rejects_nonfinite_outputs(monkeypatch: object) -> None:
    _mock_cuda_accounting(monkeypatch)

    class NonfiniteLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gain = nn.Parameter(torch.ones(()))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs * self.gain * torch.tensor(float("nan"))

    try:
        native_sm75_probe._finite_layer_probe(
            NonfiniteLayer,
            training=True,
            length=2,
            hidden_size=2,
        )
    except RuntimeError as error:
        assert "finiteness or complete-gradient" in str(error)
    else:
        raise AssertionError("non-finite native output was accepted")


def test_finite_layer_probe_requires_every_parameter_gradient(
    monkeypatch: object,
) -> None:
    _mock_cuda_accounting(monkeypatch)

    class DisconnectedLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.used = nn.Parameter(torch.ones(()))
            self.unused = nn.Parameter(torch.ones(()))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs * self.used

    try:
        native_sm75_probe._finite_layer_probe(
            DisconnectedLayer,
            training=True,
            length=2,
            hidden_size=2,
        )
    except RuntimeError as error:
        assert "parameter_grads=1/2" in str(error)
    else:
        raise AssertionError("missing native parameter gradient was accepted")
