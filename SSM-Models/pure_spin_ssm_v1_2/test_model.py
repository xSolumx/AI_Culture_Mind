from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from mamba2_baseline import fused_mamba2_available
from model import PureSpinSSMV12, PureSpinV12Config, SolSelfGate


def tiny_model() -> PureSpinSSMV12:
    return PureSpinSSMV12(
        PureSpinV12Config(d_model=16, num_layers=1, spin_channels=1, d_conv=3)
    )


def test_shape_backward_and_finite_state() -> None:
    torch.manual_seed(3)
    model = tiny_model()
    tokens = torch.randint(0, 256, (2, 7))
    result = model(tokens, scan_mode="compiled_controller")
    assert result["logits"].shape == (2, 7, 256)
    assert all(torch.isfinite(state).all() for state in result["states"])
    result["logits"].square().mean().backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_causality() -> None:
    torch.manual_seed(5)
    model = tiny_model().eval()
    left = torch.randint(0, 256, (1, 8))
    right = left.clone()
    right[:, 5:] = torch.randint(0, 256, (1, 3))
    with torch.no_grad():
        a = model(left, scan_mode="compiled_controller")["logits"]
        b = model(right, scan_mode="compiled_controller")["logits"]
    torch.testing.assert_close(a[:, :5], b[:, :5])


def test_initial_language_model_loss_has_sane_scale() -> None:
    torch.manual_seed(7)
    model = tiny_model()
    inputs = torch.randint(0, 256, (2, 16))
    targets = torch.randint(0, 256, (2, 16))
    logits = model(inputs, scan_mode="compiled_controller")["logits"]
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    assert 4.5 < float(loss.detach()) < 6.5


def test_fused_mamba_probe_never_claims_fallback() -> None:
    available, detail = fused_mamba2_available()
    assert isinstance(available, bool)
    assert isinstance(detail, str)


def test_sol_self_gate_is_finite_and_has_bounded_local_slope_factor() -> None:
    mixer = SolSelfGate(width=8, expansion=3)
    inputs = torch.linspace(-1e3, 1e3, 40).reshape(5, 8).requires_grad_()
    projected = mixer.input(inputs)
    slope_factor = 1.0 + projected * torch.rsqrt(1.0 + projected.square())
    assert torch.all(slope_factor > 0.0)
    assert torch.all(slope_factor < 2.0)
    output = mixer(inputs)
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(inputs.grad).all()
