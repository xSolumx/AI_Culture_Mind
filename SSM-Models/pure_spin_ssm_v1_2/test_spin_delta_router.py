from __future__ import annotations

import copy

import pytest
import torch

from model import PureSpinSSMV12, PureSpinV12Config
from spin_delta_capability_gate import overwrite_retrieval_batch
from spin_delta_router import (
    CausalLowEntropyRouter,
    RoutedSpinDelta,
    router_supervision_loss,
)


def test_router_controls_are_hard_and_causal() -> None:
    torch.manual_seed(202_608_61)
    router = CausalLowEntropyRouter(width=12, kernel_size=3)
    left = torch.randint(0, 64, (3, 13))
    right = left.clone()
    right[:, 8:] = torch.randint(0, 64, (3, 5))
    expected = router(left)
    actual = router(right)
    torch.testing.assert_close(actual.controls[:, :8], expected.controls[:, :8])
    assert torch.all((actual.controls[..., (0, 3)] == 0) | (actual.controls[..., (0, 3)] == 1))
    torch.testing.assert_close(actual.controls[..., 1:3].sum(dim=-1), torch.ones(3, 13))
    torch.testing.assert_close(actual.controls[..., 4:6].sum(dim=-1), torch.ones(3, 13))


def test_router_supervision_has_finite_gradients() -> None:
    torch.manual_seed(202_608_63)
    inputs, _, oracle = overwrite_retrieval_batch(
        8,
        5,
        generator=torch.Generator().manual_seed(202_608_63),
        return_oracle=True,
    )
    router = CausalLowEntropyRouter(width=16)
    losses = router_supervision_loss(router(inputs), oracle)
    losses["total"].backward()
    assert all(torch.isfinite(loss).all() for loss in losses.values())
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in router.parameters()
    )


def test_routed_spin_delta_preserves_recurrent_parallel_semantics() -> None:
    torch.manual_seed(202_608_67)
    config = PureSpinV12Config(
        d_model=16,
        num_layers=1,
        spin_channels=2,
        recurrence="spin_delta",
    )
    recurrent = RoutedSpinDelta(
        PureSpinSSMV12(config), CausalLowEntropyRouter(width=12)
    )
    parallel = copy.deepcopy(recurrent)
    tokens = torch.randint(0, 64, (2, 9))
    expected = recurrent(tokens, scan_mode="delta_recurrent")["logits"]
    actual = parallel(tokens, scan_mode="delta_parallel")["logits"]
    gradient = torch.randn_like(actual)
    expected_gradients = torch.autograd.grad(
        expected, tuple(recurrent.parameters()), gradient
    )
    actual_gradients = torch.autograd.grad(
        actual, tuple(parallel.parameters()), gradient
    )
    torch.testing.assert_close(actual, expected, rtol=8e-5, atol=8e-5)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=5e-3, atol=7e-3
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_routed_spin_delta_raw_cuda_full_model_gradient_parity() -> None:
    torch.manual_seed(202_608_71)
    config = PureSpinV12Config(
        d_model=16,
        num_layers=1,
        spin_channels=2,
        recurrence="spin_delta",
        group_schedule=(4,),
    )
    semantic = RoutedSpinDelta(
        PureSpinSSMV12(config), CausalLowEntropyRouter(width=12)
    ).cuda()
    raw = copy.deepcopy(semantic)
    tokens = torch.randint(0, 64, (2, 7), device="cuda")
    expected = semantic(tokens, scan_mode="delta_recurrent")["logits"]
    actual = raw(tokens, scan_mode="raw_cuda_delta")["logits"]
    gradient = torch.randn_like(actual)
    expected_gradients = torch.autograd.grad(
        expected, tuple(semantic.parameters()), gradient
    )
    actual_gradients = torch.autograd.grad(
        actual, tuple(raw.parameters()), gradient
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-5)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=3e-3, atol=5e-3
        )
