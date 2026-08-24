"""Correctness and fail-closed gates for the optional FLA operator adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fla_adapter import (
    DELTA_RULE_BACKENDS,
    DeltaRuleAdapter,
    FlaAvailability,
    delta_rule,
    delta_rule_state_bytes,
    delta_rule_state_scalars,
    fla_available,
    require_fla,
)


def _inputs(
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    batch: int = 2,
    length: int = 9,
    heads: int = 2,
    key_dimension: int = 4,
    value_dimension: int = 3,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(20260824)
    q = torch.randn(
        batch,
        length,
        heads,
        key_dimension,
        generator=generator,
        dtype=dtype,
        device=device,
    )
    k = F.normalize(
        torch.randn(
            batch,
            length,
            heads,
            key_dimension,
            generator=generator,
            dtype=dtype,
            device=device,
        ),
        dim=-1,
    )
    v = torch.randn(
        batch,
        length,
        heads,
        value_dimension,
        generator=generator,
        dtype=dtype,
        device=device,
    )
    beta = torch.sigmoid(
        torch.randn(
            batch,
            length,
            heads,
            generator=generator,
            dtype=dtype,
            device=device,
        )
    )
    initial = torch.randn(
        batch,
        heads,
        key_dimension,
        value_dimension,
        generator=generator,
        dtype=dtype,
        device=device,
    )
    return q, k, v, beta, initial


def test_availability_is_structured_and_cpu_contract_fails_closed() -> None:
    report = fla_available(device="cpu", dtype=torch.float64)
    assert isinstance(report, FlaAvailability)
    assert report.device == "cpu"
    assert report.dtype == "float64"
    assert not report.available
    assert not bool(report)
    assert report.reasons
    assert report.as_dict()["reasons"] == report.reasons
    assert "CUDA device" in report.summary()
    with pytest.raises(RuntimeError, match="FLA unavailable"):
        require_fla(device="cpu", dtype=torch.float64)


@pytest.mark.parametrize("backend", ("fla_chunk", "fla_recurrent"))
def test_explicit_fla_backend_never_uses_semantic_fallback(backend: str) -> None:
    q, k, v, beta, initial = _inputs()
    semantic, _ = delta_rule(q, k, v, beta, backend="semantic_recurrent")
    assert semantic.shape == v.shape
    with pytest.raises(RuntimeError, match="FLA unavailable"):
        delta_rule(
            q,
            k,
            v,
            beta,
            backend=backend,  # type: ignore[arg-type]
            initial_state=initial,
            output_final_state=True,
            chunk_size=16,
        )


def test_semantic_output_state_and_full_gradient_parity_float64() -> None:
    source = _inputs()
    recurrent_inputs = [tensor.detach().clone().requires_grad_() for tensor in source]
    parallel_inputs = [tensor.detach().clone().requires_grad_() for tensor in source]

    recurrent_output, recurrent_state = delta_rule(
        *recurrent_inputs[:4],
        backend="semantic_recurrent",
        scale=0.75,
        initial_state=recurrent_inputs[4],
        output_final_state=True,
    )
    parallel_output, parallel_state = delta_rule(
        *parallel_inputs[:4],
        backend="semantic_parallel",
        scale=0.75,
        initial_state=parallel_inputs[4],
        output_final_state=True,
    )
    assert recurrent_state is not None
    assert parallel_state is not None
    torch.testing.assert_close(
        parallel_output, recurrent_output, rtol=2e-12, atol=2e-12
    )
    torch.testing.assert_close(parallel_state, recurrent_state, rtol=2e-12, atol=2e-12)

    output_weight = torch.linspace(
        0.2, 1.2, recurrent_output.numel(), dtype=torch.float64
    ).reshape_as(recurrent_output)
    state_weight = torch.linspace(
        -0.4, 0.4, recurrent_state.numel(), dtype=torch.float64
    ).reshape_as(recurrent_state)
    recurrent_gradients = torch.autograd.grad(
        (recurrent_output * output_weight).sum()
        + (recurrent_state * state_weight).sum(),
        recurrent_inputs,
    )
    parallel_gradients = torch.autograd.grad(
        (parallel_output * output_weight).sum() + (parallel_state * state_weight).sum(),
        parallel_inputs,
    )
    for actual, expected in zip(parallel_gradients, recurrent_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=3e-11, atol=3e-11)


def test_shape_dtype_backend_and_initial_state_validation() -> None:
    q, k, v, beta, initial = _inputs()
    with pytest.raises(ValueError, match="matching B,L,H"):
        delta_rule(q[:, :-1], k, v, beta)
    with pytest.raises(ValueError, match="must have shapes"):
        delta_rule(q, k, v, beta[..., None])
    with pytest.raises(ValueError, match="initial_state must have shape"):
        delta_rule(q, k, v, beta, initial_state=initial[..., :-1])
    with pytest.raises(ValueError, match="unknown DeltaRule backend"):
        delta_rule(q, k, v, beta, backend="automatic")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="chunk_size"):
        delta_rule(q, k, v, beta, backend="fla_chunk", chunk_size=8)
    assert DELTA_RULE_BACKENDS == (
        "semantic_recurrent",
        "semantic_parallel",
        "fla_chunk",
        "fla_recurrent",
    )


def test_state_scalar_and_byte_accounting_and_module_contract() -> None:
    adapter = DeltaRuleAdapter(2, 4, 3, backend="semantic_parallel")
    assert adapter.state_shape == (2, 4, 3)
    assert adapter.state_scalars == delta_rule_state_scalars(2, 4, 3) == 24
    assert adapter.state_bytes(torch.float64, batch_size=5) == 5 * 24 * 8
    assert delta_rule_state_bytes(2, 4, 3, torch.float16, batch_size=3) == 3 * 24 * 2

    q, k, v, beta, initial = _inputs()
    output, final = adapter(
        q,
        k,
        v,
        beta,
        initial_state=initial,
        output_final_state=True,
    )
    assert output.shape == v.shape
    assert final is not None and final.shape == initial.shape
    with pytest.raises(ValueError, match="adapter state shape"):
        adapter(q[..., :-1], k[..., :-1], v, beta)


@pytest.mark.parametrize("backend", ("fla_chunk", "fla_recurrent"))
def test_guarded_cuda_fla_smoke_parity(backend: str) -> None:
    report = fla_available(device="cuda", dtype=torch.float16)
    if not report:
        pytest.skip(report.summary())

    q, k, v, beta, initial = _inputs(
        dtype=torch.float16,
        device="cuda",
        length=64,
        key_dimension=32,
        value_dimension=32,
    )
    expected_output, expected_state = delta_rule(
        q,
        k,
        v,
        beta,
        backend="semantic_recurrent",
        initial_state=initial,
        output_final_state=True,
    )
    actual_output, actual_state = delta_rule(
        q,
        k,
        v,
        beta,
        backend=backend,  # type: ignore[arg-type]
        initial_state=initial,
        output_final_state=True,
        chunk_size=16,
    )
    assert expected_state is not None
    assert actual_state is not None
    torch.testing.assert_close(
        actual_output.float(), expected_output.float(), rtol=3e-2, atol=3e-2
    )
    torch.testing.assert_close(
        actual_state.float(), expected_state.float(), rtol=3e-2, atol=3e-2
    )
