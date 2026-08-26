from __future__ import annotations

import sys

import pytest
import torch

from .albert import albert_determinant
from .primitive_action import (
    PrimitiveExceptionalAction,
    _apply_one_reference,
    build_primitive_metadata,
    dense_primitive_product_oracle,
    primitive_delta_recurrence_cuda,
    primitive_delta_recurrence_reference,
    primitive_event_layout,
    primitive_product_cuda,
    primitive_product_reference,
)


@pytest.mark.parametrize("algebra,factors,content", [("f4", 52, 0), ("e6", 78, 26)])
def test_primitive_metadata_is_exact_small_block_bank(
    algebra: str, factors: int, content: int
) -> None:
    metadata = build_primitive_metadata(algebra)  # type: ignore[arg-type]
    assert metadata.factor_count == factors
    assert metadata.permutations.shape == (factors, 27)
    assert metadata.local_generators.shape == (factors, 9, 3, 3)
    assert int(metadata.content_mask.sum()) == content
    assert torch.all(metadata.operator_norms > 0)
    for factor in range(factors):
        permutation = metadata.permutations[factor].long()
        reconstructed = torch.zeros(27, 27, dtype=torch.float64)
        for block in range(9):
            indices = permutation[3 * block : 3 * block + 3]
            reconstructed[indices[:, None], indices[None, :]] = (
                metadata.local_generators[factor, block]
            )
        torch.testing.assert_close(
            reconstructed, metadata.generators[factor], atol=2e-12, rtol=0
        )


@pytest.mark.parametrize("algebra", ["f4", "e6"])
def test_every_primitive_matches_dense_exponential_and_gradient(algebra: str) -> None:
    torch.manual_seed(20260826)
    metadata = build_primitive_metadata(algebra)  # type: ignore[arg-type]
    for factor in range(metadata.factor_count):
        values = torch.randn(1, 2, 27, dtype=torch.float64, requires_grad=True)
        angle = torch.tensor([0.07], dtype=torch.float64, requires_grad=True)
        actual = _apply_one_reference(values, angle, factor, metadata)
        action = torch.matrix_exp(angle[:, None, None] * metadata.generators[factor])
        expected = values @ action.transpose(-1, -2)
        torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
        cotangent = torch.randn_like(actual)
        actual_gradients = torch.autograd.grad(
            actual, (values, angle), cotangent, retain_graph=True
        )
        expected_gradients = torch.autograd.grad(
            expected, (values, angle), cotangent
        )
        for candidate, oracle in zip(actual_gradients, expected_gradients, strict=True):
            torch.testing.assert_close(candidate, oracle, atol=3e-9, rtol=3e-9)


@pytest.mark.parametrize("algebra", ["f4", "e6"])
def test_full_reference_product_matches_dense_oracle_and_autograd(algebra: str) -> None:
    torch.manual_seed(17)
    factors = build_primitive_metadata(algebra).factor_count  # type: ignore[arg-type]
    values = torch.randn(2, 3, 27, dtype=torch.float64, requires_grad=True)
    coordinates = (
        0.03 * torch.randn(2, factors, dtype=torch.float64)
    ).requires_grad_()
    actual = primitive_product_reference(values, coordinates, algebra)  # type: ignore[arg-type]
    expected = dense_primitive_product_oracle(values, coordinates, algebra)  # type: ignore[arg-type]
    torch.testing.assert_close(actual, expected, atol=4e-12, rtol=4e-12)
    cotangent = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(
        actual, (values, coordinates), cotangent, retain_graph=True
    )
    expected_gradients = torch.autograd.grad(
        expected, (values, coordinates), cotangent
    )
    for candidate, oracle in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(candidate, oracle, atol=4e-10, rtol=4e-10)


def test_reference_backend_contains_no_dense_matrix_exponential() -> None:
    assert "matrix_exp" not in primitive_product_reference.__code__.co_names
    assert "matrix_exp" not in _apply_one_reference.__code__.co_names


@pytest.mark.parametrize("algebra", ["f4", "e6"])
def test_full_primitive_product_preserves_cubic_and_has_exact_reverse_inverse(
    algebra: str,
) -> None:
    torch.manual_seed(20260828)
    metadata = build_primitive_metadata(algebra)  # type: ignore[arg-type]
    values = torch.randn(3, 2, 27, dtype=torch.float64)
    coordinates = 0.04 * torch.randn(
        3, metadata.factor_count, dtype=torch.float64
    )
    transported = primitive_product_reference(values, coordinates, algebra)  # type: ignore[arg-type]
    torch.testing.assert_close(
        albert_determinant(transported),
        albert_determinant(values),
        atol=2e-10,
        rtol=2e-10,
    )
    if algebra == "f4":
        torch.testing.assert_close(
            transported.square().sum(dim=-1),
            values.square().sum(dim=-1),
            atol=2e-10,
            rtol=2e-10,
        )

    recovered = transported
    for factor in reversed(range(metadata.factor_count)):
        recovered = _apply_one_reference(
            recovered, -coordinates[..., factor], factor, metadata
        )
    torch.testing.assert_close(recovered, values, atol=3e-10, rtol=3e-10)


def test_module_registers_reusable_metadata_and_exposes_e6_stability() -> None:
    action = PrimitiveExceptionalAction("e6", backend="reference")
    buffers = dict(action.named_buffers())
    assert "local_generators" in buffers
    assert "permutations" in buffers
    assert "operator_norms" in buffers
    assert "content_mask" in buffers
    assert int(action.content_mask.sum()) == 26
    assert action.operator_norms.shape == (78,)
    state = action.state_dict()
    assert "local_generators" in state and "operator_norms" in state


@pytest.mark.skipif(
    sys.platform != "linux"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (7, 5),
    reason="the native primitive backend is deliberately WSL/Linux SM75-only",
)
@pytest.mark.parametrize("algebra", ["f4", "e6"])
def test_native_sm75_full_product_forward_backward_and_buffer_reuse(algebra: str) -> None:
    torch.manual_seed(20260827)
    action = PrimitiveExceptionalAction(algebra, backend="cuda").cuda().float()
    pointers = {name: tensor.data_ptr() for name, tensor in action.named_buffers()}
    values = torch.randn(2, 4, 27, device="cuda", requires_grad=True)
    coordinates = (
        0.02 * torch.randn(2, action.coordinate_dim, device="cuda")
    ).requires_grad_()
    actual = action(values, coordinates)
    expected = dense_primitive_product_oracle(values, coordinates, algebra)  # type: ignore[arg-type]
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    cotangent = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(
        actual, (values, coordinates), cotangent, retain_graph=True
    )
    expected_gradients = torch.autograd.grad(
        expected, (values, coordinates), cotangent
    )
    torch.testing.assert_close(
        actual_gradients[0], expected_gradients[0], atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        actual_gradients[1], expected_gradients[1], atol=1e-4, rtol=2e-5
    )
    assert pointers == {
        name: tensor.data_ptr() for name, tensor in action.named_buffers()
    }


def _delta_inputs(
    algebra: str, *, batch: int = 2, length: int = 7, heads: int = 3, rank: int = 2
) -> list[torch.Tensor]:
    factors = build_primitive_metadata(algebra).factor_count  # type: ignore[arg-type]
    _, events = primitive_event_layout(length, 3, 1, 0)
    torch.manual_seed(20260828)
    return [
        torch.sigmoid(torch.randn(batch, length, heads, dtype=torch.float64)),
        0.1 * torch.randn(batch, length, rank, heads, dtype=torch.float64),
        0.1 * torch.randn(batch, length, rank, heads, dtype=torch.float64),
        0.1 * torch.randn(batch, length, rank, 27, dtype=torch.float64),
        0.1 * torch.randn(batch, heads, 27, dtype=torch.float64),
        0.1 * torch.randn(batch, length, heads, dtype=torch.float64),
        0.01 * torch.randn(batch, events, factors, dtype=torch.float64),
    ]


@pytest.mark.parametrize("algebra", ["f4", "e6"])
def test_delta_reference_chunk_continuation_and_no_event(algebra: str) -> None:
    inputs = _delta_inputs(algebra)
    full_reads, full_final = primitive_delta_recurrence_reference(
        *inputs, algebra, event_stride=3, event_phase=1
    )
    split = 3
    first_inputs = [tensor[:, :split] for tensor in inputs[:4]]
    first_inputs += [inputs[4], inputs[5][:, :split], inputs[6][:, :1]]
    first_reads, first_final = primitive_delta_recurrence_reference(
        *first_inputs, algebra, event_stride=3, event_phase=1
    )
    second_inputs = [tensor[:, split:] for tensor in inputs[:4]]
    second_inputs += [first_final, inputs[5][:, split:], inputs[6][:, 1:]]
    second_reads, second_final = primitive_delta_recurrence_reference(
        *second_inputs,
        algebra,
        event_stride=3,
        event_phase=1,
        position_offset=split,
    )
    torch.testing.assert_close(
        torch.cat((first_reads, second_reads), dim=1), full_reads
    )
    torch.testing.assert_close(second_final, full_final)

    no_event_coordinates = inputs[6][:, :0]
    reads, final = primitive_delta_recurrence_reference(
        *inputs[:6],
        no_event_coordinates,
        algebra,
        event_stride=20,
        event_phase=19,
    )
    assert reads.shape == (inputs[0].shape[0], inputs[0].shape[1], 27)
    assert final.shape == inputs[4].shape


@pytest.mark.skipif(
    sys.platform != "linux"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (7, 5),
    reason="the fused Delta recurrence is deliberately WSL/Linux SM75-only",
)
@pytest.mark.parametrize("algebra", ["f4", "e6"])
def test_native_sm75_delta_outputs_final_and_all_input_gradients(algebra: str) -> None:
    cpu_inputs = _delta_inputs(algebra, length=6)
    action = PrimitiveExceptionalAction(algebra, backend="cuda").cuda().float()
    native_inputs = [
        tensor.float().cuda().detach().requires_grad_() for tensor in cpu_inputs
    ]
    oracle_inputs = [tensor.detach().float().cuda().requires_grad_() for tensor in cpu_inputs]
    actual = primitive_delta_recurrence_cuda(
        *native_inputs, action, event_stride=3, event_phase=1
    )
    expected = primitive_delta_recurrence_reference(
        *oracle_inputs, algebra, event_stride=3, event_phase=1
    )
    torch.testing.assert_close(actual[0], expected[0], atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-7, rtol=2e-6)
    read_cotangent = torch.randn_like(actual[0])
    final_cotangent = torch.randn_like(actual[1])
    actual_gradients = torch.autograd.grad(
        actual, native_inputs, (read_cotangent, final_cotangent)
    )
    expected_gradients = torch.autograd.grad(
        expected, oracle_inputs, (read_cotangent, final_cotangent)
    )
    for candidate, oracle in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(candidate, oracle, atol=2e-6, rtol=2e-5)


@pytest.mark.skipif(
    sys.platform != "linux"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (7, 5),
    reason="the fused Delta recurrence is deliberately WSL/Linux SM75-only",
)
def test_native_sm75_delta_no_event_and_chunk_continuation() -> None:
    algebra = "f4"
    cpu_inputs = _delta_inputs(algebra)
    action = PrimitiveExceptionalAction(algebra, backend="cuda").cuda().float()
    inputs = [tensor.float().cuda() for tensor in cpu_inputs]
    full = primitive_delta_recurrence_cuda(
        *inputs, action, event_stride=3, event_phase=1
    )
    split = 3
    first_inputs = [tensor[:, :split] for tensor in inputs[:4]]
    first_inputs += [inputs[4], inputs[5][:, :split], inputs[6][:, :1]]
    first = primitive_delta_recurrence_cuda(
        *first_inputs, action, event_stride=3, event_phase=1
    )
    second_inputs = [tensor[:, split:] for tensor in inputs[:4]]
    second_inputs += [first[1], inputs[5][:, split:], inputs[6][:, 1:]]
    second = primitive_delta_recurrence_cuda(
        *second_inputs,
        action,
        event_stride=3,
        event_phase=1,
        position_offset=split,
    )
    torch.testing.assert_close(torch.cat((first[0], second[0]), dim=1), full[0])
    torch.testing.assert_close(second[1], full[1])

    no_events = inputs[6][:, :0]
    native = primitive_delta_recurrence_cuda(
        *inputs[:6], no_events, action, event_stride=20, event_phase=19
    )
    oracle = primitive_delta_recurrence_reference(
        *inputs[:6], no_events, algebra, event_stride=20, event_phase=19
    )
    torch.testing.assert_close(native[0], oracle[0], atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(native[1], oracle[1], atol=2e-7, rtol=2e-6)
