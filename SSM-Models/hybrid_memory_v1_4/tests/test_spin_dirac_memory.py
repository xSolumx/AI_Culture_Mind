"""Contracts for the content-addressed Spin/Clifford memory candidate."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pure_spin8_ssm.torch_backend import spin8_factorized_actions
from spin8_triality import (
    TRIALITY_REPRESENTATIONS,
    algebra_diagnostics,
    torch_triality_generators,
)
from spin8_triality_lift import triality_tensor

from hybrid_memory_v1_4.experiments import (
    gated_delta_association_auxiliary_loss,
    intermediate_retrieval_auxiliary_loss,
)
from hybrid_memory_v1_4.model import HybridMemoryConfig, HybridMemoryLM
from hybrid_memory_v1_4.spin_dirac_memory import SpinDiracConfig, SpinDiracMemory
from hybrid_memory_v1_4.tasks import generate_mqar_batch


def test_config_fails_closed() -> None:
    with pytest.raises(ValueError, match="transport_mode"):
        SpinDiracConfig(8, transport_mode="rotor")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="readout_mode"):
        SpinDiracConfig(8, readout_mode="learned")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="gate_mode"):
        SpinDiracConfig(8, gate_mode="diagonal")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="initial_retention"):
        SpinDiracConfig(8, initial_retention=0.999)
    with pytest.raises(ValueError, match="maximum_coordinate"):
        SpinDiracConfig(8, maximum_coordinate=0.0)


def test_recurrent_and_two_sided_parallel_scan_match() -> None:
    torch.manual_seed(20260825)
    memory = SpinDiracMemory(SpinDiracConfig(16, heads=2)).double()
    inputs = torch.randn(2, 11, 16, dtype=torch.float64)
    initial = torch.randn(2, 2, 8, 8, dtype=torch.float64) * 0.03
    recurrent, recurrent_state = memory(inputs, initial, scan_mode="recurrent")
    parallel, parallel_state = memory(inputs, initial, scan_mode="parallel")
    torch.testing.assert_close(parallel, recurrent, rtol=2e-10, atol=2e-11)
    torch.testing.assert_close(parallel_state, recurrent_state, rtol=2e-10, atol=2e-11)


def test_masked_steps_freeze_cache_and_zero_the_update() -> None:
    torch.manual_seed(7)
    memory = SpinDiracMemory(SpinDiracConfig(8, heads=1)).double()
    inputs = torch.randn(2, 5, 8, dtype=torch.float64)
    valid = torch.tensor(
        [[True, True, False, False, False], [True, False, True, False, True]]
    )
    output, final_state = memory(inputs, valid_mask=valid, scan_mode="recurrent")
    assert torch.equal(output[~valid], torch.zeros_like(output[~valid]))

    first_prefix = memory(inputs[:1, :2], scan_mode="recurrent")[1]
    torch.testing.assert_close(final_state[:1], first_prefix)


def test_clifford_readout_is_spin8_equivariant() -> None:
    torch.manual_seed(13)
    dtype = torch.float64
    coordinates = 0.2 * torch.randn(5, 28, dtype=dtype)
    actions = spin8_factorized_actions(
        coordinates,
        torch_triality_generators(TRIALITY_REPRESENTATIONS).to(dtype),
        TRIALITY_REPRESENTATIONS,
    )
    indices = {
        name: TRIALITY_REPRESENTATIONS.index(name) for name in TRIALITY_REPRESENTATIONS
    }
    vector_action = actions[:, indices["vector"]]
    positive_action = actions[:, indices["positive"]]
    negative_action = actions[:, indices["negative"]]
    vector = torch.randn(5, 8, dtype=dtype)
    positive = torch.randn(5, 8, dtype=dtype)
    rho = triality_tensor(dtype=dtype)

    negative = torch.einsum("...i,vji,...v->...j", positive, rho, vector)
    transformed_negative = torch.einsum(
        "...i,vji,...v->...j",
        torch.einsum("...ij,...j->...i", positive_action, positive),
        rho,
        torch.einsum("...ij,...j->...i", vector_action, vector),
    )
    expected = torch.einsum("...ij,...j->...i", negative_action, negative)
    torch.testing.assert_close(transformed_negative, expected, rtol=2e-12, atol=2e-12)


def test_scalar_edit_law_is_inner_conjugation_covariant() -> None:
    """Transport, address edit, state, and both reads share one Spin frame."""

    torch.manual_seed(15)
    dtype = torch.float64
    generators = torch_triality_generators(TRIALITY_REPRESENTATIONS).to(dtype)
    action = spin8_factorized_actions(
        0.2 * torch.randn(28, dtype=dtype), generators, TRIALITY_REPRESENTATIONS
    )
    frame = spin8_factorized_actions(
        0.2 * torch.randn(28, dtype=dtype), generators, TRIALITY_REPRESENTATIONS
    )
    indices = {
        name: TRIALITY_REPRESENTATIONS.index(name) for name in TRIALITY_REPRESENTATIONS
    }
    vector_action, positive_action = (
        action[indices["vector"]],
        action[indices["positive"]],
    )
    vector_frame, positive_frame, negative_frame = (
        frame[indices["vector"]],
        frame[indices["positive"]],
        frame[indices["negative"]],
    )
    conjugate_vector = vector_frame @ vector_action @ vector_frame.T
    conjugate_positive = positive_frame @ positive_action @ positive_frame.T

    key = torch.nn.functional.normalize(torch.randn(8, dtype=dtype), dim=0)
    query = torch.nn.functional.normalize(torch.randn(8, dtype=dtype), dim=0)
    value = torch.randn(8, dtype=dtype)
    state = torch.randn(8, 8, dtype=dtype)
    erase_strength = torch.tensor(0.37, dtype=dtype)
    write_strength = torch.tensor(0.61, dtype=dtype)
    retention = torch.tensor(0.97, dtype=dtype)
    identity = torch.eye(8, dtype=dtype)

    erase = identity - erase_strength * key[:, None] * key[None, :]
    left = erase @ (retention * vector_action)
    updated = (
        left @ state @ positive_action.T
        + write_strength * key[:, None] * value[None, :]
    )

    framed_key = vector_frame @ key
    framed_query = vector_frame @ query
    framed_value = positive_frame @ value
    framed_state = vector_frame @ state @ positive_frame.T
    framed_erase = identity - erase_strength * framed_key[:, None] * framed_key[None, :]
    framed_left = framed_erase @ (retention * conjugate_vector)
    framed_updated = (
        framed_left @ framed_state @ conjugate_positive.T
        + write_strength * framed_key[:, None] * framed_value[None, :]
    )
    torch.testing.assert_close(
        framed_updated,
        vector_frame @ updated @ positive_frame.T,
        rtol=3e-12,
        atol=3e-12,
    )

    positive_read = query @ updated
    framed_positive_read = framed_query @ framed_updated
    torch.testing.assert_close(
        framed_positive_read,
        positive_frame @ positive_read,
        rtol=3e-12,
        atol=3e-12,
    )
    rho = triality_tensor(dtype=dtype)
    negative_read = torch.einsum("i,vji,v->j", positive_read, rho, query)
    framed_negative_read = torch.einsum(
        "i,vji,v->j", framed_positive_read, rho, framed_query
    )
    torch.testing.assert_close(
        framed_negative_read,
        negative_frame @ negative_read,
        rtol=3e-12,
        atol=3e-12,
    )


def test_backend_distinguishes_all_four_spin8_center_signatures() -> None:
    diagnostics = algebra_diagnostics(seed=20260825)
    residuals = diagnostics["center_signature_max_abs"]
    assert isinstance(residuals, dict)
    assert set(residuals) == {"identity", "minus_one", "omega", "minus_omega"}
    assert max(float(value) for value in residuals.values()) <= 1e-10


def test_transport_actions_are_orthogonal_and_controls_are_auditable() -> None:
    torch.manual_seed(17)
    memory = SpinDiracMemory(
        SpinDiracConfig(8, heads=1, transport_mode="commuting_so2")
    ).double()
    with torch.no_grad():
        memory.coordinate_projection.weight.normal_(std=0.1)
    inputs = torch.randn(2, 4, 8, dtype=torch.float64)
    _, _, diagnostics = memory(inputs, return_diagnostics=True)
    actions = diagnostics["transport_actions"]
    assert isinstance(actions, torch.Tensor)
    identity = torch.eye(8, dtype=torch.float64)
    torch.testing.assert_close(
        actions.transpose(-1, -2) @ actions,
        identity.expand_as(actions),
        rtol=2e-12,
        atol=2e-12,
    )
    coordinates = diagnostics["transport_coordinates"]
    assert isinstance(coordinates, torch.Tensor)
    inactive = ~memory.commuting_coordinate_mask
    assert torch.equal(
        coordinates[..., inactive], torch.zeros_like(coordinates[..., inactive])
    )


def test_su3_torus_is_the_constrained_rank_two_slice() -> None:
    torch.manual_seed(18)
    memory = SpinDiracMemory(
        SpinDiracConfig(8, heads=1, transport_mode="su3_torus")
    ).double()
    with torch.no_grad():
        memory.coordinate_projection.weight.normal_(std=0.1)
    _, _, diagnostics = memory(
        torch.randn(2, 4, 8, dtype=torch.float64), return_diagnostics=True
    )
    coordinates = diagnostics["transport_coordinates"]
    assert isinstance(coordinates, torch.Tensor)
    indices = memory.su3_coordinate_indices
    active = coordinates.index_select(-1, indices)
    torch.testing.assert_close(active.sum(dim=-1), torch.zeros_like(active[..., 0]))
    inactive = torch.ones(28, dtype=torch.bool)
    inactive[indices] = False
    assert torch.equal(
        coordinates[..., inactive], torch.zeros_like(coordinates[..., inactive])
    )
    assert float(coordinates.abs().max().detach()) <= memory.config.maximum_coordinate


def test_broken_spin_control_preserves_orthogonality_but_breaks_coupling() -> None:
    torch.manual_seed(181)
    shared = SpinDiracMemory(
        SpinDiracConfig(8, heads=1, transport_mode="spin8")
    ).double()
    broken = SpinDiracMemory(
        SpinDiracConfig(8, heads=1, transport_mode="broken_spin8")
    ).double()
    broken.load_state_dict(shared.state_dict(), strict=False)
    with torch.no_grad():
        shared.coordinate_projection.weight.normal_(std=0.1)
        broken.coordinate_projection.weight.copy_(shared.coordinate_projection.weight)
    inputs = torch.randn(1, 3, 8, dtype=torch.float64)
    shared_actions = shared(inputs, return_diagnostics=True)[2]["transport_actions"]
    broken_actions = broken(inputs, return_diagnostics=True)[2]["transport_actions"]
    assert isinstance(shared_actions, torch.Tensor)
    assert isinstance(broken_actions, torch.Tensor)
    identity = torch.eye(8, dtype=torch.float64)
    torch.testing.assert_close(
        broken_actions.transpose(-1, -2) @ broken_actions,
        identity.expand_as(broken_actions),
        rtol=2e-12,
        atol=2e-12,
    )
    vector_index = TRIALITY_REPRESENTATIONS.index("vector")
    positive_index = TRIALITY_REPRESENTATIONS.index("positive")
    torch.testing.assert_close(
        broken_actions[..., vector_index, :, :],
        shared_actions[..., vector_index, :, :],
    )
    assert not torch.allclose(
        broken_actions[..., positive_index, :, :],
        shared_actions[..., positive_index, :, :],
    )


def test_full_backward_reaches_transport_and_address_controllers() -> None:
    torch.manual_seed(19)
    memory = SpinDiracMemory(SpinDiracConfig(16, heads=2)).double()
    inputs = torch.randn(2, 6, 16, dtype=torch.float64, requires_grad=True)
    output, state = memory(inputs, scan_mode="parallel")
    (output.square().mean() + 0.01 * state.square().mean()).backward()
    assert inputs.grad is not None and bool(torch.isfinite(inputs.grad).all())
    for parameter in (
        memory.coordinate_projection.weight,
        memory.query_projection.weight,
        memory.key_projection.weight,
        memory.value_projection.weight,
        memory.erase_projection.weight,
        memory.write_projection.weight,
        memory.decay_projection.weight,
    ):
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert float(parameter.grad.abs().sum()) > 0.0


def test_oracle_fast_weight_is_content_addressed() -> None:
    key = torch.nn.functional.one_hot(torch.tensor(2), num_classes=8).double()
    other = torch.nn.functional.one_hot(torch.tensor(3), num_classes=8).double()
    value = torch.arange(1, 9, dtype=torch.float64)
    state = key[:, None] * value[None, :]
    torch.testing.assert_close(key @ state, value)
    torch.testing.assert_close(other @ state, torch.zeros_like(value))


def test_state_accounting_is_explicit() -> None:
    memory = SpinDiracMemory(SpinDiracConfig(32, heads=3))
    assert memory.state_scalars == 3 * 8 * 8
    assert memory.state_bytes(torch.float32, batch_size=2) == 2 * 3 * 64 * 4


def test_default_transition_and_drive_are_bounded() -> None:
    torch.manual_seed(17)
    memory = SpinDiracMemory(SpinDiracConfig(8, heads=1)).double()
    inputs = 100.0 * torch.randn(2, 32, 8, dtype=torch.float64)
    _, key, value, erase, write, retention, coordinates = memory._controls(inputs)
    left, right, injection, _ = memory._transitions(
        key, value, erase, write, retention, coordinates, None
    )
    left_norm = torch.linalg.matrix_norm(left, ord=2)
    right_norm = torch.linalg.matrix_norm(right, ord=2)
    drive_norm = torch.linalg.matrix_norm(injection, ord="fro")
    assert float(left_norm.max().detach()) <= memory.config.maximum_retention + 1e-10
    assert float(right_norm.max().detach()) <= 1.0 + 1e-10
    assert float(drive_norm.max().detach()) <= 1.0 + 1e-10


def test_bounded_default_has_finite_long_horizon_state() -> None:
    torch.manual_seed(23)
    length = 4096
    memory = SpinDiracMemory(SpinDiracConfig(8, heads=1)).double()
    inputs = 1000.0 * torch.randn(1, length, 8, dtype=torch.float64)
    _, final_state = memory(inputs, scan_mode="recurrent")
    ceiling = (1.0 - memory.config.maximum_retention**length) / (
        1.0 - memory.config.maximum_retention
    )
    # This is an input-independent bound, not a claim that the very loose
    # asymptotic ceiling is well conditioned.
    assert bool(torch.isfinite(final_state).all())
    assert (
        float(torch.linalg.matrix_norm(final_state, ord="fro").detach())
        <= ceiling + 1e-8
    )


def test_commissioning_losses_dispatch_to_spin_dirac() -> None:
    torch.manual_seed(23)
    batch = generate_mqar_batch(3, 4, 3, 24, seed=24)
    model = HybridMemoryLM(
        HybridMemoryConfig(
            vocab_size=197,
            model_dim=16,
            layer_plan=("spin_dirac",),
            spin_dirac_heads=2,
            use_local_conv=False,
        )
    )
    output = model(batch.inputs, return_diagnostics=True)
    association = gated_delta_association_auxiliary_loss(output, batch)
    assert association.ndim == 0 and torch.isfinite(association) and association > 0
    association.backward()
    mixer = model.blocks[0].mixer
    assert isinstance(mixer, SpinDiracMemory)
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
    assert mixer.coordinate_projection.weight.grad is not None
    assert torch.count_nonzero(mixer.coordinate_projection.weight.grad) > 0
