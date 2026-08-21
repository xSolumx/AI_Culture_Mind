from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import benchmark_pure_spin8_lift_bit_calibration as benchmark
import torch

ROOT = Path(__file__).resolve().parent


def test_exact_one_bit_certificate_passes() -> None:
    certificate = benchmark.exact_one_bit_certificate()
    assert certificate["passed"]
    assert certificate["minimum_binary_measurements"] == 1
    assert certificate["checks"]["witness_bits_are_opposite"]
    assert certificate["robust_adaptive_chart"]["address_bits"] == 3
    assert certificate["robust_adaptive_chart"]["external_lift_bits_given_address"] == 1


def test_lift_bit_flips_on_every_nonexceptional_antipodal_pair() -> None:
    generator = torch.Generator().manual_seed(123)
    states = torch.randn(128, 3, 8, generator=generator)
    states[:, 1, benchmark.POSITIVE_PROBE_INDEX] += 0.123
    bits = benchmark.lift_bit_from_positive_endpoint(states)
    opposite = benchmark.lift_bit_from_positive_endpoint(-states)
    nonexceptional = states[:, 1, benchmark.POSITIVE_PROBE_INDEX] != 0
    assert torch.equal(bits[nonexceptional], ~opposite[nonexceptional])


def test_vector_plus_bit_loss_has_exact_gradient_support() -> None:
    predictions = torch.randn(5, 7, 3, 8, requires_grad=True)
    vector_targets = torch.randn(5, 8)
    bit_targets = torch.tensor([True, False, True, False, True])
    loss, components = benchmark.vector_plus_bit_loss(
        predictions, vector_targets, bit_targets
    )
    loss.backward()
    assert predictions.grad is not None
    assert torch.count_nonzero(predictions.grad[:, :-1]) == 0
    assert torch.count_nonzero(predictions.grad[:, -1, 0]) > 0
    assert (
        torch.count_nonzero(
            predictions.grad[:, -1, 1, benchmark.POSITIVE_PROBE_INDEX]
        )
        > 0
    )
    hidden_positive = predictions.grad[:, -1, 1].clone()
    hidden_positive[:, benchmark.POSITIVE_PROBE_INDEX] = 0
    assert torch.count_nonzero(hidden_positive) == 0
    assert torch.count_nonzero(predictions.grad[:, -1, 2]) == 0
    assert set(components) == {"vector_mse", "bit_bce", "bit_accuracy"}


def test_adaptive_chart_is_antipodally_invariant_with_flipped_bit() -> None:
    generator = torch.Generator().manual_seed(456)
    states = torch.randn(128, 3, 8, generator=generator)
    addresses, bits = benchmark.adaptive_lift_address_and_bit(states)
    opposite_addresses, opposite_bits = benchmark.adaptive_lift_address_and_bit(
        -states
    )
    assert torch.equal(addresses, opposite_addresses)
    assert torch.equal(bits, ~opposite_bits)


def test_adaptive_chart_has_unit_sphere_margin_and_deterministic_ties() -> None:
    generator = torch.Generator().manual_seed(789)
    states = torch.randn(256, 3, 8, generator=generator)
    states[:, 1] = torch.nn.functional.normalize(states[:, 1], dim=-1)
    addresses, _ = benchmark.adaptive_lift_address_and_bit(states)
    selected = states[:, 1].gather(-1, addresses.unsqueeze(-1)).squeeze(-1)
    assert float(selected.abs().min()) >= 1.0 / math.sqrt(8.0) - 1e-6

    tied = torch.zeros(1, 3, 8)
    tied[0, 1, :2] = torch.tensor([1.0, -1.0]) / math.sqrt(2.0)
    address, bit = benchmark.adaptive_lift_address_and_bit(tied)
    opposite_address, opposite_bit = benchmark.adaptive_lift_address_and_bit(-tied)
    assert address.item() == 0
    assert torch.equal(address, opposite_address)
    assert torch.equal(bit, ~opposite_bit)


def test_adaptive_bit_loss_touches_one_addressed_spinor_scalar_per_sample() -> None:
    predictions = torch.randn(5, 7, 3, 8, requires_grad=True)
    vector_targets = torch.randn(5, 8)
    addresses = torch.tensor([0, 2, 4, 6, 7], dtype=torch.long)
    bit_targets = torch.tensor([True, False, True, False, True])
    loss, components = benchmark.vector_plus_adaptive_bit_loss(
        predictions, vector_targets, addresses, bit_targets
    )
    loss.backward()
    assert predictions.grad is not None
    assert torch.count_nonzero(predictions.grad[:, :-1]) == 0
    assert torch.count_nonzero(predictions.grad[:, -1, 0]) > 0
    positive_grad = predictions.grad[:, -1, 1]
    assert torch.count_nonzero(positive_grad) == predictions.shape[0]
    for row, address in enumerate(addresses):
        nonzero = torch.nonzero(positive_grad[row], as_tuple=False).flatten()
        assert torch.equal(nonzero, address.reshape(1))
    assert torch.count_nonzero(predictions.grad[:, -1, 2]) == 0
    assert set(components) == {
        "vector_mse",
        "adaptive_bit_bce",
        "adaptive_bit_accuracy",
    }


def test_mode_loss_rejects_unknown_mode() -> None:
    predictions = torch.randn(2, 3, 3, 8)
    targets = torch.randn(2, 3, 8)
    try:
        benchmark.mode_loss("unknown", predictions, targets)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown calibration mode was accepted")


def test_bit_loss_rejects_full_spinor_target_tensor() -> None:
    predictions = torch.randn(2, 3, 3, 8)
    vector_targets = torch.randn(2, 3, 8)
    bits = torch.tensor([True, False])
    try:
        benchmark.vector_plus_bit_loss(predictions, vector_targets, bits)
    except ValueError:
        pass
    else:
        raise AssertionError("unsliced target tensor reached one-bit loss")


def test_development_artifacts_are_content_locked() -> None:
    artifacts = ROOT / "experiments" / "artifacts"
    expected = {
        "pure_spin8_lift_bit_calibration_development_seed0.json": (
            "5fded3a079877fabaec4d8dc7b7ae7d6d99b0ad8310f97c1a3d4911a2bfe8c53"
        ),
        "pure_spin8_lift_bit_calibration_adaptive_development_seed0.json": (
            "9565026805afb2948b18be08a1ce684b3048bf83c412435707650abe223aefef"
        ),
        "pure_spin8_lift_bit_calibration_adaptive_development_seed0_validated.json": (
            "88c38c8a819e191e99f74862ef8d568c10db47978155dc126e2893f6c6372757"
        ),
    }
    for name, digest in expected.items():
        assert hashlib.sha256((artifacts / name).read_bytes()).hexdigest() == digest


def test_protocol_freeze_is_recorded() -> None:
    assert benchmark.PROTOCOL_FROZEN_AT == "2026-08-17T07:05:22.8794763+02:00"


def test_fresh_adjudication_artifacts_are_content_locked_and_passed() -> None:
    artifacts = ROOT / "experiments" / "artifacts"
    expected = {
        "pure_spin8_lift_bit_calibration_validation_seed4.json": (
            "d25e56ac62912ab78c975c35194bae9b089dbfdf4cb013d31dc3389574970663"
        ),
        "pure_spin8_lift_bit_calibration_validation_seed5.json": (
            "fd2e8206124d732708950fcdef5937cbe7409da0f654c122a8fa16f08ad18bad"
        ),
        "pure_spin8_lift_bit_calibration_validation_seed6.json": (
            "07a3cda44449013e5047939f6443a8728006e4ff6cb466a5b46cc635d5b142b3"
        ),
        "pure_spin8_lift_bit_calibration_validation_seeds4_6.json": (
            "fb89fd75d5aa7c3b16448844225baf838aeb4bdf40cb62ba1276f07ac7503b69"
        ),
    }
    for name, digest in expected.items():
        path = artifacts / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest

    aggregate = json.loads(
        (artifacts / "pure_spin8_lift_bit_calibration_validation_seeds4_6.json")
        .read_text(encoding="utf-8")
    )
    assert aggregate["passed"]
    assert aggregate["seeds"] == [4, 5, 6]
    assert aggregate["global_checks"][
        "every_frozen_gate_passes_without_median_rescue"
    ]
