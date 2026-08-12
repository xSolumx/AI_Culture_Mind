"""Compile exact Spin(8) Q8 dynamics from behaviorally valid path centroids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compare_recurrences import GROUPS, make_group_batches
from mechanistic_group_actions import (
    PureGroupActionModel,
    representation_diagnostics,
    streaming_equivalence,
)
from q8_spinor_center_experiment import (
    INPUT_ELEMENTS,
    central_pair_evaluation,
    final_state,
)
from q8_spinor_center_long_audit import LONG_BASE_LENGTHS
from q8_spinor_joint_retraction import SMOKE_BASE_LENGTHS
from spin8_q8_joint_retraction import learned_canonical_orbit, positive_spin8_parameters
from spin8_q8_regular_orbit_retraction import (
    exact_regular_ambient_actions,
    regular_orbit_projection,
)


CALIBRATION_LENGTHS = (15, 16)
CALIBRATION_BATCHES = 32
CALIBRATION_BATCH_SIZE = 512


def minimum_change_observer(
    old_weight: np.ndarray, teacher_states: np.ndarray, exact_states: np.ndarray
) -> np.ndarray:
    """Transport a linear observer while changing it only on the exact section."""

    if teacher_states.shape != exact_states.shape:
        raise ValueError("teacher and exact state sections must have the same shape")
    if old_weight.shape[1] != teacher_states.shape[0]:
        raise ValueError("observer input width must match the state section")
    desired = old_weight @ teacher_states
    return old_weight + (desired - old_weight @ exact_states) @ np.linalg.pinv(
        exact_states, rcond=1e-12
    )


@torch.no_grad()
def path_section_centroids(
    model: PureGroupActionModel, *, seed_base: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray, float]:
    group = GROUPS["q8"]
    sums = torch.zeros(
        group.order, model.channels, 8, dtype=torch.float64, device=device
    )
    counts = torch.zeros(group.order, dtype=torch.int64, device=device)
    correct = 0
    examples = 0
    for length in CALIBRATION_LENGTHS:
        batches = make_group_batches(
            group,
            CALIBRATION_BATCHES,
            CALIBRATION_BATCH_SIZE,
            length,
            seed_base + length,
            input_elements=INPUT_ELEMENTS,
        )
        for tokens, targets in batches:
            tokens = tokens.to(device)
            endpoints = targets[:, -1].to(device)
            states = final_state(model, tokens)
            predictions = model.decode(states[:, None])[:, 0].argmax(dim=-1)
            correct += int((predictions == endpoints).sum())
            examples += endpoints.numel()
            sums.index_add_(0, endpoints, states.to(torch.float64))
            counts.index_add_(
                0, endpoints, torch.ones_like(endpoints, dtype=torch.int64)
            )
    if bool((counts == 0).any()):
        raise RuntimeError(f"path calibration missed endpoints: {counts.tolist()}")
    centroids = sums / counts[:, None, None]
    # Convert (group, channel, ambient) to the common orbit layout.
    orbit = centroids.permute(1, 2, 0).cpu().numpy()
    return orbit, counts.cpu().numpy(), correct / examples


def compile_checkpoint(
    source: Path, destination: Path, *, device: torch.device
) -> dict[str, object]:
    checkpoint = torch.load(source, map_location=device, weights_only=False)
    if checkpoint["family"] != "pure_spin8_positive":
        raise ValueError("path-section compiler requires a positive-chiral checkpoint")
    config = checkpoint["config"]
    model = PureGroupActionModel(
        len(INPUT_ELEMENTS), 8,
        family=checkpoint["family"],
        channels=int(config["channels"]),
        max_rotor_angle=float(config["max_angle"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    raw_tangent_norms = (
        model.action_parameters.detach().cpu().double().norm(dim=-1).numpy()
    )

    calibration_seed = 10_500_000 + 10_000 * int(config["seed"])
    centroid_orbit, counts, raw_accuracy = path_section_centroids(
        model, seed_base=calibration_seed, device=device
    )
    old_weight = model.output_head.weight.detach().cpu().double().numpy()
    old_flat = centroid_orbit.reshape(-1, GROUPS["q8"].order)
    desired_logits = old_weight @ old_flat

    targets, conjugations, projection_rms, gram_eigenvalues, commutant = (
        regular_orbit_projection(centroid_orbit)
    )
    target_actions = exact_regular_ambient_actions(conjugations)
    parameters, imaginary_max, tangent_projection_max = positive_spin8_parameters(
        target_actions
    )
    initial = targets[:, :, 0]
    initial /= np.linalg.norm(initial, axis=1, keepdims=True).clip(min=1e-12)
    with torch.no_grad():
        model.action_parameters.copy_(
            torch.as_tensor(parameters, dtype=model.action_parameters.dtype, device=device)
        )
        model.initial_orbit_state.copy_(
            torch.as_tensor(initial, dtype=model.initial_orbit_state.dtype, device=device)
        )

    exact_orbit = learned_canonical_orbit(model)
    exact_flat = exact_orbit.reshape(-1, GROUPS["q8"].order)
    exact_rank = int(np.linalg.matrix_rank(exact_flat, tol=1e-7))
    exact_condition = float(np.linalg.cond(exact_flat))
    transported = minimum_change_observer(old_weight, old_flat, exact_flat)
    observer_displacement = float(
        np.sqrt(np.mean(np.square(transported - old_weight)))
    )
    with torch.no_grad():
        model.output_head.weight.copy_(
            torch.as_tensor(
                transported, dtype=model.output_head.weight.dtype, device=device
            )
        )
    actual_weight = model.output_head.weight.detach().cpu().double().numpy()
    logit_transport_max = float(
        np.max(np.abs(actual_weight @ exact_flat - desired_logits))
    )
    reconstructed = model.action_matrices().detach().cpu().double().numpy()
    action_reconstruction_max = float(np.max(np.abs(reconstructed - target_actions)))
    diagnostics = representation_diagnostics(model, GROUPS["q8"], INPUT_ELEMENTS)
    dense = central_pair_evaluation(
        model, base_lengths=SMOKE_BASE_LENGTHS, batches=2, batch_size=512,
        seed_base=10_700_000 + 10_000 * int(config["seed"]), device=device,
    )
    long = central_pair_evaluation(
        model, base_lengths=LONG_BASE_LENGTHS, batches=1, batch_size=128,
        seed_base=10_800_000 + 10_000 * int(config["seed"]), device=device,
    )
    probe = torch.arange(128, device=device).reshape(4, 32) % len(INPUT_ELEMENTS)
    streaming = streaming_equivalence(model, probe)
    gates = {
        "calibration_coverage": bool(np.all(counts > 0)),
        "raw_calibration_accuracy": raw_accuracy >= 0.99,
        "commutant": float(commutant.max()) <= 1e-10,
        "spin8_action_reconstruction": action_reconstruction_max <= 1e-5,
        "full_homomorphism": diagnostics["linear_homomorphism_rms"] <= 1e-5,
        "full_section_rank": exact_rank == GROUPS["q8"].order,
        "centroid_logit_transport": logit_transport_max <= 1e-5,
        "dense_central_pair": bool(dense["gate_pass"]),
        "long_central_pair": bool(long["gate_pass"]),
        "streaming_state": max(
            streaming["chunked_state_max_abs_error"],
            streaming["streaming_state_max_abs_error"],
        ) <= 1e-5,
        "streaming_logits": max(
            streaming["chunked_logit_max_abs_error"],
            streaming["streaming_logit_max_abs_error"],
        ) <= 1e-4,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    compiled = dict(checkpoint)
    compiled["config"] = {
        **config,
        "spin8_q8_path_section_compiler": True,
        "spin8_q8_path_section_compiler_source": str(source),
        "calibration_seed_base": calibration_seed,
    }
    compiled["state_dict"] = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    torch.save(compiled, destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "seed": int(config["seed"]),
        "method": "path-section centroid regular retraction and observer transport",
        "table_oracle_used": True,
        "target_one_hot_labels_used": False,
        "gradient_steps_after_compilation": 0,
        "calibration_lengths": list(CALIBRATION_LENGTHS),
        "calibration_batches_per_length": CALIBRATION_BATCHES,
        "calibration_batch_size": CALIBRATION_BATCH_SIZE,
        "calibration_endpoint_counts": counts.tolist(),
        "raw_calibration_final_accuracy": raw_accuracy,
        "raw_tangent_norm_per_token_channel": raw_tangent_norms.tolist(),
        "raw_tangent_norm_max": float(raw_tangent_norms.max()),
        "raw_tangent_norm_mean": float(raw_tangent_norms.mean()),
        "per_channel_centroid_projection_rms": projection_rms.tolist(),
        "per_channel_projected_gram_eigenvalues": gram_eigenvalues.tolist(),
        "per_channel_commutant_max_abs": commutant.tolist(),
        "exact_section_rank": exact_rank,
        "exact_section_condition_number": exact_condition,
        "observer_displacement_rms": observer_displacement,
        "centroid_prescale_logit_transport_max_abs": logit_transport_max,
        "matrix_log_imaginary_max_abs": imaginary_max,
        "lie_tangent_projection_max_abs": tangent_projection_max,
        "spin8_action_reconstruction_max_abs": action_reconstruction_max,
        "representation_diagnostics": diagnostics,
        "dense_central_pair_evaluation": dense,
        "long_central_pair_evaluation": long,
        "streaming_equivalence": streaming,
        "gates": gates,
        "passed": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    torch.use_deterministic_algorithms(True)
    result = compile_checkpoint(args.source, args.destination, device=device)
    report = {
        "experiment": "Spin(8) Q8 path-section compiler seed-4 diagnostic",
        "original_8_of_9_cohort_result_unchanged": True,
        "result": result,
        "passed": result["passed"],
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
