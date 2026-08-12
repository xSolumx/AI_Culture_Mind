"""Preserve fixed components and variation scale in exact Spin(8) Q8 retraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compare_recurrences import GROUPS
from mechanistic_group_actions import (
    PureGroupActionModel,
    representation_diagnostics,
    streaming_equivalence,
)
from q8_spinor_center_experiment import INPUT_ELEMENTS, central_pair_evaluation
from q8_spinor_center_long_audit import LONG_BASE_LENGTHS
from q8_spinor_joint_retraction import SMOKE_BASE_LENGTHS
from spin8_q8_joint_retraction import (
    NEGATIVE_Q8_ELEMENTS,
    POSITIVE_Q8_ELEMENTS,
    exact_ambient_actions,
    learned_canonical_orbit,
    positive_spin8_parameters,
)


def affine_orbit_components(
    orbit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return frames, fixed components, variation amplitudes, and fit RMS."""

    frames = []
    fixed_components = []
    amplitudes = []
    projection_rms = []
    for channel_orbit in orbit:
        differences = 0.5 * (
            channel_orbit[:, POSITIVE_Q8_ELEMENTS]
            - channel_orbit[:, NEGATIVE_Q8_ELEMENTS]
        )
        left, values, right = np.linalg.svd(differences, full_matrices=False)
        frame = left @ right
        amplitude = float(np.mean(values))
        mean = np.mean(channel_orbit, axis=1)
        fixed = mean - frame @ (frame.T @ mean)
        candidate = fixed[:, None] + amplitude * np.concatenate((frame, -frame), axis=1)
        candidate /= np.linalg.norm(candidate, axis=0, keepdims=True).clip(min=1e-12)
        frames.append(frame)
        fixed_components.append(fixed)
        amplitudes.append(amplitude)
        projection_rms.append(
            float(np.sqrt(np.mean(np.square(candidate - channel_orbit))))
        )
    return (
        np.stack(frames),
        np.stack(fixed_components),
        np.asarray(amplitudes),
        np.asarray(projection_rms),
    )


def retract_checkpoint(
    source: Path, destination: Path, *, device: torch.device
) -> dict[str, object]:
    checkpoint = torch.load(source, map_location=device, weights_only=False)
    if checkpoint["family"] != "pure_spin8_positive":
        raise ValueError("affine-orbit retraction requires a positive-chiral checkpoint")
    config = checkpoint["config"]
    model = PureGroupActionModel(
        len(INPUT_ELEMENTS),
        8,
        family=checkpoint["family"],
        channels=int(config["channels"]),
        max_rotor_angle=float(config["max_angle"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    orbit = learned_canonical_orbit(model)
    frames, fixed, amplitudes, projection_rms = affine_orbit_components(orbit)
    targets = exact_ambient_actions(frames)
    parameters, logarithm_imaginary_max, tangent_projection_max = (
        positive_spin8_parameters(targets)
    )
    candidate_initial = fixed + amplitudes[:, None] * frames[:, :, 0]
    candidate_initial /= np.linalg.norm(candidate_initial, axis=1, keepdims=True).clip(
        min=1e-12
    )
    with torch.no_grad():
        model.action_parameters.copy_(
            torch.as_tensor(parameters, dtype=model.action_parameters.dtype, device=device)
        )
        model.initial_orbit_state.copy_(
            torch.as_tensor(
                candidate_initial,
                dtype=model.initial_orbit_state.dtype,
                device=device,
            )
        )
    after = model.action_matrices().detach().cpu().double().numpy()
    action_reconstruction_max = float(np.max(np.abs(after - targets)))
    diagnostics = representation_diagnostics(model, GROUPS["q8"], INPUT_ELEMENTS)
    dense = central_pair_evaluation(
        model,
        base_lengths=SMOKE_BASE_LENGTHS,
        batches=2,
        batch_size=512,
        seed_base=9_300_000 + 10_000 * int(config["seed"]),
        device=device,
    )
    long = central_pair_evaluation(
        model,
        base_lengths=LONG_BASE_LENGTHS,
        batches=1,
        batch_size=128,
        seed_base=9_400_000 + 10_000 * int(config["seed"]),
        device=device,
    )
    probe = torch.arange(128, device=device).reshape(4, 32) % len(INPUT_ELEMENTS)
    streaming = streaming_equivalence(model, probe)
    gates = {
        "spin8_action_reconstruction": action_reconstruction_max <= 1e-5,
        "full_homomorphism": diagnostics["linear_homomorphism_rms"] <= 1e-5,
        "dense_central_pair": bool(dense["gate_pass"]),
        "long_central_pair": bool(long["gate_pass"]),
        "streaming_state": max(
            streaming["chunked_state_max_abs_error"],
            streaming["streaming_state_max_abs_error"],
        )
        <= 1e-5,
        "streaming_logits": max(
            streaming["chunked_logit_max_abs_error"],
            streaming["streaming_logit_max_abs_error"],
        )
        <= 1e-4,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    revised = dict(checkpoint)
    revised["config"] = {
        **config,
        "spin8_q8_affine_orbit_retraction": True,
        "spin8_q8_affine_orbit_retraction_source": str(source),
    }
    revised["state_dict"] = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    torch.save(revised, destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "seed": int(config["seed"]),
        "method": "fixed-component and scaled-variation joint Q8 orbit retraction",
        "decoder_changed": False,
        "channel_selected_or_removed": False,
        "independent_token_normalization": False,
        "per_channel_fixed_component_norm": np.linalg.norm(fixed, axis=1).tolist(),
        "per_channel_variation_amplitude": amplitudes.tolist(),
        "per_channel_orbit_projection_rms": projection_rms.tolist(),
        "matrix_log_imaginary_max_abs": logarithm_imaginary_max,
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
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    torch.use_deterministic_algorithms(True)
    result = retract_checkpoint(args.source, args.destination, device=device)
    report = {
        "experiment": "Spin(8) Q8 affine-orbit retraction seed-4 diagnostic",
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
