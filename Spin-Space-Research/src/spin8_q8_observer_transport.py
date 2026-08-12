"""Compile an exact Spin(8) Q8 family and transport its linear observer."""

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
from spin8_q8_joint_retraction import learned_canonical_orbit, positive_spin8_parameters
from spin8_q8_regular_orbit_retraction import (
    exact_regular_ambient_actions,
    regular_orbit_projection,
)


def retract_and_transport(
    source: Path, destination: Path, *, device: torch.device
) -> dict[str, object]:
    checkpoint = torch.load(source, map_location=device, weights_only=False)
    if checkpoint["family"] != "pure_spin8_positive":
        raise ValueError("observer transport requires a positive-chiral checkpoint")
    config = checkpoint["config"]
    model = PureGroupActionModel(
        len(INPUT_ELEMENTS), 8,
        family=checkpoint["family"],
        channels=int(config["channels"]),
        max_rotor_angle=float(config["max_angle"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    old_orbit = learned_canonical_orbit(model)
    old_flat = old_orbit.reshape(-1, GROUPS["q8"].order)
    old_weight = model.output_head.weight.detach().cpu().double().numpy()
    old_prescale_logits = old_weight @ old_flat

    targets, conjugations, projection_rms, gram_eigenvalues, commutant = (
        regular_orbit_projection(old_orbit)
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

    new_orbit = learned_canonical_orbit(model)
    new_flat = new_orbit.reshape(-1, GROUPS["q8"].order)
    new_rank = int(np.linalg.matrix_rank(new_flat, tol=1e-7))
    new_condition = float(np.linalg.cond(new_flat))
    residual = old_prescale_logits - old_weight @ new_flat
    new_weight = old_weight + residual @ np.linalg.pinv(new_flat, rcond=1e-12)
    observer_displacement_rms = float(
        np.sqrt(np.mean(np.square(new_weight - old_weight)))
    )
    with torch.no_grad():
        model.output_head.weight.copy_(
            torch.as_tensor(
                new_weight,
                dtype=model.output_head.weight.dtype,
                device=device,
            )
        )
    actual_weight = model.output_head.weight.detach().cpu().double().numpy()
    logit_transport_max = float(
        np.max(np.abs(actual_weight @ new_flat - old_prescale_logits))
    )
    reconstructed = model.action_matrices().detach().cpu().double().numpy()
    action_reconstruction_max = float(np.max(np.abs(reconstructed - target_actions)))
    diagnostics = representation_diagnostics(model, GROUPS["q8"], INPUT_ELEMENTS)
    dense = central_pair_evaluation(
        model, base_lengths=SMOKE_BASE_LENGTHS, batches=2, batch_size=512,
        seed_base=9_700_000 + 10_000 * int(config["seed"]), device=device,
    )
    long = central_pair_evaluation(
        model, base_lengths=LONG_BASE_LENGTHS, batches=1, batch_size=128,
        seed_base=9_800_000 + 10_000 * int(config["seed"]), device=device,
    )
    probe = torch.arange(128, device=device).reshape(4, 32) % len(INPUT_ELEMENTS)
    streaming = streaming_equivalence(model, probe)
    gates = {
        "commutant": float(commutant.max()) <= 1e-10,
        "spin8_action_reconstruction": action_reconstruction_max <= 1e-5,
        "full_homomorphism": diagnostics["linear_homomorphism_rms"] <= 1e-5,
        "full_canonical_rank": new_rank == GROUPS["q8"].order,
        "canonical_logit_transport": logit_transport_max <= 1e-5,
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
    revised = dict(checkpoint)
    revised["config"] = {
        **config,
        "spin8_q8_regular_orbit_observer_transport": True,
        "spin8_q8_regular_orbit_observer_transport_source": str(source),
    }
    revised["state_dict"] = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    torch.save(revised, destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "seed": int(config["seed"]),
        "method": "full regular-orbit retraction plus reachable-subspace observer transport",
        "target_labels_used_for_observer": False,
        "gradient_steps_after_retraction": 0,
        "per_channel_orbit_projection_rms": projection_rms.tolist(),
        "per_channel_projected_gram_eigenvalues": gram_eigenvalues.tolist(),
        "per_channel_commutant_max_abs": commutant.tolist(),
        "new_canonical_orbit_rank": new_rank,
        "new_canonical_orbit_condition_number": new_condition,
        "observer_displacement_rms": observer_displacement_rms,
        "canonical_prescale_logit_transport_max_abs": logit_transport_max,
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
    result = retract_and_transport(args.source, args.destination, device=device)
    report = {
        "experiment": "Spin(8) Q8 regular retraction plus observer transport seed-4 diagnostic",
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
