"""Joint Spin(8) retraction onto the complete eight-dimensional Q8 regular orbit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compare_recurrences import GROUPS, FiniteGroup
from mechanistic_group_actions import (
    PureGroupActionModel,
    representation_diagnostics,
    streaming_equivalence,
)
from q8_spinor_center_experiment import INPUT_ELEMENTS, central_pair_evaluation
from q8_spinor_center_long_audit import LONG_BASE_LENGTHS
from q8_spinor_joint_retraction import SMOKE_BASE_LENGTHS
from spin8_q8_joint_retraction import (
    learned_canonical_orbit,
    positive_spin8_parameters,
)


def right_regular_actions(group: FiniteGroup) -> np.ndarray:
    """Return the right-regular action in the group's recovered ordering."""

    actions = np.zeros((group.order, group.order, group.order), dtype=np.float64)
    columns = np.arange(group.order)
    for element in range(group.order):
        actions[element, group.table[columns, element], columns] = 1.0
    return actions


def q8_right_regular_actions() -> np.ndarray:
    """Compatibility wrapper for the original table-aware Q8 experiment."""

    return right_regular_actions(GROUPS["q8"])


def regular_orbit_projection(
    orbit: np.ndarray,
    group: FiniteGroup | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project every channel orbit Gram matrix onto the regular commutant."""

    if group is None:
        group = GROUPS["q8"]
    if orbit.ndim != 3 or orbit.shape[1:] != (group.order, group.order):
        raise ValueError(
            "orbit must have shape (channels, group.order, group.order)"
        )
    regular = right_regular_actions(group)
    target_orbits = []
    conjugations = []
    projection_rms = []
    gram_eigenvalues = []
    commutant_residuals = []
    for learned in orbit:
        gram = learned.T @ learned
        projected_gram = np.mean(
            np.stack([action.T @ gram @ action for action in regular]), axis=0
        )
        projected_gram = 0.5 * (projected_gram + projected_gram.T)
        values, vectors = np.linalg.eigh(projected_gram)
        values = np.clip(values, 0.0, None)
        root = (vectors * np.sqrt(values)[None, :]) @ vectors.T
        left, _, right = np.linalg.svd(learned @ root.T)
        conjugation = left @ right
        target = conjugation @ root
        residual = max(
            float(np.max(np.abs(projected_gram @ action - action @ projected_gram)))
            for action in regular
        )
        target_orbits.append(target)
        conjugations.append(conjugation)
        projection_rms.append(float(np.sqrt(np.mean(np.square(target - learned)))))
        gram_eigenvalues.append(values)
        commutant_residuals.append(residual)
    return (
        np.stack(target_orbits),
        np.stack(conjugations),
        np.asarray(projection_rms),
        np.stack(gram_eigenvalues),
        np.asarray(commutant_residuals),
    )


def regular_ambient_actions(
    conjugations: np.ndarray,
    group: FiniteGroup,
    input_elements: tuple[int, ...],
) -> np.ndarray:
    """Conjugate a recovered regular action into each learned channel gauge."""

    if conjugations.shape[1:] != (group.order, group.order):
        raise ValueError("conjugations do not match recovered group order")
    regular = right_regular_actions(group)
    actions = np.zeros(
        (len(input_elements), len(conjugations), group.order, group.order)
    )
    for token, element in enumerate(input_elements):
        for channel, change in enumerate(conjugations):
            actions[token, channel] = change @ regular[element] @ change.T
    return actions


def exact_regular_ambient_actions(conjugations: np.ndarray) -> np.ndarray:
    """Compatibility wrapper for the original table-aware Q8 experiment."""

    return regular_ambient_actions(conjugations, GROUPS["q8"], INPUT_ELEMENTS)


def retract_checkpoint(
    source: Path, destination: Path, *, device: torch.device
) -> dict[str, object]:
    checkpoint = torch.load(source, map_location=device, weights_only=False)
    if checkpoint["family"] != "pure_spin8_positive":
        raise ValueError("regular-orbit retraction requires a positive-chiral checkpoint")
    config = checkpoint["config"]
    model = PureGroupActionModel(
        len(INPUT_ELEMENTS), 8,
        family=checkpoint["family"],
        channels=int(config["channels"]),
        max_rotor_angle=float(config["max_angle"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    orbit = learned_canonical_orbit(model)
    targets, conjugations, projection_rms, gram_eigenvalues, commutant = (
        regular_orbit_projection(orbit)
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
    reconstructed = model.action_matrices().detach().cpu().double().numpy()
    action_reconstruction_max = float(np.max(np.abs(reconstructed - target_actions)))
    diagnostics = representation_diagnostics(model, GROUPS["q8"], INPUT_ELEMENTS)
    dense = central_pair_evaluation(
        model, base_lengths=SMOKE_BASE_LENGTHS, batches=2, batch_size=512,
        seed_base=9_500_000 + 10_000 * int(config["seed"]), device=device,
    )
    long = central_pair_evaluation(
        model, base_lengths=LONG_BASE_LENGTHS, batches=1, batch_size=128,
        seed_base=9_600_000 + 10_000 * int(config["seed"]), device=device,
    )
    probe = torch.arange(128, device=device).reshape(4, 32) % len(INPUT_ELEMENTS)
    streaming = streaming_equivalence(model, probe)
    gates = {
        "commutant": float(commutant.max()) <= 1e-10,
        "spin8_action_reconstruction": action_reconstruction_max <= 1e-5,
        "full_homomorphism": diagnostics["linear_homomorphism_rms"] <= 1e-5,
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
        "spin8_q8_regular_orbit_retraction": True,
        "spin8_q8_regular_orbit_retraction_source": str(source),
    }
    revised["state_dict"] = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    torch.save(revised, destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "seed": int(config["seed"]),
        "method": "full regular-orbit Gram-commutant projection",
        "decoder_changed": False,
        "rank_threshold_used": False,
        "independent_token_normalization": False,
        "per_channel_orbit_projection_rms": projection_rms.tolist(),
        "per_channel_projected_gram_eigenvalues": gram_eigenvalues.tolist(),
        "per_channel_commutant_max_abs": commutant.tolist(),
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
    result = retract_checkpoint(args.source, args.destination, device=device)
    report = {
        "experiment": "Spin(8) Q8 full regular-orbit seed-4 diagnostic",
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
