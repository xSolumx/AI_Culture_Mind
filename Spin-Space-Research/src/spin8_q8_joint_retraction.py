"""Jointly retract a learned Spin(8) token family onto an exact Q8 orbit.

The retraction operates on the complete learned canonical orbit and derives all
token operators from one shared four-frame.  It never normalizes or rounds
tokens independently.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm, schur
import torch

from compare_recurrences import GROUPS
from mechanistic_group_actions import (
    PureGroupActionModel,
    canonical_group_words,
    representation_diagnostics,
    streaming_equivalence,
)
from q8_spinor_center_experiment import (
    INPUT_ELEMENTS,
    central_pair_evaluation,
)
from q8_spinor_joint_retraction import SMOKE_BASE_LENGTHS
from spin8_triality import build_spin8_triality_algebra


POSITIVE_Q8_ELEMENTS = (0, 1, 2, 3)
NEGATIVE_Q8_ELEMENTS = (4, 5, 6, 7)


def learned_canonical_orbit(model: PureGroupActionModel) -> np.ndarray:
    """Return learned states as ``(channel, ambient, group_element)``."""

    group = GROUPS["q8"]
    words = canonical_group_words(group, INPUT_ELEMENTS)
    actions = model.action_matrices().detach().cpu().double().numpy()
    initial = model.initial_state(1)[0].detach().cpu().double().numpy()
    channels = initial.shape[0]
    orbit = np.zeros((channels, 8, group.order), dtype=np.float64)
    for channel in range(channels):
        for element, word in enumerate(words):
            state = initial[channel].copy()
            for token in word:
                state = actions[token, channel] @ state
            orbit[channel, :, element] = state
    return orbit


def polar_q8_frames(orbit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover one shared orthonormal four-frame from all central pairs."""

    if orbit.ndim != 3 or orbit.shape[1:] != (8, 8):
        raise ValueError("orbit must have shape (channel, 8, 8)")
    frames = []
    singular_values = []
    projection_rms = []
    for channel_orbit in orbit:
        differences = 0.5 * (
            channel_orbit[:, POSITIVE_Q8_ELEMENTS]
            - channel_orbit[:, NEGATIVE_Q8_ELEMENTS]
        )
        left, values, right = np.linalg.svd(differences, full_matrices=False)
        frame = left @ right
        frames.append(frame)
        singular_values.append(np.linalg.svd(channel_orbit, compute_uv=False))
        target_orbit = np.concatenate((frame, -frame), axis=1)
        projection_rms.append(
            math.sqrt(float(np.mean(np.square(target_orbit - channel_orbit))))
        )
    return np.stack(frames), np.stack(singular_values), np.asarray(projection_rms)


def q8_active_actions() -> np.ndarray:
    """Exact right-multiplication actions on the signed `(1,i,j,k)` frame."""

    group = GROUPS["q8"]
    result = np.zeros((len(INPUT_ELEMENTS), 4, 4), dtype=np.float64)
    for token, generator in enumerate(INPUT_ELEMENTS):
        for column, element in enumerate(POSITIVE_Q8_ELEMENTS):
            product = int(group.table[element, generator])
            sign = -1.0 if product >= 4 else 1.0
            row = product - 4 if product >= 4 else product
            result[token, row, column] = sign
    return result


def exact_ambient_actions(frames: np.ndarray) -> np.ndarray:
    """Induce the entire token family from each shared active frame."""

    active = q8_active_actions()
    identity = np.eye(8)
    channels = frames.shape[0]
    result = np.zeros((len(INPUT_ELEMENTS), channels, 8, 8), dtype=np.float64)
    for channel, frame in enumerate(frames):
        projector = frame @ frame.T
        complement = identity - projector
        for token in range(len(INPUT_ELEMENTS)):
            result[token, channel] = frame @ active[token] @ frame.T + complement
    return result


def real_orthogonal_logarithm(action: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
    """Return a real skew logarithm of an SO(n) matrix, including paired -1 modes.

    A principal complex logarithm treats each `-1` eigenvalue separately.  An
    SO(n) matrix has even `-1` multiplicity, so pairing those modes into real
    pi-rotation planes yields the required real Lie-algebra logarithm.
    """

    if action.ndim != 2 or action.shape[0] != action.shape[1]:
        raise ValueError("orthogonal logarithm requires one square matrix")
    identity = np.eye(action.shape[0])
    if np.max(np.abs(action.T @ action - identity)) > 1e-7:
        raise ValueError("orthogonal logarithm input is not orthogonal")
    if np.linalg.det(action) < 0.0:
        raise ValueError("a determinant-negative matrix has no real skew logarithm")
    triangular, basis = schur(action, output="real")
    tangent_schur = np.zeros_like(triangular)
    negative_indices: list[int] = []
    index = 0
    while index < action.shape[0]:
        if index + 1 < action.shape[0] and abs(triangular[index + 1, index]) > tolerance:
            cosine = 0.5 * (
                triangular[index, index] + triangular[index + 1, index + 1]
            )
            sine = 0.5 * (
                triangular[index, index + 1] - triangular[index + 1, index]
            )
            angle = math.atan2(sine, cosine)
            tangent_schur[index, index + 1] = angle
            tangent_schur[index + 1, index] = -angle
            index += 2
            continue
        value = triangular[index, index]
        if value < -1.0 + tolerance:
            negative_indices.append(index)
        elif abs(value - 1.0) > 1e-7:
            raise ValueError(f"unexpected real Schur eigenvalue {value}")
        index += 1
    if len(negative_indices) % 2:
        raise ValueError("SO(n) -1 eigenspace must have even dimension")
    for left, right in zip(negative_indices[::2], negative_indices[1::2]):
        tangent_schur[left, right] = math.pi
        tangent_schur[right, left] = -math.pi
    tangent = basis @ tangent_schur @ basis.T
    tangent = 0.5 * (tangent - tangent.T)
    residual = float(np.max(np.abs(expm(tangent) - action)))
    if residual > 1e-7:
        raise RuntimeError(f"real orthogonal logarithm reconstruction residual {residual}")
    return tangent


def positive_spin8_parameters(actions: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Map exact ambient SO(8) actions through the fixed positive-spin Lie basis."""

    generators = build_spin8_triality_algebra().positive_generators
    basis = generators.reshape(28, -1).T
    parameters = np.zeros((*actions.shape[:2], 28), dtype=np.float64)
    logarithm_imaginary_max = 0.0
    tangent_projection_max = 0.0
    for token in range(actions.shape[0]):
        for channel in range(actions.shape[1]):
            tangent = real_orthogonal_logarithm(actions[token, channel])
            coefficients, *_ = np.linalg.lstsq(basis, tangent.ravel(), rcond=None)
            reconstructed = np.einsum("p,pij->ij", coefficients, generators)
            tangent_projection_max = max(
                tangent_projection_max,
                float(np.max(np.abs(reconstructed - tangent))),
            )
            parameters[token, channel] = coefficients
    return parameters, logarithm_imaginary_max, tangent_projection_max


def retract_checkpoint(
    source: Path,
    destination: Path,
    *,
    device: torch.device,
) -> dict[str, object]:
    checkpoint = torch.load(source, map_location=device, weights_only=False)
    if checkpoint["family"] != "pure_spin8_positive":
        raise ValueError(f"{source} is not a positive-chiral Spin(8) checkpoint")
    config = checkpoint["config"]
    model = PureGroupActionModel(
        len(INPUT_ELEMENTS),
        GROUPS["q8"].order,
        family="pure_spin8_positive",
        channels=int(config["channels"]),
        max_rotor_angle=float(config["max_angle"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    before_actions = model.action_matrices().detach().cpu().double().numpy()
    before_orbit = learned_canonical_orbit(model)
    frames, orbit_singular_values, frame_projection_rms = polar_q8_frames(before_orbit)
    target_actions = exact_ambient_actions(frames)
    parameters, logarithm_imaginary_max, tangent_projection_max = (
        positive_spin8_parameters(target_actions)
    )

    with torch.no_grad():
        model.action_parameters.copy_(
            torch.as_tensor(parameters, dtype=model.action_parameters.dtype, device=device)
        )
        model.initial_orbit_state.copy_(
            torch.as_tensor(frames[:, :, 0], dtype=model.initial_orbit_state.dtype, device=device)
        )
    after_actions = model.action_matrices().detach().cpu().double().numpy()
    action_reconstruction_max = float(np.max(np.abs(after_actions - target_actions)))
    action_projection_rms = np.sqrt(
        np.mean(np.square(after_actions - before_actions), axis=(0, 2, 3))
    )

    diagnostics = representation_diagnostics(model, GROUPS["q8"], INPUT_ELEMENTS)
    dense = central_pair_evaluation(
        model,
        base_lengths=SMOKE_BASE_LENGTHS,
        batches=2,
        batch_size=512,
        seed_base=7_300_000 + 10_000 * int(config["seed"]),
        device=device,
    )
    probe = torch.arange(128, device=device).reshape(4, 32) % len(INPUT_ELEMENTS)
    streaming = streaming_equivalence(model, probe)

    gates = {
        "spin8_action_reconstruction": action_reconstruction_max <= 1e-5,
        "full_homomorphism": diagnostics["linear_homomorphism_rms"] <= 1e-5,
        "dense_central_pair": bool(dense["gate_pass"]),
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
    retracted = dict(checkpoint)
    retracted["config"] = {
        **config,
        "spin8_q8_orbit_family_retraction": True,
        "spin8_q8_orbit_family_retraction_source": str(source),
    }
    retracted["state_dict"] = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    torch.save(retracted, destination)

    return {
        "source": str(source),
        "destination": str(destination),
        "seed": int(config["seed"]),
        "method": "joint canonical-orbit polar frame and exact Q8 family induction",
        "independent_token_normalization": False,
        "decoder_changed": False,
        "per_channel_orbit_singular_values": orbit_singular_values.tolist(),
        "per_channel_frame_projection_rms": frame_projection_rms.tolist(),
        "per_channel_action_projection_rms": action_projection_rms.tolist(),
        "matrix_log_imaginary_max_abs": logarithm_imaginary_max,
        "lie_tangent_projection_max_abs": tangent_projection_max,
        "spin8_action_reconstruction_max_abs": action_reconstruction_max,
        "representation_diagnostics": diagnostics,
        "dense_central_pair_evaluation": dense,
        "streaming_equivalence": streaming,
        "gates": gates,
        "passed": all(gates.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    report = {
        "experiment": "joint Spin(8) Q8 orbit-family retraction smoke",
        "result": retract_checkpoint(args.source, args.destination, device=device),
    }
    report["passed"] = report["result"]["passed"]
    rendered = json.dumps(report, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
