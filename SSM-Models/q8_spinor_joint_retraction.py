"""Jointly retract learned quaternion token families onto exact Q8 frames."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from compare_recurrences import GROUPS
from mechanistic_group_actions import PureGroupActionModel, representation_diagnostics
from q8_spinor_center_experiment import (
    INPUT_ELEMENTS,
    central_pair_evaluation,
)


SMOKE_BASE_LENGTHS = (15, 16, 31, 32, 63, 64, 127, 128, 255, 256)


def exact_q8_targets(quaternions: np.ndarray) -> np.ndarray:
    """Couple all four token actions through one orthonormal frame per channel."""

    if quaternions.ndim != 3 or quaternions.shape[0] != 4 or quaternions.shape[2] != 4:
        raise ValueError("expected four token quaternions with shape (4, channel, 4)")
    targets = np.zeros_like(quaternions, dtype=np.float64)
    for channel in range(quaternions.shape[1]):
        first = quaternions[0, channel, 1:] - quaternions[1, channel, 1:]
        second = quaternions[2, channel, 1:] - quaternions[3, channel, 1:]
        first_norm = np.linalg.norm(first)
        if first_norm <= 1e-8:
            raise ValueError(f"channel {channel} has a degenerate first inverse pair")
        first = first / first_norm
        second = second - np.dot(second, first) * first
        second_norm = np.linalg.norm(second)
        if second_norm <= 1e-8:
            raise ValueError(f"channel {channel} has collinear generator axes")
        second = second / second_norm
        targets[0, channel, 1:] = first
        targets[1, channel, 1:] = -first
        targets[2, channel, 1:] = second
        targets[3, channel, 1:] = -second
    return targets


def target_parameters(targets: np.ndarray, max_angle: float) -> np.ndarray:
    if not max_angle > math.pi:
        raise ValueError("the quaternion chart must extend strictly beyond pi")
    magnitude = np.arctanh(math.pi / max_angle)
    # unit_quaternion_from_bivector emits a leading minus on its vector part.
    return -magnitude * targets[..., 1:]


def retract_checkpoint(
    source: Path,
    destination: Path,
    *,
    device: torch.device,
) -> dict[str, object]:
    checkpoint = torch.load(source, map_location=device, weights_only=False)
    config = checkpoint["config"]
    if checkpoint["family"] != "pure_quaternion_spinor":
        raise ValueError(f"{source} is not a quaternion-spinor checkpoint")
    model = PureGroupActionModel(
        len(INPUT_ELEMENTS),
        8,
        family=checkpoint["family"],
        channels=int(config["channels"]),
        max_rotor_angle=float(config["max_angle"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    token_ids = torch.arange(4, device=device)
    before = model.token_actions(token_ids).detach().cpu().double().numpy()
    targets = exact_q8_targets(before)
    parameters = target_parameters(targets, float(config["max_angle"]))
    with torch.no_grad():
        model.action_parameters.copy_(
            torch.as_tensor(
                parameters,
                dtype=model.action_parameters.dtype,
                device=device,
            )
        )
    after = model.token_actions(token_ids).detach().cpu().double().numpy()
    per_channel_projection = np.sqrt(np.mean(np.square(after - before), axis=(0, 2)))
    diagnostics = representation_diagnostics(model, GROUPS["q8"], INPUT_ELEMENTS)
    dense = central_pair_evaluation(
        model,
        base_lengths=SMOKE_BASE_LENGTHS,
        batches=2,
        batch_size=512,
        seed_base=5_300_000 + 10_000 * int(config["seed"]),
        device=device,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    retracted = dict(checkpoint)
    retracted["config"] = {
        **config,
        "joint_q8_retraction": True,
        "joint_q8_retraction_source": str(source),
    }
    retracted["state_dict"] = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    torch.save(retracted, destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "seed": int(config["seed"]),
        "per_channel_action_projection_rms": [
            float(value) for value in per_channel_projection
        ],
        "maximum_action_projection_rms": float(per_channel_projection.max()),
        "dense_central_pair_evaluation": dense,
        "representation_diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    torch.use_deterministic_algorithms(True)
    results = []
    for source in args.sources:
        destination = args.output_directory / f"{source.stem}_q8_retracted.pt"
        results.append(
            retract_checkpoint(source, destination, device=device)
        )
        print(source.name, results[-1]["dense_central_pair_evaluation"]["gate_pass"], flush=True)
    report = {
        "experiment": "joint Q8 spinor representation retraction",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
