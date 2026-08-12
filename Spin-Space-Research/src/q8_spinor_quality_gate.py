"""Gate decoder channels using only distance to the joint Q8 manifold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compare_recurrences import GROUPS
from mechanistic_group_actions import PureGroupActionModel, representation_diagnostics
from q8_spinor_center_experiment import INPUT_ELEMENTS, central_pair_evaluation


SMOKE_BASE_LENGTHS = (15, 16, 31, 32, 63, 64, 127, 128, 255, 256)


def gate_pair(
    learned_path: Path,
    retracted_path: Path,
    destination: Path,
    *,
    threshold: float,
    device: torch.device,
) -> dict[str, object]:
    learned = torch.load(learned_path, map_location=device, weights_only=False)
    retracted = torch.load(retracted_path, map_location=device, weights_only=False)
    if learned["family"] != "pure_quaternion_spinor" or retracted["family"] != learned["family"]:
        raise ValueError("quality gating requires matching quaternion-spinor checkpoints")
    if int(learned["config"]["seed"]) != int(retracted["config"]["seed"]):
        raise ValueError("learned and retracted checkpoint seeds differ")
    config = retracted["config"]
    model = PureGroupActionModel(
        len(INPUT_ELEMENTS),
        8,
        family=retracted["family"],
        channels=int(config["channels"]),
        max_rotor_angle=float(config["max_angle"]),
    ).to(device)
    model.load_state_dict(retracted["state_dict"])
    learned_model = PureGroupActionModel(
        len(INPUT_ELEMENTS),
        8,
        family=learned["family"],
        channels=int(config["channels"]),
        max_rotor_angle=float(config["max_angle"]),
    ).to(device)
    learned_model.load_state_dict(learned["state_dict"])
    token_ids = torch.arange(4, device=device)
    before = learned_model.token_actions(token_ids).detach().cpu().double().numpy()
    after = model.token_actions(token_ids).detach().cpu().double().numpy()
    distances = np.sqrt(np.mean(np.square(after - before), axis=(0, 2)))
    mask = distances <= threshold
    if not np.any(mask):
        raise ValueError("quality gate rejected every channel")
    with torch.no_grad():
        weights = model.output_head.weight.reshape(8, model.channels, 8)
        rejected = torch.as_tensor(~mask, device=device)
        weights[:, rejected, :] = 0.0
    model.eval()
    dense = central_pair_evaluation(
        model,
        base_lengths=SMOKE_BASE_LENGTHS,
        batches=2,
        batch_size=512,
        seed_base=6_300_000 + 10_000 * int(config["seed"]),
        device=device,
    )
    diagnostics = representation_diagnostics(model, GROUPS["q8"], INPUT_ELEMENTS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    gated = dict(retracted)
    gated["config"] = {
        **config,
        "representation_quality_gate_threshold": threshold,
        "representation_quality_gate_mask": [bool(value) for value in mask],
        "representation_quality_gate_learned_source": str(learned_path),
    }
    gated["state_dict"] = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }
    torch.save(gated, destination)
    return {
        "seed": int(config["seed"]),
        "learned_source": str(learned_path),
        "retracted_source": str(retracted_path),
        "destination": str(destination),
        "threshold": threshold,
        "projection_rms": [float(value) for value in distances],
        "channel_mask": [bool(value) for value in mask],
        "retained_channels": int(mask.sum()),
        "dense_central_pair_evaluation": dense,
        "representation_diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned-directory", type=Path, required=True)
    parser.add_argument("--retracted-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--threshold", type=float, default=0.10)
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
    for seed in args.seeds:
        pattern = f"q8_pure_quaternion_spinor_angle*_seed{seed}.pt"
        learned_matches = list(args.learned_directory.glob(pattern))
        retracted_matches = list(args.retracted_directory.glob(
            f"q8_pure_quaternion_spinor_angle*_seed{seed}_q8_retracted.pt"
        ))
        if len(learned_matches) != 1 or len(retracted_matches) != 1:
            raise ValueError(f"expected one learned/retracted checkpoint for seed {seed}")
        destination = args.output_directory / f"q8_spinor_seed{seed}_quality_gated.pt"
        result = gate_pair(
            learned_matches[0],
            retracted_matches[0],
            destination,
            threshold=args.threshold,
            device=device,
        )
        results.append(result)
        print(seed, result["channel_mask"], result["dense_central_pair_evaluation"]["gate_pass"], flush=True)
    report = {
        "experiment": "Q8 representation-quality decoder gate",
        "threshold": args.threshold,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
