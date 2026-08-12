"""Dense long-horizon audit for endpoint-only retraction checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from compare_recurrences import GROUPS
from mechanistic_group_actions import PureGroupActionModel
from train_self_compiling_retraction import evaluate_anchor


AUDIT_LENGTHS = tuple(range(4_096, 16_385, 1_024))


def evaluate_seed(
    seed: int,
    checkpoint_directory: Path,
    device: torch.device,
    *,
    generator_class_index: int,
    expected_curriculum: bool,
) -> dict[str, object]:
    path = checkpoint_directory / f"self_compiling_retraction_seed{seed}.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    if not config.get("endpoint_only"):
        raise ValueError(f"checkpoint {path} is not marked endpoint-only")
    if bool(config.get("endpoint_length_curriculum")) != expected_curriculum:
        raise ValueError(f"checkpoint {path} has the wrong curriculum protocol")
    group = GROUPS["a5"]
    inputs = tuple(checkpoint["input_elements"])
    model = PureGroupActionModel(
        len(inputs),
        group.order,
        family=checkpoint["family"],
        channels=config["channels"],
        max_rotor_angle=config["max_rotor_angle"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    result = evaluate_anchor(
        model,
        group,
        inputs,
        config["anchor_channel"],
        generator_class=generator_class_index,
        lengths=AUDIT_LENGTHS,
        batches=1,
        batch_size=256,
        seed_base=1_810_000,
        device=device,
    )
    return {
        "training_seed": seed,
        "checkpoint": str(path),
        "minimum_accuracy": result["minimum_accuracy"],
        "gate_pass": result["gate_pass"],
        "maximum_path_drift": result["path_vs_canonical_state_drift_max"],
        "evaluation": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generator-class-index", type=int, default=59)
    parser.add_argument("--expected-curriculum", action="store_true")
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    results = [
        evaluate_seed(
            seed,
            args.checkpoint_directory,
            device,
            generator_class_index=args.generator_class_index,
            expected_curriculum=args.expected_curriculum,
        )
        for seed in args.seeds
    ]
    report = {
        "experiment": "endpoint-only dense long-horizon audit",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "lengths": list(AUDIT_LENGTHS),
        "generator_class_index": args.generator_class_index,
        "expected_curriculum": args.expected_curriculum,
        "results": results,
        "all_pass": all(result["gate_pass"] for result in results),
        "minimum_accuracy": min(result["minimum_accuracy"] for result in results),
        "maximum_path_drift": max(result["maximum_path_drift"] for result in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "all_pass": report["all_pass"],
        "minimum_accuracy": report["minimum_accuracy"],
        "maximum_path_drift": report["maximum_path_drift"],
    }))


if __name__ == "__main__":
    main()
