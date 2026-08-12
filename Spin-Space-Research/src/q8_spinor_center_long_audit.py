"""Long, parity-matched central-pair audit for trained Q8 checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mechanistic_group_actions import PureGroupActionModel, streaming_equivalence
from q8_spinor_center_experiment import INPUT_ELEMENTS, central_pair_evaluation


LONG_BASE_LENGTHS = (4095, 4096, 16383, 16384)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
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
    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        if tuple(checkpoint["input_elements"]) != INPUT_ELEMENTS:
            raise ValueError(f"unexpected Q8 alphabet in {path}")
        model = PureGroupActionModel(
            len(INPUT_ELEMENTS),
            8,
            family=checkpoint["family"],
            channels=int(config["channels"]),
            max_rotor_angle=float(config["max_angle"]),
        ).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        evaluation = central_pair_evaluation(
            model,
            base_lengths=LONG_BASE_LENGTHS,
            batches=1,
            batch_size=args.batch_size,
            seed_base=3_300_000 + 10_000 * int(config["seed"]),
            device=device,
        )
        probe = torch.arange(192, device=device).reshape(3, 64) % len(INPUT_ELEMENTS)
        parity = streaming_equivalence(model, probe)
        results.append(
            {
                "checkpoint": str(path),
                "family": checkpoint["family"],
                "max_angle": float(config["max_angle"]),
                "seed": int(config["seed"]),
                "central_pair_evaluation": evaluation,
                "streaming_equivalence": parity,
            }
        )
        print(
            path.name,
            evaluation["minimum_pair_member_accuracy"],
            evaluation["minimum_both_members_correct_accuracy"],
        )
    report = {
        "experiment": "Q8 long central-pair audit",
        "base_lengths": list(LONG_BASE_LENGTHS),
        "batch_size": args.batch_size,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "checkpoints": len(results)}))


if __name__ == "__main__":
    main()
