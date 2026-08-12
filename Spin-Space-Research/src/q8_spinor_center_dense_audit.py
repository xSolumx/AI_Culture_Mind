"""Full preregistered odd/even dense audit for trained Q8 checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mechanistic_group_actions import PureGroupActionModel, streaming_equivalence
from q8_spinor_center_experiment import INPUT_ELEMENTS, central_pair_evaluation


DENSE_BASE_LENGTHS = tuple(range(15, 33)) + tuple(
    length
    for multiplier in range(3, 17)
    for length in (16 * multiplier - 1, 16 * multiplier)
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--batches", type=int, default=2)
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
            base_lengths=DENSE_BASE_LENGTHS,
            batches=args.batches,
            batch_size=args.batch_size,
            seed_base=4_300_000 + 10_000 * int(config["seed"]),
            device=device,
        )
        probe = torch.arange(192, device=device).reshape(3, 64) % len(INPUT_ELEMENTS)
        results.append(
            {
                "checkpoint": str(path),
                "family": checkpoint["family"],
                "max_angle": float(config["max_angle"]),
                "seed": int(config["seed"]),
                "central_pair_evaluation": evaluation,
                "streaming_equivalence": streaming_equivalence(model, probe),
            }
        )
        print(
            path.name,
            evaluation["minimum_pair_member_accuracy"],
            evaluation["minimum_both_members_correct_accuracy"],
            flush=True,
        )
    report = {
        "experiment": "Q8 full dense central-pair audit",
        "base_lengths": list(DENSE_BASE_LENGTHS),
        "batches": args.batches,
        "batch_size": args.batch_size,
        "all_pass": all(
            item["central_pair_evaluation"]["gate_pass"] for item in results
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "all_pass": report["all_pass"]}))


if __name__ == "__main__":
    main()
