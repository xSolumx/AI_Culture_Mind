"""Apply the frozen Spin(8) Q8 retraction and long audit to a checkpoint cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mechanistic_group_actions import PureGroupActionModel
from q8_spinor_center_experiment import INPUT_ELEMENTS, central_pair_evaluation
from q8_spinor_center_long_audit import LONG_BASE_LENGTHS
from spin8_q8_joint_retraction import retract_checkpoint


def long_audit(path: Path, device: torch.device, batch_size: int) -> dict[str, object]:
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
    return central_pair_evaluation(
        model,
        base_lengths=LONG_BASE_LENGTHS,
        batches=1,
        batch_size=batch_size,
        seed_base=8_300_000 + 10_000 * int(config["seed"]),
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--long-batch-size", type=int, default=128)
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
    results = []
    for source in args.sources:
        destination = args.output_directory / f"{source.stem}_retracted.pt"
        result = retract_checkpoint(source, destination, device=device)
        result["long_central_pair_evaluation"] = long_audit(
            destination, device, args.long_batch_size
        )
        result["fresh_cohort_passed"] = bool(
            result["passed"] and result["long_central_pair_evaluation"]["gate_pass"]
        )
        results.append(result)
        print(
            f"seed={result['seed']} raw_projection={max(result['per_channel_frame_projection_rms']):.6g} "
            f"dense={result['dense_central_pair_evaluation']['gate_pass']} "
            f"long={result['long_central_pair_evaluation']['gate_pass']}",
            flush=True,
        )
    passed = sum(bool(result["fresh_cohort_passed"]) for result in results)
    report = {
        "experiment": "Spin(8) Q8 fresh joint-retraction cohort",
        "seed0_excluded_from_confirmatory_count": True,
        "fresh_seed_count": len(results),
        "fresh_pass_count": passed,
        "reliability_gate_at_least_8_of_9": len(results) == 9 and passed >= 8,
        "uniform_reliability_9_of_9": len(results) == 9 and passed == 9,
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
