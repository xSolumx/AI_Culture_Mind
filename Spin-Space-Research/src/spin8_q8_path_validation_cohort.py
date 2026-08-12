"""Apply the frozen path-section compiler to an untouched Spin(8) cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from spin8_q8_path_section_compiler import compile_checkpoint


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
        else "cpu" if args.device == "auto" else args.device
    )
    torch.use_deterministic_algorithms(True)
    results = []
    for source in args.sources:
        destination = args.output_directory / f"{source.stem}_path_compiled.pt"
        try:
            result = compile_checkpoint(source, destination, device=device)
        except Exception as error:
            checkpoint = torch.load(source, map_location="cpu", weights_only=False)
            result = {
                "source": str(source),
                "seed": int(checkpoint["config"]["seed"]),
                "passed": False,
                "pipeline_exception": f"{type(error).__name__}: {error}",
            }
        results.append(result)
        print(
            f"seed={result['seed']} passed={result['passed']} "
            f"calibration={result.get('raw_calibration_final_accuracy')} ",
            flush=True,
        )
    passed = sum(bool(result["passed"]) for result in results)
    report = {
        "experiment": "Spin(8) Q8 path-section untouched validation cohort",
        "seeds_0_through_9_excluded": True,
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
