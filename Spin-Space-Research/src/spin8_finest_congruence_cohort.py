"""Apply the frozen finest-congruence Spin(8) compiler to a checkpoint cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from spin8_finest_congruence_compiler import compile_finest_congruence_checkpoint
from spin8_state_only_compiler import posthoc_state_only_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    torch.use_deterministic_algorithms(True)
    results = []
    for source in args.sources:
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        seed = int(checkpoint["config"]["seed"])
        destination = args.output_directory / f"q8_spin8_seed{seed}_finest.pt"
        try:
            result, recovered = compile_finest_congruence_checkpoint(
                source, destination, device=device
            )
            result = posthoc_state_only_score(
                result, recovered, destination, device=device
            )
        except Exception as error:
            result = {
                "source": str(source), "seed": seed, "passed": False,
                "pipeline_exception": f"{type(error).__name__}: {error}",
            }
        results.append(result)
        print(
            f"seed={seed} passed={result['passed']} "
            f"selected={result.get('state_cardinality_selected')} "
            f"viable={result.get('finest_congruence_discovery', {}).get('viable_cardinalities')}",
            flush=True,
        )
    passed = sum(bool(result["passed"]) for result in results)
    report = {
        "experiment": "Spin8 state-only finest-congruence fresh cohort",
        "development_seeds_38_43_46_excluded": True,
        "prospective_smoke_seed_48_excluded": True,
        "fresh_seed_count": len(results),
        "fresh_pass_count": passed,
        "reliability_gate_at_least_8_of_9": len(results) == 9 and passed >= 8,
        "uniform_reliability_9_of_9": len(results) == 9 and passed == 9,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
