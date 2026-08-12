"""Run the frozen table-blind family compiler over checkpoint cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from spin8_table_blind_compiler import posthoc_q8_score
from table_blind_family_compiler import compile_family_checkpoint


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
        family = checkpoint["family"]
        destination = args.output_directory / f"q8_{family}_seed{seed}_blind.pt"
        try:
            result, recovered = compile_family_checkpoint(
                source, destination, device=device
            )
            result = posthoc_q8_score(
                result, recovered, destination, device=device
            )
        except Exception as error:
            result = {
                "source": str(source), "seed": seed, "family": family,
                "passed": False,
                "pipeline_exception": f"{type(error).__name__}: {error}",
            }
        results.append(result)
        print(
            f"family={family} seed={seed} passed={result['passed']} "
            f"rank={result.get('exact_section_rank')} "
            f"discarded={result.get('discarded_teacher_logit_rms')}",
            flush=True,
        )
    families = sorted({result["family"] for result in results})
    summaries = {}
    for family in families:
        members = [result for result in results if result["family"] == family]
        passed = sum(bool(result["passed"]) for result in members)
        summaries[family] = {
            "count": len(members), "passed": passed,
            "reliability_gate_at_least_8_of_9": len(members) == 9 and passed >= 8,
            "uniform_reliability_9_of_9": len(members) == 9 and passed == 9,
        }
    report = {
        "experiment": "table-blind family fresh validation cohort",
        "seeds": sorted({int(result["seed"]) for result in results}),
        "family_summaries": summaries,
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
