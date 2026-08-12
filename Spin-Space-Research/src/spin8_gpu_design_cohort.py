"""Run a sequential, resource-bounded cohort of CUDA design falsifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spin8_gpu_design_audit import run


def run_cohort(
    seeds: list[int],
    *,
    workers: int,
    samples_per_allocation: int,
    batch_size: int,
    restarts_per_allocation: int,
    optimization_steps: int,
    sensitivity_samples: int,
    reweight_steps: int,
    noise_samples: int,
) -> dict[str, object]:
    if len(set(seeds)) != len(seeds):
        raise ValueError("cohort seeds must be unique")
    reports = [
        run(
            seed=seed,
            workers=workers,
            samples_per_allocation=samples_per_allocation,
            batch_size=batch_size,
            restarts_per_allocation=restarts_per_allocation,
            optimization_steps=optimization_steps,
            sensitivity_samples=sensitivity_samples,
            reweight_steps=reweight_steps,
            noise_samples=noise_samples,
        )
        for seed in seeds
    ]
    rows = []
    for report in reports:
        sensitivity = report["continuous_sensitivity_and_reweighting"]
        reweighting = sensitivity["fixed_support_gradient_reweighting"]
        rows.append(
            {
                "seed": report["seed"],
                "passed": report["passed"],
                "dense_best_log_determinant": report["dense_interior_falsifier"][
                    "best_observed_log_determinant"
                ],
                "dense_candidate_count": report["dense_interior_falsifier"][
                    "candidate_above_target_plus_1e_minus_9"
                ],
                "gradient_best_log_determinant": report["gradient_challenger_search"][
                    "best_refined_log_determinant"
                ],
                "gradient_candidate_count": report["gradient_challenger_search"][
                    "candidate_above_target_plus_1e_minus_8"
                ],
                "sensitivity_maximum": sensitivity[
                    "global_sampled_or_optimized_maximum"
                ],
                "reweighting_maximum_weight_error": reweighting[
                    "maximum_weight_error_from_exact_symmetric_solution"
                ],
                "noise_uphill_count": sum(
                    row["samples_above_exact_local_optimum_plus_1e_minus_9"]
                    for row in report["monte_carlo_noise_profile"]["rows"]
                ),
                "maximum_float32_logdet_error": max(
                    row["maximum_absolute_float32_logdet_error"]
                    for row in report["monte_carlo_noise_profile"]["rows"]
                ),
                "peak_cuda_memory_mib": report["hardware"]["peak_cuda_memory_mib"],
            }
        )
    return {
        "experiment": "multi-seed CUDA continuous Spin8 design falsification cohort",
        "seeds": seeds,
        "seed_count": len(seeds),
        "configuration": {
            "workers": workers,
            "samples_per_allocation": samples_per_allocation,
            "batch_size": batch_size,
            "restarts_per_allocation": restarts_per_allocation,
            "optimization_steps": optimization_steps,
            "sensitivity_samples": sensitivity_samples,
            "reweight_steps": reweight_steps,
            "noise_samples": noise_samples,
        },
        "total_dense_interior_samples": sum(
            report["dense_interior_falsifier"]["total_samples"] for report in reports
        ),
        "total_gradient_restarts": len(seeds)
        * len(reports[0]["gradient_challenger_search"]["allocations"])
        * restarts_per_allocation,
        "pass_count": sum(row["passed"] for row in rows),
        "rows": rows,
        "raw_seed_reports": reports,
        "claim_boundary": (
            "This cohort can find numerical counterexample candidates but cannot "
            "prove global optimality. Any candidate must be rationalized and "
            "replayed exactly."
        ),
        "passed": all(row["passed"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--samples-per-allocation", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--restarts-per-allocation", type=int, default=8)
    parser.add_argument("--optimization-steps", type=int, default=250)
    parser.add_argument("--sensitivity-samples", type=int, default=65536)
    parser.add_argument("--reweight-steps", type=int, default=1000)
    parser.add_argument("--noise-samples", type=int, default=4096)
    arguments = parser.parse_args()
    report = run_cohort(
        arguments.seeds,
        workers=arguments.workers,
        samples_per_allocation=arguments.samples_per_allocation,
        batch_size=arguments.batch_size,
        restarts_per_allocation=arguments.restarts_per_allocation,
        optimization_steps=arguments.optimization_steps,
        sensitivity_samples=arguments.sensitivity_samples,
        reweight_steps=arguments.reweight_steps,
        noise_samples=arguments.noise_samples,
    )
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "experiment": report["experiment"],
                "seed_count": report["seed_count"],
                "pass_count": report["pass_count"],
                "total_dense_interior_samples": report["total_dense_interior_samples"],
                "total_gradient_restarts": report["total_gradient_restarts"],
                "rows": report["rows"],
                "passed": report["passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not report["passed"]:
        raise SystemExit("one or more cohort falsification gates failed")


if __name__ == "__main__":
    main()
