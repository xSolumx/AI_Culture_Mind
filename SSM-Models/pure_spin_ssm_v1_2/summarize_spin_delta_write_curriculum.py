"""Summarize the frozen paired Spin-Delta write-curriculum cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

INIT_SEEDS = (541, 547, 557)
DATA_SEEDS = (563, 569, 571)
ARMS = ("fixed", "curriculum")
EXPECTED_SCHEDULES = {
    "fixed": [{"writes": 8, "steps": 800}],
    "curriculum": [
        {"writes": 2, "steps": 100},
        {"writes": 3, "steps": 100},
        {"writes": 5, "steps": 200},
        {"writes": 8, "steps": 400},
    ],
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ranges(rows: list[dict[str, object]], arm: str):
    arm_rows = [row for row in rows if row["arm"] == arm]
    init_ranges = {
        str(data): max(row["accuracy"]["16"] for row in arm_rows if row["data_seed"] == data)
        - min(row["accuracy"]["16"] for row in arm_rows if row["data_seed"] == data)
        for data in DATA_SEEDS
    }
    data_ranges = {
        str(init): max(row["accuracy"]["16"] for row in arm_rows if row["init_seed"] == init)
        - min(row["accuracy"]["16"] for row in arm_rows if row["init_seed"] == init)
        for init in INIT_SEEDS
    }
    return init_ranges, data_ranges


def summarize(paths: list[Path]) -> dict[str, object]:
    if len(paths) != 18:
        raise ValueError("write-curriculum cohort requires 18 artifacts")
    rows = []
    hashes = None
    seen = set()
    initial_digests: dict[int, str] = {}
    initial_metrics: dict[int, object] = {}
    exposure: dict[str, tuple[int, int]] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        if payload["stage"] != "spin_delta_write_curriculum":
            raise ValueError(f"wrong stage: {path}")
        config = payload["config"]
        cell = (config["init_seed"], config["data_seed"], config["arm"])
        if cell in seen:
            raise ValueError(f"duplicate cell {cell}")
        seen.add(cell)
        if payload["training_schedule"] != EXPECTED_SCHEDULES[config["arm"]]:
            raise ValueError(f"schedule differs from frozen protocol: {path}")
        frozen_config = {
            "steps": 800,
            "batch_size": 128,
            "evaluation_writes": [8, 16, 32],
            "evaluation_batches": 16,
            "learning_rate": 0.003,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
            "d_model": 64,
            "layers": 2,
        }
        for name, expected in frozen_config.items():
            if config[name] != expected:
                raise ValueError(f"{name} differs from frozen protocol: {path}")
        if hashes is None:
            hashes = payload["implementation_sha256"]
        if payload["implementation_sha256"] != hashes:
            raise ValueError("implementation hashes differ")
        digest = payload["initial_state_sha256"]
        prior_digest = initial_digests.setdefault(config["init_seed"], digest)
        if digest != prior_digest:
            raise ValueError("paired initial model states differ")
        prior_metrics = initial_metrics.setdefault(config["init_seed"], payload["initial"])
        if payload["initial"] != prior_metrics:
            raise ValueError("paired initial evaluations differ")
        arm_exposure = (payload["training_examples"], payload["training_tokens"])
        prior_exposure = exposure.setdefault(config["arm"], arm_exposure)
        if arm_exposure != prior_exposure:
            raise ValueError("training exposure differs within an arm")
        rows.append(
            {
                "init_seed": cell[0],
                "data_seed": cell[1],
                "arm": cell[2],
                "accuracy": {
                    writes: payload["final"][writes]["accuracy"]
                    for writes in ("8", "16", "32")
                },
                "source": path.as_posix(),
                "source_sha256": _sha(path),
            }
        )
    expected_cells = {
        (init, data, arm)
        for init in INIT_SEEDS
        for data in DATA_SEEDS
        for arm in ARMS
    }
    if seen != expected_cells:
        raise ValueError("cohort cells do not match the frozen grid")
    rows.sort(key=lambda row: (row["init_seed"], row["data_seed"], row["arm"]))

    arm_summaries = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        init_ranges, data_ranges = _ranges(rows, arm)
        arm_summaries[arm] = {
            "mean_accuracy": {
                writes: sum(row["accuracy"][writes] for row in arm_rows) / len(arm_rows)
                for writes in ("8", "16", "32")
            },
            "minimum_accuracy": {
                writes: min(row["accuracy"][writes] for row in arm_rows)
                for writes in ("8", "16", "32")
            },
            "maximum_accuracy": {
                writes: max(row["accuracy"][writes] for row in arm_rows)
                for writes in ("8", "16", "32")
            },
            "initialization_ranges_at_16": init_ranges,
            "data_order_ranges_at_16": data_ranges,
            "robustness_pass": all(
                row["accuracy"]["8"] >= 0.95
                and row["accuracy"]["16"] >= 0.95
                and row["accuracy"]["32"] >= 0.93
                for row in arm_rows
            ),
        }

    by_cell = {
        (row["init_seed"], row["data_seed"], row["arm"]): row for row in rows
    }
    paired = []
    for init in INIT_SEEDS:
        for data in DATA_SEEDS:
            fixed = by_cell[(init, data, "fixed")]
            curriculum = by_cell[(init, data, "curriculum")]
            paired.append(
                {
                    "init_seed": init,
                    "data_seed": data,
                    "improvement": {
                        writes: curriculum["accuracy"][writes] - fixed["accuracy"][writes]
                        for writes in ("8", "16", "32")
                    },
                }
            )
    mean_improvement_16 = sum(row["improvement"]["16"] for row in paired) / 9
    worst_improvement_16 = min(row["improvement"]["16"] for row in paired)
    fixed_minimum_16 = arm_summaries["fixed"]["minimum_accuracy"]["16"]
    curriculum_minimum_16 = arm_summaries["curriculum"]["minimum_accuracy"]["16"]
    curriculum_init_max = max(
        arm_summaries["curriculum"]["initialization_ranges_at_16"].values()
    )
    curriculum_data_max = max(
        arm_summaries["curriculum"]["data_order_ranges_at_16"].values()
    )
    worst_cell_rescue = curriculum_minimum_16 >= fixed_minimum_16 + 0.02
    variance_contraction = curriculum_init_max < 0.05 and curriculum_data_max < 0.05
    no_large_regression = worst_improvement_16 >= -0.02
    robust_repair = (
        arm_summaries["curriculum"]["robustness_pass"]
        and worst_cell_rescue
        and variance_contraction
        and mean_improvement_16 >= 0.0
        and no_large_regression
    )
    return {
        "schema_version": 1,
        "rows": rows,
        "arms": arm_summaries,
        "paired_improvements": paired,
        "mean_paired_improvement_at_16": mean_improvement_16,
        "worst_paired_improvement_at_16": worst_improvement_16,
        "worst_cell_rescue_pass": worst_cell_rescue,
        "variance_contraction_pass": variance_contraction,
        "no_large_paired_regression_pass": no_large_regression,
        "robust_core_repair_pass": robust_repair,
        "training_exposure": {
            arm: {"examples": values[0], "tokens": values[1]}
            for arm, values in exposure.items()
        },
        "implementation_sha256": hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
