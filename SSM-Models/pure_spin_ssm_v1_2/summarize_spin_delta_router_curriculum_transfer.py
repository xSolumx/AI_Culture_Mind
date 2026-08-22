"""Summarize the frozen learned-router Spin-Delta curriculum transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

INIT_SEEDS = (587, 593, 599)
DATA_SEEDS = (601, 607, 613)
ARMS = ("fixed", "curriculum")
READINESS_WRITES = (2, 3, 5, 8, 16, 32)
SCORED_WRITES = (8, 16, 32)
ROUTER_METRICS = (
    "write_event_f1",
    "query_event_f1",
    "write_slot_accuracy",
    "query_slot_accuracy",
)
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
        str(data): max(
            row["accuracy"]["16"] for row in arm_rows if row["data_seed"] == data
        )
        - min(row["accuracy"]["16"] for row in arm_rows if row["data_seed"] == data)
        for data in DATA_SEEDS
    }
    data_ranges = {
        str(init): max(
            row["accuracy"]["16"] for row in arm_rows if row["init_seed"] == init
        )
        - min(row["accuracy"]["16"] for row in arm_rows if row["init_seed"] == init)
        for init in INIT_SEEDS
    }
    return init_ranges, data_ranges


def summarize(paths: list[Path]) -> dict[str, object]:
    if len(paths) != 9:
        raise ValueError("router-curriculum transfer requires nine artifacts")
    rows = []
    hashes = None
    seen = set()
    router_digests: dict[int, str] = {}
    readiness_by_init: dict[int, object] = {}
    readiness_values = []
    training_router_values = []
    exposure: dict[str, tuple[int, int]] = {}
    for path in paths:
        payload = json.loads(path.read_text())
        if payload["stage"] != "spin_delta_router_curriculum_transfer":
            raise ValueError(f"wrong stage: {path}")
        config = payload["config"]
        cell = (config["init_seed"], config["data_seed"])
        if cell in seen:
            raise ValueError(f"duplicate cell {cell}")
        seen.add(cell)
        if (
            payload["autonomous_evaluation"] is not True
            or payload["oracle_controls_supplied_to_model"] is not False
        ):
            raise ValueError(f"autonomous evaluation contract failed: {path}")
        frozen = {
            "router_steps": 100,
            "core_steps": 800,
            "batch_size": 128,
            "router_training_writes": 8,
            "evaluation_writes": [2, 3, 5, 8, 16, 32],
            "evaluation_batches": 16,
            "learning_rate": 0.003,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
            "d_model": 64,
            "layers": 2,
            "router_width": 32,
            "router_kernel_size": 3,
            "router_temperature": 1.0,
        }
        for name, expected in frozen.items():
            if config[name] != expected:
                raise ValueError(f"{name} differs from frozen protocol: {path}")
        if hashes is None:
            hashes = payload["implementation_sha256"]
        if payload["implementation_sha256"] != hashes:
            raise ValueError("implementation hashes differ")
        router_phase = payload["router_phase"]
        if router_phase["core_untouched"] is not True:
            raise ValueError(f"router phase modified the core: {path}")
        prior_digest = router_digests.setdefault(
            config["init_seed"], router_phase["post_router_state_sha256"]
        )
        if router_phase["post_router_state_sha256"] != prior_digest:
            raise ValueError("post-router state differs across core data seeds")
        prior_readiness = readiness_by_init.setdefault(
            config["init_seed"], router_phase["readiness"]
        )
        if router_phase["readiness"] != prior_readiness:
            raise ValueError("router readiness differs across core data seeds")
        for writes in READINESS_WRITES:
            metrics = router_phase["readiness"][str(writes)]["router"]
            readiness_values.extend(metrics[name] for name in ROUTER_METRICS)
        arms = payload["arms"]
        if [arm["arm"] for arm in arms] != list(ARMS):
            raise ValueError(f"wrong arm order: {path}")
        if arms[0]["initial_state_sha256"] != arms[1]["initial_state_sha256"]:
            raise ValueError(f"cloned arm states differ: {path}")
        for arm in arms:
            name = arm["arm"]
            if arm["training_schedule"] != EXPECTED_SCHEDULES[name]:
                raise ValueError(f"schedule differs from frozen protocol: {path}")
            if arm["router_frozen"] is not True:
                raise ValueError(f"router was not frozen: {path}")
            controls = arm["training_router_metrics"]
            training_router_values.extend(controls[metric] for metric in ROUTER_METRICS)
            values = [arm["final"][str(writes)]["accuracy"] for writes in SCORED_WRITES]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"nonfinite metric: {path}")
            arm_exposure = (arm["training_examples"], arm["training_tokens"])
            prior_exposure = exposure.setdefault(name, arm_exposure)
            if arm_exposure != prior_exposure:
                raise ValueError("training exposure differs within an arm")
            rows.append(
                {
                    "init_seed": config["init_seed"],
                    "data_seed": config["data_seed"],
                    "arm": name,
                    "accuracy": {
                        str(writes): arm["final"][str(writes)]["accuracy"]
                        for writes in SCORED_WRITES
                    },
                    "source": path.as_posix(),
                    "source_sha256": _sha(path),
                }
            )
    expected = {(init, data) for init in INIT_SEEDS for data in DATA_SEEDS}
    if seen != expected:
        raise ValueError("cohort cells do not match the frozen grid")
    rows.sort(key=lambda row: (row["init_seed"], row["data_seed"], row["arm"]))

    arm_summaries = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        init_ranges, data_ranges = _ranges(rows, arm)
        arm_summaries[arm] = {
            "mean_accuracy": {
                str(writes): sum(row["accuracy"][str(writes)] for row in arm_rows) / 9
                for writes in SCORED_WRITES
            },
            "minimum_accuracy": {
                str(writes): min(row["accuracy"][str(writes)] for row in arm_rows)
                for writes in SCORED_WRITES
            },
            "maximum_accuracy": {
                str(writes): max(row["accuracy"][str(writes)] for row in arm_rows)
                for writes in SCORED_WRITES
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
    by_cell = {(row["init_seed"], row["data_seed"], row["arm"]): row for row in rows}
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
                        str(writes): curriculum["accuracy"][str(writes)]
                        - fixed["accuracy"][str(writes)]
                        for writes in SCORED_WRITES
                    },
                }
            )
    mean_improvement = sum(row["improvement"]["16"] for row in paired) / 9
    worst_improvement = min(row["improvement"]["16"] for row in paired)
    fixed_minimum = arm_summaries["fixed"]["minimum_accuracy"]["16"]
    curriculum_minimum = arm_summaries["curriculum"]["minimum_accuracy"]["16"]
    curriculum_init_max = max(
        arm_summaries["curriculum"]["initialization_ranges_at_16"].values()
    )
    curriculum_data_max = max(
        arm_summaries["curriculum"]["data_order_ranges_at_16"].values()
    )
    readiness_pass = min(readiness_values) >= 0.99
    training_controls_pass = min(training_router_values) == 1.0
    autonomous_validity = readiness_pass and training_controls_pass
    worst_cell_rescue = curriculum_minimum >= fixed_minimum + 0.02
    variance_contraction = curriculum_init_max < 0.05 and curriculum_data_max < 0.05
    no_large_regression = worst_improvement >= -0.02
    transfer_pass = (
        autonomous_validity
        and arm_summaries["curriculum"]["robustness_pass"]
        and worst_cell_rescue
        and variance_contraction
        and mean_improvement >= 0.0
        and no_large_regression
    )
    return {
        "schema_version": 1,
        "rows": rows,
        "arms": arm_summaries,
        "paired_improvements": paired,
        "router_readiness_pass": readiness_pass,
        "training_controls_exact_pass": training_controls_pass,
        "autonomous_router_validity_pass": autonomous_validity,
        "mean_paired_improvement_at_16": mean_improvement,
        "worst_paired_improvement_at_16": worst_improvement,
        "worst_cell_rescue_pass": worst_cell_rescue,
        "variance_contraction_pass": variance_contraction,
        "no_large_paired_regression_pass": no_large_regression,
        "curriculum_transfer_pass": transfer_pass,
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
