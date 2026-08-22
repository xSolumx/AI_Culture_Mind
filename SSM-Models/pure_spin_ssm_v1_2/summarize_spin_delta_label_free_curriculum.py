"""Summarize the frozen retrieval-only Spin-Delta curriculum cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from spin_delta_label_free_curriculum import (
    ARMS,
    DATA_SEEDS,
    INIT_SEEDS,
    PROTOCOL,
    READINESS_WRITES,
    SCHEDULES,
    SCORED_WRITES,
)

ROUTER_METRICS = (
    "write_event_f1",
    "query_event_f1",
    "write_slot_accuracy",
    "query_slot_accuracy",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slot_gauge(readiness) -> tuple[str, dict[str, dict[str, float]]]:
    correct = total = 0
    for writes in READINESS_WRITES:
        metrics = readiness[str(writes)]["router"]
        correct += metrics["write_slot_correct"] + metrics["query_slot_correct"]
        total += metrics["write_slot_total"] + metrics["query_slot_total"]
    gauge = "identity" if correct >= total - correct else "swap"
    oriented = {}
    for writes in READINESS_WRITES:
        metrics = readiness[str(writes)]["router"]
        write_slot = metrics["write_slot_accuracy"]
        query_slot = metrics["query_slot_accuracy"]
        if gauge == "swap":
            write_slot = 1.0 - write_slot
            query_slot = 1.0 - query_slot
        oriented[str(writes)] = {
            "write_event_f1": metrics["write_event_f1"],
            "query_event_f1": metrics["query_event_f1"],
            "write_slot_accuracy": write_slot,
            "query_slot_accuracy": query_slot,
        }
    return gauge, oriented


def _ranges(rows, arm):
    selected = [row for row in rows if row["arm"] == arm]
    init_ranges = {
        str(data): max(
            row["accuracy"]["16"] for row in selected if row["data_seed"] == data
        )
        - min(row["accuracy"]["16"] for row in selected if row["data_seed"] == data)
        for data in DATA_SEEDS
    }
    data_ranges = {
        str(init): max(
            row["accuracy"]["16"] for row in selected if row["init_seed"] == init
        )
        - min(row["accuracy"]["16"] for row in selected if row["init_seed"] == init)
        for init in INIT_SEEDS
    }
    return init_ranges, data_ranges


def summarize(paths: list[Path]) -> dict[str, object]:
    if len(paths) != 9:
        raise ValueError("label-free curriculum requires nine artifacts")
    rows = []
    hashes = None
    seen = set()
    execution_ids = {}
    source_digests = {}
    exposure = {}
    contract_pass = True
    for path in paths:
        payload = json.loads(path.read_text())
        if (
            payload["stage"] != "spin_delta_label_free_curriculum"
            or payload["protocol"] != PROTOCOL
        ):
            raise ValueError(f"wrong protocol: {path}")
        config = payload["config"]
        cell = (config["init_seed"], config["data_seed"])
        if cell in seen:
            raise ValueError(f"duplicate cell {cell}")
        seen.add(cell)
        frozen = {
            "steps": 800,
            "batch_size": 128,
            "evaluation_writes": list(READINESS_WRITES),
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
        if any(config[name] != expected for name, expected in frozen.items()):
            raise ValueError(f"configuration differs from frozen protocol: {path}")
        if hashes is None:
            hashes = payload["implementation_sha256"]
        if payload["implementation_sha256"] != hashes:
            raise ValueError("implementation hashes differ")
        execution = payload["cohort_execution"]
        if (
            execution["shared_initial_single_execution"] is not True
            or execution["init_seed"] != config["init_seed"]
            or execution["data_seeds"] != list(DATA_SEEDS)
        ):
            raise ValueError(f"shared-initial execution failed: {path}")
        prior_execution = execution_ids.setdefault(
            config["init_seed"], execution["execution_id"]
        )
        prior_digest = source_digests.setdefault(
            config["init_seed"], payload["source_state_sha256"]
        )
        if (
            execution["execution_id"] != prior_execution
            or payload["source_state_sha256"] != prior_digest
        ):
            raise ValueError("shared initial state differs within an init cohort")
        required_contract = {
            "training_uses_router_labels": False,
            "router_auxiliary_loss_weight": 0.0,
            "oracle_controls_supplied_to_model": False,
            "audit_labels_detached_from_loss": True,
            "router_and_core_jointly_trainable": True,
        }
        if payload["contract"] != required_contract:
            contract_pass = False
        if [arm["arm"] for arm in payload["arms"]] != list(ARMS):
            raise ValueError(f"wrong arm order: {path}")
        for arm in payload["arms"]:
            name = arm["arm"]
            expected_schedule = [
                {"writes": writes, "steps": steps} for writes, steps in SCHEDULES[name]
            ]
            if arm["training_schedule"] != expected_schedule:
                raise ValueError(f"wrong schedule: {path}")
            if arm["initial_state_sha256"] != payload["source_state_sha256"]:
                raise ValueError(f"arm clone differs: {path}")
            arm_contract = (
                arm["training_uses_router_labels"] is False
                and arm["router_auxiliary_loss_weight"] == 0.0
                and arm["oracle_controls_supplied_to_model"] is False
                and arm["audit_labels_detached_from_loss"] is True
            )
            contract_pass = contract_pass and arm_contract
            values = [arm["final"][str(writes)]["accuracy"] for writes in SCORED_WRITES]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"nonfinite metric: {path}")
            gauge, oriented = _slot_gauge(arm["router_readiness"])
            arm_exposure = (arm["training_examples"], arm["training_tokens"])
            if name in exposure and exposure[name] != arm_exposure:
                raise ValueError("training exposure differs within an arm")
            exposure[name] = arm_exposure
            rows.append(
                {
                    "init_seed": config["init_seed"],
                    "data_seed": config["data_seed"],
                    "arm": name,
                    "accuracy": {
                        str(writes): arm["final"][str(writes)]["accuracy"]
                        for writes in SCORED_WRITES
                    },
                    "slot_gauge": gauge,
                    "oriented_router_readiness": oriented,
                    "source": path.as_posix(),
                    "source_sha256": _sha(path),
                }
            )
    expected = {(init, data) for init in INIT_SEEDS for data in DATA_SEEDS}
    if seen != expected:
        raise ValueError("cells do not match the frozen grid")
    rows.sort(key=lambda row: (row["init_seed"], row["data_seed"], row["arm"]))

    arm_summaries = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        init_ranges, data_ranges = _ranges(rows, arm)
        router_minimum = min(
            row["oriented_router_readiness"][str(writes)][metric]
            for row in selected
            for writes in READINESS_WRITES
            for metric in ROUTER_METRICS
        )
        arm_summaries[arm] = {
            "mean_accuracy": {
                str(writes): sum(row["accuracy"][str(writes)] for row in selected) / 9
                for writes in SCORED_WRITES
            },
            "minimum_accuracy": {
                str(writes): min(row["accuracy"][str(writes)] for row in selected)
                for writes in SCORED_WRITES
            },
            "maximum_accuracy": {
                str(writes): max(row["accuracy"][str(writes)] for row in selected)
                for writes in SCORED_WRITES
            },
            "initialization_ranges_at_16": init_ranges,
            "data_order_ranges_at_16": data_ranges,
            "router_readiness_minimum_after_slot_gauge": router_minimum,
            "router_identification_pass": router_minimum >= 0.99,
            "robustness_pass": all(
                row["accuracy"]["8"] >= 0.95
                and row["accuracy"]["16"] >= 0.95
                and row["accuracy"]["32"] >= 0.93
                for row in selected
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
    init_max = max(arm_summaries["curriculum"]["initialization_ranges_at_16"].values())
    data_max = max(arm_summaries["curriculum"]["data_order_ranges_at_16"].values())
    worst_rescue = curriculum_minimum >= fixed_minimum + 0.02
    variance_pass = init_max < 0.05 and data_max < 0.05
    no_regression = worst_improvement >= -0.02
    retrieval_pass = (
        arm_summaries["curriculum"]["robustness_pass"]
        and worst_rescue
        and variance_pass
        and mean_improvement >= 0.0
        and no_regression
    )
    identification_pass = arm_summaries["curriculum"]["router_identification_pass"]
    return {
        "schema_version": 1,
        "rows": rows,
        "arms": arm_summaries,
        "paired_improvements": paired,
        "label_free_contract_pass": contract_pass,
        "mean_paired_improvement_at_16": mean_improvement,
        "worst_paired_improvement_at_16": worst_improvement,
        "worst_cell_rescue_pass": worst_rescue,
        "variance_contraction_pass": variance_pass,
        "no_large_paired_regression_pass": no_regression,
        "retrieval_autonomy_pass": retrieval_pass,
        "router_identification_pass": identification_pass,
        "learning_autonomy_pass": contract_pass
        and retrieval_pass
        and identification_pass,
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
