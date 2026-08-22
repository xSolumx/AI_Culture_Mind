"""Apply the frozen decisions to the Spin-Delta query continuation cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from spin_delta_query_continuation import (
    ARMS,
    DATA_SEEDS,
    INIT_SEEDS,
    PROTOCOL,
    READINESS_WRITES,
    SCHEDULE,
    SCORED_WRITES,
    continuation_alpha,
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
        oriented[str(writes)] = {
            "write_event_f1": metrics["write_event_f1"],
            "query_event_f1": metrics["query_event_f1"],
            "write_slot_accuracy": (
                metrics["write_slot_accuracy"]
                if gauge == "identity"
                else 1.0 - metrics["write_slot_accuracy"]
            ),
            "query_slot_accuracy": (
                metrics["query_slot_accuracy"]
                if gauge == "identity"
                else 1.0 - metrics["query_slot_accuracy"]
            ),
        }
    return gauge, oriented


def _expected_alpha_ranges(arm: str) -> dict[str, tuple[float, float]]:
    ranges = {}
    start = 1
    end = 0
    for _, stage_steps in SCHEDULE:
        end += stage_steps
        ranges[str(end)] = (
            1.0 if arm == "hard_event" else continuation_alpha(start, 800),
            1.0 if arm == "hard_event" else continuation_alpha(end, 800),
        )
        start = end + 1
    return ranges


def _factorial_ranges(rows, arm: str):
    selected = [row for row in rows if row["arm"] == arm]
    initialization = {
        str(data): max(
            row["accuracy"]["16"] for row in selected if row["data_seed"] == data
        )
        - min(row["accuracy"]["16"] for row in selected if row["data_seed"] == data)
        for data in DATA_SEEDS
    }
    data_order = {
        str(init): max(
            row["accuracy"]["16"] for row in selected if row["init_seed"] == init
        )
        - min(row["accuracy"]["16"] for row in selected if row["init_seed"] == init)
        for init in INIT_SEEDS
    }
    return initialization, data_order


def summarize(paths: list[Path]) -> dict[str, object]:
    if len(paths) != 9:
        raise ValueError("query continuation requires nine artifacts")
    rows = []
    seen = set()
    hashes = None
    execution_ids = {}
    source_digests = {}
    exposure = {}
    contract_pass = True
    for path in paths:
        payload = json.loads(path.read_text())
        if (
            payload["stage"] != "spin_delta_query_continuation"
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
            or execution["identical_batch_generators_between_arms"] is not True
            or execution["init_seed"] != config["init_seed"]
            or execution["data_seeds"] != list(DATA_SEEDS)
        ):
            raise ValueError(f"paired execution contract failed: {path}")
        previous_execution = execution_ids.setdefault(
            config["init_seed"], execution["execution_id"]
        )
        previous_digest = source_digests.setdefault(
            config["init_seed"], payload["source_state_sha256"]
        )
        if (
            execution["execution_id"] != previous_execution
            or payload["source_state_sha256"] != previous_digest
        ):
            raise ValueError("shared initial state differs within an init cohort")
        required_contract = {
            "only_query_event_forward_path_differs": True,
            "training_uses_router_labels": False,
            "router_auxiliary_loss_weight": 0.0,
            "oracle_controls_supplied_to_model": False,
            "audit_labels_detached_from_loss": True,
            "router_and_core_jointly_trainable": True,
            "evaluation_uses_hard_router": True,
        }
        contract_pass = contract_pass and payload["contract"] == required_contract
        if [arm["arm"] for arm in payload["arms"]] != list(ARMS):
            raise ValueError(f"wrong arm order: {path}")
        for arm in payload["arms"]:
            name = arm["arm"]
            expected_schedule = [
                {"writes": writes, "steps": steps} for writes, steps in SCHEDULE
            ]
            if arm["training_schedule"] != expected_schedule:
                raise ValueError(f"wrong schedule: {path}")
            if arm["initial_state_sha256"] != payload["source_state_sha256"]:
                raise ValueError(f"arm clone differs: {path}")
            if (
                arm["query_event_training_path"] != name
                or arm["evaluation_uses_hard_router"] is not True
                or arm["training_uses_router_labels"] is not False
                or arm["router_auxiliary_loss_weight"] != 0.0
                or arm["oracle_controls_supplied_to_model"] is not False
                or arm["audit_labels_detached_from_loss"] is not True
            ):
                contract_pass = False
            expected_ranges = _expected_alpha_ranges(name)
            for end, (alpha_start, alpha_end) in expected_ranges.items():
                observed = arm["event_forward_ranges"][end]
                if (
                    observed["alpha_start"] != alpha_start
                    or observed["alpha_end"] != alpha_end
                ):
                    raise ValueError(f"continuation schedule differs: {path}")
            values = [arm["final"][str(writes)]["accuracy"] for writes in SCORED_WRITES]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"nonfinite retrieval metric: {path}")
            gauge, oriented = _slot_gauge(arm["router_readiness"])
            if not all(
                math.isfinite(oriented[str(writes)][metric])
                for writes in READINESS_WRITES
                for metric in ROUTER_METRICS
            ):
                raise ValueError(f"nonfinite router metric: {path}")
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
    expected_cells = {(init, data) for init in INIT_SEEDS for data in DATA_SEEDS}
    if seen != expected_cells:
        raise ValueError("cells do not match the frozen grid")
    if exposure[ARMS[0]] != exposure[ARMS[1]]:
        raise ValueError("paired arms have unequal exposure")
    rows.sort(key=lambda row: (row["init_seed"], row["data_seed"], row["arm"]))

    summaries = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        init_ranges, data_ranges = _factorial_ranges(rows, arm)
        metric_minimum = {
            metric: min(
                row["oriented_router_readiness"][str(writes)][metric]
                for row in selected
                for writes in READINESS_WRITES
            )
            for metric in ROUTER_METRICS
        }
        summaries[arm] = {
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
            "router_metric_minimum_after_slot_gauge": metric_minimum,
        }

    by_cell = {(row["init_seed"], row["data_seed"], row["arm"]): row for row in rows}
    paired = []
    for init in INIT_SEEDS:
        for data in DATA_SEEDS:
            hard = by_cell[(init, data, "hard_event")]
            continuation = by_cell[(init, data, "linear_continuation")]
            paired.append(
                {
                    "init_seed": init,
                    "data_seed": data,
                    "improvement": {
                        str(writes): continuation["accuracy"][str(writes)]
                        - hard["accuracy"][str(writes)]
                        for writes in SCORED_WRITES
                    },
                }
            )
    mean_improvement = sum(row["improvement"]["16"] for row in paired) / 9
    worst_improvement = min(row["improvement"]["16"] for row in paired)
    hard = summaries["hard_event"]
    continuation = summaries["linear_continuation"]
    query_event_gain = (
        continuation["router_metric_minimum_after_slot_gauge"]["query_event_f1"]
        - hard["router_metric_minimum_after_slot_gauge"]["query_event_f1"]
    )
    query_slot_gain = (
        continuation["router_metric_minimum_after_slot_gauge"]["query_slot_accuracy"]
        - hard["router_metric_minimum_after_slot_gauge"]["query_slot_accuracy"]
    )
    component_decisions = {
        "mean_retrieval_gain_pass": mean_improvement >= 0.10,
        "paired_regression_limit_pass": worst_improvement >= -0.10,
        "worst_cell_rescue_pass": continuation["minimum_accuracy"]["16"]
        >= hard["minimum_accuracy"]["16"] + 0.05,
        "query_event_repair_pass": continuation[
            "router_metric_minimum_after_slot_gauge"
        ]["query_event_f1"]
        >= 0.50
        and query_event_gain >= 0.25,
        "query_slot_repair_pass": continuation[
            "router_metric_minimum_after_slot_gauge"
        ]["query_slot_accuracy"]
        >= 0.60
        and query_slot_gain >= 0.10,
    }
    mechanism_repair = contract_pass and all(component_decisions.values())
    robustness = all(
        row["accuracy"]["8"] >= 0.95
        and row["accuracy"]["16"] >= 0.95
        and row["accuracy"]["32"] >= 0.93
        for row in rows
        if row["arm"] == "linear_continuation"
    )
    variance = (
        max(continuation["initialization_ranges_at_16"].values()) < 0.05
        and max(continuation["data_order_ranges_at_16"].values()) < 0.05
    )
    identification = all(
        value >= 0.99
        for value in continuation["router_metric_minimum_after_slot_gauge"].values()
    )
    return {
        "schema_version": 1,
        "rows": rows,
        "arms": summaries,
        "paired_improvements": paired,
        "label_free_contract_pass": contract_pass,
        "mean_paired_improvement_at_16": mean_improvement,
        "worst_paired_improvement_at_16": worst_improvement,
        "query_event_minimum_gain": query_event_gain,
        "query_slot_minimum_gain": query_slot_gain,
        "component_decisions": component_decisions,
        "mechanism_repair_pass": mechanism_repair,
        "autonomy_robustness_pass": robustness,
        "autonomy_variance_pass": variance,
        "autonomy_identification_pass": identification,
        "learning_autonomy_promotion_pass": contract_pass
        and robustness
        and variance
        and identification,
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
