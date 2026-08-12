"""Aggregate recurrence-ladder reports without mixing incompatible protocols."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


FAMILIES = ("complex_unitary", "ga_rotor_selective")
CONFIG_KEYS = (
    "steps",
    "batch_size",
    "sequence_length",
    "validation_batches",
    "validation_batch_size",
    "channels",
    "layers",
    "expansion",
    "learning_rate",
)


def load_reports(paths: Iterable[Path]) -> list[dict[str, Any]]:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not reports:
        raise ValueError("at least one report is required")
    reference = {key: reports[0]["config"][key] for key in CONFIG_KEYS}
    seen = set()
    for report in reports:
        # The original 1000-step Q8 report predates generic finite-group
        # metadata. Accept only that exact legacy schema.
        if "group" not in report:
            elements = report.get("q8_element_order")
            if elements != ["1", "i", "j", "k", "-1", "-i", "-j", "-k"]:
                raise ValueError("unrecognized report without group metadata")
            report["group"] = {
                "key": "q8",
                "name": "quaternion group Q8",
                "order": 8,
                "element_order": elements,
            }
        current = {key: report["config"][key] for key in CONFIG_KEYS}
        if current != reference:
            raise ValueError(
                f"protocol mismatch for {report['group']['key']} seed "
                f"{report['config']['seed']}: {current} != {reference}"
            )
        contract = report["fairness_contract"]
        required = (
            "identical_parameter_count",
            "identical_parameter_shapes",
            "identical_initial_parameters",
            "identical_initial_function",
            "same_training_batches",
            "same_validation_batches",
        )
        if not all(contract[key] for key in required):
            raise ValueError("a report violates the fairness contract")
        key = (report["group"]["key"], report["config"]["seed"])
        if key in seen:
            raise ValueError(f"duplicate report for group/seed {key}")
        seen.add(key)
        present = {result["family"] for result in report["results"]}
        missing = set(FAMILIES) - present
        if missing:
            raise ValueError(f"report {key} is missing {sorted(missing)}")
    return reports


def mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def summarize(reports: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        by_group[report["group"]["key"]].append(report)

    groups: dict[str, Any] = {}
    for group_key, group_reports in sorted(by_group.items()):
        group_reports.sort(key=lambda report: report["config"]["seed"])
        family_metrics: dict[str, Any] = {}
        per_seed: list[dict[str, Any]] = []
        values: dict[str, dict[str, list[float]]] = {
            family: defaultdict(list) for family in FAMILIES
        }
        margins = defaultdict(list)
        for report in group_reports:
            results = {result["family"]: result for result in report["results"]}
            seed_row = {"seed": report["config"]["seed"], "families": {}}
            for family in FAMILIES:
                result = results[family]
                metrics = {
                    "validation_loss": result["final_validation_loss"],
                    "final_accuracy_l16": result["length_generalization"]["16"][
                        "final_position_accuracy"
                    ],
                    "final_accuracy_l32": result["length_generalization"]["32"][
                        "final_position_accuracy"
                    ],
                    "steps_per_second": result["steps_per_second"],
                }
                seed_row["families"][family] = metrics
                for name, value in metrics.items():
                    values[family][name].append(value)
            for metric in ("final_accuracy_l16", "final_accuracy_l32"):
                margins[metric].append(
                    seed_row["families"]["ga_rotor_selective"][metric]
                    - seed_row["families"]["complex_unitary"][metric]
                )
            per_seed.append(seed_row)
        for family in FAMILIES:
            family_metrics[family] = {
                metric: mean_sd(metric_values)
                for metric, metric_values in values[family].items()
            }
        groups[group_key] = {
            "name": group_reports[0]["group"]["name"],
            "chance_accuracy": 1.0 / group_reports[0]["group"]["order"],
            "seeds": [report["config"]["seed"] for report in group_reports],
            "families": family_metrics,
            "ga_minus_complex": {
                metric: mean_sd(metric_values)
                for metric, metric_values in margins.items()
            },
            "per_seed": per_seed,
        }
    return {
        "protocol": {key: reports[0]["config"][key] for key in CONFIG_KEYS},
        "groups": groups,
    }


def percent_summary(metric: dict[str, float]) -> str:
    return f"{100 * metric['mean']:.2f} +/- {100 * metric['sample_sd']:.2f}"


def render_markdown(summary: dict[str, Any], sources: list[Path]) -> str:
    lines = [
        "# Recurrence ladder: 1000-step multi-group replication",
        "",
        "All rows use the same optimizer, batches per seed, eight-real state width,",
        "parameter shapes, initial parameters, and initial function. Values are mean",
        "+/- sample standard deviation across seeds 0, 1, and 2.",
        "",
        "| Group | Family | Loss | L16 final accuracy | L32 final accuracy | steps/s |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for group_key, group in summary["groups"].items():
        for family in FAMILIES:
            metrics = group["families"][family]
            lines.append(
                f"| {group_key.upper()} | `{family}` | "
                f"{metrics['validation_loss']['mean']:.3f} +/- "
                f"{metrics['validation_loss']['sample_sd']:.3f} | "
                f"{percent_summary(metrics['final_accuracy_l16'])}% | "
                f"{percent_summary(metrics['final_accuracy_l32'])}% | "
                f"{metrics['steps_per_second']['mean']:.2f} |"
            )
    lines.extend(["", "## GA minus complex margin", ""])
    for group_key, group in summary["groups"].items():
        l16 = group["ga_minus_complex"]["final_accuracy_l16"]
        l32 = group["ga_minus_complex"]["final_accuracy_l32"]
        lines.append(
            f"- {group_key.upper()}: L16 {100*l16['mean']:+.2f} points; "
            f"L32 {100*l32['mean']:+.2f} points."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The selective GA rotor has the higher mean L32 final accuracy in all",
            "three groups, but variance is large and individual D4/S3 seeds reverse",
            "the ordering. This is promising evidence for long-horizon behavior, not",
            "yet a robust universal win. Q8 is the most consistent result: GA wins L32",
            "on every seed. More seeds and harder tasks remain necessary.",
            "",
            "The grade-decay family is intentionally excluded from this two-family",
            "replication summary because only seed 0 has been run.",
            "",
            "## Source reports",
            "",
        ]
    )
    lines.extend(f"- `{path.as_posix()}`" for path in sources)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    reports = load_reports(args.reports)
    summary = summarize(reports)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.output_markdown.write_text(
        render_markdown(summary, args.reports), encoding="utf-8"
    )
    print(args.output_markdown.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
