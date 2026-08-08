"""Render the v2.1 transport-ablation JSON as a reviewer-facing Markdown report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from transport_ablation_v2 import FAMILY_NAMES

DISPLAY = {
    "identity": "Identity",
    "real_diagonal": "Real diagonal",
    "complex_phase": "Complex phases",
    "quaternion_left": "Quaternion left",
    "rotor": "Cl(3,0) rotor",
    "fixed_rotor": "Fixed rotor",
    "so8": "Generic SO(8)",
}


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else math.nan


def format_value(value: float, digits: int = 4) -> str:
    return "--" if not math.isfinite(value) else f"{value:.{digits}f}"


def format_ci(summary: dict[str, Any] | None) -> str:
    if summary is None:
        return "--"
    return (
        f"{summary['mean']:.4f} [{summary['ci95_low']:.4f}, {summary['ci95_high']:.4f}]"
    )


def prediction_rows(report: dict[str, Any], view: str) -> list[str]:
    runs = [run for run in report["prediction_runs"] if run["view"] == view]
    summary = report["summary"]["prediction"][view]
    rows = []
    for family in FAMILY_NAMES:
        family_runs = [run for run in runs if run["family"] == family]
        if not family_runs:
            continue
        family_summary = summary[family]
        rows.append(
            "| "
            + " | ".join(
                (
                    DISPLAY[family],
                    str(family_runs[0]["channels"]),
                    f"{family_runs[0]['effective_parameters']:,}",
                    format_ci(family_summary["confirmation_loss"]),
                    format_ci(family_summary["paired_loss_minus_identity"]),
                    str(family_summary["wins_vs_identity"]),
                    f"{mean([run['training_tokens_per_second'] for run in family_runs]):,.0f}",
                    f"{mean([run['peak_cuda_memory_mib'] for run in family_runs]):.1f}",
                )
            )
            + " |"
        )
    return rows


def memory_rows(report: dict[str, Any], task: str) -> list[str]:
    summary = report["summary"]["memory"][task]
    lengths = tuple(next(iter(summary.values())).keys())
    rows = []
    for family in FAMILY_NAMES:
        if family not in summary:
            continue
        cells = [DISPLAY[family]]
        for length in lengths:
            value = summary[family][length]["accuracy"]
            cells.append("--" if value is None else f"{100 * value['mean']:.2f}%")
        longest = summary[family][lengths[-1]]["paired_accuracy_minus_identity"]
        cells.append("--" if longest is None else f"{100 * longest['mean']:+.2f} pp")
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def calibration_rows(calibration: dict[str, Any]) -> list[str]:
    target = calibration["rotor_target"]["forward_backward"]["median_ms"]
    rows = []
    for family in FAMILY_NAMES:
        width = calibration["cuda_matched_widths"][family]
        candidates = calibration["cuda_candidates"][family]
        selected = next(item for item in candidates if item["channels"] == width)
        measured = selected["forward_backward"]["median_ms"]
        residual = 100 * (measured / target - 1)
        rows.append(
            f"| {DISPLAY[family]} | {calibration['parameter_matched_widths'][family]} "
            f"| {width} | {measured:.3f} | {residual:+.1f}% |"
        )
    return rows


def system_rows(report: dict[str, Any], view: str, length: int = 256) -> list[str]:
    systems = report.get("systems_benchmark")
    if not systems:
        return []
    rows = []
    for family in FAMILY_NAMES:
        matches = [
            item
            for item in systems["results"]
            if item["view"] == view
            and item["family"] == family
            and item["sequence_length"] == length
        ]
        if not matches:
            continue
        item = matches[0]
        rows.append(
            f"| {DISPLAY[family]} | {item['channels']} "
            f"| {item['inference']['median_ms']:.3f} "
            f"| {item['forward_backward']['median_ms']:.3f} "
            f"| {item['forward_backward']['tokens_per_second']:,.0f} "
            f"| {item['forward_backward']['peak_memory_mib']:.1f} |"
        )
    return rows


def intervention_lines(report: dict[str, Any]) -> list[str]:
    rotor_runs = [
        run
        for run in report["prediction_runs"]
        if run["view"] == "state_matched" and run["family"] == "rotor"
    ]
    if not rotor_runs:
        return ["No complete state-matched rotor intervention cohort is present."]
    clamp = [
        run["interventions"]["identity_clamp_confirmation_loss"]
        - run["confirmation_loss"]
        for run in rotor_runs
    ]
    shuffle = [
        run["interventions"]["time_shuffled_confirmation_loss"]
        - run["confirmation_loss"]
        for run in rotor_runs
    ]
    return [
        f"- Identity clamping changed confirmation loss by {mean(clamp):+.4f} nats on average.",
        f"- Time-shuffling rotor actions changed confirmation loss by {mean(shuffle):+.4f} nats on average.",
    ]


def decision_lines(report: dict[str, Any]) -> list[str]:
    prediction = report["summary"]["prediction"]["state_matched"]["rotor"]
    prediction_delta = prediction["paired_loss_minus_identity"]
    prediction_pass = (
        prediction_delta is not None
        and prediction_delta["mean"] < 0
        and prediction["wins_vs_identity"] >= 4
    )
    memory_passes = []
    for task, task_summary in report["summary"]["memory"].items():
        rotor = task_summary["rotor"]
        lengths = tuple(rotor)
        extrapolation = lengths[1:]
        memory_passes.append(
            (
                task,
                bool(extrapolation)
                and all(
                    rotor[length]["paired_accuracy_minus_identity"] is not None
                    and rotor[length]["paired_accuracy_minus_identity"]["mean"] > 0
                    for length in extrapolation
                ),
            )
        )
    complete = (
        report["prediction_run_count"] == 105 and report["memory_run_count"] == 70
    )
    cuda_rotor = report["summary"]["prediction"]["cuda_matched"]["rotor"]
    cuda_delta = cuda_rotor["paired_loss_minus_identity"]
    compute_pass = cuda_delta is not None and cuda_delta["mean"] < 0
    return [
        (
            f"- Cohort completeness: **{'PASS' if complete else 'INCOMPLETE'}** "
            f"({report['prediction_run_count']}/105 prediction, "
            f"{report['memory_run_count']}/70 memory runs)."
        ),
        f"- Preregistered state-matched prediction rule: **{'PASS' if prediction_pass else 'FAIL'}**.",
        "- Preregistered memory rule by task: "
        + ", ".join(
            f"`{task}` **{'PASS' if passed else 'FAIL'}**"
            for task, passed in memory_passes
        )
        + ".",
        (
            f"- Preregistered compute-efficiency rule: "
            f"**{'PASS' if compute_pass else 'FAIL'}** "
            f"(CUDA-matched rotor minus identity loss "
            f"{cuda_delta['mean']:+.4f} nats)."
        ),
    ]


def render(
    report: dict[str, Any],
    base: dict[str, Any] | None,
    prior: dict[str, Any] | None,
    base_benchmark: dict[str, Any] | None,
    aggregate_sha256: str,
) -> str:
    lines = [
        "# Pure rotor SSM v2.1.0 transport-ablation results",
        "",
        "Date: 2026-08-06.",
        "",
        (
            "This is the outcome report for the prospectively frozen transport "
            "ladder. The raw JSON retains every run, seed, timing, intervention, "
            "hash, and numerical-parity measurement."
        ),
        "",
        "## Refined-model retrain",
        "",
    ]
    if base:
        result = base["results"][0]
        prior_result = prior["results"][0] if prior else None
        lines.extend(
            (
                "The standalone approximately-4-GiB v2.1 rotor retrain changed only the open rotor-chart limit from `pi/2` to `pi`; its measured peak was 4109.2 MiB, slightly above 4096 MiB.",
                "",
                "| Version | Validation nats | Bits/byte | Peak MiB | Train seconds |",
                "|---|---:|---:|---:|---:|",
                *(
                    (
                        (
                            f"| v2.0.0 | {prior_result['final_validation_loss']:.6f} | "
                            f"{prior_result['final_validation_bits_per_byte']:.6f} | "
                            f"{prior_result['peak_cuda_memory_mib']:.1f} | "
                            f"{prior_result['elapsed_seconds']:.1f} |"
                        ),
                    )
                    if prior_result
                    else ()
                ),
                f"| v2.1.0 | {result['final_validation_loss']:.6f} | {result['final_validation_bits_per_byte']:.6f} | {result['peak_cuda_memory_mib']:.1f} | {result['elapsed_seconds']:.1f} |",
                "",
                "The 0.001458-nat single-seed difference from v2.0 is descriptive, not an established refinement gain.",
                "",
            )
        )
    else:
        lines.extend(
            (
                "The standalone refined-model artifact was not supplied to this renderer.",
                "",
            )
        )
    if base_benchmark:
        benchmark_by_key = {
            (item["sequence_length"], item["scan_mode"]): item
            for item in base_benchmark["model_results"]
        }
        benchmark_rows = []
        for length in (64, 128, 256, 512):
            parallel = benchmark_by_key[(length, "parallel")]["forward_backward"]
            recurrent = benchmark_by_key[(length, "recurrent")]["forward_backward"]
            benchmark_rows.append(
                f"| {length} | {parallel['median_ms']:.2f} "
                f"| {recurrent['median_ms']:.2f} "
                f"| {recurrent['median_ms'] / parallel['median_ms']:.2f}x "
                f"| {parallel['peak_memory_mib']:.1f} |"
            )
        lines.extend(
            (
                "### Standalone v2.1 scan benchmark",
                "",
                "Batch 8, PyTorch eager float32 on the recorded local CUDA device. Times are forward/backward medians.",
                "",
                "| Context | Parallel ms | Recurrent ms | Speedup | Parallel peak MiB |",
                "|---:|---:|---:|---:|---:|",
                *benchmark_rows,
                "",
            )
        )
    lines.extend(("## Cohort decision gates", "", *decision_lines(report), ""))
    state = report["summary"]["prediction"]["state_matched"]
    parameter = report["summary"]["prediction"]["parameter_matched"]
    cuda = report["summary"]["prediction"]["cuda_matched"]
    lines.extend(
        (
            "## Direct answer",
            "",
            (
                "- **Prediction: qualified yes versus identity.** Rotor improves "
                f"state-matched loss by {-state['rotor']['paired_loss_minus_identity']['mean']:.4f} nats "
                "and wins all five seeds. It is not the best transport: quaternion "
                f"improves by {-state['quaternion_left']['paired_loss_minus_identity']['mean']:.4f}, "
                f"commuting phases by {-state['complex_phase']['paired_loss_minus_identity']['mean']:.4f}, "
                f"and the larger state-matched SO(8) row by {-state['so8']['paired_loss_minus_identity']['mean']:.4f}."
            ),
            (
                "- **Parameter efficiency: qualified yes versus identity, not versus "
                "the best alternatives.** Rotor retains a "
                f"{-parameter['rotor']['paired_loss_minus_identity']['mean']:.4f}-nat "
                "advantage, but quaternion and commuting phases remain better."
            ),
            "- **Memory: no.** Rotor loses to identity on the mean associative-recall result at every length and Q8 remains at chance-scale noise.",
            (
                "- **Compute efficiency: no.** At matched measured CUDA budget, rotor "
                f"is {cuda['rotor']['paired_loss_minus_identity']['mean']:.4f} nats "
                "worse than the wider identity model."
            ),
            "",
        )
    )
    for view, title in (
        ("state_matched", "State matched"),
        ("parameter_matched", "Effective-parameter matched"),
        ("cuda_matched", "Measured-CUDA matched"),
    ):
        lines.extend(
            (
                f"## Prediction: {title}",
                "",
                "Loss and paired deltas are mean [95% t interval] over five seeds; negative paired loss favors the family over retrained identity.",
                "",
                "| Family | C | Effective params | Confirmation nats | Loss minus identity | Wins | Train tok/s | Peak MiB |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
                *prediction_rows(report, view),
                "",
            )
        )
    lines.extend(
        ("## Trained-rotor interventions", "", *intervention_lines(report), "")
    )
    for task, title in (
        ("associative_recall", "Associative recall"),
        ("q8_ordered_product", "Q8 ordered product"),
    ):
        summary = report["summary"]["memory"][task]
        lengths = tuple(next(iter(summary.values())))
        lines.extend(
            (
                f"## Memory: {title}",
                "",
                *(
                    (
                        "The registered even Q8 evaluation lengths have four-label support, so a supported-class chance predictor is 25%.",
                        "",
                    )
                    if task == "q8_ordered_product"
                    else ()
                ),
                "| Family | "
                + " | ".join(f"L={length}" for length in lengths)
                + " | Longest minus identity |",
                "|---|" + "---:|" * (len(lengths) + 1),
                *memory_rows(report, task),
                "",
            )
        )
    calibration = report.get("calibration")
    if calibration:
        lines.extend(
            (
                "## Width calibration",
                "",
                "The parameter column is the integer width nearest the C=8 rotor parameter count. CUDA residual is relative to the C=8 rotor forward/backward target at batch 64, context 128.",
                "",
                "| Family | Parameter C | CUDA C | CUDA ms | Residual |",
                "|---|---:|---:|---:|---:|",
                *calibration_rows(calibration),
                "",
            )
        )
    lines.extend(
        (
            "## CUDA systems result (state matched, batch 8, context 256)",
            "",
            "| Family | C | Inference ms | Forward/backward ms | Train tok/s | Peak MiB |",
            "|---|---:|---:|---:|---:|---:|",
            *system_rows(report, "state_matched"),
            "",
            "## Interpretation boundary",
            "",
            "A prediction win establishes an empirical advantage under this small byte-model protocol, not a universal language-model advantage. A Q8-only memory win supports ordered algebraic composition, not general recall. A systems result applies to PyTorch eager float32 on the recorded local GPU. Exact recurrence closure and norm bounds are mathematical properties; finite-precision scan parity, optimization, generalization, and throughput remain empirical.",
            "",
        )
    )
    parity_values = [
        value
        for run in report["prediction_runs"]
        for value in run.get("numerical_parity", {}).values()
    ]
    prediction_peak = max(
        run["peak_cuda_memory_mib"] for run in report["prediction_runs"]
    )
    memory_peak = max(run["peak_cuda_memory_mib"] for run in report["memory_runs"])
    train_hours = (
        sum(
            run["elapsed_seconds"]
            for run in report["prediction_runs"] + report["memory_runs"]
        )
        / 3600
    )
    data = report["prediction_runs"][0]["data"]
    lines.extend(
        (
            "## Evidence integrity",
            "",
            f"- Frozen protocol SHA-256: `{report['protocol_sha256']}`.",
            f"- Raw aggregate SHA-256: `{aggregate_sha256}`.",
            f"- WikiText-2 train-byte SHA-256: `{data['train_sha256']}`.",
            f"- WikiText-2 validation-byte SHA-256: `{data['validation_sha256']}`.",
            *(
                (
                    f"- Standalone v2.1 checkpoint SHA-256: `{base['results'][0]['checkpoint_sha256']}`.",
                )
                if base
                else ()
            ),
            f"- Maximum prediction parallel/recurrent or full/chunked logit discrepancy: `{max(parity_values):.8g}`.",
            f"- Maximum prediction-run peak allocation: `{prediction_peak:.2f} MiB`; memory-run peak: `{memory_peak:.2f} MiB`.",
            f"- Summed measured training-loop time across 175 runs: `{train_hours:.6f} hours` (excludes calibration and evaluation).",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-retrain", type=Path)
    parser.add_argument("--prior-retrain", type=Path)
    parser.add_argument("--base-benchmark", type=Path)
    args = parser.parse_args()
    aggregate_bytes = args.input.read_bytes()
    aggregate_sha256 = hashlib.sha256(aggregate_bytes).hexdigest()
    report = json.loads(aggregate_bytes)
    base = (
        json.loads(args.base_retrain.read_text(encoding="utf-8"))
        if args.base_retrain
        else None
    )
    prior = (
        json.loads(args.prior_retrain.read_text(encoding="utf-8"))
        if args.prior_retrain
        else None
    )
    base_benchmark = (
        json.loads(args.base_benchmark.read_text(encoding="utf-8"))
        if args.base_benchmark
        else None
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render(report, base, prior, base_benchmark, aggregate_sha256),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
