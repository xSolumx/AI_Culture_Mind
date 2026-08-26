"""G15B-R2 retained-checkpoint collision-only erase diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F

from .g15b_interleaved_cohort import (
    EVALUATION_LENGTHS,
    NEEDLE_DISTANCES,
    _evaluation_batch_size,
    _gather_time,
    _sha256,
    _stable_seed,
)
from .g15b_interleaved_tasks import InterleavedBatch, generate_interleaved_batch
from .g15br1_event_erase import (
    EXPECTED_G15B_SHA256,
    EXPECTED_R0_SHA256,
    PRESERVED_CONTROL_NAMES,
    R0_ARTIFACT,
    _validate_parent,
    _validate_quality_artifact,
    _validate_r0,
    event_erase_forward,
)
from .g15br_checkpoint_repair import (
    PARENT_ARTIFACT,
    QUALITY_SEEDS,
    ROOT,
    _expected_identity,
    _git_provenance,
    _load_checkpoint,
    _sync,
    local_write_event_mask,
    temporal_observability_witness,
)
from .model import HybridMemoryLM

PROTOCOL = ROOT / "G15BR2_COLLISION_ERASE_PROTOCOL_2026-08-26.md"
R1_ARTIFACT = ROOT / "artifacts/g15br1_event_erase_sm75_2026-08-26.json"
EXPECTED_R1_SHA256 = "c015b128846e4b5c63d927778815a87728a7d613369163b1027ed3dd9f0b2912"
INTERVENTIONS = ("learned", "soft_collision_erase", "exact_collision_erase")
Intervention = Literal["learned", "soft_collision_erase", "exact_collision_erase"]
STRATA = (
    "before_any_overwrite",
    "after_unrelated_overwrite_only",
    "after_same_key_overwrite",
)


def collision_erase_forward(
    model: HybridMemoryLM,
    batch: InterleavedBatch,
    intervention: Intervention,
) -> dict[str, Any]:
    """Preserve learned writes and move only erase to true collisions."""

    if intervention not in INTERVENTIONS:
        raise ValueError(f"unknown G15B-R2 intervention {intervention!r}")
    renamed = replace(
        batch,
        write_event_mask=batch.erase_event_mask,
        _skip_validation=True,
    )
    mapped = {
        "learned": "learned",
        "soft_collision_erase": "soft_event_erase",
        "exact_collision_erase": "exact_event_erase",
    }[intervention]
    return event_erase_forward(model, renamed, mapped)  # type: ignore[arg-type]


def overwrite_query_strata(batch: InterleavedBatch) -> dict[str, torch.Tensor]:
    """Partition queries by same-key and unrelated prior overwrite history."""

    write_before = batch.write_positions[:, None, :] < batch.query_positions[:, :, None]
    overwrite_before = write_before & batch.overwrite_mask[:, None, :]
    same_key = batch.write_keys[:, None, :] == batch.query_keys[:, :, None]
    same_before = (overwrite_before & same_key).any(dim=-1)
    unrelated_before = (overwrite_before & ~same_key).any(dim=-1)
    strata = {
        "before_any_overwrite": ~same_before & ~unrelated_before,
        "after_unrelated_overwrite_only": ~same_before & unrelated_before,
        "after_same_key_overwrite": same_before,
    }
    assigned = sum(mask.to(torch.int8) for mask in strata.values())
    if not bool(assigned.eq(1).all()):
        raise RuntimeError("overwrite query strata do not form a partition")
    return strata


def _new_integrity() -> dict[str, Any]:
    return {
        "local_decoder_batches_checked": 0,
        "collision_mask_batches_checked": 0,
        "model_forward_maximum_absolute_logit_residual": 0.0,
        "preserved_controls": {
            name: {"bitwise_equal": True, "maximum_absolute_residual": 0.0}
            for name in PRESERVED_CONTROL_NAMES
        },
    }


@torch.no_grad()
def evaluate_checkpoint(
    model: HybridMemoryLM,
    *,
    seed: int,
    decisions: int,
    batch_cap: int,
) -> dict[str, Any]:
    model.eval()
    device = model.embedding.weight.device
    cells: dict[str, Any] = {}
    integrity = _new_integrity()
    for task in ("mqar", "overwrite", "selective", "needle"):
        for length in EVALUATION_LENGTHS:
            batch_size = _evaluation_batch_size(
                task, decisions=decisions, cap=batch_cap
            )
            per_batch = batch_size * (1 if task == "needle" else 8)
            if decisions % per_batch:
                raise ValueError("decisions must contain complete evaluation batches")
            correct = {name: 0 for name in INTERVENTIONS}
            episodes = {name: 0 for name in INTERVENTIONS}
            nll_sum = {name: 0.0 for name in INTERVENTIONS}
            stratum_correct = {
                name: {stratum: 0 for stratum in STRATA} for name in INTERVENTIONS
            }
            stratum_total = {stratum: 0 for stratum in STRATA}
            total = 0
            episode_total = 0
            batch_index = 0
            while total < decisions:
                batch = generate_interleaved_batch(
                    task,
                    batch_size,
                    length,
                    8,
                    24,
                    8,
                    seed=_stable_seed("g15b-eval", seed, task, length, batch_index),
                    needle_distance=(
                        NEEDLE_DISTANCES[length] if task == "needle" else None
                    ),
                ).to(device)
                if not torch.equal(
                    local_write_event_mask(batch.token_ids), batch.write_event_mask
                ):
                    raise RuntimeError("valid-write target is not locally observable")
                integrity["local_decoder_batches_checked"] += 1
                expected_collisions = torch.zeros_like(batch.erase_event_mask)
                expected_collisions.scatter_(
                    1, batch.write_positions, batch.overwrite_mask
                )
                if not torch.equal(expected_collisions, batch.erase_event_mask):
                    raise RuntimeError(
                        "collision mask does not match overwrite history"
                    )
                integrity["collision_mask_batches_checked"] += 1

                results = {
                    intervention: collision_erase_forward(model, batch, intervention)
                    for intervention in INTERVENTIONS
                }
                learned_result = results["learned"]
                learned_controls = learned_result["controls"]
                ordinary_logits = model(batch.token_ids)["logits"]
                integrity["model_forward_maximum_absolute_logit_residual"] = max(
                    float(integrity["model_forward_maximum_absolute_logit_residual"]),
                    float((ordinary_logits - learned_result["logits"]).abs().max()),
                )
                strata = overwrite_query_strata(batch) if task == "overwrite" else {}
                for stratum, mask in strata.items():
                    stratum_total[stratum] += int(mask.sum())
                for intervention, result in results.items():
                    selected_logits = _gather_time(
                        result["logits"], batch.query_positions
                    )
                    prediction = selected_logits.argmax(-1)
                    match = prediction == batch.targets
                    correct[intervention] += int(match.sum())
                    episodes[intervention] += int(match.all(dim=1).sum())
                    nll_sum[intervention] += float(
                        F.cross_entropy(
                            selected_logits.flatten(0, 1),
                            batch.targets.flatten(),
                            reduction="sum",
                        )
                    )
                    for stratum, mask in strata.items():
                        stratum_correct[intervention][stratum] += int(
                            (match & mask).sum()
                        )
                    if intervention == "learned":
                        continue
                    for name in PRESERVED_CONTROL_NAMES:
                        learned_control = learned_controls[name]
                        repaired_control = result["controls"][name]
                        report = integrity["preserved_controls"][name]
                        report["bitwise_equal"] = bool(
                            report["bitwise_equal"]
                        ) and torch.equal(learned_control, repaired_control)
                        report["maximum_absolute_residual"] = max(
                            float(report["maximum_absolute_residual"]),
                            float((learned_control - repaired_control).abs().max()),
                        )
                total += batch.targets.numel()
                episode_total += batch.batch_size
                batch_index += 1
            cell = {
                "task": task,
                "length": length,
                "query_decisions": total,
                "interventions": {
                    name: {
                        "query_accuracy": correct[name] / total,
                        "exact_episode_accuracy": episodes[name] / episode_total,
                        "bits_per_query": nll_sum[name] / total / math.log(2.0),
                    }
                    for name in INTERVENTIONS
                },
            }
            if task == "overwrite":
                cell["query_strata"] = {
                    stratum: {
                        "query_decisions": stratum_total[stratum],
                        "accuracy": {
                            name: (
                                stratum_correct[name][stratum] / stratum_total[stratum]
                                if stratum_total[stratum]
                                else None
                            )
                            for name in INTERVENTIONS
                        },
                    }
                    for stratum in STRATA
                }
            cells[f"{task}:L{length}"] = cell
    integrity["preserved_controls_bitwise_equal"] = all(
        row["bitwise_equal"] for row in integrity["preserved_controls"].values()
    )
    return {"cells": cells, "runtime_integrity": integrity}


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot calculate an empty mean")
    return sum(values) / len(values)


def _adjudicate(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    cell_names = list(seed_reports[0]["evaluation"]["cells"])
    replay_residuals = [
        residual
        for report in seed_reports
        for cell in report["evaluation"]["cells"].values()
        for residual in (
            cell["baseline_query_accuracy_absolute_residual"],
            cell["baseline_exact_episode_accuracy_absolute_residual"],
            cell["baseline_bits_per_query_absolute_residual"],
        )
    ]
    means: dict[str, Any] = {}
    for cell_name in cell_names:
        rows = [report["evaluation"]["cells"][cell_name] for report in seed_reports]
        accuracy = {
            intervention: _mean(
                [row["interventions"][intervention]["query_accuracy"] for row in rows]
            )
            for intervention in INTERVENTIONS
        }
        cell_report: dict[str, Any] = {
            "mean_query_accuracy": accuracy,
            **{
                f"{mode}_minus_learned": accuracy[mode] - accuracy["learned"]
                for mode in INTERVENTIONS[1:]
            },
        }
        if cell_name.startswith("overwrite:"):
            cell_report["query_strata"] = {}
            for stratum in STRATA:
                stratum_accuracy = {}
                for intervention in INTERVENTIONS:
                    values = [
                        row["query_strata"][stratum]["accuracy"][intervention]
                        for row in rows
                        if row["query_strata"][stratum]["accuracy"][intervention]
                        is not None
                    ]
                    stratum_accuracy[intervention] = _mean(values) if values else None
                cell_report["query_strata"][stratum] = {
                    "mean_accuracy": stratum_accuracy,
                    **{
                        f"{mode}_minus_learned": (
                            stratum_accuracy[mode] - stratum_accuracy["learned"]
                            if stratum_accuracy[mode] is not None
                            and stratum_accuracy["learned"] is not None
                            else None
                        )
                        for mode in INTERVENTIONS[1:]
                    },
                }
        means[cell_name] = cell_report

    replay_passed = max(replay_residuals, default=math.inf) <= 1e-12
    witness_passed = all(row["observability_witness"]["passed"] for row in seed_reports)
    runtime_integrity_passed = all(
        report["evaluation"]["runtime_integrity"][
            "model_forward_maximum_absolute_logit_residual"
        ]
        == 0.0
        and report["evaluation"]["runtime_integrity"][
            "preserved_controls_bitwise_equal"
        ]
        and report["evaluation"]["runtime_integrity"]["local_decoder_batches_checked"]
        > 0
        and report["evaluation"]["runtime_integrity"]["collision_mask_batches_checked"]
        > 0
        for report in seed_reports
    )
    mode_checks: dict[str, dict[str, bool]] = {}
    stratum_checks: dict[str, dict[str, bool]] = {}
    for mode in INTERVENTIONS[1:]:
        delta_name = f"{mode}_minus_learned"
        checks = {}
        guard_checks = {}
        for name, row in means.items():
            task = name.split(":", 1)[0]
            if task == "overwrite":
                checks[name] = row[delta_name] >= 0.10
                for stratum, stratum_row in row["query_strata"].items():
                    key = f"{name}:{stratum}"
                    threshold = 0.10 if stratum == "after_same_key_overwrite" else -0.02
                    delta = stratum_row[delta_name]
                    guard_checks[key] = (
                        delta >= threshold
                        if delta is not None
                        else stratum != "after_same_key_overwrite"
                    )
            elif task in ("mqar", "selective"):
                checks[name] = row[delta_name] >= -0.02
            else:
                checks[name] = row["mean_query_accuracy"][mode] >= 0.999
        mode_checks[mode] = checks
        stratum_checks[mode] = guard_checks
    passed_modes = [
        mode
        for mode in INTERVENTIONS[1:]
        if replay_passed
        and witness_passed
        and runtime_integrity_passed
        and all(mode_checks[mode].values())
        and all(stratum_checks[mode].values())
    ]
    selected = None
    if passed_modes:
        degradations = {}
        for mode in passed_modes:
            delta_name = f"{mode}_minus_learned"
            deltas = [
                row[delta_name]
                for name, row in means.items()
                if name.startswith(("mqar:", "selective:"))
            ]
            deltas.extend(
                stratum_row[delta_name]
                for name, row in means.items()
                if name.startswith("overwrite:")
                for stratum, stratum_row in row["query_strata"].items()
                if stratum != "after_same_key_overwrite"
                and stratum_row[delta_name] is not None
            )
            degradations[mode] = _mean([max(0.0, -delta) for delta in deltas])
        selected = min(
            passed_modes,
            key=lambda mode: (
                degradations[mode],
                mode != "soft_collision_erase",
            ),
        )

    post_same_improved = {
        mode: all(
            (
                row["query_strata"]["after_same_key_overwrite"][f"{mode}_minus_learned"]
                is not None
                and row["query_strata"]["after_same_key_overwrite"][
                    f"{mode}_minus_learned"
                ]
                >= 0.10
            )
            for name, row in means.items()
            if name.startswith("overwrite:")
        )
        for mode in INTERVENTIONS[1:]
    }
    if selected is not None:
        decision = (
            f"support an explicit occupancy-state successor with {selected}; "
            "do not retrain the token-local controller"
        )
    elif any(post_same_improved.values()):
        decision = "do not train; test an oblique or separate erase address"
    else:
        decision = "do not train; test exact logical-component replacement"
    return {
        "baseline_replay_maximum_absolute_residual": max(
            replay_residuals, default=math.inf
        ),
        "baseline_replay_passed": replay_passed,
        "observability_witness_passed": witness_passed,
        "runtime_integrity_passed": runtime_integrity_passed,
        "three_seed_means": means,
        "mode_checks": mode_checks,
        "stratum_checks": stratum_checks,
        "post_same_key_improved": post_same_improved,
        "passed_modes": passed_modes,
        "selected_mode": selected,
        "passed": selected is not None,
        "decision": decision,
    }


def _validate_r1(path: Path) -> tuple[dict[str, Any], str]:
    actual_sha256 = _sha256(path)
    if actual_sha256 != EXPECTED_R1_SHA256:
        raise ValueError("R1 artifact hash does not match the frozen input")
    report = json.loads(path.read_text(encoding="utf-8"))
    _validate_quality_artifact(report, name="R1")
    if report.get("parent_g15b_sha256") != EXPECTED_G15B_SHA256:
        raise ValueError("R1 does not bind the frozen G15B artifact")
    if report.get("parent_r0_sha256") != EXPECTED_R0_SHA256:
        raise ValueError("R1 does not bind the frozen R0 artifact")
    if report.get("adjudication", {}).get("passed") is not False:
        raise ValueError("R2 requires the failed R1 adjudication")
    if report["adjudication"].get("runtime_integrity_passed") is not True:
        raise ValueError("R2 requires R1 runtime integrity")
    return report, actual_sha256


def run(
    *,
    mode: Literal["smoke", "quality"],
    device: torch.device,
    parent_path: Path,
    r0_path: Path,
    r1_path: Path,
    checkpoint_directory: Path,
    commit: str,
    status_at_start: list[str],
) -> dict[str, Any]:
    parent, parent_sha256 = _validate_parent(parent_path)
    _, r0_sha256 = _validate_r0(r0_path, parent_sha256=parent_sha256)
    _, r1_sha256 = _validate_r1(r1_path)
    expected = _expected_identity(parent)
    seeds = QUALITY_SEEDS if mode == "quality" else QUALITY_SEEDS[:1]
    decisions = 4096 if mode == "quality" else 16
    batch_cap = 16 if mode == "quality" else 2
    seed_reports = []
    started = time.perf_counter()
    for seed in seeds:
        checkpoint_path = checkpoint_directory / f"g15b_I_seed{seed}.pt"
        model, checkpoint = _load_checkpoint(
            checkpoint_path,
            seed=seed,
            expected=expected[seed],
            device=device,
        )
        witness = temporal_observability_witness(model)
        _sync(device)
        evaluation_started = time.perf_counter()
        evaluation = evaluate_checkpoint(
            model, seed=seed, decisions=decisions, batch_cap=batch_cap
        )
        _sync(device)
        for name, cell in evaluation["cells"].items():
            recorded = checkpoint["evaluation"]["cells"][name]
            replayed = cell["interventions"]["learned"]
            for metric in (
                "query_accuracy",
                "exact_episode_accuracy",
                "bits_per_query",
            ):
                cell[f"recorded_g15b_{metric}"] = recorded[metric]
                cell[f"baseline_{metric}_absolute_residual"] = abs(
                    replayed[metric] - recorded[metric]
                )
        seed_reports.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "observability_witness": witness,
                "evaluation_wall_seconds": time.perf_counter() - evaluation_started,
                "evaluation": evaluation,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    adjudication = _adjudicate(seed_reports)
    source_paths = (
        Path(__file__),
        PROTOCOL,
        ROOT / "g15br1_event_erase.py",
        ROOT / "g15br_checkpoint_repair.py",
        ROOT / "g15b_interleaved_cohort.py",
        ROOT / "g15b_interleaved_tasks.py",
        ROOT / "spin_dirac_memory.py",
        ROOT / "model.py",
    )
    return {
        "schema_version": 1,
        "experiment": "G15B-R2 collision-only erase checkpoint diagnostic",
        "mode": mode,
        "evidentiary": mode == "quality" and not status_at_start,
        "git_commit_at_start": commit,
        "git_status_at_start": status_at_start,
        "elapsed_wall_seconds": time.perf_counter() - started,
        "parent_g15b_artifact": str(parent_path),
        "parent_g15b_sha256": parent_sha256,
        "parent_r0_artifact": str(r0_path),
        "parent_r0_sha256": r0_sha256,
        "parent_r1_artifact": str(r1_path),
        "parent_r1_sha256": r1_sha256,
        "protocol": {
            "seeds": list(seeds),
            "evaluation_decisions_per_cell": decisions,
            "evaluation_batch_cap": batch_cap,
            "tasks": ["mqar", "overwrite", "selective", "needle"],
            "lengths": list(EVALUATION_LENGTHS),
            "interventions": list(INTERVENTIONS),
            "overwrite_query_strata": list(STRATA),
            "optimizer_updates": 0,
        },
        "source_files": {
            str(path.relative_to(ROOT)): _sha256(path) for path in source_paths
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "compute_capability": (
                list(torch.cuda.get_device_capability(device))
                if device.type == "cuda"
                else None
            ),
        },
        "seed_reports": seed_reports,
        "adjudication": adjudication,
        "explicit_nonclaims": [
            "no parameter is trained or updated",
            "collision masks use commissioned causal task history",
            "replayed held-out G15B cells are not fresh generalization evidence",
            "no G15C, token-local controller, natural-text, optimizer, Spin, scaling, or model-family promotion follows",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "quality"), required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--parent-artifact", type=Path, default=PARENT_ARTIFACT)
    parser.add_argument("--r0-artifact", type=Path, default=R0_ARTIFACT)
    parser.add_argument("--r1-artifact", type=Path, default=R1_ARTIFACT)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    device = torch.device(args.device)
    commit, status = _git_provenance()
    if args.mode == "quality":
        if status:
            raise RuntimeError("G15B-R2 quality requires a clean git tree at start")
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("G15B-R2 quality requires CUDA")
        if torch.cuda.get_device_capability(device) != (7, 5):
            raise RuntimeError("G15B-R2 quality is frozen to SM75")
    report = run(
        mode=args.mode,
        device=device,
        parent_path=args.parent_artifact,
        r0_path=args.r0_artifact,
        r1_path=args.r1_artifact,
        checkpoint_directory=args.checkpoint_directory,
        commit=commit,
        status_at_start=status,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": _sha256(args.output),
                "decision": report["adjudication"]["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "INTERVENTIONS",
    "STRATA",
    "collision_erase_forward",
    "evaluate_checkpoint",
    "overwrite_query_strata",
]
