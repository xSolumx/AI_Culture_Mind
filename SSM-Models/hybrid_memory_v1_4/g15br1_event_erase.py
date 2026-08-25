"""G15B-R1 retained-checkpoint event-anchored erase diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Literal

import torch

from .g15b_interleaved_cohort import (
    EVALUATION_LENGTHS,
    NEEDLE_DISTANCES,
    _address_prototypes,
    _evaluation_batch_size,
    _gather_time,
    _sha256,
    _stable_seed,
)
from .g15b_interleaved_tasks import InterleavedBatch, generate_interleaved_batch
from .g15br_checkpoint_repair import (
    PARENT_ARTIFACT,
    QUALITY_SEEDS,
    ROOT,
    _event_mask,
    _expected_identity,
    _git_provenance,
    _hidden_controls,
    _load_checkpoint,
    _load_parent,
    _sync,
    local_write_event_mask,
    temporal_observability_witness,
)
from .model import HybridMemoryLM

PROTOCOL = ROOT / "G15BR1_EVENT_ERASE_PROTOCOL_2026-08-26.md"
R0_ARTIFACT = ROOT / "artifacts/g15br_checkpoint_repair_sm75_2026-08-26.json"
INTERVENTIONS = ("learned", "soft_event_erase", "exact_event_erase")
Intervention = Literal["learned", "soft_event_erase", "exact_event_erase"]


def event_erase_forward(
    model: HybridMemoryLM,
    batch: InterleavedBatch,
    intervention: Intervention,
) -> dict[str, Any]:
    """Preserve learned writes and change only event-anchored erase."""

    if intervention not in INTERVENTIONS:
        raise ValueError(f"unknown G15B-R1 intervention {intervention!r}")
    hidden, outer_gate, mixed, mixer, controls = _hidden_controls(
        model, batch.token_ids
    )
    if intervention != "learned":
        event = _event_mask(batch.write_event_mask, controls[3])
        if intervention == "soft_event_erase":
            controls[3] = controls[4] * event
        else:
            controls[3] = event

    read, _ = mixer.forward_controls(*controls, scan_mode="parallel")
    output_gate = 1.0 + torch.tanh(mixer.output_gate(mixed).view_as(read))
    update = mixer.output_projection(
        (mixer.output_norm(read) * output_gate).flatten(start_dim=2)
    )
    block = model.blocks[0]
    hidden = hidden + torch.sigmoid(block.residual_scale) * block.dropout(
        update * torch.sigmoid(outer_gate)
    )
    hidden = hidden + block.dropout(block.ffn(block.ffn_norm(hidden)))
    return {
        "logits": model.lm_head(model.final_norm(hidden)),
        "controls": {
            name: tensor
            for name, tensor in zip(
                (
                    "query_vector",
                    "key_vector",
                    "value_positive",
                    "erase_strength",
                    "write_strength",
                    "retention",
                    "transport_coordinates",
                ),
                controls,
                strict=True,
            )
        },
    }


def _new_cross_accumulator() -> dict[str, float | int]:
    return {"absolute_sum": 0.0, "count": 0, "maximum_absolute": 0.0}


def _accumulate_prototype_cross(
    accumulator: dict[str, float | int],
    controls: dict[str, torch.Tensor],
    batch: InterleavedBatch,
) -> None:
    if batch.live_keys.shape[1] <= 1:
        return
    prototypes, _, _ = _address_prototypes(controls, batch)
    gram = torch.einsum("bkhd,blhd->bhkl", prototypes, prototypes)
    keys = gram.shape[-1]
    mask = ~torch.eye(keys, dtype=torch.bool, device=gram.device)
    off_diagonal = gram[..., mask].abs().float()
    accumulator["absolute_sum"] = float(accumulator["absolute_sum"]) + float(
        off_diagonal.sum()
    )
    accumulator["count"] = int(accumulator["count"]) + off_diagonal.numel()
    accumulator["maximum_absolute"] = max(
        float(accumulator["maximum_absolute"]), float(off_diagonal.max())
    )


def _finish_prototype_cross(
    accumulator: dict[str, float | int],
) -> dict[str, float | int]:
    count = int(accumulator["count"])
    return {
        "pairs": count,
        "mean_absolute_off_diagonal_cosine": (
            float(accumulator["absolute_sum"]) / max(1, count)
        ),
        "maximum_absolute_off_diagonal_cosine": float(accumulator["maximum_absolute"]),
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
    cross = _new_cross_accumulator()
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
                learned_controls: dict[str, torch.Tensor] | None = None
                for intervention in INTERVENTIONS:
                    result = event_erase_forward(model, batch, intervention)
                    prediction = _gather_time(
                        result["logits"], batch.query_positions
                    ).argmax(-1)
                    match = prediction == batch.targets
                    correct[intervention] += int(match.sum())
                    episodes[intervention] += int(match.all(dim=1).sum())
                    if intervention == "learned":
                        learned_controls = result["controls"]
                if learned_controls is None:
                    raise RuntimeError("learned controls were not evaluated")
                _accumulate_prototype_cross(cross, learned_controls, batch)
                total += batch.targets.numel()
                episode_total += batch.batch_size
                batch_index += 1
            cells[f"{task}:L{length}"] = {
                "task": task,
                "length": length,
                "query_decisions": total,
                "interventions": {
                    name: {
                        "query_accuracy": correct[name] / total,
                        "exact_episode_accuracy": episodes[name] / episode_total,
                    }
                    for name in INTERVENTIONS
                },
            }
    return {"cells": cells, "prototype_cross": _finish_prototype_cross(cross)}


def _adjudicate(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    cell_names = list(seed_reports[0]["evaluation"]["cells"])
    replay_residuals = [
        cell["baseline_replay_absolute_residual"]
        for report in seed_reports
        for cell in report["evaluation"]["cells"].values()
    ]
    means: dict[str, Any] = {}
    for cell_name in cell_names:
        rows = [report["evaluation"]["cells"][cell_name] for report in seed_reports]
        accuracy = {
            intervention: sum(
                row["interventions"][intervention]["query_accuracy"] for row in rows
            )
            / len(rows)
            for intervention in INTERVENTIONS
        }
        means[cell_name] = {
            "mean_query_accuracy": accuracy,
            "soft_event_erase_minus_learned": (
                accuracy["soft_event_erase"] - accuracy["learned"]
            ),
            "exact_event_erase_minus_learned": (
                accuracy["exact_event_erase"] - accuracy["learned"]
            ),
        }

    replay_passed = max(replay_residuals, default=math.inf) <= 1e-12
    witness_passed = all(row["observability_witness"]["passed"] for row in seed_reports)
    mode_checks: dict[str, dict[str, bool]] = {}
    for intervention in ("soft_event_erase", "exact_event_erase"):
        checks = {}
        delta_name = f"{intervention}_minus_learned"
        for name, row in means.items():
            task = name.split(":", 1)[0]
            if task == "overwrite":
                checks[name] = row[delta_name] >= 0.10
            elif task in ("mqar", "selective"):
                checks[name] = row[delta_name] >= -0.02
            else:
                checks[name] = row["mean_query_accuracy"][intervention] >= 0.999
        mode_checks[intervention] = checks
    passed_modes = [
        mode
        for mode, checks in mode_checks.items()
        if replay_passed and witness_passed and all(checks.values())
    ]
    selected = None
    if passed_modes:
        regressions = {}
        for mode in passed_modes:
            delta_name = f"{mode}_minus_learned"
            regressions[mode] = sum(
                max(0.0, -row[delta_name])
                for name, row in means.items()
                if not name.startswith("overwrite:")
            )
        selected = min(
            passed_modes,
            key=lambda mode: (regressions[mode], mode != "soft_event_erase"),
        )
    prototype_rows = [
        report["evaluation"]["prototype_cross"] for report in seed_reports
    ]
    prototype_cross = {
        "seed_mean_absolute_off_diagonal_cosine": sum(
            row["mean_absolute_off_diagonal_cosine"] for row in prototype_rows
        )
        / len(prototype_rows),
        "maximum_absolute_off_diagonal_cosine": max(
            row["maximum_absolute_off_diagonal_cosine"] for row in prototype_rows
        ),
    }
    return {
        "baseline_replay_maximum_absolute_residual": max(
            replay_residuals, default=math.inf
        ),
        "baseline_replay_passed": replay_passed,
        "observability_witness_passed": witness_passed,
        "three_seed_means": means,
        "mode_checks": mode_checks,
        "passed_modes": passed_modes,
        "selected_mode": selected,
        "prototype_cross": prototype_cross,
        "passed": selected is not None,
        "decision": (
            f"authorize fresh identity event-erase training with {selected}"
            if selected is not None
            else "do not train event-anchored erase; inspect prototype cross-talk and state-aware correction"
        ),
    }


def _validate_r0(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("evidentiary") is not True:
        raise ValueError("R0 artifact is not evidentiary")
    if report["adjudication"]["passed"] is not False:
        raise ValueError("R1 requires the failed R0 adjudication")
    if report["adjudication"]["baseline_replay_passed"] is not True:
        raise ValueError("R1 requires exact R0 baseline replay")
    return report


def run(
    *,
    mode: Literal["smoke", "quality"],
    device: torch.device,
    parent_path: Path,
    r0_path: Path,
    checkpoint_directory: Path,
    commit: str,
    status_at_start: list[str],
) -> dict[str, Any]:
    parent = _load_parent(parent_path)
    _validate_r0(r0_path)
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
            recorded = checkpoint["evaluation"]["cells"][name]["query_accuracy"]
            replayed = cell["interventions"]["learned"]["query_accuracy"]
            cell["recorded_g15b_query_accuracy"] = recorded
            cell["baseline_replay_absolute_residual"] = abs(replayed - recorded)
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
        ROOT / "g15br_checkpoint_repair.py",
        ROOT / "g15b_interleaved_cohort.py",
        ROOT / "g15b_interleaved_tasks.py",
        ROOT / "spin_dirac_memory.py",
        ROOT / "model.py",
    )
    return {
        "schema_version": 1,
        "experiment": "G15B-R1 event-anchored erase checkpoint diagnostic",
        "mode": mode,
        "evidentiary": mode == "quality" and not status_at_start,
        "git_commit_at_start": commit,
        "git_status_at_start": status_at_start,
        "elapsed_wall_seconds": time.perf_counter() - started,
        "parent_g15b_artifact": str(parent_path),
        "parent_g15b_sha256": _sha256(parent_path),
        "parent_r0_artifact": str(r0_path),
        "parent_r0_sha256": _sha256(r0_path),
        "protocol": {
            "seeds": list(seeds),
            "evaluation_decisions_per_cell": decisions,
            "evaluation_batch_cap": batch_cap,
            "tasks": ["mqar", "overwrite", "selective", "needle"],
            "lengths": list(EVALUATION_LENGTHS),
            "interventions": list(INTERVENTIONS),
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
            "exact event masks use commissioned task grammar",
            "replayed held-out G15B cells are not fresh generalization evidence",
            "no G15C, natural-text, optimizer, Spin, scaling, or model-family promotion follows",
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
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    device = torch.device(args.device)
    commit, status = _git_provenance()
    if args.mode == "quality":
        if status:
            raise RuntimeError("G15B-R1 quality requires a clean git tree at start")
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("G15B-R1 quality requires CUDA")
        if torch.cuda.get_device_capability(device) != (7, 5):
            raise RuntimeError("G15B-R1 quality is frozen to SM75")
    report = run(
        mode=args.mode,
        device=device,
        parent_path=args.parent_artifact,
        r0_path=args.r0_artifact,
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
    "evaluate_checkpoint",
    "event_erase_forward",
]
