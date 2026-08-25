"""Prospective gauge-preserving G15B-R0 checkpoint intervention.

This runner never updates parameters.  It replays the retained G15B identity
checkpoints while replacing only erase/write timing, keeping the learned
address, value, read, and decoder gauge fixed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

import torch
from torch.nn import functional as F

from .g15b_interleaved_cohort import (
    EVALUATION_LENGTHS,
    NEEDLE_DISTANCES,
    _evaluation_batch_size,
    _gather_time,
    _sha256,
    _stable_seed,
)
from .g15b_interleaved_tasks import (
    PAYLOAD_START,
    ROLE_FILLER,
    ROLE_ITEM_KEY,
    ROLE_ITEM_MARKER,
    ROLE_ITEM_VALUE,
    ROLE_QUERY_KEY,
    ROLE_QUERY_MARKER,
    ROLE_WRITE_KEY,
    ROLE_WRITE_MARKER,
    ROLE_WRITE_VALUE,
    SELECT_TOKEN,
    WRITE_TOKEN,
    InterleavedBatch,
    generate_interleaved_batch,
)
from .model import HybridMemoryConfig, HybridMemoryLM
from .spin_dirac_memory import SpinDiracMemory

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "G15BR_CHECKPOINT_REPAIR_PROTOCOL_2026-08-26.md"
PARENT_ARTIFACT = ROOT / "artifacts/g15b_interleaved_controller_sm75_2026-08-26.json"
QUALITY_SEEDS = (2309, 2311, 2333)
INTERVENTIONS = (
    "learned",
    "soft_delta",
    "exact_collision_timing",
    "exact_delta_timing",
)
Intervention = Literal[
    "learned", "soft_delta", "exact_collision_timing", "exact_delta_timing"
]
ROLE_NAMES = {
    ROLE_FILLER: "filler",
    ROLE_WRITE_MARKER: "write_marker",
    ROLE_WRITE_KEY: "write_key",
    ROLE_WRITE_VALUE: "write_value",
    ROLE_QUERY_MARKER: "query_marker",
    ROLE_QUERY_KEY: "query_key",
    ROLE_ITEM_MARKER: "item_marker",
    ROLE_ITEM_KEY: "item_key",
    ROLE_ITEM_VALUE: "item_value",
}


def _git(args: list[str]) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _git_provenance() -> tuple[str, list[str]]:
    commit = _git(["rev-parse", "HEAD"])
    status = [line for line in _git(["status", "--porcelain"]).splitlines() if line]
    return commit, status


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _hidden_controls(
    model: HybridMemoryLM, token_ids: torch.Tensor
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    SpinDiracMemory,
    list[torch.Tensor],
]:
    if model.layer_plan != ("spin_dirac",):
        raise ValueError("G15B-R0 requires the one-block Spin-Dirac shell")
    block = model.blocks[0]
    mixer = block.mixer
    if not isinstance(mixer, SpinDiracMemory):
        raise TypeError("G15B-R0 block does not contain SpinDiracMemory")
    hidden = model.embedding(token_ids)
    value, outer_gate = block.input_projection(block.mixer_norm(hidden)).chunk(
        2, dim=-1
    )
    if block.local_conv is None:
        raise ValueError("G15B-R0 requires the frozen width-four local convolution")
    mixed, _ = block.local_conv(value, None, None)
    mixed = F.silu(mixed)
    controls = [tensor.clone() for tensor in mixer._controls(mixed, None)]
    return hidden, outer_gate, mixed, mixer, controls


def _event_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return mask[..., None, None].to(reference).expand_as(reference).clone()


def repair_control_forward(
    model: HybridMemoryLM,
    batch: InterleavedBatch,
    intervention: Intervention,
) -> dict[str, Any]:
    """Run one gauge-preserving edit-timing intervention."""

    if intervention not in INTERVENTIONS:
        raise ValueError(f"unknown G15B-R0 intervention {intervention!r}")
    hidden, outer_gate, mixed, mixer, controls = _hidden_controls(
        model, batch.token_ids
    )
    if intervention == "soft_delta":
        controls[3] = controls[4].clone()
    elif intervention == "exact_collision_timing":
        controls[4] = _event_mask(batch.write_event_mask, controls[4])
        controls[3] = _event_mask(batch.erase_event_mask, controls[3])
    elif intervention == "exact_delta_timing":
        exact_write = _event_mask(batch.write_event_mask, controls[4])
        controls[4] = exact_write
        controls[3] = exact_write.clone()

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
    logits = model.lm_head(model.final_norm(hidden))
    return {
        "logits": logits,
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


def local_write_event_mask(token_ids: torch.Tensor) -> torch.Tensor:
    """Decode the repaired edit target from the local causal grammar."""

    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape (batch, length)")
    decoded = torch.zeros_like(token_ids, dtype=torch.bool)
    decoded[:, 2:] = token_ids[:, :-2].eq(WRITE_TOKEN) | token_ids[:, :-2].eq(
        SELECT_TOKEN
    )
    return decoded


def temporal_observability_witness(model: HybridMemoryLM) -> dict[str, Any]:
    """Exhibit equal local observations with opposite collision labels."""

    filler = PAYLOAD_START + 30
    repeated_key = PAYLOAD_START + 3
    other_key = PAYLOAD_START + 4
    old_value = PAYLOAD_START + 40
    new_value = PAYLOAD_START + 41
    length = 16
    target = 14
    tokens = torch.full(
        (2, length), filler, dtype=torch.long, device=model.embedding.weight.device
    )
    tokens[0, 0:3] = torch.tensor(
        [WRITE_TOKEN, repeated_key, old_value], device=tokens.device
    )
    tokens[1, 0:3] = torch.tensor(
        [WRITE_TOKEN, other_key, old_value], device=tokens.device
    )
    common = torch.tensor(
        [filler, WRITE_TOKEN, repeated_key, new_value], device=tokens.device
    )
    tokens[:, target - 3 : target + 1] = common
    with torch.no_grad():
        _, _, _, _, controls = _hidden_controls(model, tokens)
    residuals = {
        name: float((tensor[0, target] - tensor[1, target]).abs().max())
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
    }
    windows_equal = torch.equal(
        tokens[0, target - 3 : target + 1], tokens[1, target - 3 : target + 1]
    )
    maximum = max(residuals.values())
    return {
        "width": 4,
        "target_position": target,
        "local_windows_equal": windows_equal,
        "collision_labels": [True, False],
        "labels_differ": True,
        "control_residuals": residuals,
        "maximum_control_residual": maximum,
        "passed": windows_equal and maximum <= 5e-7,
    }


def _new_role_accumulator(heads: int) -> dict[str, Any]:
    return {
        "roles": {
            name: {
                "count": 0,
                "probability_sum_by_head": [0.0] * heads,
                "positive_by_head": [0] * heads,
            }
            for name in ROLE_NAMES.values()
        },
        "one_step_after_write": {
            "count": 0,
            "probability_sum_by_head": [0.0] * heads,
            "positive_by_head": [0] * heads,
        },
    }


def _accumulate_write_roles(
    accumulator: dict[str, Any],
    write_strength: torch.Tensor,
    batch: InterleavedBatch,
) -> None:
    values = write_strength.squeeze(-1).float()
    heads = values.shape[-1]
    for role, name in ROLE_NAMES.items():
        mask = batch.roles == role
        count = int(mask.sum())
        if not count:
            continue
        selected = values[mask]
        row = accumulator["roles"][name]
        row["count"] += count
        for head in range(heads):
            row["probability_sum_by_head"][head] += float(selected[:, head].sum())
            row["positive_by_head"][head] += int((selected[:, head] >= 0.5).sum())

    after = batch.write_positions + 1
    valid = after < batch.length
    rows = torch.arange(batch.batch_size, device=after.device)[:, None].expand_as(after)
    selected = values[rows[valid], after[valid]]
    row = accumulator["one_step_after_write"]
    count = selected.shape[0]
    row["count"] += count
    for head in range(heads):
        row["probability_sum_by_head"][head] += float(selected[:, head].sum())
        row["positive_by_head"][head] += int((selected[:, head] >= 0.5).sum())


def _finish_write_roles(accumulator: dict[str, Any]) -> dict[str, Any]:
    for row in [*accumulator["roles"].values(), accumulator["one_step_after_write"]]:
        count = max(1, row["count"])
        row["mean_probability_by_head"] = [
            value / count for value in row.pop("probability_sum_by_head")
        ]
        row["positive_rate_by_head"] = [
            value / count for value in row.pop("positive_by_head")
        ]
    return accumulator


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
    write_roles = _new_role_accumulator(model.config.spin_dirac_heads)
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
                decoded_write = local_write_event_mask(batch.token_ids)
                if not torch.equal(decoded_write, batch.write_event_mask):
                    raise RuntimeError(
                        "repaired write target is not exactly locally observable"
                    )
                for intervention in INTERVENTIONS:
                    result = repair_control_forward(model, batch, intervention)
                    prediction = _gather_time(
                        result["logits"], batch.query_positions
                    ).argmax(-1)
                    match = prediction == batch.targets
                    correct[intervention] += int(match.sum())
                    episodes[intervention] += int(match.all(dim=1).sum())
                    if intervention == "learned":
                        _accumulate_write_roles(
                            write_roles,
                            result["controls"]["write_strength"],
                            batch,
                        )
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
    return {"cells": cells, "learned_write_roles": _finish_write_roles(write_roles)}


def _load_parent(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("evidentiary") is not True or report["adjudication"]["passed"]:
        raise ValueError("parent artifact is not the completed failed G15B cohort")
    return report


def _expected_identity(parent: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {row["seed"]: row["arms"]["I"] for row in parent["seed_reports"]}


def _load_checkpoint(
    path: Path,
    *,
    seed: int,
    expected: dict[str, Any],
    device: torch.device,
) -> tuple[HybridMemoryLM, dict[str, Any]]:
    actual_hash = _sha256(path)
    if actual_hash != expected["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash mismatch for seed {seed}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("arm") != "I" or checkpoint.get("seed") != seed:
        raise ValueError("checkpoint arm/seed mismatch")
    model = HybridMemoryLM(HybridMemoryConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    if (
        model.config.spin_dirac_transport_mode != "identity"
        or model.config.spin_dirac_readout_mode != "identity"
        or model.config.conv_kernel != 4
    ):
        raise ValueError("checkpoint is not the frozen identity width-four shell")
    return model.to(device), checkpoint


def _adjudicate(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    cell_names = list(seed_reports[0]["evaluation"]["cells"])
    means: dict[str, Any] = {}
    replay_residuals = []
    for seed_report in seed_reports:
        for cell in seed_report["evaluation"]["cells"].values():
            replay_residuals.append(cell["baseline_replay_absolute_residual"])
    for cell_name in cell_names:
        rows = [report["evaluation"]["cells"][cell_name] for report in seed_reports]
        mean_accuracy = {
            intervention: sum(
                row["interventions"][intervention]["query_accuracy"] for row in rows
            )
            / len(rows)
            for intervention in INTERVENTIONS
        }
        means[cell_name] = {
            "mean_query_accuracy": mean_accuracy,
            "soft_delta_minus_learned": (
                mean_accuracy["soft_delta"] - mean_accuracy["learned"]
            ),
        }
    soft_checks: dict[str, bool] = {}
    exact_checks: dict[str, bool] = {}
    for name, row in means.items():
        task = name.split(":", 1)[0]
        if task == "overwrite":
            soft_checks[name] = row["soft_delta_minus_learned"] >= 0.10
        elif task in ("mqar", "selective"):
            soft_checks[name] = row["soft_delta_minus_learned"] >= -0.02
        else:
            soft_checks[name] = row["mean_query_accuracy"]["soft_delta"] >= 0.999
        threshold = 0.999 if task == "needle" else 0.95
        exact_checks[name] = (
            row["mean_query_accuracy"]["exact_delta_timing"] >= threshold
        )
    replay_passed = max(replay_residuals, default=math.inf) <= 1e-12
    witness_passed = all(row["observability_witness"]["passed"] for row in seed_reports)
    soft_passed = replay_passed and witness_passed and all(soft_checks.values())
    exact_passed = replay_passed and witness_passed and all(exact_checks.values())
    if soft_passed:
        decision = "authorize fresh identity delta-law training; retained learned write timing already supports the repair"
    elif exact_passed:
        decision = "authorize fresh identity delta-law training; exact timing passes but learned write-role control still requires repair"
    else:
        decision = "do not train the delta-law successor; inspect address orthogonality, values, decoder, and write-tail interference"
    return {
        "baseline_replay_maximum_absolute_residual": max(
            replay_residuals, default=math.inf
        ),
        "baseline_replay_passed": replay_passed,
        "observability_witness_passed": witness_passed,
        "three_seed_means": means,
        "soft_delta_checks": soft_checks,
        "soft_delta_passed": soft_passed,
        "exact_delta_timing_checks": exact_checks,
        "exact_delta_timing_passed": exact_passed,
        "decision": decision,
        "passed": soft_passed or exact_passed,
    }


def run(
    *,
    mode: Literal["smoke", "quality"],
    device: torch.device,
    parent_path: Path,
    checkpoint_directory: Path,
    commit: str,
    status_at_start: list[str],
) -> dict[str, Any]:
    parent = _load_parent(parent_path)
    expected = _expected_identity(parent)
    seeds = QUALITY_SEEDS if mode == "quality" else QUALITY_SEEDS[:1]
    decisions = 4096 if mode == "quality" else 16
    batch_cap = 16 if mode == "quality" else 2
    seed_reports = []
    started = time.perf_counter()
    for seed in seeds:
        path = checkpoint_directory / f"g15b_I_seed{seed}.pt"
        model, checkpoint = _load_checkpoint(
            path, seed=seed, expected=expected[seed], device=device
        )
        witness = temporal_observability_witness(model)
        _sync(device)
        evaluation_started = time.perf_counter()
        evaluation = evaluate_checkpoint(
            model, seed=seed, decisions=decisions, batch_cap=batch_cap
        )
        _sync(device)
        evaluation_seconds = time.perf_counter() - evaluation_started
        for name, cell in evaluation["cells"].items():
            recorded = checkpoint["evaluation"]["cells"][name]["query_accuracy"]
            replayed = cell["interventions"]["learned"]["query_accuracy"]
            cell["recorded_g15b_query_accuracy"] = recorded
            cell["baseline_replay_absolute_residual"] = abs(replayed - recorded)
        seed_reports.append(
            {
                "seed": seed,
                "checkpoint": str(path),
                "checkpoint_sha256": _sha256(path),
                "observability_witness": witness,
                "evaluation_wall_seconds": evaluation_seconds,
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
        ROOT / "g15b_interleaved_cohort.py",
        ROOT / "g15b_interleaved_tasks.py",
        ROOT / "spin_dirac_memory.py",
        ROOT / "model.py",
    )
    return {
        "schema_version": 1,
        "experiment": "G15B-R0 gauge-preserving checkpoint repair",
        "mode": mode,
        "evidentiary": mode == "quality" and not status_at_start,
        "git_commit_at_start": commit,
        "git_status_at_start": status_at_start,
        "elapsed_wall_seconds": time.perf_counter() - started,
        "parent_artifact": str(parent_path),
        "parent_artifact_sha256": _sha256(parent_path),
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
            "replayed held-out G15B cells are a paired checkpoint diagnostic, not fresh generalization evidence",
            "exact timing interventions use task labels and are not deployable controllers",
            "no G15C, natural-text, Spin, optimizer, scaling, or model-family promotion follows",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "quality"), required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--parent-artifact", type=Path, default=PARENT_ARTIFACT)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    device = torch.device(args.device)
    commit, status = _git_provenance()
    if args.mode == "quality":
        if status:
            raise RuntimeError("G15B-R0 quality requires a clean git tree at start")
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("G15B-R0 quality requires CUDA")
        if torch.cuda.get_device_capability(device) != (7, 5):
            raise RuntimeError("G15B-R0 quality is frozen to SM75")
    report = run(
        mode=args.mode,
        device=device,
        parent_path=args.parent_artifact,
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
    "QUALITY_SEEDS",
    "evaluate_checkpoint",
    "local_write_event_mask",
    "repair_control_forward",
    "temporal_observability_witness",
]
