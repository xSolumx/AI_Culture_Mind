"""Run the frozen G15A Clifford-read and broken-coupling controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

if __package__:
    from .g15a_spin_dirac_cohort import (
        CONDITIONAL_ARM_SPECS,
        QUALITY_SEEDS,
        _atomic_json,
        _build_model,
        _git_state,
        _macro_accuracy,
        _now,
        _run_symmetry_arm,
        _sha256,
        _train_no_symmetry,
        quality_config,
    )
    from .model import parameter_count
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4.g15a_spin_dirac_cohort import (  # type: ignore[no-redef]
        CONDITIONAL_ARM_SPECS,
        QUALITY_SEEDS,
        _atomic_json,
        _build_model,
        _git_state,
        _macro_accuracy,
        _now,
        _run_symmetry_arm,
        _sha256,
        _train_no_symmetry,
        quality_config,
    )
    from hybrid_memory_v1_4.model import parameter_count  # type: ignore[no-redef]


PROTOCOL = Path(__file__).with_name("G15A_CONDITIONAL_CONTROLS_PROTOCOL_2026-08-25.md")
PRIMARY_ARTIFACT_SHA256 = (
    "6e2f5e58d8c411ce9c1a594e71197c76327145de00fc3af7023f884e788dd43a"
)


def _load_primary(path: Path) -> dict[str, Any]:
    if _sha256(path) != PRIMARY_ARTIFACT_SHA256:
        raise RuntimeError("the primary G15A artifact does not match the frozen hash")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("mode") != "quality"
        or report.get("evidentiary") is not True
        or report.get("adjudication", {}).get("passed") is not True
        or tuple(report.get("protocol", {}).get("seeds", ())) != QUALITY_SEEDS
    ):
        raise RuntimeError("the primary G15A artifact is not the passed quality cohort")
    for name in ("g15a_tasks.py", "model.py", "optimizers.py", "spin_dirac_memory.py"):
        current = Path(__file__).with_name(name)
        if _sha256(current) != report["source_files"][name]:
            raise RuntimeError(f"core source {name} changed after the primary cohort")
    return report


def _primary_s_by_seed(primary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(report["seed"]): report["arms"]["S"] for report in primary["seed_reports"]
    }


def _state_dict_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _adjudicate(
    primary: dict[str, Any], conditional_reports: list[dict[str, Any]]
) -> dict[str, Any]:
    primary_s = _primary_s_by_seed(primary)
    per_seed = []
    for report in conditional_reports:
        seed = int(report["seed"])
        spin = primary_s[seed]
        controls = report["arms"]
        spin_symmetry = _macro_accuracy(spin["symmetry"]["evaluation"])
        control_symmetry = {
            arm: _macro_accuracy(result["symmetry"]["evaluation"])
            for arm, result in controls.items()
        }
        spin_no_symmetry = _macro_accuracy(spin["no_symmetry"]["evaluation"])
        control_no_symmetry = {
            arm: _macro_accuracy(result["no_symmetry"]["evaluation"])
            for arm, result in controls.items()
        }
        symmetry_margins = {
            arm: spin_symmetry - accuracy for arm, accuracy in control_symmetry.items()
        }
        schedule_matches = {
            arm: result["no_symmetry"]["training_schedule_sha256"]
            == spin["no_symmetry"]["training_schedule_sha256"]
            for arm, result in controls.items()
        }
        inner_residuals = {
            arm: result["symmetry"]["inner_conjugation_max_abs_residual_float64"]
            for arm, result in controls.items()
        }
        checks = {
            "clifford_read_margin_at_least_0_02": (
                symmetry_margins["S+identity-read"] + 1e-12 >= 0.02
            ),
            "shared_triality_margin_at_least_0_02": (
                symmetry_margins["S-broken"] + 1e-12 >= 0.02
            ),
            "paired_training_schedule_hashes_match": all(schedule_matches.values()),
            "conditional_inner_replays_at_most_1e_9": all(
                residual <= 1e-9 for residual in inner_residuals.values()
            ),
        }
        per_seed.append(
            {
                "seed": seed,
                "spin_symmetry_accuracy": spin_symmetry,
                "conditional_symmetry_accuracy": control_symmetry,
                "symmetry_margins": symmetry_margins,
                "spin_no_symmetry_accuracy": spin_no_symmetry,
                "conditional_no_symmetry_accuracy": control_no_symmetry,
                "conditional_no_symmetry_deltas_from_spin": {
                    arm: accuracy - spin_no_symmetry
                    for arm, accuracy in control_no_symmetry.items()
                },
                "training_schedule_matches": schedule_matches,
                "inner_conjugation_max_abs_residual_float64": inner_residuals,
                "checks": checks,
            }
        )
    clifford_supported = all(
        row["checks"]["clifford_read_margin_at_least_0_02"] for row in per_seed
    )
    triality_supported = all(
        row["checks"]["shared_triality_margin_at_least_0_02"] for row in per_seed
    )
    integrity_passed = all(
        row["checks"]["paired_training_schedule_hashes_match"]
        and row["checks"]["conditional_inner_replays_at_most_1e_9"]
        for row in per_seed
    )
    if triality_supported:
        decision = "the shared-triality attribution survives the conditional control"
    else:
        decision = (
            "triality-specific attribution fails; G15A supports richer two-sided "
            "orthogonal transport only"
        )
    return {
        "completed": True,
        "integrity_passed": integrity_passed,
        "clifford_read_contribution_supported": clifford_supported,
        "shared_triality_coupling_contribution_supported": triality_supported,
        "decision": decision,
        "per_seed": per_seed,
    }


def run(
    primary: dict[str, Any],
    *,
    device: torch.device,
    checkpoint_directory: Path,
    commit: str,
    status_at_start: list[str],
) -> dict[str, Any]:
    config = quality_config()
    started = time.perf_counter()
    started_at = _now()
    primary_s = _primary_s_by_seed(primary)
    conditional_reports = []
    for seed in config.seeds:
        arms = {}
        spin_reference = _build_model("S", config, seed, torch.device("cpu"))
        reference_initial_state_hash = _state_dict_sha256(spin_reference)
        del spin_reference
        for arm, (transport, readout) in CONDITIONAL_ARM_SPECS.items():
            shape_model = _build_model(arm, config, seed, torch.device("cpu"))
            shapes = {
                name: tuple(parameter.shape)
                for name, parameter in shape_model.named_parameters()
            }
            shape_hash = hashlib.sha256(
                json.dumps(shapes, sort_keys=True).encode()
            ).hexdigest()
            parameters = parameter_count(shape_model)
            initial_state_hash = _state_dict_sha256(shape_model)
            del shape_model
            if (
                parameters != primary_s[seed]["parameters"]
                or shape_hash != primary_s[seed]["parameter_shapes_sha256"]
            ):
                raise RuntimeError(
                    f"conditional arm {arm} is not parameter-matched to primary S"
                )
            if initial_state_hash != reference_initial_state_hash:
                raise RuntimeError(
                    f"conditional arm {arm} initialization differs from S"
                )
            arms[arm] = {
                "transport_mode": transport,
                "readout_mode": readout,
                "parameters": parameters,
                "parameter_shapes_sha256": shape_hash,
                "initial_state_sha256": initial_state_hash,
                "initial_state_matches_regenerated_spin": True,
                "symmetry": _run_symmetry_arm(arm, config, seed=seed, device=device),
                "no_symmetry": _train_no_symmetry(
                    arm,
                    config,
                    seed=seed,
                    device=device,
                    checkpoint_directory=checkpoint_directory,
                ),
            }
        conditional_reports.append({"seed": seed, "arms": arms})
    adjudication = _adjudicate(primary, conditional_reports)
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("g15a_spin_dirac_cohort.py"),
        Path(__file__).with_name("g15a_tasks.py"),
        Path(__file__).with_name("spin_dirac_memory.py"),
        Path(__file__).with_name("model.py"),
        Path(__file__).with_name("optimizers.py"),
        PROTOCOL,
    )
    return {
        "schema_version": 1,
        "experiment": "G15A conditional Clifford-read and triality controls",
        "claim_status": (
            "conditional attribution diagnostics for the finite oracle-controlled "
            "G15A mechanism task"
        ),
        "mode": "quality",
        "evidentiary": not status_at_start,
        "started_at": started_at,
        "finished_at": _now(),
        "elapsed_wall_seconds": time.perf_counter() - started,
        "git_commit_at_start": commit,
        "git_status_at_start": status_at_start,
        "primary_artifact_sha256": PRIMARY_ARTIFACT_SHA256,
        "primary_artifact_commit": primary["git_commit_at_start"],
        "protocol": asdict(config),
        "conditional_arm_specs": {
            arm: {"transport": modes[0], "readout": modes[1]}
            for arm, modes in CONDITIONAL_ARM_SPECS.items()
        },
        "protocol_file_sha256": _sha256(PROTOCOL),
        "source_files": {
            str(path.relative_to(Path(__file__).parent)): _sha256(path)
            for path in source_paths
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "dtype": "float32",
        },
        "seed_reports": conditional_reports,
        "adjudication": adjudication,
        "explicit_nonclaims": [
            "coordinates and carrier controls are supplied on the symmetry task",
            "the controls do not establish learned geometry or generic association",
            "no natural-text, long-recall, scaling, or fused-kernel claim follows",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = _parser().parse_args()
    primary = _load_primary(args.primary_artifact)
    commit, status_at_start = _git_state()
    if status_at_start:
        raise RuntimeError(
            "evidentiary G15A conditional controls require a clean committed worktree"
        )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the frozen conditional controls require CUDA")
    if torch.cuda.get_device_capability(device) != (7, 5):
        raise RuntimeError("the frozen conditional controls require exact SM75")
    report = run(
        primary,
        device=device,
        checkpoint_directory=args.checkpoint_directory,
        commit=commit,
        status_at_start=status_at_start,
    )
    _atomic_json(args.output, report)
    print(args.output)
    print(json.dumps(report["adjudication"], indent=2, sort_keys=True))
    if not report["adjudication"]["integrity_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
