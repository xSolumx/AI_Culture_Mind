"""Frozen G15B-D product versus coupled residual-delta cohort."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal

import torch

from .g15b_interleaved_cohort import _sha256, _stable_seed
from .g15b_interleaved_tasks import generate_interleaved_batch
from .g15be_effective_edit_cohort import (
    CohortConfig,
    EVALUATION_LENGTHS,
    EVALUATION_TASKS,
    QUALITY_SEEDS,
    _diagnostics,
    _enforce_execution_eligibility,
    _finite_tree,
    _model_state_sha256,
    _optimizer_partition,
    _tensor,
    _train_arm,
    build_model,
    commissioned_losses,
    effective_edit_intervention_forward,
    frozen_config as _g15be_frozen_config,
)
from .g15br3_logical_component import STRATA
from .model import parameter_count

ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[1]
PROTOCOL = ROOT / "G15BD_RESIDUAL_DELTA_PROTOCOL_2026-08-26.md"
PHASE0_ARTIFACT = (
    ROOT / "artifacts/g15bd_phase0_qualification_sm75_2026-08-26.json"
)
EXPECTED_PHASE0_SHA256 = (
    "44a8556b60db7cb8c5e1edc239255dc510b62f03960e4514a49b645e79123921"
)
ARMS = ("P", "D")
Arm = Literal["P", "D"]
EXPECTED_UPDATES = 3400
EXPECTED_TOKENS = 13_926_400


def frozen_config(mode: Literal["smoke", "quality"]) -> CohortConfig:
    """Return the frozen cohort with an SM75-safe paired evaluation cap."""

    config = _g15be_frozen_config(mode)
    if mode == "quality":
        return replace(config, evaluation_batch_cap=4)
    return config


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        text=True,
        encoding="utf-8",
    ).strip()


def run_preflight(device: torch.device) -> dict[str, Any]:
    phase0_sha = _sha256(PHASE0_ARTIFACT)
    phase0 = json.loads(PHASE0_ARTIFACT.read_text(encoding="utf-8"))
    phase0_sources = phase0.get("source_files", {})
    core_paths = (
        ROOT / "transactional_delta.py",
        ROOT / "model.py",
        ROOT / "optimizers.py",
        ROOT / "g15bd_phase0_qualification.py",
    )
    core_matches = {
        str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"): (
            phase0_sources.get(
                str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/")
            )
            == _sha256(path)
        )
        for path in core_paths
    }
    models = {
        arm: build_model(arm, 29, device)  # type: ignore[arg-type]
        for arm in ARMS
    }
    counts = {arm: parameter_count(model) for arm, model in models.items()}
    active = {
        arm: sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        for arm, model in models.items()
    }
    state_bytes = {
        arm: model.state_capacity_bytes(1, torch.float32)
        for arm, model in models.items()
    }
    hashes = {arm: _model_state_sha256(model) for arm, model in models.items()}
    batch = generate_interleaved_batch(
        "overwrite",
        2,
        128,
        4,
        8,
        4,
        seed=_stable_seed("g15bd-preflight"),
    ).to(device)
    gradients = {}
    reconstruction = {}
    coupling = {}
    for arm, model in models.items():
        model.train()
        model.zero_grad(set_to_none=True)
        output = model(batch.token_ids, return_diagnostics=True)
        loss, _ = commissioned_losses(output, batch)
        loss.backward()
        gradients[arm] = all(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            and bool(torch.count_nonzero(parameter.grad))
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        diagnostics = _diagnostics(output)
        coupling[arm] = float(
            (
                _tensor(diagnostics, "effective_erase_strength")
                - _tensor(diagnostics, "effective_write_strength")
            )
            .abs()
            .max()
            .detach()
        )
        model.eval()
        with torch.no_grad():
            ordinary = model(batch.token_ids)["logits"]
            replay = effective_edit_intervention_forward(
                model, batch, "learned_reconstruction"
            )["logits"]
        reconstruction[arm] = {
            "maximum_absolute_logit_residual": float(
                (ordinary - replay).abs().max()
            ),
            "predictions_equal": torch.equal(
                ordinary.argmax(-1), replay.argmax(-1)
            ),
        }
    optimizer, partition = _optimizer_partition(models["D"])
    del optimizer
    checks = {
        "sealed_phase0": phase0_sha == EXPECTED_PHASE0_SHA256
        and phase0.get("adjudication", {}).get("passed") is True,
        "phase0_core_sources_unchanged": all(core_matches.values()),
        "matched_parameters": len(set(counts.values())) == 1
        and len(set(active.values())) == 1
        and len(set(state_bytes.values())) == 1
        and len(set(hashes.values())) == 1,
        "finite_nonzero_gradients": all(gradients.values()),
        "optimizer_partition": partition["passed"],
        "D_coupled_edit": coupling["D"] == 0.0,
        "learned_reconstruction": all(
            row["maximum_absolute_logit_residual"] <= 5e-4
            and row["predictions_equal"]
            for row in reconstruction.values()
        ),
    }
    return {
        "phase0_sha256": phase0_sha,
        "phase0_core_source_matches": core_matches,
        "parameter_counts": counts,
        "active_parameter_counts": active,
        "state_bytes_per_sequence_fp32": state_bytes,
        "initial_parameter_sha256": hashes,
        "gradients_finite_nonzero": gradients,
        "optimizer_partition": partition,
        "effective_erase_write_maximum_absolute_residual": coupling,
        "learned_reconstruction": reconstruction,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _ordinary_quality_checks(
    reports: dict[tuple[str, int], dict[str, Any]], arm: Arm
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for seed in QUALITY_SEEDS:
        cells = reports[(arm, seed)]["evaluation"]["cells"]
        for length in (128, 512, 1024):
            overwrite = cells[f"overwrite:L{length}"]
            post = overwrite["query_strata"]["after_same_key_overwrite"][
                "accuracy"
            ]
            checks[f"{arm}:{seed}:overwrite:L{length}"] = (
                overwrite["query_accuracy"] >= 0.93
            )
            checks[f"{arm}:{seed}:post_same:L{length}"] = (
                post is not None and post >= 0.92
            )
            checks[f"{arm}:{seed}:mqar_selective:L{length}"] = all(
                cells[f"{task}:L{length}"]["query_accuracy"] >= 0.98
                for task in ("mqar", "selective")
            )
        checks[f"{arm}:{seed}:needle_guard"] = all(
            cells[f"needle:L{length}"]["query_accuracy"] == 1.0
            and cells[f"overwrite_guard:L{length}"]["query_accuracy"] >= 0.99
            and all(
                cells[f"overwrite_guard:L{length}"]["query_strata"][name][
                    "query_decisions"
                ]
                > 0
                and cells[f"overwrite_guard:L{length}"]["query_strata"][name][
                    "accuracy"
                ]
                is not None
                and cells[f"overwrite_guard:L{length}"]["query_strata"][name][
                    "accuracy"
                ]
                >= 0.99
                for name in STRATA
            )
            for length in EVALUATION_LENGTHS
        )
    return checks


def _adjudicate(config: Any, reports: list[dict[str, Any]]) -> dict[str, Any]:
    if config.mode == "smoke":
        return {
            "passed": False,
            "eligible_for_promotion": False,
            "decision": "smoke execution completed; run the frozen quality cohort",
        }
    by_arm_seed = {(row["arm"], row["seed"]): row for row in reports}
    expected = {(arm, seed) for arm in ARMS for seed in QUALITY_SEEDS}
    shared: dict[str, bool] = {
        "complete_seed_arm_matrix": set(by_arm_seed) == expected
        and len(reports) == len(expected)
    }
    if not shared["complete_seed_arm_matrix"]:
        return {
            "passed": False,
            "eligible_for_promotion": False,
            "shared_integrity_checks": shared,
            "decision": "stop G15B-D after an incomplete Phase-1 cohort",
        }
    expected_cells = {
        f"{task}:L{length}"
        for task in EVALUATION_TASKS
        for length in EVALUATION_LENGTHS
    }
    expected_interventions = {
        f"{task}:L{length}"
        for task in ("mqar", "overwrite", "selective")
        for length in (512, 1024)
    }
    for (arm, seed), row in by_arm_seed.items():
        checkpoint = Path(row["checkpoint"])
        shared[f"provenance:{arm}:{seed}:checkpoint"] = (
            checkpoint.is_file()
            and _sha256(checkpoint) == row["checkpoint_sha256"]
        )
        shared[f"numerical:{arm}:{seed}:finite"] = _finite_tree(row)
        shared[f"boundary:{arm}:{seed}"] = row["evaluation"]["boundary_audit"][
            "passed"
        ]
        shared[f"complete:{arm}:{seed}:ordinary"] = (
            set(row["evaluation"]["cells"]) == expected_cells
        )
        shared[f"complete:{arm}:{seed}:interventions"] = set(
            row["evaluation"]["intervention_cells"]
        ) == (expected_interventions if arm == "D" else set())
        for cell_name, cell in row["evaluation"]["cells"].items():
            shared[f"decisions:{arm}:{seed}:{cell_name}"] = (
                cell["query_decisions"] == config.evaluation_decisions
            )
            if cell["task"] in ("overwrite", "overwrite_guard"):
                shared[f"strata:{arm}:{seed}:{cell_name}"] = sum(
                    value["query_decisions"]
                    for value in cell["query_strata"].values()
                ) == cell["query_decisions"]
        for cell_name, cell in row["evaluation"]["intervention_cells"].items():
            shared[f"intervention_decisions:{arm}:{seed}:{cell_name}"] = (
                cell["query_decisions"] == config.intervention_decisions
            )
            if cell["task"] == "overwrite":
                shared[f"intervention_strata:{arm}:{seed}:{cell_name}"] = sum(
                    value["query_decisions"]
                    for value in cell["query_strata"].values()
                ) == cell["query_decisions"]
    for seed in QUALITY_SEEDS:
        paired = [by_arm_seed[(arm, seed)] for arm in ARMS]
        for name, getter in (
            ("initial", lambda row: row["initial_parameter_sha256"]),
            ("training", lambda row: row["training_schedule_sha256"]),
            (
                "evaluation",
                lambda row: row["evaluation"][
                    "standard_evaluation_schedule_sha256"
                ],
            ),
            (
                "boundary",
                lambda row: row["evaluation"]["boundary_batch_sha256"],
            ),
            (
                "parameters",
                lambda row: (
                    row["parameters"],
                    row["active_parameters"],
                    row["state_bytes_per_sequence_fp32"],
                ),
            ),
        ):
            shared[f"paired:{seed}:{name}"] = len(
                {getter(row) for row in paired}
            ) == 1
        shared[f"paired:{seed}:budget"] = all(
            row["training_updates"] == EXPECTED_UPDATES
            and row["training_tokens"] == EXPECTED_TOKENS
            for row in paired
        )
        shared[f"paired:{seed}:fingerprints"] = all(
            not row["train_evaluation_hash_intersection"] for row in paired
        )

    product = _ordinary_quality_checks(by_arm_seed, "P")
    delta = _ordinary_quality_checks(by_arm_seed, "D")
    for seed in QUALITY_SEEDS:
        interventions = by_arm_seed[("D", seed)]["evaluation"][
            "intervention_cells"
        ]
        for length in (512, 1024):
            for task in ("mqar", "overwrite", "selective"):
                cell = interventions[f"{task}:L{length}"]
                for name in (
                    "memory_zero",
                    "valid_event_edit_zero",
                    "permuted_write_binding",
                ):
                    delta[f"D:{seed}:{task}:{name}:L{length}"] = (
                        cell["interventions"][name]["drop_from_learned"] >= 0.50
                    )
                delta[f"D:{seed}:{task}:event_only:L{length}"] = (
                    abs(
                        cell["interventions"]["valid_event_only"][
                            "drop_from_learned"
                        ]
                    )
                    <= 0.02
                )
                delta[f"D:{seed}:{task}:non_event_only:L{length}"] = (
                    cell["interventions"]["non_event_only"]["query_accuracy"]
                    <= 0.50
                )
                delta[f"D:{seed}:{task}:reconstruction:L{length}"] = (
                    cell["reconstruction_maximum_absolute_logit_residual"]
                    <= 5e-4
                    and cell["reconstruction_query_predictions_equal"]
                )
            overwrite = interventions[f"overwrite:L{length}"]
            gates = overwrite["effective_gate_statistics"]
            delta[f"D:{seed}:event_edit_mean:L{length}"] = (
                gates["event_write"]["mean"] >= 0.25
            )
            delta[f"D:{seed}:non_event_edit_mean:L{length}"] = (
                gates["non_event_write"]["mean"] <= 0.05
            )

    comparative: dict[str, bool] = {}
    for length in (128, 512, 1024):
        means = {
            arm: sum(
                by_arm_seed[(arm, seed)]["evaluation"]["cells"][
                    f"overwrite:L{length}"
                ]["query_accuracy"]
                for seed in QUALITY_SEEDS
            )
            / len(QUALITY_SEEDS)
            for arm in ARMS
        }
        comparative[f"D:mean_minus_P:L{length}"] = (
            means["D"] - means["P"] >= 0.02
        )
        delta[f"D:worst_seed:L{length}"] = all(
            means["D"]
            - by_arm_seed[("D", seed)]["evaluation"]["cells"][
                f"overwrite:L{length}"
            ]["query_accuracy"]
            <= 0.03
            for seed in QUALITY_SEEDS
        )
        for seed in QUALITY_SEEDS:
            for task in EVALUATION_TASKS:
                product_norm = by_arm_seed[("P", seed)]["evaluation"]["cells"][
                    f"{task}:L{length}"
                ]["state_norm_maximum"]
                delta_norm = by_arm_seed[("D", seed)]["evaluation"]["cells"][
                    f"{task}:L{length}"
                ]["state_norm_maximum"]
                comparative[f"D:state_norm:{seed}:{task}:L{length}"] = (
                    delta_norm <= 1.25 * max(product_norm, 1e-12)
                )
    product_passed = all(shared.values()) and all(product.values())
    delta_passed = all(shared.values()) and all(delta.values())
    passed = delta_passed and all(comparative.values())
    if passed and product_passed:
        decision = "both pass; run a separately compiled efficiency comparison"
    elif passed:
        decision = "D passes and P fails; freeze a natural-text identity protocol"
    elif delta_passed:
        decision = "D passes absolute gates but lacks the frozen comparative margin"
    elif product_passed:
        decision = "P passes while D fails; reject coupled residual delta"
    else:
        decision = "both arms fail; stop G15B-D and redesign event-local write targets"
    return {
        "passed": passed,
        "eligible_for_promotion": passed,
        "product_absolute_quality_passed": product_passed,
        "delta_absolute_and_causal_passed": delta_passed,
        "shared_integrity_checks": shared,
        "product_absolute_checks": product,
        "delta_absolute_and_causal_checks": delta,
        "comparative_checks": comparative,
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "quality"), required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-commit")
    parser.add_argument("--allow-dirty", action="store_true")
    arguments = parser.parse_args()
    config = frozen_config(arguments.mode)
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain").splitlines()
    if arguments.expected_commit and commit != arguments.expected_commit:
        raise RuntimeError("HEAD does not match --expected-commit")
    if status and not arguments.allow_dirty:
        raise RuntimeError("G15B-D cohort requires a clean checkout")
    device = torch.device(arguments.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("G15B-D requires the declared CUDA device")
    if torch.cuda.get_device_capability(device) != (7, 5):
        raise RuntimeError("G15B-D requires exact compute capability (7, 5)")
    release = platform.release().lower()
    if platform.system() != "Linux" or not (
        "microsoft" in release or "wsl" in release
    ):
        raise RuntimeError("G15B-D evidentiary execution requires WSL2 Linux")
    preflight = run_preflight(device)
    if not preflight["passed"]:
        raise RuntimeError("G15B-D preflight failed")
    reports = []
    for seed in config.seeds:
        for arm in ARMS:
            reports.append(
                _train_arm(
                    arm,  # type: ignore[arg-type]
                    config,
                    seed=seed,
                    device=device,
                    checkpoint_directory=arguments.checkpoint_dir,
                    experiment_name="G15B-D coupled residual delta",
                    namespace="g15bd",
                    checkpoint_prefix="g15bd",
                )
            )
    adjudication = _enforce_execution_eligibility(
        _adjudicate(config, reports), status
    )
    source_paths = (
        PROTOCOL,
        Path(__file__).resolve(),
        ROOT / "g15be_effective_edit_cohort.py",
        ROOT / "transactional_delta.py",
        ROOT / "model.py",
        ROOT / "optimizers.py",
        ROOT / "g15b_interleaved_tasks.py",
        ROOT / "g15b_interleaved_cohort.py",
        ROOT / "g15bt_transactional_cohort.py",
        ROOT / "g15br2_collision_erase.py",
        ROOT / "g15br3_logical_component.py",
    )
    report = {
        "schema_version": 1,
        "experiment": "G15B-D coupled residual delta",
        "mode": config.mode,
        "evidentiary": not status,
        "git_commit_at_start": commit,
        "git_status_at_start": status,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        },
        "protocol": asdict(config),
        "source_files": {
            str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"): _sha256(path)
            for path in source_paths
        },
        "phase0_artifact_sha256": _sha256(PHASE0_ARTIFACT),
        "preflight": preflight,
        "seed_arm_reports": reports,
        "adjudication": adjudication,
    }
    arguments.artifact.parent.mkdir(parents=True, exist_ok=True)
    arguments.artifact.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(adjudication, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["ARMS", "_adjudicate", "frozen_config", "run_preflight"]
