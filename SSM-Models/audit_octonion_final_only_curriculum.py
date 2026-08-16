"""Post-protocol audit for the final-only composition-depth curriculum."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from audit_octonion_basis_identification import g2_gauge_audit
from benchmark_octonion_associator_tracking import AssociatorConfig, tensor_hash
from benchmark_octonion_basis_transport import (
    LearnedBasisOperatorTracker,
    haar_special_orthogonal,
)
from benchmark_octonion_final_only import terminal_metrics
from benchmark_octonion_final_only_curriculum import (
    make_curriculum_schedules,
    terminal_batch,
)

ROOT = Path(__file__).resolve().parent
FIXED_ARTIFACT = (
    ROOT / "experiments" / "artifacts" / "octonion_final_only_replication1000.json"
)
FIXED_ARTIFACT_SHA256 = (
    "7def2ca25ce7b04f11f282c06c16dea51ccf86d2a18eae1a4db679fa4d9e8f4a"
)
CURRICULUM_ARTIFACT = (
    ROOT / "experiments" / "artifacts" / "octonion_final_only_curriculum1000.json"
)
CURRICULUM_ARTIFACT_SHA256 = (
    "4ddabe92d532361e146ee0c7c156237a30061c0c46299f0cb29e3b4951def322"
)
DEFAULT_OUTPUT = (
    ROOT / "experiments" / "artifacts" / "octonion_final_only_curriculum_audit.json"
)
LONG_LENGTHS = (256, 512, 1024)
LONG_BATCH_SIZE = 32


def now() -> str:
    return datetime.now().astimezone().isoformat()


def checkpoint_path(serialized: str) -> Path:
    path = Path(serialized)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    payload = path.read_bytes()
    return payload, json.loads(payload)


def long_schedules(
    basis_seed: int, basis: torch.Tensor
) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], str]:
    generator = torch.Generator().manual_seed(160_000 + basis_seed)
    schedules = {}
    hash_tensors = [basis]
    for length in LONG_LENGTHS:
        batch = terminal_batch(LONG_BATCH_SIZE, length, basis, generator)
        schedules[length] = batch
        hash_tensors.extend(batch)
    return schedules, tensor_hash(hash_tensors)


@torch.no_grad()
def evaluate_long(
    model: LearnedBasisOperatorTracker,
    schedules: dict[int, tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    model = model.to(device).eval()
    result = {}
    for length, (inputs, targets) in schedules.items():
        predictions = model(inputs.to(device))[:, -1]
        result[str(length)] = terminal_metrics(predictions, targets.to(device))
    return result


def audit_basis(
    curriculum_report: dict[str, Any],
    fixed_report: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    basis_seed = int(curriculum_report["basis_seed"])
    basis = haar_special_orthogonal(basis_seed)
    config = AssociatorConfig(seed=basis_seed, steps=1_000)
    _, _, replay_schedules = make_curriculum_schedules(config, basis)
    schedule_replay = {
        key: replay_schedules[key] == curriculum_report["schedules"][key]
        for key in replay_schedules
    }
    long_data, long_hash = long_schedules(basis_seed, basis)

    checkpoint_rehash = {}
    structured_audits = {}
    for model_seed, record in curriculum_report["learned_basis_operator"].items():
        checkpoint = checkpoint_path(record["checkpoint"])
        actual_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        checkpoint_rehash[f"learned_basis_operator_{model_seed}"] = (
            actual_hash == record["checkpoint_sha256"]
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = LearnedBasisOperatorTracker()
        model.load_state_dict(payload["state_dict"])
        learned_basis = model.basis().detach().double()
        structured_audits[model_seed] = {
            "gauge": g2_gauge_audit(learned_basis, basis.double()),
            "long_evaluation": evaluate_long(model, long_data, device),
        }

    for family_name in ("dense_linear_operator", "reference_results"):
        for model_seed, record in curriculum_report[family_name].items():
            checkpoint = checkpoint_path(record["checkpoint"])
            checkpoint_rehash[f"{family_name}_{model_seed}"] = (
                hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                == record["checkpoint_sha256"]
            )

    fixed_successes = sum(
        record["evaluation"]["128"]["mse"] < 1e-3
        for record in fixed_report["learned_basis_operator"].values()
    )
    curriculum_successes = sum(
        record["evaluation"]["128"]["mse"] < 1e-3
        for record in curriculum_report["learned_basis_operator"].values()
    )
    dense_successes = sum(
        record["evaluation"]["128"]["mse"] < 1e-3
        for record in curriculum_report["dense_linear_operator"].values()
    )
    maximum_g2_residual = max(
        record["gauge"]["g2_left_action_intertwiner_residual"]
        for record in structured_audits.values()
    )
    maximum_l1024_mse = max(
        record["long_evaluation"]["1024"]["mse"]
        for record in structured_audits.values()
    )
    checks = {
        "all_schedule_hashes_replay": all(schedule_replay.values()),
        "all_eight_checkpoints_rehash": all(checkpoint_rehash.values()),
        "fixed_success_count_is_recorded": fixed_successes
        == (3 if basis_seed == 1 else 0),
        "curriculum_structured_success_count_is_3": curriculum_successes == 3,
        "curriculum_dense_success_count_is_3": dense_successes == 3,
        "all_learned_gauges_are_g2_intertwiners": maximum_g2_residual < 1e-3,
        "all_structured_l1024_mse_below_1e_8": maximum_l1024_mse < 1e-8,
    }
    return {
        "basis_seed": basis_seed,
        "schedule_replay": schedule_replay,
        "long_schedule_sha256": long_hash,
        "checkpoint_rehash": checkpoint_rehash,
        "fixed_structured_successes_out_of_3": fixed_successes,
        "curriculum_structured_successes_out_of_3": curriculum_successes,
        "curriculum_dense_successes_out_of_3": dense_successes,
        "maximum_g2_intertwiner_residual": maximum_g2_residual,
        "maximum_structured_l1024_mse": maximum_l1024_mse,
        "structured_audits": structured_audits,
        "checks": checks,
        "all_required_checks_passed": all(checks.values()),
    }


def run() -> dict[str, Any]:
    fixed_bytes, fixed = load_json(FIXED_ARTIFACT)
    curriculum_bytes, curriculum = load_json(CURRICULUM_ARTIFACT)
    fixed_hash = hashlib.sha256(fixed_bytes).hexdigest()
    curriculum_hash = hashlib.sha256(curriculum_bytes).hexdigest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixed_by_seed = {row["basis_seed"]: row for row in fixed["basis_reports"]}
    audits = [
        audit_basis(row, fixed_by_seed[row["basis_seed"]], device)
        for row in curriculum["basis_reports"]
    ]
    source_hashes_match = (
        fixed_hash == FIXED_ARTIFACT_SHA256
        and curriculum_hash == CURRICULUM_ARTIFACT_SHA256
    )
    return {
        "schema_version": 1,
        "experiment": "final-only curriculum replay, G2, and L1024 audit",
        "finished_at": now(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor()
            ),
        },
        "fixed_artifact_sha256": fixed_hash,
        "curriculum_artifact_sha256": curriculum_hash,
        "source_hashes_match_frozen": source_hashes_match,
        "basis_audits": audits,
        "all_required_checks_passed": source_hashes_match
        and all(row["all_required_checks_passed"] for row in audits),
        "interpretation": (
            "The L2-to-L16 terminal-only homotopy changes structured recovery "
            "from 3/9 to 9/9 and dense recovery from 0/9 to 9/9; it improves "
            "optimization for the composition object generally, while the "
            "28-parameter gauge remains materially more accurate."
        ),
        "claim_boundary": (
            "Post-protocol replay and longer-length diagnostic; not a global "
            "optimization theorem or natural-task result."
        ),
    }


def main() -> None:
    report = run()
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_required_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
