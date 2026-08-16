"""Post-protocol identification and G2-gauge audit for the Haar-basis cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from benchmark_octonion_associator_tracking import AssociatorConfig, metrics
from benchmark_octonion_basis_transport import (
    BASIS_SEEDS,
    LearnedBasisOperatorTracker,
    haar_special_orthogonal,
    make_transported_schedules,
)
from pure_rotor_ssm.octonion_operator_scan import (
    OCTONION_DIM,
    associative_matrix_prefix_scan,
    octonion_left_multiplication_matrix,
)

SOURCE_ARTIFACT = (
    Path(__file__).resolve().parent
    / "experiments"
    / "artifacts"
    / "octonion_basis_transport_replication300.json"
)
SOURCE_ARTIFACT_SHA256 = (
    "b96bed5d0e4c33e229816f6ce2db24d2c42c5b33ae1b978950a2a0bd9960daf5"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "experiments"
    / "artifacts"
    / "octonion_basis_identification_audit.json"
)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def checkpoint_path(serialized: str) -> Path:
    path = Path(serialized)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def dense_identification(
    training: list[tuple[torch.Tensor, torch.Tensor]],
    evaluations: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, Any]:
    design = torch.cat([inputs[:, 0] for inputs, _ in training]).double()
    response = torch.cat([targets[:, 0] for _, targets in training]).double()
    solution = torch.linalg.lstsq(design, response).solution
    singular_values = torch.linalg.svdvals(design)
    evaluation = {}
    for length, (inputs, targets) in evaluations.items():
        leaves = (inputs.double() @ solution).reshape(
            inputs.shape[0], length, OCTONION_DIM, OCTONION_DIM
        )
        predictions = associative_matrix_prefix_scan(
            leaves, backend="work_efficient"
        ).flatten(start_dim=-2)
        evaluation[str(length)] = metrics(predictions, targets.double())
    return {
        "method": "float64 least squares on legal position-1 training targets",
        "design_rows": int(design.shape[0]),
        "design_rank": int(torch.linalg.matrix_rank(design)),
        "design_condition_number": float(singular_values.max() / singular_values.min()),
        "leaf_training_mse": float(torch.mean((design @ solution - response) ** 2)),
        "evaluation": evaluation,
        "identified_weight_sha256": hashlib.sha256(
            solution.contiguous().numpy().tobytes()
        ).hexdigest(),
    }


def g2_gauge_audit(
    learned_basis: torch.Tensor, true_basis: torch.Tensor
) -> dict[str, float]:
    gauge = learned_basis.T @ true_basis
    canonical_tokens = torch.eye(OCTONION_DIM, dtype=gauge.dtype)
    canonical_actions = octonion_left_multiplication_matrix(canonical_tokens)
    transported_tokens = torch.einsum("ij,tj->ti", gauge, canonical_tokens)
    transported_actions = octonion_left_multiplication_matrix(transported_tokens)
    conjugated_actions = torch.einsum("ij,tjk,lk->til", gauge, canonical_actions, gauge)
    return {
        "gauge_determinant": float(torch.linalg.det(gauge)),
        "gauge_orthogonality_residual": float(
            (gauge.T @ gauge - torch.eye(OCTONION_DIM)).abs().max()
        ),
        "gauge_identity_residual": float(
            (gauge[:, 0] - canonical_tokens[0]).abs().max()
        ),
        "g2_left_action_intertwiner_residual": float(
            (conjugated_actions - transported_actions).abs().max()
        ),
    }


def audit_basis(source_report: dict[str, Any]) -> dict[str, Any]:
    seed = int(source_report["basis_seed"])
    config = AssociatorConfig(seed=seed, steps=source_report["config"]["steps"])
    basis = haar_special_orthogonal(seed)
    training, evaluations, schedules = make_transported_schedules(config, basis)
    schedule_replay = {
        key: schedules[key] == source_report["schedules"][key] for key in schedules
    }

    learned_record = source_report["results"]["learned_basis_operator"]
    learned_checkpoint = checkpoint_path(learned_record["checkpoint"])
    learned_payload = torch.load(
        learned_checkpoint, map_location="cpu", weights_only=False
    )
    learned_model = LearnedBasisOperatorTracker()
    learned_model.load_state_dict(learned_payload["state_dict"])
    learned_basis = learned_model.basis().detach().double()

    checkpoints_rehash = {}
    for name, record in source_report["results"].items():
        if "checkpoint" not in record:
            continue
        path = checkpoint_path(record["checkpoint"])
        checkpoints_rehash[name] = (
            path.is_file()
            and hashlib.sha256(path.read_bytes()).hexdigest()
            == record["checkpoint_sha256"]
        )

    identified = dense_identification(training, evaluations)
    gauge = g2_gauge_audit(learned_basis, basis.double())
    checks = {
        "all_schedule_hashes_replay": all(schedule_replay.values()),
        "all_learned_checkpoints_rehash": all(checkpoints_rehash.values()),
        "dense_design_has_rank_8": identified["design_rank"] == OCTONION_DIM,
        "identified_dense_l128_below_1e_10": identified["evaluation"]["128"]["mse"]
        < 1e-10,
        "learned_gauge_is_orthogonal": gauge["gauge_orthogonality_residual"] < 1e-5,
        "learned_gauge_fixes_identity": gauge["gauge_identity_residual"] < 1e-3,
        "learned_gauge_is_g2_intertwiner": gauge["g2_left_action_intertwiner_residual"]
        < 1e-3,
    }
    return {
        "basis_seed": seed,
        "schedule_replay": schedule_replay,
        "checkpoint_rehash": checkpoints_rehash,
        "identified_dense_operator": identified,
        "learned_gauge": gauge,
        "checks": checks,
        "all_required_checks_passed": all(checks.values()),
    }


def run() -> dict[str, Any]:
    source_bytes = SOURCE_ARTIFACT.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    source = json.loads(source_bytes)
    reports_by_seed = {
        int(report["basis_seed"]): report for report in source["basis_reports"]
    }
    basis_audits = [audit_basis(reports_by_seed[seed]) for seed in BASIS_SEEDS]
    source_hash_matches = source_hash == SOURCE_ARTIFACT_SHA256
    return {
        "schema_version": 1,
        "experiment": "post-protocol dense identification and G2 gauge audit",
        "finished_at": now(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu float64 identification and audit",
        },
        "source_artifact": str(
            SOURCE_ARTIFACT.relative_to(Path(__file__).resolve().parent)
        ),
        "source_artifact_sha256": source_hash,
        "source_artifact_hash_matches_frozen": source_hash_matches,
        "basis_audits": basis_audits,
        "all_required_checks_passed": source_hash_matches
        and all(report["all_required_checks_passed"] for report in basis_audits),
        "interpretation": (
            "The frozen dense AdamW failure is optimization, not realizability: "
            "legal position-1 prefix supervision identifies the unrestricted "
            "linear leaf map. The 28-parameter model instead learns the hidden "
            "basis up to an approximately G2 automorphism under the same blind "
            "gradient schedule."
        ),
        "claim_boundary": (
            "Post-protocol diagnostic using legal training prefixes; not a "
            "preregistered learned-baseline win or natural-task result."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_required_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
