"""Resolve the signed-G2 parity ambiguity in the final-only curriculum."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from benchmark_octonion_basis_transport import (
    LearnedBasisOperatorTracker,
    haar_special_orthogonal,
)
from benchmark_octonion_final_only_curriculum import terminal_batch
from pure_rotor_ssm.octonion_operator_scan import (
    OCTONION_DIM,
    octonion_left_multiplication_matrix,
)
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
SOURCE_ARTIFACT = (
    ROOT / "experiments" / "artifacts" / "octonion_final_only_curriculum1000.json"
)
SOURCE_ARTIFACT_SHA256 = (
    "4ddabe92d532361e146ee0c7c156237a30061c0c46299f0cb29e3b4951def322"
)
DEFAULT_OUTPUT = (
    ROOT / "experiments" / "artifacts" / "octonion_final_only_parity_gauge_audit.json"
)
ODD_LENGTH = 17
ODD_BATCH_SIZE = 128


def now() -> str:
    return datetime.now().astimezone().isoformat()


def checkpoint_path(serialized: str) -> Path:
    path = Path(serialized)
    return path if path.is_absolute() else ROOT / path


def signed_gauge_audit(
    learned_basis: torch.Tensor, true_basis: torch.Tensor
) -> dict[str, float | int]:
    """Test whether the recovered gauge belongs to G2 or its negative coset."""
    gauge = learned_basis.T @ true_basis
    identity = torch.zeros(OCTONION_DIM, dtype=gauge.dtype)
    identity[0] = 1
    identity_overlap = torch.dot(gauge @ identity, identity)
    sign = 1 if identity_overlap >= 0 else -1

    canonical_tokens = torch.eye(OCTONION_DIM, dtype=gauge.dtype)
    canonical_actions = octonion_left_multiplication_matrix(canonical_tokens)
    transported_tokens = torch.einsum("ij,tj->ti", gauge, canonical_tokens)
    transported_actions = octonion_left_multiplication_matrix(transported_tokens)
    conjugated_actions = torch.einsum("ij,tjk,lk->til", gauge, canonical_actions, gauge)
    signed_residual = conjugated_actions - sign * transported_actions
    unsigned_residual = conjugated_actions - transported_actions
    return {
        "parity_sign": sign,
        "identity_overlap": float(identity_overlap),
        "signed_identity_residual": float(
            (gauge @ identity - sign * identity).abs().max()
        ),
        "signed_left_action_intertwiner_residual": float(signed_residual.abs().max()),
        "unsigned_left_action_intertwiner_residual": float(
            unsigned_residual.abs().max()
        ),
        "orthogonality_residual": float(
            (gauge.T @ gauge - torch.eye(OCTONION_DIM, dtype=gauge.dtype)).abs().max()
        ),
        "determinant": float(torch.linalg.det(gauge)),
    }


@torch.no_grad()
def odd_length_audit(
    model: LearnedBasisOperatorTracker,
    basis: torch.Tensor,
    basis_seed: int,
    model_seed: int,
    sign: int,
    device: torch.device,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(190_000 + 1_000 * basis_seed + model_seed)
    inputs, targets = terminal_batch(ODD_BATCH_SIZE, ODD_LENGTH, basis, generator)
    predictions = model.to(device).eval()(inputs.to(device))[:, -1]
    targets = targets.to(device)
    signed_targets = (sign**ODD_LENGTH) * targets
    return {
        "ordinary_mse": float(F.mse_loss(predictions, targets)),
        "parity_corrected_mse": float(F.mse_loss(predictions, signed_targets)),
        "maximum_parity_corrected_absolute_error": float(
            (predictions - signed_targets).abs().max()
        ),
    }


def audit_model(
    basis_seed: int,
    basis: torch.Tensor,
    model_seed: int,
    record: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = checkpoint_path(record["checkpoint"])
    actual_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = LearnedBasisOperatorTracker()
    model.load_state_dict(payload["state_dict"])
    gauge = signed_gauge_audit(model.basis().detach().double(), basis.double())
    odd = odd_length_audit(
        model,
        basis,
        basis_seed,
        model_seed,
        int(gauge["parity_sign"]),
        device,
    )
    sign = int(gauge["parity_sign"])
    checks = {
        "checkpoint_rehashes": actual_hash == record["checkpoint_sha256"],
        "gauge_is_orthogonal": gauge["orthogonality_residual"] < 1e-5,
        "gauge_is_in_signed_g2": gauge["signed_identity_residual"] < 1e-3
        and gauge["signed_left_action_intertwiner_residual"] < 1e-3,
        "odd_length_obeys_predicted_sign": odd["parity_corrected_mse"] < 1e-8,
        "ordinary_odd_length_distinguishes_cosets": (
            odd["ordinary_mse"] < 1e-8 if sign == 1 else odd["ordinary_mse"] > 0.49
        ),
    }
    return {
        "model_seed": model_seed,
        "checkpoint_sha256": actual_hash,
        "gauge": gauge,
        "odd_length_evaluation": odd,
        "checks": checks,
        "all_required_checks_passed": all(checks.values()),
    }


def run() -> dict[str, Any]:
    source_bytes = SOURCE_ARTIFACT.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    source = json.loads(source_bytes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    basis_audits = []
    for basis_report in source["basis_reports"]:
        basis_seed = int(basis_report["basis_seed"])
        basis = haar_special_orthogonal(basis_seed)
        models = [
            audit_model(
                basis_seed,
                basis,
                int(model_seed),
                record,
                device,
            )
            for model_seed, record in basis_report["learned_basis_operator"].items()
        ]
        basis_audits.append(
            {
                "basis_seed": basis_seed,
                "positive_coset_models": sum(
                    row["gauge"]["parity_sign"] == 1 for row in models
                ),
                "negative_coset_models": sum(
                    row["gauge"]["parity_sign"] == -1 for row in models
                ),
                "models": models,
                "all_required_checks_passed": all(
                    row["all_required_checks_passed"] for row in models
                ),
            }
        )

    positive = sum(row["positive_coset_models"] for row in basis_audits)
    negative = sum(row["negative_coset_models"] for row in basis_audits)
    source_hash_matches = source_hash == SOURCE_ARTIFACT_SHA256
    return {
        "schema_version": 1,
        "experiment": "signed-G2 parity audit for terminal-only curriculum",
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
        "source_artifact_sha256": source_hash,
        "source_artifact_hash_matches_frozen": source_hash_matches,
        "training_length_parity": "all frozen curriculum lengths are even",
        "odd_length_diagnostic": {
            "length": ODD_LENGTH,
            "batch_size": ODD_BATCH_SIZE,
        },
        "positive_g2_coset_models": positive,
        "negative_g2_coset_models": negative,
        "basis_audits": basis_audits,
        "all_required_checks_passed": source_hash_matches
        and positive > 0
        and negative > 0
        and all(row["all_required_checks_passed"] for row in basis_audits),
        "interpretation": (
            "Even-only terminal supervision identifies the octonion law only up "
            "to the two cosets G2 and -G2. A negative-coset leaf contributes one "
            "global minus sign, which cancels at every even training and evaluation "
            "length but is exposed exactly by the held-out odd length."
        ),
        "claim_boundary": (
            "Post-protocol identifiability audit; it explains the failed unsigned "
            "G2 check and does not convert curriculum convergence into a global "
            "optimization theorem."
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
