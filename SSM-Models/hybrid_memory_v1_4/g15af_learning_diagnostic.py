"""Posthoc chart-error decomposition for the failed G15A-F cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
from spin8_triality import SPIN8_PAIRS

if __package__:
    from .g15a_spin_dirac_cohort import (
        OFF_TORUS_PAIRS,
        _atomic_json,
        _now,
        _oracle_memory,
        _sha256,
        _stable_seed,
    )
    from .g15af_full_frame_cohort import (
        _frame_prediction,
        _metrics,
        _teacher_target,
        generate_frame_batch,
    )
    from .g15al_learned_coordinate_cohort import (
        ACTION_ANGLE,
        ACTION_VOCABULARY,
        MAXIMUM_COORDINATE,
        VOCABULARY_SIZE,
        _token_map,
    )
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4.g15a_spin_dirac_cohort import (  # type: ignore[no-redef]
        OFF_TORUS_PAIRS,
        _atomic_json,
        _now,
        _oracle_memory,
        _sha256,
        _stable_seed,
    )
    from hybrid_memory_v1_4.g15af_full_frame_cohort import (  # type: ignore[no-redef]
        _frame_prediction,
        _metrics,
        _teacher_target,
        generate_frame_batch,
    )
    from hybrid_memory_v1_4.g15al_learned_coordinate_cohort import (  # type: ignore[no-redef]
        ACTION_ANGLE,
        ACTION_VOCABULARY,
        MAXIMUM_COORDINATE,
        VOCABULARY_SIZE,
        _token_map,
    )


INPUT_ARTIFACT_SHA256 = (
    "cdfdcb1785e2bf2a85ea592e2100a61596d1a06ea219a9d75c058f1d11e74296"
)


def _load(path: Path) -> dict[str, Any]:
    if _sha256(path) != INPUT_ARTIFACT_SHA256:
        raise RuntimeError("G15A-F artifact does not match the bound hash")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("evidentiary") is not True
        or report.get("adjudication", {}).get("passed") is not False
    ):
        raise RuntimeError("input must be the failed evidentiary G15A-F cohort")
    return report


def _expected_table(seed: int) -> torch.Tensor:
    expected = torch.zeros(VOCABULARY_SIZE, len(SPIN8_PAIRS))
    mapping = _token_map(seed)
    for semantic in range(ACTION_VOCABULARY):
        token = int(mapping[semantic])
        pair = OFF_TORUS_PAIRS[semantic % len(OFF_TORUS_PAIRS)]
        coordinate = SPIN8_PAIRS.index(pair)
        sign = 1.0 if semantic < len(OFF_TORUS_PAIRS) else -1.0
        expected[token, coordinate] = sign * ACTION_ANGLE
    return expected


def _table_variants(
    raw_coordinates: list[list[float]], seed: int
) -> dict[str, torch.Tensor]:
    learned = MAXIMUM_COORDINATE * torch.tanh(torch.tensor(raw_coordinates))
    expected = _expected_table(seed)
    support = expected != 0
    return {
        "learned": learned,
        "active_only_learned_amplitude": torch.where(
            support, learned, torch.zeros_like(learned)
        ),
        "exact_amplitude_with_learned_leakage": torch.where(support, expected, learned),
        "oracle_exact": expected,
    }


def _chart_diagnostics(variants: dict[str, torch.Tensor]) -> dict[str, float]:
    learned = variants["learned"]
    expected = variants["oracle_exact"]
    support = expected != 0
    active_error = learned[support] - expected[support]
    inactive = learned[(~support) & (torch.arange(VOCABULARY_SIZE)[:, None] != 0)]
    return {
        "active_coordinate_mae": float(active_error.abs().mean()),
        "active_coordinate_max_abs_error": float(active_error.abs().max()),
        "inactive_coordinate_rms": float(inactive.square().mean().sqrt()),
        "inactive_coordinate_max_abs": float(inactive.abs().max()),
        "full_active_token_chart_rmse": float(
            (learned[1:] - expected[1:]).square().mean().sqrt()
        ),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    relative = torch.tensor(
        [value for row in rows for value in row["relative_frobenius_errors"]],
        dtype=torch.float64,
    )
    return {
        "mean_relative_frobenius_error": float(relative.mean()),
        "p95_relative_frobenius_error": float(torch.quantile(relative, 0.95)),
        "maximum_relative_frobenius_error": float(relative.max()),
        "raw_elementwise_mse": sum(row["raw_elementwise_mse"] for row in rows)
        / len(rows),
        "mean_matrix_cosine": sum(row["mean_matrix_cosine"] for row in rows)
        / len(rows),
        "minimum_matrix_cosine": min(row["minimum_matrix_cosine"] for row in rows),
    }


@torch.no_grad()
def run(report: dict[str, Any], *, device: torch.device) -> dict[str, Any]:
    started = time.perf_counter()
    diagnostics = []
    for seed_report in report["seed_reports"]:
        seed = int(seed_report["seed"])
        variants = _table_variants(
            seed_report["arms"]["S"]["learned_raw_coordinates"], seed
        )
        memory = _oracle_memory("S", dtype=torch.float32, device=device)
        teacher = _oracle_memory("S", dtype=torch.float32, device=device)
        per_length = {}
        for length, actions in ((64, 8), (256, 12), (1024, 16)):
            variant_rows: dict[str, list[dict[str, Any]]] = {
                name: [] for name in variants
            }
            for offset in range(0, 80, 8):
                batch = generate_frame_batch(
                    8,
                    length,
                    seed=_stable_seed("g15af-eval", seed, length, offset),
                    model_seed=seed,
                    minimum_actions=actions,
                    maximum_actions=actions,
                ).to(device)
                target = _teacher_target(teacher, batch, device=device)
                for name, table in variants.items():
                    coordinates = table.to(device)[batch.token_ids].unsqueeze(2)
                    prediction = _frame_prediction(
                        memory, batch, coordinates, device=device
                    )
                    variant_rows[name].append(_metrics(prediction, target))
            per_length[str(length)] = {
                name: _aggregate(rows) for name, rows in variant_rows.items()
            }
        diagnostics.append(
            {
                "seed": seed,
                "chart": _chart_diagnostics(variants),
                "training_loss_samples": seed_report["arms"]["S"]["loss_samples"],
                "per_length": per_length,
            }
        )
    return {
        "schema_version": 1,
        "experiment": "G15A-F posthoc learned-chart error decomposition",
        "claim_status": "descriptive posthoc diagnostic; no promotion gate",
        "input_artifact_sha256": INPUT_ARTIFACT_SHA256,
        "started_at": _now(),
        "elapsed_wall_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        },
        "seed_diagnostics": diagnostics,
        "analytic_observation": (
            "active-only keeps learned target-plane amplitudes and removes all "
            "inactive coordinates; exact-amplitude-with-leakage does the converse"
        ),
        "explicit_nonclaims": [
            "the decompositions use oracle primitive support after the frozen failure",
            "posthoc ablations do not change the G15A-F decision",
            "the diagnostic does not validate a new optimizer or curriculum",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = _load(args.input)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the local diagnostic requires CUDA")
    if torch.cuda.get_device_capability(device) != (7, 5):
        raise RuntimeError("the local diagnostic requires exact SM75")
    result = run(report, device=device)
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    _atomic_json(args.output, result)
    print(args.output)
    print(json.dumps(result["seed_diagnostics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
