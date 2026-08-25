"""Posthoc observability audit for the failed G15A-L quality cohort."""

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
from torch.nn import functional as F

if __package__:
    from .g15a_spin_dirac_cohort import (
        OFF_TORUS_PAIRS,
        _atomic_json,
        _now,
        _oracle_memory,
        _sha256,
        _stable_seed,
    )
    from .g15al_learned_coordinate_cohort import (
        MAXIMUM_COORDINATE,
        TokenCoordinateController,
        _carrier_totals,
        _event_sparse_prediction,
        _teacher_target,
        _token_map,
        generate_batch,
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
    from hybrid_memory_v1_4.g15al_learned_coordinate_cohort import (  # type: ignore[no-redef]
        MAXIMUM_COORDINATE,
        TokenCoordinateController,
        _carrier_totals,
        _event_sparse_prediction,
        _teacher_target,
        _token_map,
        generate_batch,
    )


INPUT_ARTIFACT_SHA256 = (
    "7716b75e43964d479bd5fef0cfbd06d0328a315c00a4a0fa107b26f785af1108"
)


def _load(path: Path) -> dict[str, Any]:
    if _sha256(path) != INPUT_ARTIFACT_SHA256:
        raise RuntimeError("G15A-L artifact does not match the bound hash")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("evidentiary") is not True
        or report.get("adjudication", {}).get("passed") is not False
    ):
        raise RuntimeError("input must be the failed evidentiary G15A-L cohort")
    return report


def _restore(raw: list[list[float]], device: torch.device) -> TokenCoordinateController:
    controller = TokenCoordinateController().to(device)
    with torch.no_grad():
        controller.raw_coordinates.copy_(torch.tensor(raw, device=device))
    controller.eval()
    return controller


def _chart_diagnostics(
    seed: int,
    spin: TokenCoordinateController,
    broken: TokenCoordinateController,
    broken_memory: Any,
) -> dict[str, float]:
    spin_coordinates = MAXIMUM_COORDINATE * torch.tanh(
        spin.raw_coordinates.detach().cpu()
    )
    broken_coordinates = MAXIMUM_COORDINATE * torch.tanh(
        broken.raw_coordinates.detach().cpu()
    )
    effective_broken = (
        broken_coordinates.index_select(
            -1, broken_memory.broken_coordinate_permutation.cpu()
        )
        * broken_memory.broken_coordinate_signs.cpu()
    )
    expected = torch.zeros_like(spin_coordinates)
    mapping = _token_map(seed)
    target_errors = []
    leakage = []
    for semantic in range(16):
        token = int(mapping[semantic])
        pair = OFF_TORUS_PAIRS[semantic % len(OFF_TORUS_PAIRS)]
        coordinate = SPIN8_PAIRS.index(pair)
        target = 0.12 if semantic < len(OFF_TORUS_PAIRS) else -0.12
        expected[token, coordinate] = target
        target_errors.append(float(spin_coordinates[token, coordinate] - target))
        mask = torch.ones(len(SPIN8_PAIRS), dtype=torch.bool)
        mask[coordinate] = False
        leakage.extend(spin_coordinates[token, mask].tolist())
    return {
        "spin_vs_broken_effective_chart_max_abs": float(
            (spin_coordinates[1:] - effective_broken[1:]).abs().max()
        ),
        "spin_exact_active_coordinate_mae": sum(map(abs, target_errors))
        / len(target_errors),
        "spin_exact_active_coordinate_max_abs": max(map(abs, target_errors)),
        "spin_inactive_coordinate_rms": (
            sum(value * value for value in leakage) / len(leakage)
        )
        ** 0.5,
        "spin_full_chart_rmse": float(
            F.mse_loss(spin_coordinates[1:], expected[1:]).sqrt()
        ),
    }


@torch.no_grad()
def run(report: dict[str, Any], *, device: torch.device) -> dict[str, Any]:
    started = time.perf_counter()
    rows = []
    for seed_report in report["seed_reports"]:
        seed = int(seed_report["seed"])
        spin = _restore(seed_report["arms"]["S"]["learned_raw_coordinates"], device)
        broken = _restore(
            seed_report["arms"]["S-broken"]["learned_raw_coordinates"], device
        )
        spin_memory = _oracle_memory("S", dtype=torch.float32, device=device)
        broken_memory = _oracle_memory("S-broken", dtype=torch.float32, device=device)
        teacher = _oracle_memory("S", dtype=torch.float32, device=device)
        chart = _chart_diagnostics(seed, spin, broken, broken_memory)
        lengths = {}
        for length, actions in ((64, 8), (256, 12), (1024, 16)):
            accumulators: dict[str, list[float]] = {
                "spin_relative_l2": [],
                "broken_relative_l2": [],
                "spin_alignment_ratio": [],
                "broken_alignment_ratio": [],
                "spin_broken_prediction_max_abs": [],
                "spin_broken_cosine": [],
            }
            for offset in range(0, 80, 8):
                batch = generate_batch(
                    8,
                    length,
                    seed=_stable_seed("g15al-eval", seed, length, offset),
                    model_seed=seed,
                    minimum_actions=actions,
                    maximum_actions=actions,
                ).to(device)
                exact_query, target = _teacher_target(teacher, batch, device=device)
                spin_coordinates = spin(batch.token_ids)
                broken_coordinates = broken(batch.token_ids)
                spin_prediction = _event_sparse_prediction(
                    spin_memory,
                    batch,
                    spin_coordinates,
                    exact_query,
                    device=device,
                )
                broken_prediction = _event_sparse_prediction(
                    broken_memory,
                    batch,
                    broken_coordinates,
                    exact_query,
                    device=device,
                )
                spin_vector, _ = _carrier_totals(
                    spin_memory, batch, spin_coordinates, device=device
                )
                broken_vector, _ = _carrier_totals(
                    broken_memory, batch, broken_coordinates, device=device
                )
                denominator = exact_query.square().sum(dim=-1)
                for name, prediction, vector in (
                    ("spin", spin_prediction, spin_vector),
                    ("broken", broken_prediction, broken_vector),
                ):
                    relative = torch.linalg.vector_norm(
                        prediction - target, dim=-1
                    ) / torch.linalg.vector_norm(target, dim=-1)
                    transported_key = torch.einsum("bij,bj->bi", vector, batch.keys)
                    alignment = (exact_query * transported_key).sum(
                        dim=-1
                    ) / denominator
                    accumulators[f"{name}_relative_l2"].extend(relative.cpu().tolist())
                    accumulators[f"{name}_alignment_ratio"].extend(
                        alignment.cpu().tolist()
                    )
                accumulators["spin_broken_prediction_max_abs"].append(
                    float((spin_prediction - broken_prediction).abs().max())
                )
                cosine = (
                    F.normalize(spin_prediction, dim=-1)
                    * F.normalize(broken_prediction, dim=-1)
                ).sum(dim=-1)
                accumulators["spin_broken_cosine"].extend(cosine.cpu().tolist())
            lengths[str(length)] = {
                "spin_mean_relative_l2": sum(accumulators["spin_relative_l2"]) / 80,
                "broken_mean_relative_l2": sum(accumulators["broken_relative_l2"]) / 80,
                "spin_alignment_ratio_mean": sum(accumulators["spin_alignment_ratio"])
                / 80,
                "spin_alignment_ratio_minimum": min(
                    accumulators["spin_alignment_ratio"]
                ),
                "broken_alignment_ratio_mean": sum(
                    accumulators["broken_alignment_ratio"]
                )
                / 80,
                "broken_alignment_ratio_minimum": min(
                    accumulators["broken_alignment_ratio"]
                ),
                "spin_broken_prediction_max_abs": max(
                    accumulators["spin_broken_prediction_max_abs"]
                ),
                "spin_broken_cosine_minimum": min(accumulators["spin_broken_cosine"]),
            }
        rows.append({"seed": seed, "chart": chart, "per_length": lengths})
    return {
        "schema_version": 1,
        "experiment": "G15A-L posthoc observability diagnostic",
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
        "seed_diagnostics": rows,
        "analytic_observation": (
            "cosine scoring cancels the positive vector-side alignment scalar; "
            "the signed coordinate permutation is invertible for the positive carrier"
        ),
        "explicit_nonclaims": [
            "posthoc diagnostics do not change the frozen G15A-L failure",
            "chart equivalence here is under the tested observation and controller",
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
