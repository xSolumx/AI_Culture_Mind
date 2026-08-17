"""Certify direct gradient identifiability in the adaptive lift-bit cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Any

import benchmark_pure_spin8_continuous_observation as continuous
import benchmark_pure_spin8_lift_bit_calibration as calibration
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCES = tuple(
    ROOT
    / "experiments"
    / "artifacts"
    / f"pure_spin8_lift_bit_calibration_validation_seed{seed}.json"
    for seed in (4, 5, 6)
)
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "artifacts"
    / "pure_spin8_lift_gradient_identifiability_certificate.json"
)
MODE = "vector_plus_adaptive_lift_bit"
INDEPENDENT = "independent_so8_triplet"
SHARED = "shared_pure_spin8"
REPRESENTATIONS = ("vector", "positive", "negative")
DECAY_RESIDUAL_TOLERANCE = 2e-6
DATA_UPDATE_MINIMUM = 1e-5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_from_mapping(mapping: dict[str, Any]) -> continuous.ContinuousObservationConfig:
    allowed = {field.name for field in fields(continuous.ContinuousObservationConfig)}
    values = {key: value for key, value in mapping.items() if key in allowed}
    if "evaluation_lengths" in values:
        values["evaluation_lengths"] = tuple(values["evaluation_lengths"])
    return continuous.ContinuousObservationConfig(**values)


def _first_training_batch(
    config: continuous.ContinuousObservationConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild only the first frozen batch without materializing 2,000 steps."""

    system = continuous.make_observation_system(config.seed)
    coordinate_generator = np.random.default_rng(820_000 + config.seed)
    noise_generator = torch.Generator().manual_seed(830_000 + config.seed)
    coordinates, _ = continuous._sample_coordinate_sequence(
        coordinate_generator,
        config.batch_size,
        config.training_length,
        config,
    )
    observations = continuous.observe_coordinates(
        coordinates,
        system,
        noise_std=config.observation_noise_std,
        generator=noise_generator,
    )
    endpoints = continuous.teacher_outputs(coordinates, device)[:, -1]
    return observations, endpoints


def _gradient_statistics(value: torch.Tensor) -> dict[str, float | int]:
    detached = value.detach().double().cpu()
    return {
        "l2": float(torch.linalg.vector_norm(detached)),
        "max_abs": float(detached.abs().max()),
        "nonzero": int(torch.count_nonzero(detached)),
        "elements": detached.numel(),
    }


def first_batch_gradient_audit(
    *,
    seed: int,
    config: continuous.ContinuousObservationConfig,
    device: torch.device,
) -> dict[str, Any]:
    observations, endpoints = _first_training_batch(config, device)
    results: dict[str, Any] = {}
    for candidate in (SHARED, INDEPENDENT):
        candidate_offset = calibration.CANDIDATES.index(candidate)
        continuous.seed_everything(1_160_000 + 1_000 * seed + candidate_offset)
        model = calibration.FACTORIES[candidate]().to(device)
        predictions = model(observations.to(device))
        loss, components = calibration.mode_loss(MODE, predictions, endpoints)
        loss.backward()
        if candidate == SHARED:
            gradient = model.coordinate_head.weight.grad
            bias_gradient = model.coordinate_head.bias.grad
            if gradient is None or bias_gradient is None:
                raise RuntimeError("shared coordinate head did not receive gradients")
            row_norms = torch.linalg.vector_norm(gradient.detach().double(), dim=-1)
            results[candidate] = {
                "loss": float(loss.detach()),
                "components": {
                    key: float(value.detach()) for key, value in components.items()
                },
                "coordinate_head_weight": _gradient_statistics(gradient),
                "coordinate_head_bias": _gradient_statistics(bias_gradient),
                "coordinate_row_min_l2": float(row_norms.min()),
                "coordinate_rows_nonzero": int(torch.count_nonzero(row_norms)),
            }
        else:
            gradient = model.coordinate_head.weight.grad
            bias_gradient = model.coordinate_head.bias.grad
            trunk_gradient = model.observation_hidden.weight.grad
            if gradient is None or bias_gradient is None or trunk_gradient is None:
                raise RuntimeError("independent router did not produce expected gradients")
            weight_blocks = gradient.reshape(3, 28, gradient.shape[-1])
            bias_blocks = bias_gradient.reshape(3, 28)
            results[candidate] = {
                "loss": float(loss.detach()),
                "components": {
                    key: float(value.detach()) for key, value in components.items()
                },
                "coordinate_head_weight_by_representation": {
                    representation: _gradient_statistics(weight_blocks[index])
                    for index, representation in enumerate(REPRESENTATIONS)
                },
                "coordinate_head_bias_by_representation": {
                    representation: _gradient_statistics(bias_blocks[index])
                    for index, representation in enumerate(REPRESENTATIONS)
                },
                "shared_observation_trunk_weight": _gradient_statistics(
                    trunk_gradient
                ),
            }
        del model, predictions, loss
    independent = results[INDEPENDENT]
    shared = results[SHARED]
    checks = {
        "independent_vector_head_has_data_gradient": (
            independent["coordinate_head_weight_by_representation"]["vector"][
                "nonzero"
            ]
            > 0
        ),
        "independent_positive_head_has_data_gradient": (
            independent["coordinate_head_weight_by_representation"]["positive"][
                "nonzero"
            ]
            > 0
        ),
        "independent_negative_weight_gradient_exactly_zero": (
            independent["coordinate_head_weight_by_representation"]["negative"][
                "nonzero"
            ]
            == 0
        ),
        "independent_negative_bias_gradient_exactly_zero": (
            independent["coordinate_head_bias_by_representation"]["negative"][
                "nonzero"
            ]
            == 0
        ),
        "independent_shared_trunk_has_data_gradient": (
            independent["shared_observation_trunk_weight"]["nonzero"] > 0
        ),
        "shared_all_28_coordinate_rows_have_data_gradient": (
            shared["coordinate_rows_nonzero"] == 28
            and shared["coordinate_row_min_l2"] > 0.0
        ),
    }
    return {"seed": seed, "results": results, "checks": checks, "passed": all(checks.values())}


def _repeated_adamw_decay(
    value: torch.Tensor, *, learning_rate: float, weight_decay: float, steps: int
) -> torch.Tensor:
    result = value.detach().clone()
    factor = 1.0 - learning_rate * weight_decay
    for _ in range(steps):
        result.mul_(factor)
    return result


def checkpoint_decay_audit(source: Path) -> dict[str, Any]:
    source_payload = json.loads(source.read_text(encoding="utf-8"))
    seed = int(source_payload["config"]["seed"])
    result = source_payload["results"][MODE][INDEPENDENT]
    checkpoint = Path(result["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    checkpoint_digest = _sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = _config_from_mapping(payload["config"])

    candidate_offset = calibration.CANDIDATES.index(INDEPENDENT)
    continuous.seed_everything(1_160_000 + 1_000 * seed + candidate_offset)
    initial = calibration.FACTORIES[INDEPENDENT]()
    initial_weight = initial.coordinate_head.weight.detach().reshape(3, 28, 9)
    initial_bias = initial.coordinate_head.bias.detach().reshape(3, 28)
    final_weight = payload["state_dict"]["coordinate_head.weight"].reshape(3, 28, 9)
    final_bias = payload["state_dict"]["coordinate_head.bias"].reshape(3, 28)
    expected_weight = _repeated_adamw_decay(
        initial_weight,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        steps=config.steps,
    )
    expected_bias = _repeated_adamw_decay(
        initial_bias,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        steps=config.steps,
    )

    residuals = {
        representation: {
            "weight_max_abs_from_decay_only": float(
                (final_weight[index] - expected_weight[index]).abs().max()
            ),
            "bias_max_abs_from_decay_only": float(
                (final_bias[index] - expected_bias[index]).abs().max()
            ),
            "weight_max_abs_change_from_initial": float(
                (final_weight[index] - initial_weight[index]).abs().max()
            ),
        }
        for index, representation in enumerate(REPRESENTATIONS)
    }
    checks = {
        "source_checkpoint_hash_matches": checkpoint_digest
        == result["checkpoint_sha256"],
        "checkpoint_candidate_matches": payload["candidate"] == INDEPENDENT,
        "checkpoint_mode_matches": payload["mode"] == MODE,
        "checkpoint_seed_matches": config.seed == seed,
        "negative_weight_is_decay_only": (
            residuals["negative"]["weight_max_abs_from_decay_only"]
            <= DECAY_RESIDUAL_TOLERANCE
        ),
        "negative_bias_is_decay_only": (
            residuals["negative"]["bias_max_abs_from_decay_only"]
            <= DECAY_RESIDUAL_TOLERANCE
        ),
        "vector_weight_has_data_update": (
            residuals["vector"]["weight_max_abs_from_decay_only"]
            > DATA_UPDATE_MINIMUM
        ),
        "positive_weight_has_data_update": (
            residuals["positive"]["weight_max_abs_from_decay_only"]
            > DATA_UPDATE_MINIMUM
        ),
    }
    return {
        "seed": seed,
        "source": str(source),
        "source_sha256": _sha256(source),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "optimizer": {
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "steps": config.steps,
            "nominal_decay_factor": math.pow(
                1.0 - config.learning_rate * config.weight_decay, config.steps
            ),
        },
        "residuals": residuals,
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_certificate(sources: tuple[Path, ...], device: torch.device) -> dict[str, Any]:
    if len(sources) != 3:
        raise ValueError("exactly three frozen seed sources are required")
    checkpoint_audits = [checkpoint_decay_audit(source) for source in sources]
    seeds = [audit["seed"] for audit in checkpoint_audits]
    gradient_audits = []
    for source, seed in zip(sources, seeds, strict=True):
        payload = json.loads(source.read_text(encoding="utf-8"))
        config = _config_from_mapping(payload["config"])
        gradient_audits.append(
            first_batch_gradient_audit(seed=seed, config=config, device=device)
        )
    global_checks = {
        "fresh_seeds_exact": seeds == [4, 5, 6],
        "all_checkpoint_decay_audits_pass": all(
            audit["passed"] for audit in checkpoint_audits
        ),
        "all_first_batch_gradient_audits_pass": all(
            audit["passed"] for audit in gradient_audits
        ),
    }
    return {
        "schema_version": 1,
        "experiment": "Pure Spin8 adaptive lift gradient identifiability certificate",
        "status": "certified",
        "recorded_at": continuous.now(),
        "mode": MODE,
        "seeds": seeds,
        "structural_statement": {
            "independent_negative_specific_head": (
                "zero data gradient; AdamW decay only over all 2,000 updates"
            ),
            "independent_shared_trunk": (
                "receives data gradient, so negative predictions are not frozen"
            ),
            "shared_spin8_head": (
                "all 28 common bivector-coordinate rows receive data gradient"
            ),
        },
        "global_checks": global_checks,
        "checkpoint_decay_audits": checkpoint_audits,
        "first_batch_gradient_audits": gradient_audits,
        "passed": all(global_checks.values()),
        "claim_boundary": (
            "This certifies direct parameter identifiability under the frozen loss. "
            "It does not prove a global optimizer theorem or prevent indirect "
            "negative-output changes through the independent shared trunk."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="*", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = tuple(path.resolve() for path in args.sources)
    result = run_certificate(sources, torch.device(args.device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"global_checks": result["global_checks"], "passed": result["passed"]}, indent=2))
    print(f"wrote {args.output}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
