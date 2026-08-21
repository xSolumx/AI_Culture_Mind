"""Matched shared-latent control with trainable scrambled Spin(8) alignments."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any

import benchmark_pure_spin8_continuous_observation as continuous
import benchmark_pure_spin8_endpoint_observability as observability
import benchmark_pure_spin8_endpoint_supervision as endpoint
import benchmark_pure_spin8_lift_bit_calibration as calibration
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
SCRAMBLED = "shared_latent_scrambled_alignment"
SHARED = "shared_pure_spin8"
CANDIDATES = (SHARED, SCRAMBLED)
MODES = ("vector_plus_adaptive_lift_bit", "full_triality")
ALIGNMENT_INITIALIZATION_STD = 0.35
SEED_BASE = 1_260_000
PROTOCOL_FROZEN_AT = "2026-08-17T08:40:40.5270068+02:00"
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "artifacts"
    / "pure_spin8_scrambled_alignment_development_seed0.json"
)
DEFAULT_CHECKPOINT_DIRECTORY = (
    ROOT / "checkpoints" / "pure_spin8_scrambled_alignment_development"
)


class ScrambledSharedLatentSO8TripletTracker(nn.Module):
    """One common bivector router with independently conjugated spinor views."""

    recurrent_state_scalars = 24

    def __init__(self, alignment_std: float = ALIGNMENT_INITIALIZATION_STD) -> None:
        super().__init__()
        if alignment_std < 0:
            raise ValueError("alignment_std must be nonnegative")
        self.router = continuous.SharedPureSpin8Tracker()
        self.spinor_alignment_coordinates = nn.Parameter(torch.empty(2, 28))
        nn.init.normal_(self.spinor_alignment_coordinates, std=alignment_std)

    def observation_coordinates(self, observations: torch.Tensor) -> torch.Tensor:
        return self.router.observation_coordinates(observations)

    def alignment_actions(self) -> torch.Tensor:
        actions = []
        generators = self.router.layer.generators
        for local_index, representation_index in enumerate((1, 2)):
            representation = continuous.TRIALITY_REPRESENTATIONS[
                representation_index
            ]
            action = continuous.spin8_factorized_actions(
                self.spinor_alignment_coordinates[local_index],
                generators[representation_index : representation_index + 1].to(
                    self.spinor_alignment_coordinates
                ),
                (representation,),
            )[0]
            actions.append(action)
        return torch.stack(actions)

    def observation_actions(self, observations: torch.Tensor) -> torch.Tensor:
        coordinates = self.observation_coordinates(observations)
        base = continuous.spin8_group_actions(
            coordinates[..., None, :],
            self.router.layer.generators.to(observations),
            continuous.TRIALITY_REPRESENTATIONS,
            mode="factorized",
        )[..., 0, :, :, :]
        alignments = self.alignment_actions().to(observations)
        aligned = [base[..., 0, :, :]]
        for representation_index in (1, 2):
            transform = alignments[representation_index - 1]
            aligned.append(
                transform
                @ base[..., representation_index, :, :]
                @ transform.transpose(-1, -2)
            )
        return torch.stack(aligned, dim=-3)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        actions = self.observation_actions(observations)
        batch, length = observations.shape[:2]
        transition = continuous.Spin8AffineTransition(
            scale=torch.ones(batch, length, 1, device=observations.device),
            action=actions[:, :, None],
            drive=torch.zeros(
                batch,
                length,
                1,
                len(continuous.TRIALITY_REPRESENTATIONS),
                continuous.SPIN8_DIM,
                device=observations.device,
            ),
        )
        prefixes = continuous.work_efficient_spin8_scan(transition)
        initial = self.router.layer.initial_state[0].to(observations).expand(
            batch, -1, -1
        )
        return continuous.apply_spin8_affine(
            prefixes, initial[:, None, None]
        )[:, :, 0]


FACTORIES: dict[str, type[nn.Module]] = {
    SHARED: continuous.SharedPureSpin8Tracker,
    SCRAMBLED: ScrambledSharedLatentSO8TripletTracker,
}


def _trainable_router_tensors(model: nn.Module) -> tuple[torch.Tensor, ...]:
    router = model.router if isinstance(model, ScrambledSharedLatentSO8TripletTracker) else model
    return (
        router.observation_hidden.weight,
        router.observation_hidden.bias,
        router.coordinate_head.weight,
        router.coordinate_head.bias,
    )


def matched_initialization_audit(seed: int) -> dict[str, Any]:
    hashes = {}
    counts = {}
    for candidate in CANDIDATES:
        continuous.seed_everything(SEED_BASE + 1_000 * seed)
        model = FACTORIES[candidate]()
        tensors = _trainable_router_tensors(model)
        hashes[candidate] = continuous.tensor_hash(tensors)
        counts[candidate] = continuous.parameter_count(model)
    checks = {
        "router_initialization_identical": len(set(hashes.values())) == 1,
        "state_scalars_equal": (
            continuous.SharedPureSpin8Tracker.recurrent_state_scalars
            == ScrambledSharedLatentSO8TripletTracker.recurrent_state_scalars
            == 24
        ),
        "scrambled_has_only_56_extra_trainable_scalars": (
            counts[SCRAMBLED] - counts[SHARED] == 56
        ),
    }
    return {
        "router_sha256": hashes,
        "parameter_counts": counts,
        "checks": checks,
        "passed": all(checks.values()),
    }


def gradient_route_audit(
    config: continuous.ContinuousObservationConfig,
    device: torch.device,
) -> dict[str, Any]:
    system = continuous.make_observation_system(config.seed)
    schedule = endpoint.make_endpoint_training_schedule(
        continuous.ContinuousObservationConfig(
            **{**asdict(config), "steps": 1}
        ),
        system,
        device,
    )
    batch = schedule[0]
    results = {}
    for mode in MODES:
        continuous.seed_everything(SEED_BASE + 1_000 * config.seed)
        model = ScrambledSharedLatentSO8TripletTracker().to(device)
        predictions = model(batch.observations.to(device))
        loss, _ = calibration.mode_loss(mode, predictions, batch.endpoint_targets)
        loss.backward()
        gradient = model.spinor_alignment_coordinates.grad
        head_gradient = model.router.coordinate_head.weight.grad
        if gradient is None or head_gradient is None:
            raise RuntimeError("scrambled control did not produce gradients")
        row_norms = torch.linalg.vector_norm(head_gradient.detach().double(), dim=-1)
        results[mode] = {
            "positive_alignment_gradient_l2": float(
                torch.linalg.vector_norm(gradient[0].detach().double())
            ),
            "negative_alignment_gradient_l2": float(
                torch.linalg.vector_norm(gradient[1].detach().double())
            ),
            "shared_coordinate_rows_nonzero": int(torch.count_nonzero(row_norms)),
            "shared_coordinate_minimum_row_l2": float(row_norms.min()),
        }
    checks = {
        "adaptive_positive_alignment_has_gradient": (
            results["vector_plus_adaptive_lift_bit"][
                "positive_alignment_gradient_l2"
            ]
            > 0.0
        ),
        "adaptive_negative_alignment_gradient_exactly_zero": (
            results["vector_plus_adaptive_lift_bit"][
                "negative_alignment_gradient_l2"
            ]
            == 0.0
        ),
        "adaptive_all_shared_coordinate_rows_have_gradient": (
            results["vector_plus_adaptive_lift_bit"][
                "shared_coordinate_rows_nonzero"
            ]
            == 28
        ),
        "full_both_alignments_have_gradient": (
            results["full_triality"]["positive_alignment_gradient_l2"] > 0.0
            and results["full_triality"]["negative_alignment_gradient_l2"] > 0.0
        ),
    }
    return {"results": results, "checks": checks, "passed": all(checks.values())}


def run_benchmark(
    config: continuous.ContinuousObservationConfig,
    *,
    device: torch.device,
    modes: tuple[str, ...],
    candidates: tuple[str, ...],
    checkpoint_directory: Path | None,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    if not modes or len(set(modes)) != len(modes) or set(modes) - set(MODES):
        raise ValueError("modes must be a nonempty unique subset of known modes")
    if (
        not candidates
        or len(set(candidates)) != len(candidates)
        or set(candidates) - set(CANDIDATES)
    ):
        raise ValueError("candidates must be a nonempty unique subset")

    certificate = calibration.exact_one_bit_certificate()
    teacher_contract = continuous.teacher_contract(device)
    initialization = matched_initialization_audit(config.seed)
    gradient_routes = gradient_route_audit(config, device)
    system = continuous.make_observation_system(config.seed)
    schedule = endpoint.make_endpoint_training_schedule(config, system, device)
    split = endpoint.endpoint_training_split_audit(schedule)
    adaptive_bit_audit = calibration.adaptive_lift_training_audit(schedule)
    if not all(
        item["passed"]
        for item in (
            certificate,
            teacher_contract,
            initialization,
            gradient_routes,
            split,
            adaptive_bit_audit,
        )
    ):
        raise RuntimeError("scrambled-alignment task audit failed")

    evaluations = {}
    evaluation_audits = {}
    evaluation_hashes = {}
    for length in config.evaluation_lengths:
        for position in ("early", "late"):
            key = f"{position}_L{length}"
            batch = continuous.make_relation_batch(
                config, system, length, position, device
            )
            evaluations[key] = [batch]
            evaluation_audits[key] = continuous.relation_batch_audit(batch)
            evaluation_hashes[key] = continuous.tensor_hash(
                (
                    batch.observations,
                    batch.targets,
                    batch.coordinates,
                    batch.post_relation_mask,
                )
            )
    if not all(audit["passed"] for audit in evaluation_audits.values()):
        raise RuntimeError("scrambled-alignment evaluation audit failed")

    results = {}
    for mode in modes:
        results[mode] = {}
        for candidate in candidates:
            continuous.seed_everything(SEED_BASE + 1_000 * config.seed)
            results[mode][candidate] = calibration.train_candidate(
                candidate,
                mode,
                schedule,
                evaluations,
                config,
                device,
                checkpoint_directory,
                factory=FACTORIES[candidate],
            )
    all_finite = observability._all_numeric_values_finite(results)
    return {
        "schema_version": 1,
        "experiment": "Pure Spin8 shared-latent scrambled-alignment control",
        "status": "development" if config.seed == 0 else "unadjudicated",
        "protocol_frozen_at": PROTOCOL_FROZEN_AT,
        "recorded_at": continuous.now(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor()
            ),
        },
        "config": asdict(config),
        "modes": modes,
        "candidates": candidates,
        "task": {
            "training_split": split,
            "adaptive_lift_bit_audit": adaptive_bit_audit,
            "evaluation_audits": evaluation_audits,
            "evaluation_schedule_sha256": evaluation_hashes,
            "observation_system_sha256": continuous.tensor_hash(
                (system.projection, system.bias)
            ),
        },
        "architecture": {
            "alignment_initialization_std": ALIGNMENT_INITIALIZATION_STD,
            "alignment_parameterization": (
                "independent positive/negative inner conjugations, 28 scalars each"
            ),
            "initialization_audit": initialization,
            "gradient_route_audit": gradient_routes,
        },
        "integrity": {
            "same_schedule_for_every_mode_and_candidate": True,
            "shared_router_initialization_identical_across_candidates": True,
            "candidate_initialization_identical_across_modes": True,
            "coordinates_and_events_never_passed_to_models": True,
            "all_metrics_finite": all_finite,
        },
        "results": results,
        "passed": all_finite,
        "claim_scope": {
            "question": (
                "Does the correct triality alignment add transfer beyond a matched "
                "shared 28-coordinate bottleneck?"
            ),
            "development_only": config.seed == 0,
            "nonclaims": [
                "not a natural-data result",
                "not a fused-throughput comparison",
                "not a global optimization theorem",
            ],
        },
    }


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-length", type=int, default=16)
    parser.add_argument("--evaluation-pairs", type=int, default=64)
    parser.add_argument("--evaluation-lengths", default="16,64,128")
    parser.add_argument("--evaluation-microbatch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--observation-noise-std", type=float, default=0.01)
    parser.add_argument("--half-center-probability", type=float, default=0.12)
    parser.add_argument("--regular-coordinate-std", type=float, default=0.32)
    parser.add_argument("--half-center-delta", type=float, default=0.25)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--checkpoint-directory", type=Path, default=DEFAULT_CHECKPOINT_DIRECTORY
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = continuous.ContinuousObservationConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        training_length=args.training_length,
        evaluation_pairs=args.evaluation_pairs,
        evaluation_lengths=tuple(
            int(value) for value in args.evaluation_lengths.split(",") if value
        ),
        evaluation_microbatch_size=args.evaluation_microbatch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        observation_noise_std=args.observation_noise_std,
        half_center_probability=args.half_center_probability,
        regular_coordinate_std=args.regular_coordinate_std,
        half_center_delta=args.half_center_delta,
        seed=args.seed,
    )
    result = run_benchmark(
        config,
        device=torch.device(args.device),
        modes=_parse_csv(args.modes),
        candidates=_parse_csv(args.candidates),
        checkpoint_directory=args.checkpoint_directory,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
