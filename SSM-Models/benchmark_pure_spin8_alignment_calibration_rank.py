"""Train the scrambled Spin(8) control along an exact calibration-rank curve.

The endpoint task remains vector plus one adaptive positive-spinor lift bit.
The only new information is an external calibration frame for the otherwise
hidden negative-spinor alignment: for ``m`` probes, the learner is told that
``T e_j = e_j`` for ``j < m``.  Each probe transmits eight scalar values, while
the exact independent ranks are ``7,13,18,22,25,27,28,28``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import analyze_pure_spin8_alignment_calibration_rank as rank_analysis
import benchmark_pure_spin8_continuous_observation as continuous
import benchmark_pure_spin8_endpoint_observability as observability
import benchmark_pure_spin8_endpoint_supervision as endpoint
import benchmark_pure_spin8_lift_bit_calibration as calibration
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parent
MODE = "vector_plus_adaptive_lift_bit"
ANCHOR_COUNTS = tuple(range(9))
ANCHOR_LOSS_WEIGHT = 1.0
ALIGNMENT_INITIALIZATION_STD = 0.35
SEED_BASE = 1_360_000
PROTOCOL_FROZEN_AT = "2026-08-17T09:52:34.5014289+02:00"
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "artifacts"
    / "pure_spin8_alignment_calibration_rank_development_seed0.json"
)
DEFAULT_CHECKPOINT_DIRECTORY = (
    ROOT / "checkpoints" / "pure_spin8_alignment_calibration_rank_development"
)


class NegativeOnlyScrambledSpin8Tracker(nn.Module):
    """Correct observed views with one hidden trainable negative alignment."""

    recurrent_state_scalars = 24

    def __init__(
        self, alignment_std: float = ALIGNMENT_INITIALIZATION_STD
    ) -> None:
        super().__init__()
        if alignment_std < 0:
            raise ValueError("alignment_std must be nonnegative")
        self.router = continuous.SharedPureSpin8Tracker()
        self.negative_alignment_coordinates = nn.Parameter(torch.empty(28))
        nn.init.normal_(self.negative_alignment_coordinates, std=alignment_std)

    def observation_coordinates(self, observations: torch.Tensor) -> torch.Tensor:
        return self.router.observation_coordinates(observations)

    def negative_alignment_action(self) -> torch.Tensor:
        generators = self.router.layer.generators[2:3].to(
            self.negative_alignment_coordinates
        )
        return continuous.spin8_factorized_actions(
            self.negative_alignment_coordinates,
            generators,
            ("negative",),
        )[0]

    def observation_actions(self, observations: torch.Tensor) -> torch.Tensor:
        coordinates = self.observation_coordinates(observations)
        base = continuous.spin8_group_actions(
            coordinates[..., None, :],
            self.router.layer.generators.to(observations),
            continuous.TRIALITY_REPRESENTATIONS,
            mode="factorized",
        )[..., 0, :, :, :]
        transform = self.negative_alignment_action().to(observations)
        negative = transform @ base[..., 2, :, :] @ transform.transpose(-1, -2)
        return torch.stack((base[..., 0, :, :], base[..., 1, :, :], negative), dim=-3)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        base_states = self.router(observations)
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
        aligned_states = continuous.apply_spin8_affine(
            prefixes, initial[:, None, None]
        )[:, :, 0]
        return torch.stack(
            (
                base_states[..., 0, :],
                base_states[..., 1, :],
                aligned_states[..., 2, :],
            ),
            dim=-2,
        )


def _router_hash(model: nn.Module) -> str:
    router = model.router if isinstance(model, NegativeOnlyScrambledSpin8Tracker) else model
    return continuous.tensor_hash(
        tuple(parameter.detach() for parameter in router.parameters() if parameter.requires_grad)
    )


def matched_initialization_audit(seed: int) -> dict[str, Any]:
    continuous.seed_everything(SEED_BASE + 1_000 * seed)
    shared = continuous.SharedPureSpin8Tracker()
    continuous.seed_everything(SEED_BASE + 1_000 * seed)
    scrambled_model = NegativeOnlyScrambledSpin8Tracker()
    generator = torch.Generator().manual_seed(SEED_BASE + 1_000 * seed + 99)
    observations = torch.randn(2, 3, 12, generator=generator)
    shared_actions = shared.observation_actions(observations)
    scrambled_actions = scrambled_model.observation_actions(observations)
    counts = {
        "shared": continuous.parameter_count(shared),
        "negative_only_scrambled": continuous.parameter_count(scrambled_model),
    }
    checks = {
        "router_initialization_identical": _router_hash(shared)
        == _router_hash(scrambled_model),
        "state_scalars_equal": shared.recurrent_state_scalars
        == scrambled_model.recurrent_state_scalars
        == 24,
        "only_28_extra_trainable_scalars": counts["negative_only_scrambled"]
        - counts["shared"]
        == 28,
        "vector_actions_initially_identical": torch.equal(
            shared_actions[..., 0, :, :], scrambled_actions[..., 0, :, :]
        ),
        "positive_actions_initially_identical": torch.equal(
            shared_actions[..., 1, :, :], scrambled_actions[..., 1, :, :]
        ),
        "negative_actions_initially_distinct": not torch.allclose(
            shared_actions[..., 2, :, :], scrambled_actions[..., 2, :, :]
        ),
    }
    return {
        "parameter_counts": counts,
        "shared_router_sha256": _router_hash(shared),
        "scrambled_router_sha256": _router_hash(scrambled_model),
        "checks": checks,
        "passed": all(checks.values()),
    }


def negative_alignment_calibration_loss(
    model: NegativeOnlyScrambledSpin8Tracker,
    probe_count: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """MSE on the first ``probe_count`` columns of the negative alignment."""

    if not 0 <= probe_count <= continuous.SPIN8_DIM:
        raise ValueError("probe_count must lie in [0,8]")
    negative_action = model.negative_alignment_action()
    if probe_count == 0:
        raw = negative_action.sum() * 0.0
    else:
        target = torch.eye(
            continuous.SPIN8_DIM,
            device=negative_action.device,
            dtype=negative_action.dtype,
        )[:, :probe_count]
        raw = F.mse_loss(negative_action[:, :probe_count], target)
    weighted = ANCHOR_LOSS_WEIGHT * raw
    return weighted, {
        "negative_anchor_mse": raw,
        "weighted_negative_anchor_loss": weighted,
    }


def calibration_gradient_audit(seed: int = 0) -> dict[str, Any]:
    """Verify that anchors touch only the intended alignment and have exact ranks."""

    rows = {}
    for probe_count in ANCHOR_COUNTS:
        continuous.seed_everything(SEED_BASE + 1_000 * seed)
        model = NegativeOnlyScrambledSpin8Tracker().double()
        loss, _ = negative_alignment_calibration_loss(model, probe_count)
        loss.backward()
        gradient = model.negative_alignment_coordinates.grad
        if gradient is None:
            raise RuntimeError("calibration loss did not produce an alignment gradient")
        rows[str(probe_count)] = {
            "expected_jacobian_rank": rank_analysis.frame_orbit_rank(probe_count),
            "negative_alignment_gradient_l2": float(torch.linalg.vector_norm(gradient)),
            "router_gradient_present": any(
                parameter.grad is not None for parameter in model.router.parameters()
            ),
        }
    checks = {
        "zero_probe_gradient_is_exactly_zero": rows["0"][
            "negative_alignment_gradient_l2"
        ]
        == 0.0,
        "router_never_touched": all(
            not row["router_gradient_present"] for row in rows.values()
        ),
        "every_nonzero_probe_count_updates_negative_alignment": all(
            rows[str(count)]["negative_alignment_gradient_l2"] > 0.0
            for count in range(1, 9)
        ),
        "seven_and_eight_probe_expected_ranks_are_full": rows["7"][
            "expected_jacobian_rank"
        ]
        == rows["8"]["expected_jacobian_rank"]
        == continuous.SPIN8_BIVECTOR_DIM,
    }
    return {"rows": rows, "checks": checks, "passed": all(checks.values())}


@torch.no_grad()
def alignment_diagnostics(
    model: NegativeOnlyScrambledSpin8Tracker,
    probe_count: int,
) -> dict[str, float | int | bool]:
    action = model.negative_alignment_action().double()
    identity = torch.eye(continuous.SPIN8_DIM, dtype=torch.float64, device=action.device)
    error = action - identity
    selected = error[:, :probe_count]
    unselected = error[:, probe_count:]
    return {
        "probe_count": probe_count,
        "transmitted_scalar_values": continuous.SPIN8_DIM * probe_count,
        "exact_independent_rank": rank_analysis.frame_orbit_rank(probe_count),
        "full_identity_rmse": float(error.square().mean().sqrt()),
        "selected_probe_rmse": (
            float(selected.square().mean().sqrt()) if selected.numel() else 0.0
        ),
        "unselected_probe_rmse": (
            float(unselected.square().mean().sqrt()) if unselected.numel() else 0.0
        ),
        "orthogonality_max_abs": float((action.T @ action - identity).abs().max()),
        "determinant": float(torch.linalg.det(action)),
        "all_values_finite": bool(torch.isfinite(action).all()),
    }


def _evaluation_batches(
    config: continuous.ContinuousObservationConfig,
    system: continuous.ObservationSystem,
    device: torch.device,
) -> tuple[
    dict[str, list[continuous.ContinuousRelationBatch]],
    dict[str, Any],
    dict[str, str],
]:
    evaluations = {}
    audits = {}
    hashes = {}
    for length in config.evaluation_lengths:
        for position in ("early", "late"):
            key = f"{position}_L{length}"
            batch = continuous.make_relation_batch(config, system, length, position, device)
            evaluations[key] = [batch]
            audits[key] = continuous.relation_batch_audit(batch)
            hashes[key] = continuous.tensor_hash(
                (
                    batch.observations,
                    batch.targets,
                    batch.coordinates,
                    batch.post_relation_mask,
                )
            )
    return evaluations, audits, hashes


def train_scrambled_anchor_count(
    *,
    probe_count: int,
    schedule: Sequence[endpoint.EndpointTrainingBatch],
    evaluations: dict[str, list[continuous.ContinuousRelationBatch]],
    config: continuous.ContinuousObservationConfig,
    device: torch.device,
    checkpoint_directory: Path | None,
) -> dict[str, Any]:
    model = NegativeOnlyScrambledSpin8Tracker().to(device)
    router_optimizer = torch.optim.AdamW(
        model.router.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    alignment_optimizer = torch.optim.AdamW(
        (model.negative_alignment_coordinates,),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    samples = {}
    component_samples = {}
    router_gradient_norm = torch.tensor(0.0)
    alignment_gradient_norm = torch.tensor(0.0)
    initial_alignment = alignment_diagnostics(model, probe_count)
    started = time.perf_counter()
    model.train()
    for step, batch in enumerate(schedule, start=1):
        observations = batch.observations.to(device)
        router_optimizer.zero_grad(set_to_none=True)
        alignment_optimizer.zero_grad(set_to_none=True)
        predictions = model(observations)
        base_loss, components = calibration.mode_loss(
            MODE, predictions, batch.endpoint_targets
        )
        if not torch.isfinite(base_loss):
            raise RuntimeError(
                f"m={probe_count} produced nonfinite base loss at step {step}"
            )
        base_loss.backward()
        router_gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.router.parameters(), config.gradient_clip
        )
        router_optimizer.step()

        alignment_optimizer.zero_grad(set_to_none=True)
        anchor_loss, anchor_components = negative_alignment_calibration_loss(
            model, probe_count
        )
        if not torch.isfinite(anchor_loss):
            raise RuntimeError(
                f"m={probe_count} produced nonfinite anchor loss at step {step}"
            )
        anchor_loss.backward()
        alignment_gradient_norm = torch.nn.utils.clip_grad_norm_(
            (model.negative_alignment_coordinates,), config.gradient_clip
        )
        alignment_optimizer.step()
        loss = base_loss.detach() + anchor_loss.detach()
        components = {**components, **anchor_components}
        if step == 1 or step % 100 == 0 or step == config.steps:
            samples[str(step)] = float(loss)
            component_samples[str(step)] = {
                key: float(value.detach()) for key, value in components.items()
            }
            print(
                f"scrambled m={probe_count} seed={config.seed} "
                f"step={step}/{config.steps} loss={samples[str(step)]:.8f} "
                f"components={component_samples[str(step)]}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    final_alignment = alignment_diagnostics(model, probe_count)
    result: dict[str, Any] = {
        "parameters": continuous.parameter_count(model),
        "recurrent_state_scalars": int(model.recurrent_state_scalars),
        "probe_count": probe_count,
        "transmitted_scalar_values": continuous.SPIN8_DIM * probe_count,
        "exact_independent_rank": rank_analysis.frame_orbit_rank(probe_count),
        "loss_samples": samples,
        "component_samples": component_samples,
        "final_training_loss": samples[str(config.steps)],
        "last_router_preclip_gradient_norm": float(router_gradient_norm),
        "last_alignment_preclip_gradient_norm": float(alignment_gradient_norm),
        "training_wall_seconds": elapsed,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "initial_alignment": initial_alignment,
        "final_alignment": final_alignment,
        "final_router_sha256": _router_hash(model),
        "evaluation": {
            key: observability.evaluate_by_representation(
                model, batches, device, config.evaluation_microbatch_size
            )
            for key, batches in evaluations.items()
        },
        "adaptive_lift_evaluation": {
            key: calibration.evaluate_adaptive_lift_bit(
                model, batches, device, config.evaluation_microbatch_size
            )
            for key, batches in evaluations.items()
        },
        "action_rmse_by_representation": observability.action_rmse_by_representation(
            model, evaluations, device
        ),
    }
    if checkpoint_directory is not None:
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_directory / (
            f"shared_latent_scrambled_alignment_anchor{probe_count}_"
            f"seed{config.seed}_step{config.steps}.pt"
        )
        torch.save(
            {
                "format_version": 1,
                "candidate": "negative_only_scrambled_alignment",
                "mode": MODE,
                "training_supervision": (
                    "endpoint_vector_plus_adaptive_positive_lift_bit_plus_"
                    "external_negative_alignment_basis_probes"
                ),
                "probe_count": probe_count,
                "anchor_loss_weight": ANCHOR_LOSS_WEIGHT,
                "config": asdict(config),
                "state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "result": result,
            },
            checkpoint,
        )
        result["checkpoint"] = str(checkpoint)
        result["checkpoint_sha256"] = hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest()

    del router_optimizer, alignment_optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_benchmark(
    config: continuous.ContinuousObservationConfig,
    *,
    device: torch.device,
    anchor_counts: tuple[int, ...],
    checkpoint_directory: Path | None,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    if (
        not anchor_counts
        or len(set(anchor_counts)) != len(anchor_counts)
        or set(anchor_counts) - set(ANCHOR_COUNTS)
    ):
        raise ValueError("anchor_counts must be a nonempty unique subset of 0..8")

    rank_certificate = rank_analysis.build_certificate(config.seed)
    teacher_contract = continuous.teacher_contract(device)
    initialization = matched_initialization_audit(config.seed)
    gradient_audit = calibration_gradient_audit(config.seed)
    system = continuous.make_observation_system(config.seed)
    schedule = endpoint.make_endpoint_training_schedule(config, system, device)
    split = endpoint.endpoint_training_split_audit(schedule)
    adaptive_bit_audit = calibration.adaptive_lift_training_audit(schedule)
    evaluations, evaluation_audits, evaluation_hashes = _evaluation_batches(
        config, system, device
    )
    audits = (
        rank_certificate,
        teacher_contract,
        initialization,
        gradient_audit,
        split,
        adaptive_bit_audit,
        *evaluation_audits.values(),
    )
    if not all(audit["passed"] for audit in audits):
        raise RuntimeError("alignment calibration-rank task audit failed")

    continuous.seed_everything(SEED_BASE + 1_000 * config.seed)
    shared_reference = calibration.train_candidate(
        "shared_pure_spin8",
        MODE,
        schedule,
        evaluations,
        config,
        device,
        checkpoint_directory,
        factory=continuous.SharedPureSpin8Tracker,
    )
    curve = {}
    for probe_count in anchor_counts:
        continuous.seed_everything(SEED_BASE + 1_000 * config.seed)
        curve[str(probe_count)] = train_scrambled_anchor_count(
            probe_count=probe_count,
            schedule=schedule,
            evaluations=evaluations,
            config=config,
            device=device,
            checkpoint_directory=checkpoint_directory,
        )

    shared_router_hash = shared_reference["final_trainable_parameter_sha256"]
    router_hashes = {
        probe_count: curve[str(probe_count)]["final_router_sha256"]
        for probe_count in anchor_counts
    }
    router_trajectory_checks = {
        "every_anchor_rank_matches_shared_router_bitwise": all(
            digest == shared_router_hash for digest in router_hashes.values()
        ),
        "all_anchor_ranks_match_each_other_bitwise": len(set(router_hashes.values()))
        == 1,
    }

    all_finite = observability._all_numeric_values_finite(
        (shared_reference, curve)
    )
    return {
        "schema_version": 1,
        "experiment": "Pure Spin8 negative-alignment calibration-rank curve",
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
        "anchor_counts": anchor_counts,
        "anchor_loss_weight": ANCHOR_LOSS_WEIGHT,
        "rank_certificate": rank_certificate,
        "task": {
            "training_split": split,
            "adaptive_lift_bit_audit": adaptive_bit_audit,
            "gradient_audit": gradient_audit,
            "evaluation_audits": evaluation_audits,
            "evaluation_schedule_sha256": evaluation_hashes,
            "observation_system_sha256": continuous.tensor_hash(
                (system.projection, system.bias)
            ),
            "negative_endpoint_targets_transferred": False,
            "negative_alignment_basis_targets_are_external": True,
            "scalar_values_per_probe": continuous.SPIN8_DIM,
        },
        "architecture": {
            "initialization_audit": initialization,
            "same_24_scalar_recurrent_state": True,
            "same_958_parameter_negative_scramble_at_every_rank": True,
            "disjoint_router_and_alignment_optimizers": True,
            "router_trajectory_sha256": {
                "shared_reference": shared_router_hash,
                "scrambled_by_probe_count": router_hashes,
            },
            "router_trajectory_checks": router_trajectory_checks,
        },
        "results": {
            "shared_aligned_reference": shared_reference,
            "scrambled_anchor_curve": curve,
        },
        "integrity": {
            "same_schedule_for_every_anchor_count": True,
            "identical_scrambled_initialization_for_every_anchor_count": True,
            "coordinates_and_events_never_passed_to_models": True,
            "router_trajectory_bitwise_identical_at_every_rank": all(
                router_trajectory_checks.values()
            ),
            "all_metrics_finite": all_finite,
        },
        "passed": all_finite and all(router_trajectory_checks.values()),
        "claim_scope": {
            "question": (
                "At what exact calibration rank does the hidden negative alignment "
                "become trainable and recover cross-view sequence behavior?"
            ),
            "nonclaims": [
                "the calibration frame is not inferred from vector observations",
                "eight transmitted scalars per probe are not eight independent constraints",
                "local Jacobian rank does not guarantee global optimizer convergence",
                "not a natural-data or throughput result",
            ],
        },
    }


def _parse_anchor_counts(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2_000)
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
    parser.add_argument("--anchor-counts", default=",".join(map(str, ANCHOR_COUNTS)))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--checkpoint-directory", type=Path, default=DEFAULT_CHECKPOINT_DIRECTORY
    )
    parser.add_argument("--no-checkpoints", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = continuous.ContinuousObservationConfig(
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        training_length=args.training_length,
        evaluation_pairs=args.evaluation_pairs,
        evaluation_lengths=continuous.parse_lengths(args.evaluation_lengths),
        evaluation_microbatch_size=args.evaluation_microbatch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        observation_noise_std=args.observation_noise_std,
        half_center_probability=args.half_center_probability,
        regular_coordinate_std=args.regular_coordinate_std,
        half_center_delta=args.half_center_delta,
    )
    result = run_benchmark(
        config,
        device=torch.device(args.device),
        anchor_counts=_parse_anchor_counts(args.anchor_counts),
        checkpoint_directory=(
            None if args.no_checkpoints else args.checkpoint_directory
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
