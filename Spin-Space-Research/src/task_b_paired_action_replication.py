"""Prospective paired shared/independent Spin(8) Task-B replication."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

from spin8_blind_alias_action import (
    TrainedCombined,
    combined_design_audit,
    negative_calibration_basis,
    negative_subspace_metrics,
    train_combined,
)
from spin8_blind_shared_action import observed_action, sample_teacher
from spin8_continuous_alias import (
    DENSE_LENGTHS,
    TEST_RADIUS,
    FrozenSlotPolicy,
    alias_diagnostics,
)
from spin8_triality import spin8_actions, torch_triality_generators
from spin8_triality_lift import triality_tensor
from task_b_delta_action_replay import (
    paired_scan_parity,
    paired_sequence_evaluation,
)


def _evaluations(
    actions: torch.Tensor,
    oracle_actions: torch.Tensor,
    policy: FrozenSlotPolicy,
    *,
    seed: int,
    lengths: tuple[int, ...],
) -> list[dict[str, object]]:
    return [
        paired_sequence_evaluation(
            actions,
            oracle_actions,
            policy,
            seed=seed,
            length=length,
            batch_size=160 if length <= 16 else 48,
        )
        for length in lengths
    ]


def _family_report(
    trained: TrainedCombined,
    oracle_actions: torch.Tensor,
    basis: torch.Tensor,
    *,
    seed: int,
    lengths: tuple[int, ...],
) -> dict[str, object]:
    return {
        "training": trained.report,
        "observed_column_mse": float(
            F.mse_loss(
                observed_action(trained.actions), observed_action(oracle_actions)
            )
        ),
        "negative_subspaces": negative_subspace_metrics(
            trained.actions, oracle_actions, basis
        ),
        "routing": alias_diagnostics(
            trained.policy, seed=seed, radius=TEST_RADIUS
        ),
        "evaluation": _evaluations(
            trained.actions,
            oracle_actions,
            trained.policy,
            seed=seed,
            lengths=lengths,
        ),
        "scan": paired_scan_parity(trained.actions, trained.policy, seed=seed),
    }


def _parameter_payload(trained: TrainedCombined) -> dict[str, object]:
    if trained.policy.write_weight is None or trained.policy.query_weight is None:
        raise RuntimeError("paired replication requires learned router weights")
    return {
        "actions": trained.actions.tolist(),
        "coordinates": trained.coordinates.tolist(),
        "write_weight": trained.policy.write_weight.tolist(),
        "query_weight": trained.policy.query_weight.tolist(),
        "temperature": trained.policy.temperature,
    }


def _policy_from_payload(payload: dict[str, object]) -> FrozenSlotPolicy:
    return FrozenSlotPolicy(
        variant="learned_both_joint",
        write_weight=torch.tensor(payload["write_weight"], dtype=torch.float64),
        query_weight=torch.tensor(payload["query_weight"], dtype=torch.float64),
        temperature=float(payload["temperature"]),
    )


def _actions_from_payload(
    payload: dict[str, object], *, shared: bool
) -> tuple[torch.Tensor, float]:
    coordinates = torch.tensor(payload["coordinates"], dtype=torch.float64)
    generators = torch_triality_generators(dtype=torch.float64)
    if shared:
        reconstructed = spin8_actions(coordinates, generators)
    else:
        tangent = torch.einsum(
            "trk,rkij->trij", coordinates, generators
        ).contiguous()
        reconstructed = torch.matrix_exp(tangent)
    stored = torch.tensor(payload["actions"], dtype=torch.float64)
    return reconstructed, float((reconstructed - stored).abs().max())


def _max_pair_error(rows: list[dict[str, object]]) -> float:
    return max(
        max(
            float(row["direct_delta_max_state_error"]),
            float(row["direct_delta_max_prediction_error"]),
        )
        for row in rows
    )


def _routing_passed(routing: dict[str, object]) -> bool:
    return (
        int(routing["write_center_collisions"]) == 0
        and int(routing["query_center_collisions"]) == 0
        and float(routing["center_cross_encoder_agreement"]) == 1.0
        and float(routing["write_alias_agreement"]) >= 0.99
        and float(routing["query_alias_agreement"]) >= 0.99
    )


def _oracle_passed(rows: list[dict[str, object]]) -> bool:
    return all(
        float(row["delta"]["mean_query_cosine"]) >= 0.995
        and float(row["delta"]["minimum_query_cosine"]) >= 0.98
        for row in rows
    )


def _scan_passed(scan: dict[str, object]) -> bool:
    return (
        float(scan["direct_parallel_recurrent_max_error"]) <= 1e-9
        and float(scan["delta_parallel_recurrent_max_error"]) <= 1e-9
        and float(scan["direct_delta_parallel_max_error"]) <= 1e-10
        and float(scan["direct_delta_recurrent_max_error"]) <= 1e-10
    )


def retained_parameter_replay(
    *,
    seed: int,
    teacher_actions: torch.Tensor,
    shared_payload: dict[str, object],
    independent_payload: dict[str, object],
) -> dict[str, object]:
    shared_actions, _ = _actions_from_payload(shared_payload, shared=True)
    independent_actions, _ = _actions_from_payload(
        independent_payload, shared=False
    )
    shared_policy = _policy_from_payload(shared_payload)
    independent_policy = _policy_from_payload(independent_payload)
    return {
        "shared": paired_sequence_evaluation(
            shared_actions,
            teacher_actions,
            shared_policy,
            seed=seed,
            length=32,
            batch_size=8,
        ),
        "independent": paired_sequence_evaluation(
            independent_actions,
            teacher_actions,
            independent_policy,
            seed=seed,
            length=32,
            batch_size=8,
        ),
        "shared_scan": paired_scan_parity(shared_actions, shared_policy, seed=seed),
        "independent_scan": paired_scan_parity(
            independent_actions, independent_policy, seed=seed
        ),
    }


def verify_retained_parameters(report: dict[str, object]) -> dict[str, float | bool]:
    payload = report["learned_parameters"]
    teacher_actions = torch.tensor(payload["teacher_actions"], dtype=torch.float64)
    replay = retained_parameter_replay(
        seed=int(report["seed"]),
        teacher_actions=teacher_actions,
        shared_payload=payload["shared"],
        independent_payload=payload["independent"],
    )
    expected = report["retained_parameter_replay"]
    _, shared_action_difference = _actions_from_payload(payload["shared"], shared=True)
    _, independent_action_difference = _actions_from_payload(
        payload["independent"], shared=False
    )
    differences = [shared_action_difference, independent_action_difference]
    for family in ("shared", "independent"):
        for memory in ("direct", "delta"):
            for metric in (
                "mean_query_cosine",
                "minimum_query_cosine",
                "mean_relative_squared_error",
                "maximum_relative_squared_error",
            ):
                differences.append(
                    abs(
                        float(replay[family][memory][metric])
                        - float(expected[family][memory][metric])
                    )
                )
    for family in ("shared_scan", "independent_scan"):
        for metric in (
            "direct_parallel_recurrent_max_error",
            "delta_parallel_recurrent_max_error",
            "direct_delta_parallel_max_error",
            "direct_delta_recurrent_max_error",
        ):
            differences.append(
                abs(float(replay[family][metric]) - float(expected[family][metric]))
            )
    maximum_difference = max(differences, default=0.0)
    return {
        "maximum_metric_difference": maximum_difference,
        "passed": maximum_difference <= 1e-12,
    }


def run_seed(
    seed: int,
    *,
    adam_steps_per_stage: int,
    lbfgs_steps: int,
    dense: bool,
) -> dict[str, object]:
    dtype = torch.float64
    device = torch.device("cpu")
    generators = torch_triality_generators(dtype=dtype, device=device)
    rho = triality_tensor(dtype=dtype, device=device)
    teacher = sample_teacher(seed=seed, generators=generators)
    basis = negative_calibration_basis(seed, dtype=dtype, device=device)
    design = combined_design_audit(teacher.coefficients, generators, basis)
    shared = train_combined(
        "joint",
        seed=seed,
        generators=generators,
        rho=rho,
        teacher_actions=teacher.actions,
        basis=basis,
        adam_steps_per_stage=adam_steps_per_stage,
        lbfgs_steps=lbfgs_steps,
    )
    independent = train_combined(
        "independent",
        seed=seed,
        generators=generators,
        rho=rho,
        teacher_actions=teacher.actions,
        basis=basis,
        adam_steps_per_stage=adam_steps_per_stage,
        lbfgs_steps=lbfgs_steps,
    )
    oracle_actions = teacher.actions.detach().cpu()
    basis_cpu = basis.cpu()
    lengths = DENSE_LENGTHS if dense else (32, 128, 512, 2048)
    shared_report = _family_report(
        shared, oracle_actions, basis_cpu, seed=seed, lengths=lengths
    )
    independent_report = _family_report(
        independent, oracle_actions, basis_cpu, seed=seed, lengths=lengths
    )
    independent_shared_router = _evaluations(
        independent.actions,
        oracle_actions,
        shared.policy,
        seed=seed,
        lengths=lengths,
    )
    oracle_shared_router = _evaluations(
        oracle_actions,
        oracle_actions,
        shared.policy,
        seed=seed,
        lengths=lengths,
    )
    oracle_independent_router = _evaluations(
        oracle_actions,
        oracle_actions,
        independent.policy,
        seed=seed,
        lengths=lengths,
    )
    shared_payload = _parameter_payload(shared)
    independent_payload = _parameter_payload(independent)
    replay = retained_parameter_replay(
        seed=seed,
        teacher_actions=oracle_actions,
        shared_payload=shared_payload,
        independent_payload=independent_payload,
    )
    all_rows = (
        shared_report["evaluation"]
        + independent_report["evaluation"]
        + independent_shared_router
        + oracle_shared_router
        + oracle_independent_router
    )
    implementation_passed = (
        _max_pair_error(all_rows) <= 1e-10
        and _scan_passed(shared_report["scan"])
        and _scan_passed(independent_report["scan"])
        and _routing_passed(shared_report["routing"])
        and _routing_passed(independent_report["routing"])
        and _oracle_passed(oracle_shared_router)
        and _oracle_passed(oracle_independent_router)
        and float(shared_report["observed_column_mse"]) < 1e-6
        and float(independent_report["observed_column_mse"]) < 1e-6
    )
    shared_long = float(
        shared_report["evaluation"][-1]["delta"]["mean_query_cosine"]
    )
    independent_long = float(
        independent_report["evaluation"][-1]["delta"]["mean_query_cosine"]
    )
    routing_matched_independent_long = float(
        independent_shared_router[-1]["delta"]["mean_query_cosine"]
    )
    representation_prior_win = (
        shared_long >= 0.995
        and independent_long <= 0.90
        and routing_matched_independent_long <= 0.90
    )
    report = {
        "experiment": "Task-B prospective paired-action replication",
        "protocol": "TASK_B_PAIRED_ACTION_REPLICATION_PREREGISTRATION.md",
        "seed": seed,
        "device": "cpu",
        "dtype": "float64",
        "adam_steps_per_stage": adam_steps_per_stage,
        "lbfgs_steps": lbfgs_steps,
        "dense": dense,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "teacher_resamples": teacher.resamples,
        "design": design,
        "shared": shared_report,
        "independent": independent_report,
        "independent_action_shared_router": independent_shared_router,
        "oracle_action_shared_router": oracle_shared_router,
        "oracle_action_independent_router": oracle_independent_router,
        "learned_parameters": {
            "teacher_actions": oracle_actions.tolist(),
            "shared": shared_payload,
            "independent": independent_payload,
        },
        "retained_parameter_replay": replay,
        "decision": {
            "implementation_passed": implementation_passed,
            "representation_prior_win": representation_prior_win,
            "shared_delta_length2048_mean_cosine": shared_long,
            "independent_delta_length2048_mean_cosine": independent_long,
            "routing_matched_independent_delta_length2048_mean_cosine": (
                routing_matched_independent_long
            ),
            "maximum_direct_delta_error": _max_pair_error(all_rows),
        },
    }
    verification = verify_retained_parameters(report)
    report["retained_parameter_verification"] = verification
    report["decision"]["implementation_passed"] = (
        implementation_passed and bool(verification["passed"])
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--adam-steps-per-stage", type=int, default=500)
    parser.add_argument("--lbfgs-steps", type=int, default=150)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        report = json.loads(args.verify.read_text(encoding="utf-8"))
        verification = verify_retained_parameters(report)
        print(json.dumps(verification, indent=2))
        raise SystemExit(0 if verification["passed"] else 1)
    if args.seed is None or args.output is None:
        parser.error("--seed and --output are required unless --verify is used")
    report = run_seed(
        args.seed,
        adam_steps_per_stage=args.adam_steps_per_stage,
        lbfgs_steps=args.lbfgs_steps,
        dense=args.dense,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
