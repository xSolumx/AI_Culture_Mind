"""Materialize the preregistered Task-B independent-action delta row.

The historical direct row uses a low-temperature soft route.  Standard delta
overwrite is exactly the same memory only for a one-hot key, so this module
retains the soft row as a determinism audit and evaluates direct/delta memory
in lockstep after applying the same learned argmax route to both.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from schurscan_delta_memory import (
    DeltaTransition,
    apply_delta,
    delta_read,
    delta_write_transitions,
    recurrent_delta_states,
    scanned_delta_states,
    value_transport_transitions,
)
from spin8_blind_alias_action import (
    combined_design_audit,
    evaluate_sequences,
    negative_calibration_basis,
    negative_subspace_metrics,
    train_combined,
)
from spin8_blind_shared_action import (
    TOKEN_COUNT,
    observed_action,
    sample_teacher,
)
from spin8_continuous_alias import (
    DENSE_LENGTHS,
    SLOTS,
    TEST_RADIUS,
    AliasWorld,
    FrozenSlotPolicy,
    alias_diagnostics,
    random_unit,
)
from spin8_triality import torch_triality_generators
from spin8_triality_lift import triality_tensor
from spin8_triality_memory import SlotTransition, apply_slot, associative_slot_scan

METRIC_NAMES = (
    "mean_query_cosine",
    "minimum_query_cosine",
    "mean_relative_squared_error",
    "maximum_relative_squared_error",
)


def _hard_route(
    policy: FrozenSlotPolicy,
    aliases: torch.Tensor,
    labels: torch.Tensor,
    *,
    side: str,
    expected_slots: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    soft = policy.routes(aliases, labels, side=side)
    indices = soft.argmax(dim=-1)
    hard = F.one_hot(indices, SLOTS).to(dtype=aliases.dtype)
    return hard, int((indices == expected_slots[labels]).sum())


def _retrieval_metrics(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine = F.cosine_similarity(prediction, target, dim=-1)
    error = (
        (prediction - target).square().sum(dim=-1)
        / target.square().sum(dim=-1).clamp_min(1e-30)
    )
    return cosine, error


@torch.no_grad()
def paired_sequence_evaluation(
    actions: torch.Tensor,
    oracle_actions: torch.Tensor,
    policy: FrozenSlotPolicy,
    *,
    seed: int,
    length: int,
    batch_size: int,
) -> dict[str, object]:
    """Evaluate direct and standard delta memory on one identical event stream."""

    dtype = torch.float64
    device = torch.device("cpu")
    actions = actions.to(dtype=dtype, device=device)
    oracle_actions = oracle_actions.to(dtype=dtype, device=device)
    world = AliasWorld.create(seed, dtype=dtype, device=device)
    center_labels = torch.arange(SLOTS)
    write_center_slots = policy.routes(
        world.centers, center_labels, side="write"
    ).argmax(dim=-1)
    query_center_slots = policy.routes(
        world.centers, center_labels, side="query"
    ).argmax(dim=-1)
    generator = torch.Generator().manual_seed(770_000 + seed + 17 * length)
    alias_generator = torch.Generator().manual_seed(780_000 + seed + 19 * length)
    model_keys = random_unit(
        (batch_size, SLOTS, 8),
        generator=generator,
        dtype=dtype,
        device=device,
    )
    oracle_keys = model_keys.clone()
    values = torch.zeros(batch_size, SLOTS, 8, dtype=dtype)
    direct = torch.zeros_like(values)
    delta = torch.zeros_like(values)
    direct_cosines: list[torch.Tensor] = []
    delta_cosines: list[torch.Tensor] = []
    direct_errors: list[torch.Tensor] = []
    delta_errors: list[torch.Tensor] = []
    max_state_error = 0.0
    max_prediction_error = 0.0
    write_correct = 0
    write_total = 0
    query_correct = 0
    query_total = 0

    def write(batch_index: torch.Tensor, labels: torch.Tensor) -> None:
        nonlocal direct, delta, model_keys, values
        nonlocal max_state_error, write_correct, write_total
        count = int(batch_index.numel())
        if count == 0:
            return
        aliases = world.sample(labels, radius=TEST_RADIUS, generator=alias_generator)
        route, correct = _hard_route(
            policy,
            aliases,
            labels,
            side="write",
            expected_slots=write_center_slots,
        )
        value = random_unit(
            (count, 8),
            generator=generator,
            dtype=dtype,
            device=device,
        )
        direct[batch_index] = (
            (1.0 - route[..., None]) * direct[batch_index]
            + route[..., None] * value[:, None, :]
        )
        transition = delta_write_transitions(
            route[:, None],
            value[:, None],
            torch.ones(count, 1, dtype=dtype),
        )
        delta[batch_index] = apply_delta(
            transition, delta[batch_index, None]
        )[:, 0]
        max_state_error = max(
            max_state_error,
            float((direct[batch_index] - delta[batch_index]).abs().max()),
        )
        model_keys[batch_index, labels] = oracle_keys[batch_index, labels]
        values[batch_index, labels] = value
        write_correct += correct
        write_total += count

    all_batch = torch.arange(batch_size)
    for label in range(SLOTS):
        write(all_batch, torch.full((batch_size,), label, dtype=torch.long))

    for _ in range(max(0, length - SLOTS)):
        event = torch.rand(batch_size, generator=generator)
        rotate_batch = torch.nonzero(event < 0.35, as_tuple=False).flatten()
        write_batch = torch.nonzero(
            (event >= 0.35) & (event < 0.70), as_tuple=False
        ).flatten()
        query_batch = torch.nonzero(event >= 0.70, as_tuple=False).flatten()
        if rotate_batch.numel():
            tokens = torch.randint(
                TOKEN_COUNT, (rotate_batch.numel(),), generator=generator
            )
            learned = actions[tokens]
            teacher = oracle_actions[tokens]
            negative_action = learned[:, 2]
            direct[rotate_batch] = torch.einsum(
                "bij,bhj->bhi", negative_action, direct[rotate_batch]
            )
            transition = value_transport_transitions(negative_action[:, None])
            delta[rotate_batch] = apply_delta(
                transition, delta[rotate_batch, None]
            )[:, 0]
            max_state_error = max(
                max_state_error,
                float((direct[rotate_batch] - delta[rotate_batch]).abs().max()),
            )
            model_keys[rotate_batch] = torch.einsum(
                "bij,bkj->bki", learned[:, 1], model_keys[rotate_batch]
            )
            oracle_keys[rotate_batch] = torch.einsum(
                "bij,bkj->bki", teacher[:, 1], oracle_keys[rotate_batch]
            )
            values[rotate_batch] = torch.einsum(
                "bij,bkj->bki", teacher[:, 2], values[rotate_batch]
            )
        write_labels = torch.randint(
            SLOTS, (write_batch.numel(),), generator=generator
        )
        write(write_batch, write_labels)
        if query_batch.numel():
            labels = torch.randint(
                SLOTS, (query_batch.numel(),), generator=generator
            )
            aliases = world.sample(
                labels, radius=TEST_RADIUS, generator=alias_generator
            )
            route, correct = _hard_route(
                policy,
                aliases,
                labels,
                side="query",
                expected_slots=query_center_slots,
            )
            direct_prediction = torch.einsum(
                "bh,bhv->bv", route, direct[query_batch]
            )
            delta_prediction = delta_read(delta[query_batch], route)
            target = values[query_batch, labels]
            direct_cosine, direct_error = _retrieval_metrics(
                direct_prediction, target
            )
            delta_cosine, delta_error = _retrieval_metrics(delta_prediction, target)
            direct_cosines.append(direct_cosine)
            delta_cosines.append(delta_cosine)
            direct_errors.append(direct_error)
            delta_errors.append(delta_error)
            max_prediction_error = max(
                max_prediction_error,
                float((direct_prediction - delta_prediction).abs().max()),
            )
            query_correct += correct
            query_total += int(query_batch.numel())

    direct_cosine = torch.cat(direct_cosines)
    delta_cosine = torch.cat(delta_cosines)
    direct_error = torch.cat(direct_errors)
    delta_error = torch.cat(delta_errors)

    def metrics(cosine: torch.Tensor, error: torch.Tensor) -> dict[str, float | int]:
        return {
            "length": length,
            "queries": int(cosine.numel()),
            "mean_query_cosine": float(cosine.mean()),
            "minimum_query_cosine": float(cosine.min()),
            "mean_relative_squared_error": float(error.mean()),
            "maximum_relative_squared_error": float(error.max()),
        }

    return {
        "length": length,
        "direct": metrics(direct_cosine, direct_error),
        "delta": metrics(delta_cosine, delta_error),
        "direct_delta_max_state_error": max_state_error,
        "direct_delta_max_prediction_error": max_prediction_error,
        "direct_delta_max_cosine_difference": float(
            (direct_cosine - delta_cosine).abs().max()
        ),
        "direct_delta_max_relative_squared_error_difference": float(
            (direct_error - delta_error).abs().max()
        ),
        "write_route_agreement": write_correct / write_total,
        "query_route_agreement": query_correct / query_total,
        "write_center_slots": [int(value) for value in write_center_slots],
        "query_center_slots": [int(value) for value in query_center_slots],
    }


@torch.no_grad()
def paired_scan_parity(
    actions: torch.Tensor,
    policy: FrozenSlotPolicy,
    *,
    seed: int,
) -> dict[str, float | int]:
    """Compare direct and delta prefix scans on identical one-hot events."""

    dtype = torch.float64
    actions = actions.to(dtype=dtype, device=torch.device("cpu"))
    batch, length = 3, 31
    world = AliasWorld.create(seed, dtype=dtype, device=torch.device("cpu"))
    center_labels = torch.arange(SLOTS)
    write_center_slots = policy.routes(
        world.centers, center_labels, side="write"
    ).argmax(dim=-1)
    generator = torch.Generator().manual_seed(790_000 + seed)
    alias_generator = torch.Generator().manual_seed(800_000 + seed)
    identity = torch.eye(SLOTS, dtype=dtype).expand(batch, -1, -1)
    direct_retention = []
    direct_action = []
    direct_drive = []
    delta_key_action = []
    delta_value_action = []
    delta_drive = []
    route_correct = 0
    route_total = 0
    for position in range(length):
        if position % 3 == 1:
            tokens = torch.randint(TOKEN_COUNT, (batch,), generator=generator)
            negative_action = actions[tokens, 2]
            zeros = torch.zeros(batch, SLOTS, 8, dtype=dtype)
            direct_retention.append(torch.ones(batch, SLOTS, dtype=dtype))
            direct_action.append(negative_action)
            direct_drive.append(zeros)
            delta_key_action.append(identity)
            delta_value_action.append(negative_action)
            delta_drive.append(zeros)
        else:
            labels = torch.randint(SLOTS, (batch,), generator=generator)
            aliases = world.sample(
                labels, radius=TEST_RADIUS, generator=alias_generator
            )
            route, correct = _hard_route(
                policy,
                aliases,
                labels,
                side="write",
                expected_slots=write_center_slots,
            )
            value = random_unit(
                (batch, 8),
                generator=generator,
                dtype=dtype,
                device=torch.device("cpu"),
            )
            drive = route[..., None] * value[:, None, :]
            direct_retention.append(1.0 - route)
            direct_action.append(identity)
            direct_drive.append(drive)
            delta_key_action.append(
                identity - route[..., :, None] * route[..., None, :]
            )
            delta_value_action.append(identity)
            delta_drive.append(drive)
            route_correct += correct
            route_total += batch

    direct_transition = SlotTransition(
        torch.stack(direct_retention, dim=1),
        torch.stack(direct_action, dim=1),
        torch.stack(direct_drive, dim=1),
    )
    delta_transition = DeltaTransition(
        torch.stack(delta_key_action, dim=1),
        torch.stack(delta_value_action, dim=1),
        torch.stack(delta_drive, dim=1),
    )
    initial = torch.zeros(batch, SLOTS, 8, dtype=dtype)
    direct_parallel = apply_slot(
        associative_slot_scan(direct_transition), initial[:, None]
    )
    direct_state = initial
    direct_recurrent = []
    for position in range(length):
        direct_state = apply_slot(
            SlotTransition(
                direct_transition.retention[:, position],
                direct_transition.action[:, position],
                direct_transition.drive[:, position],
            ),
            direct_state,
        )
        direct_recurrent.append(direct_state)
    direct_recurrent_tensor = torch.stack(direct_recurrent, dim=1)
    delta_parallel = scanned_delta_states(
        delta_transition, initial, backend="work_efficient"
    )
    delta_recurrent = recurrent_delta_states(delta_transition, initial)
    return {
        "length": length,
        "streaming_state_scalars": SLOTS * 8,
        "hard_route_agreement": route_correct / route_total,
        "direct_parallel_recurrent_max_error": float(
            (direct_parallel - direct_recurrent_tensor).abs().max()
        ),
        "delta_parallel_recurrent_max_error": float(
            (delta_parallel - delta_recurrent).abs().max()
        ),
        "direct_delta_parallel_max_error": float(
            (direct_parallel - delta_parallel).abs().max()
        ),
        "direct_delta_recurrent_max_error": float(
            (direct_recurrent_tensor - delta_recurrent).abs().max()
        ),
    }


def _source_row(
    source: dict[str, object], seed: int
) -> dict[str, object] | None:
    matches = [row for row in source["results"] if int(row["seed"]) == seed]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"source artifact contains duplicate rows for seed {seed}")
    return matches[0]


def _soft_reproduction(
    replay: list[dict[str, object]], stored: list[dict[str, object]]
) -> dict[str, object]:
    stored_by_length = {int(row["length"]): row for row in stored}
    differences: dict[str, float] = {name: 0.0 for name in METRIC_NAMES}
    for row in replay:
        expected = stored_by_length[int(row["length"])]
        for name in METRIC_NAMES:
            differences[name] = max(
                differences[name], abs(float(row[name]) - float(expected[name]))
            )
    return {
        "max_absolute_differences": differences,
        "mean_cosine_reproduced_within_1e_12": (
            differences["mean_query_cosine"] <= 1e-12
        ),
    }


def run_seed(
    seed: int,
    *,
    source: dict[str, object],
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
    lengths = DENSE_LENGTHS if dense else (32, 128, 512, 2048)
    soft_replay = [
        evaluate_sequences(
            independent.actions,
            oracle_actions,
            independent.policy,
            mode="direct",
            seed=seed,
            length=length,
            batch_size=160 if length <= 16 else 48,
        )
        for length in lengths
    ]
    independent_paired = [
        paired_sequence_evaluation(
            independent.actions,
            oracle_actions,
            independent.policy,
            seed=seed,
            length=length,
            batch_size=160 if length <= 16 else 48,
        )
        for length in lengths
    ]
    oracle_paired = [
        paired_sequence_evaluation(
            oracle_actions,
            oracle_actions,
            independent.policy,
            seed=seed,
            length=length,
            batch_size=160 if length <= 16 else 48,
        )
        for length in lengths
    ]
    source_row = _source_row(source, seed)
    reproduction = (
        _soft_reproduction(
            soft_replay,
            source_row["variants"]["independent_direct"]["evaluation"],
        )
        if source_row is not None
        else {
            "status": "source seed unavailable; development smoke only",
            "mean_cosine_reproduced_within_1e_12": False,
        }
    )
    observed_mse = float(
        F.mse_loss(
            observed_action(independent.actions), observed_action(oracle_actions)
        )
    )
    complement = negative_subspace_metrics(
        independent.actions, oracle_actions, basis.cpu()
    )
    routing = alias_diagnostics(independent.policy, seed=seed, radius=TEST_RADIUS)
    scan = paired_scan_parity(independent.actions, independent.policy, seed=seed)
    max_state_error = max(
        float(row["direct_delta_max_state_error"]) for row in independent_paired
    )
    max_prediction_error = max(
        float(row["direct_delta_max_prediction_error"])
        for row in independent_paired
    )
    minimum_write_agreement = min(
        float(row["write_route_agreement"]) for row in independent_paired
    )
    minimum_query_agreement = min(
        float(row["query_route_agreement"]) for row in independent_paired
    )
    oracle_minimum_cosine = min(
        float(row["delta"]["minimum_query_cosine"]) for row in oracle_paired
    )
    oracle_mean_cosine = min(
        float(row["delta"]["mean_query_cosine"]) for row in oracle_paired
    )
    independent_long = float(independent_paired[-1]["delta"]["mean_query_cosine"])
    shared_long = (
        float(
            source_row["variants"]["joint_direct"]["evaluation"][-1][
                "mean_query_cosine"
            ]
        )
        if source_row is not None
        else None
    )
    implementation_smoke_passed = (
        max_state_error <= 1e-10
        and max_prediction_error <= 1e-10
        and float(scan["direct_parallel_recurrent_max_error"]) <= 1e-9
        and float(scan["delta_parallel_recurrent_max_error"]) <= 1e-9
        and minimum_write_agreement >= 0.99
        and minimum_query_agreement >= 0.99
        and int(routing["write_center_collisions"]) == 0
        and int(routing["query_center_collisions"]) == 0
        and float(routing["center_cross_encoder_agreement"]) == 1.0
        and oracle_mean_cosine >= 0.995
        and oracle_minimum_cosine >= 0.98
    )
    implementation_passed = (
        implementation_smoke_passed
        and bool(reproduction["mean_cosine_reproduced_within_1e_12"])
    )
    representation_prior_win = (
        shared_long is not None
        and observed_mse < 1e-6
        and shared_long >= 0.995
        and independent_long <= 0.90
    )
    return {
        "experiment": "Task-B independent-action delta replay",
        "protocol": "TASK_B_DELTA_ACTION_REPLAY_PREREGISTRATION.md",
        "source_commit": "c4b6310",
        "seed": seed,
        "device": "cpu",
        "dtype": "float64",
        "adam_steps_per_stage": adam_steps_per_stage,
        "lbfgs_steps": lbfgs_steps,
        "dense": dense,
        "teacher_resamples": teacher.resamples,
        "design": design,
        "training": independent.report,
        "observed_column_mse": observed_mse,
        "negative_subspaces": complement,
        "routing": routing,
        "soft_independent_direct_replay": soft_replay,
        "soft_reproduction": reproduction,
        "independent_hard_direct_delta": independent_paired,
        "oracle_negative_hard_direct_delta": oracle_paired,
        "scan": scan,
        "decision": {
            "implementation_smoke_passed": implementation_smoke_passed,
            "implementation_passed": implementation_passed,
            "representation_prior_win": representation_prior_win,
            "shared_direct_length2048_mean_cosine": shared_long,
            "independent_delta_length2048_mean_cosine": independent_long,
            "maximum_direct_delta_state_error": max_state_error,
            "maximum_direct_delta_prediction_error": max_prediction_error,
            "minimum_write_route_agreement": minimum_write_agreement,
            "minimum_query_route_agreement": minimum_query_agreement,
            "write_center_collisions": routing["write_center_collisions"],
            "query_center_collisions": routing["query_center_collisions"],
            "center_cross_encoder_agreement": routing[
                "center_cross_encoder_agreement"
            ],
            "minimum_oracle_delta_mean_cosine": oracle_mean_cosine,
            "minimum_oracle_delta_query_cosine": oracle_minimum_cosine,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("artifacts/spin8_blind_alias_action_seeds0_9.json"),
    )
    parser.add_argument("--adam-steps-per-stage", type=int, default=500)
    parser.add_argument("--lbfgs-steps", type=int, default=150)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    report = run_seed(
        args.seed,
        source=source,
        adam_steps_per_stage=args.adam_steps_per_stage,
        lbfgs_steps=args.lbfgs_steps,
        dense=args.dense,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
