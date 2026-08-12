"""Joint blind Spin(8)-action and continuous-alias completion gate."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from spin8_blind_shared_action import (
    OBSERVED_COLUMNS,
    REPRESENTATION_NAMES,
    TOKEN_COUNT,
    action_design_audit,
    observed_action,
    one_step_metrics,
    sample_teacher,
    scan_parity as action_scan_parity,
    triality_residual,
    vector_commutator,
)
from spin8_continuous_alias import (
    AliasEncoders,
    AliasWorld,
    DENSE_LENGTHS,
    TEST_RADIUS,
    TRAIN_RADII,
    VALIDATION_RADIUS,
    FrozenSlotPolicy,
    alias_diagnostics,
    balanced_labels,
    random_unit,
)
from spin8_triality import SPIN8_BIVECTOR_DIM, spin8_actions, torch_triality_generators
from spin8_triality_lift import triality_bind, triality_tensor, triality_unbind_negative
from spin8_triality_memory import SlotTransition, apply_slot, associative_slot_scan


NEGATIVE_CALIBRATION_RANK = 2


@dataclass(frozen=True)
class CalibrationBatch:
    labels: torch.Tensor
    write_alias: torch.Tensor
    query_alias: torch.Tensor
    positive: torch.Tensor
    negative: torch.Tensor
    tokens: torch.Tensor
    target_negative: torch.Tensor


def negative_calibration_basis(
    seed: int, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(730_000 + seed)
    matrix = torch.randn(8, NEGATIVE_CALIBRATION_RANK, generator=generator, dtype=torch.float64)
    basis, _ = torch.linalg.qr(matrix, mode="reduced")
    return basis.to(device=device, dtype=dtype)


def calibration_complement(basis: torch.Tensor) -> torch.Tensor:
    complete, _ = torch.linalg.qr(basis, mode="complete")
    return complete[:, NEGATIVE_CALIBRATION_RANK:]


class CombinedFamily(nn.Module):
    def __init__(
        self,
        family: str,
        *,
        seed: int,
        generators: torch.Tensor,
    ) -> None:
        super().__init__()
        if family not in {"joint", "independent"}:
            raise ValueError(f"unknown action family: {family}")
        self.family = family
        self.register_buffer("generators", generators)
        generator = torch.Generator(device=generators.device).manual_seed(740_000 + seed)
        shape = (
            (TOKEN_COUNT, SPIN8_BIVECTOR_DIM)
            if family == "joint"
            else (TOKEN_COUNT, 3, SPIN8_BIVECTOR_DIM)
        )
        initialization = 0.01 * torch.randn(
            *shape,
            generator=generator,
            dtype=generators.dtype,
            device=generators.device,
        )
        self.coordinates = nn.Parameter(initialization)
        self.alias = AliasEncoders(
            seed=seed, dtype=generators.dtype, device=generators.device
        )

    def actions(self) -> torch.Tensor:
        if self.family == "joint":
            return spin8_actions(self.coordinates, self.generators)
        tangent = torch.einsum(
            "trk,rkij->trij", self.coordinates, self.generators
        ).contiguous()
        return torch.matrix_exp(tangent)

    def routes(
        self, aliases: torch.Tensor, *, side: str, temperature: float
    ) -> torch.Tensor:
        return self.alias.routes(aliases, side=side, temperature=temperature)

    def frozen_policy(self) -> FrozenSlotPolicy:
        return FrozenSlotPolicy(
            variant="learned_both_joint",
            write_weight=self.alias.write_weight.detach().cpu(),
            query_weight=self.alias.query_weight.detach().cpu(),
        )


def sample_calibration_batch(
    *,
    world: AliasWorld,
    basis: torch.Tensor,
    teacher_actions: torch.Tensor,
    radius: float,
    repeats: int,
    generator: torch.Generator,
) -> CalibrationBatch:
    device, dtype = world.centers.device, world.centers.dtype
    labels = balanced_labels(repeats, generator=generator, device=device)
    write_alias = world.sample(labels, radius=radius, generator=generator)
    query_alias = world.sample(labels, radius=radius, generator=generator)
    positive = random_unit(
        (labels.numel(), 8), generator=generator, dtype=dtype, device=device
    )
    coefficients = random_unit(
        (labels.numel(), NEGATIVE_CALIBRATION_RANK),
        generator=generator,
        dtype=dtype,
        device=device,
    )
    negative = torch.einsum("bi,di->bd", coefficients, basis)
    tokens = torch.randint(
        0, TOKEN_COUNT, (labels.numel(),), generator=generator, device=device
    )
    target_negative = torch.einsum(
        "bij,bj->bi", teacher_actions[tokens, 2], negative
    )
    return CalibrationBatch(
        labels,
        write_alias,
        query_alias,
        positive,
        negative,
        tokens,
        target_negative,
    )


def combined_objective(
    model: CombinedFamily,
    batch: CalibrationBatch,
    *,
    target_observation: torch.Tensor,
    rho: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    actions = model.actions()
    write_routes = model.routes(
        batch.write_alias, side="write", temperature=temperature
    )
    query_routes = model.routes(
        batch.query_alias, side="query", temperature=temperature
    )
    selected = actions[batch.tokens]
    vector, positive_action, negative_action = selected.unbind(dim=1)

    bound = triality_bind(batch.positive, batch.negative, rho)
    memory = write_routes[..., None] * bound[:, None, :]
    transported_memory = torch.einsum("bij,bhj->bhi", vector, memory)
    transported_positive = torch.einsum(
        "bij,bj->bi", positive_action, batch.positive
    )
    candidates = triality_unbind_negative(
        transported_positive[:, None, :], transported_memory, rho
    )
    binding_prediction = (query_routes[..., None] * candidates).sum(dim=1)

    overlap = (write_routes * query_routes).sum(dim=-1)
    direct_prediction = overlap[:, None] * torch.einsum(
        "bij,bj->bi", negative_action, batch.negative
    )
    action_observation = F.mse_loss(observed_action(actions), target_observation)
    binding_endpoint = F.mse_loss(binding_prediction, batch.target_negative)
    direct_endpoint = F.mse_loss(direct_prediction, batch.target_negative)
    vertex = 0.5 * (
        (write_routes * (1.0 - write_routes)).mean()
        + (query_routes * (1.0 - query_routes)).mean()
    )
    uniform = 1.0 / write_routes.shape[-1]
    balance = (
        (write_routes.mean(dim=0) - uniform).square().sum()
        + (query_routes.mean(dim=0) - uniform).square().sum()
    )
    loss = (
        20.0 * action_observation
        + binding_endpoint
        + direct_endpoint
        + 0.02 * vertex
        + 8.0 * balance
    )
    return loss, {
        "action_observation": action_observation,
        "binding_endpoint": binding_endpoint,
        "direct_endpoint": direct_endpoint,
        "vertex": vertex,
        "balance": balance,
    }


@dataclass(frozen=True)
class TrainedCombined:
    actions: torch.Tensor
    policy: FrozenSlotPolicy
    coordinates: torch.Tensor
    report: dict[str, object]


def train_combined(
    family: str,
    *,
    seed: int,
    generators: torch.Tensor,
    rho: torch.Tensor,
    teacher_actions: torch.Tensor,
    basis: torch.Tensor,
    adam_steps_per_stage: int,
    lbfgs_steps: int,
    repeats: int = 16,
) -> TrainedCombined:
    device, dtype = generators.device, generators.dtype
    world = AliasWorld.create(seed, dtype=dtype, device=device)
    model = CombinedFamily(family, seed=seed, generators=generators)
    target_observation = observed_action(teacher_actions).detach()
    optimizer = torch.optim.Adam(
        [
            {"params": (model.coordinates,), "lr": 0.03},
            {
                "params": (model.alias.write_weight, model.alias.query_weight),
                "lr": 0.06,
            },
        ]
    )
    generator = torch.Generator(device=device).manual_seed(750_000 + seed)
    total_steps = len(TRAIN_RADII) * adam_steps_per_stage
    trajectory: list[dict[str, float | int | str]] = []
    final_terms: dict[str, torch.Tensor] = {}
    for stage, radius in enumerate(TRAIN_RADII):
        for step in range(adam_steps_per_stage):
            index = stage * adam_steps_per_stage + step
            fraction = index / max(1, total_steps - 1)
            temperature = 0.50 * (0.07 / 0.50) ** fraction
            batch = sample_calibration_batch(
                world=world,
                basis=basis,
                teacher_actions=teacher_actions,
                radius=radius,
                repeats=repeats,
                generator=generator,
            )
            loss, terms = combined_objective(
                model,
                batch,
                target_observation=target_observation,
                rho=rho,
                temperature=temperature,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_terms = terms
            if index in {0, 49, 199, 499, total_steps - 1}:
                trajectory.append(
                    {"stage": "adam", "step": index + 1, "loss": float(loss.detach())}
                )

    fixed_generator = torch.Generator(device=device).manual_seed(760_000 + seed)
    fixed = sample_calibration_batch(
        world=world,
        basis=basis,
        teacher_actions=teacher_actions,
        radius=TRAIN_RADII[-1],
        repeats=32,
        generator=fixed_generator,
    )
    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=lbfgs_steps,
        tolerance_grad=1e-13,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        lbfgs.zero_grad(set_to_none=True)
        loss, _ = combined_objective(
            model,
            fixed,
            target_observation=target_observation,
            rho=rho,
            temperature=0.05,
        )
        loss.backward()
        return loss

    lbfgs.step(closure)
    final_loss, final_terms = combined_objective(
        model,
        fixed,
        target_observation=target_observation,
        rho=rho,
        temperature=0.05,
    )
    trajectory.append(
        {"stage": "lbfgs", "step": lbfgs_steps, "loss": float(final_loss.detach())}
    )
    return TrainedCombined(
        actions=model.actions().detach().cpu(),
        policy=model.frozen_policy(),
        coordinates=model.coordinates.detach().cpu(),
        report={
            "family": family,
            "adam_steps": total_steps,
            "lbfgs_steps": lbfgs_steps,
            "trajectory": trajectory,
            "final_loss": float(final_loss.detach()),
            **{f"final_{key}": float(value.detach()) for key, value in final_terms.items()},
        },
    )


def jacobian_rank(function, point: torch.Tensor) -> tuple[int, float]:
    variable = point.detach().clone().requires_grad_(True)
    jacobian = torch.autograd.functional.jacobian(function, variable, vectorize=True)
    singular = torch.linalg.svdvals(jacobian.reshape(-1, variable.numel()))
    nonzero = singular[singular > 1e-9 * singular[0]]
    return int(nonzero.numel()), float(nonzero[0] / nonzero[-1])


def combined_design_audit(
    teacher_coefficients: torch.Tensor,
    generators: torch.Tensor,
    basis: torch.Tensor,
) -> dict[str, object]:
    shared = action_design_audit(teacher_coefficients, generators)
    rows = []
    for token in range(TOKEN_COUNT):
        point = teacher_coefficients[token]
        vector_rank, vector_condition = jacobian_rank(
            lambda coordinate: spin8_actions(coordinate, generators)[0, :, :OBSERVED_COLUMNS],
            point,
        )
        positive_rank, positive_condition = jacobian_rank(
            lambda coordinate: spin8_actions(coordinate, generators)[1, :, :OBSERVED_COLUMNS],
            point,
        )
        negative_rank, negative_condition = jacobian_rank(
            lambda coordinate: spin8_actions(coordinate, generators)[2] @ basis,
            point,
        )
        rows.append(
            {
                "token": token,
                "vector_rank": vector_rank,
                "positive_rank": positive_rank,
                "negative_rank2_endpoint_rank": negative_rank,
                "independent_total_rank": vector_rank + positive_rank + negative_rank,
                "independent_slack_dimension": 3 * SPIN8_BIVECTOR_DIM
                - vector_rank
                - positive_rank
                - negative_rank,
                "maximum_condition_number": max(
                    vector_condition, positive_condition, negative_condition
                ),
            }
        )
    return {
        "shared": shared,
        "independent": rows,
        "minimum_shared_rank": int(shared["minimum_rank"]),
        "independent_rank_pattern": sorted(
            {
                (
                    row["vector_rank"],
                    row["positive_rank"],
                    row["negative_rank2_endpoint_rank"],
                )
                for row in rows
            }
        ),
        "minimum_independent_slack": min(
            int(row["independent_slack_dimension"]) for row in rows
        ),
    }


@torch.no_grad()
def negative_subspace_metrics(
    actions: torch.Tensor, oracle: torch.Tensor, basis: torch.Tensor
) -> dict[str, float]:
    complement = calibration_complement(basis)

    def metrics(vectors: torch.Tensor) -> tuple[float, float]:
        predicted = torch.einsum("tij,jk->tki", actions[:, 2], vectors)
        expected = torch.einsum("tij,jk->tki", oracle[:, 2], vectors)
        cosine = F.cosine_similarity(predicted, expected, dim=-1)
        return float(cosine.mean()), float(cosine.min())

    calibration_mean, calibration_min = metrics(basis)
    complement_mean, complement_min = metrics(complement)
    return {
        "calibration_mean_cosine": calibration_mean,
        "calibration_minimum_cosine": calibration_min,
        "complement_mean_cosine": complement_mean,
        "complement_minimum_cosine": complement_min,
    }


@torch.no_grad()
def evaluate_sequences(
    actions: torch.Tensor,
    oracle_actions: torch.Tensor,
    policy: FrozenSlotPolicy,
    *,
    mode: str,
    seed: int,
    length: int,
    batch_size: int,
) -> dict[str, float | int]:
    if mode not in {"binding", "direct"}:
        raise ValueError(f"unknown memory mode: {mode}")
    dtype = torch.float64
    device = torch.device("cpu")
    actions = actions.to(dtype=dtype, device=device)
    oracle_actions = oracle_actions.to(dtype=dtype, device=device)
    world = AliasWorld.create(seed, dtype=dtype, device=device)
    rho = triality_tensor(dtype=dtype)
    generator = torch.Generator().manual_seed(770_000 + seed + 17 * length)
    alias_generator = torch.Generator().manual_seed(780_000 + seed + 19 * length)
    model_keys = random_unit(
        (batch_size, 8, 8), generator=generator, dtype=dtype, device=device
    )
    oracle_keys = model_keys.clone()
    values = torch.zeros(batch_size, 8, 8, dtype=dtype)
    memory = torch.zeros(batch_size, 8, 8, dtype=dtype)
    cosines: list[torch.Tensor] = []
    errors: list[torch.Tensor] = []

    def write(batch_index: torch.Tensor, labels: torch.Tensor) -> None:
        nonlocal memory, model_keys, oracle_keys, values
        count = int(batch_index.numel())
        if count == 0:
            return
        aliases = world.sample(labels, radius=TEST_RADIUS, generator=alias_generator)
        route = policy.routes(aliases, labels, side="write")
        value = random_unit(
            (count, 8), generator=generator, dtype=dtype, device=device
        )
        supplied_key = oracle_keys[batch_index, labels]
        payload = (
            triality_bind(supplied_key, value, rho) if mode == "binding" else value
        )
        memory[batch_index] = (
            (1.0 - route[..., None]) * memory[batch_index]
            + route[..., None] * payload[:, None, :]
        )
        model_keys[batch_index, labels] = supplied_key
        values[batch_index, labels] = value

    all_batch = torch.arange(batch_size)
    for label in range(8):
        write(all_batch, torch.full((batch_size,), label, dtype=torch.long))

    for _ in range(max(0, length - 8)):
        event = torch.rand(batch_size, generator=generator)
        rotate_batch = torch.nonzero(event < 0.35, as_tuple=False).flatten()
        write_batch = torch.nonzero((event >= 0.35) & (event < 0.70), as_tuple=False).flatten()
        query_batch = torch.nonzero(event >= 0.70, as_tuple=False).flatten()
        if rotate_batch.numel():
            tokens = torch.randint(TOKEN_COUNT, (rotate_batch.numel(),), generator=generator)
            learned = actions[tokens]
            teacher = oracle_actions[tokens]
            memory_action = learned[:, 0] if mode == "binding" else learned[:, 2]
            memory[rotate_batch] = torch.einsum(
                "bij,bhj->bhi", memory_action, memory[rotate_batch]
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
        write_labels = torch.randint(8, (write_batch.numel(),), generator=generator)
        write(write_batch, write_labels)
        if query_batch.numel():
            labels = torch.randint(8, (query_batch.numel(),), generator=generator)
            aliases = world.sample(labels, radius=TEST_RADIUS, generator=alias_generator)
            route = policy.routes(aliases, labels, side="query")
            if mode == "binding":
                candidates = triality_unbind_negative(
                    model_keys[query_batch, labels, None, :], memory[query_batch], rho
                )
            else:
                candidates = memory[query_batch]
            prediction = (route[..., None] * candidates).sum(dim=1)
            target = values[query_batch, labels]
            cosines.append(F.cosine_similarity(prediction, target, dim=-1))
            errors.append(
                (prediction - target).square().sum(dim=-1)
                / target.square().sum(dim=-1).clamp_min(1e-30)
            )
    cosine = torch.cat(cosines)
    error = torch.cat(errors)
    return {
        "length": length,
        "queries": int(cosine.numel()),
        "mean_query_cosine": float(cosine.mean()),
        "minimum_query_cosine": float(cosine.min()),
        "mean_relative_squared_error": float(error.mean()),
        "maximum_relative_squared_error": float(error.max()),
    }


@torch.no_grad()
def slot_scan_parity(
    actions: torch.Tensor,
    policy: FrozenSlotPolicy,
    *,
    mode: str,
    seed: int,
) -> dict[str, float | int]:
    dtype = torch.float64
    batch, length = 3, 31
    actions = actions.to(dtype=dtype)
    world = AliasWorld.create(seed, dtype=dtype, device=torch.device("cpu"))
    rho = triality_tensor(dtype=dtype)
    generator = torch.Generator().manual_seed(790_000 + seed)
    alias_generator = torch.Generator().manual_seed(800_000 + seed)
    keys = random_unit(
        (batch, 8, 8),
        generator=generator,
        dtype=dtype,
        device=torch.device("cpu"),
    )
    retention_steps, action_steps, drive_steps = [], [], []
    eye = torch.eye(8, dtype=dtype).expand(batch, -1, -1)
    for position in range(length):
        if position % 3 == 1:
            tokens = torch.randint(TOKEN_COUNT, (batch,), generator=generator)
            selected = actions[tokens]
            retention_steps.append(torch.ones(batch, 8, dtype=dtype))
            action_steps.append(selected[:, 0] if mode == "binding" else selected[:, 2])
            drive_steps.append(torch.zeros(batch, 8, 8, dtype=dtype))
            keys = torch.einsum("bij,bkj->bki", selected[:, 1], keys)
        else:
            labels = torch.randint(8, (batch,), generator=generator)
            aliases = world.sample(labels, radius=TEST_RADIUS, generator=alias_generator)
            route = policy.routes(aliases, labels, side="write")
            value = random_unit(
                (batch, 8),
                generator=generator,
                dtype=dtype,
                device=torch.device("cpu"),
            )
            key = keys[torch.arange(batch), labels]
            payload = triality_bind(key, value, rho) if mode == "binding" else value
            retention_steps.append(1.0 - route)
            action_steps.append(eye)
            drive_steps.append(route[..., None] * payload[:, None, :])
    transition = SlotTransition(
        torch.stack(retention_steps, dim=1),
        torch.stack(action_steps, dim=1),
        torch.stack(drive_steps, dim=1),
    )
    initial = torch.zeros(batch, 8, 8, dtype=dtype)
    parallel = apply_slot(associative_slot_scan(transition), initial[:, None])
    state = initial
    recurrent = []
    for position in range(length):
        state = apply_slot(
            SlotTransition(
                transition.retention[:, position],
                transition.action[:, position],
                transition.drive[:, position],
            ),
            state,
        )
        recurrent.append(state)
    return {
        "streaming_state_scalars": 64,
        "parallel_recurrent_max_error": float(
            (parallel - torch.stack(recurrent, dim=1)).abs().max()
        ),
    }


def variant_report(
    name: str,
    actions: torch.Tensor,
    oracle_actions: torch.Tensor,
    policy: FrozenSlotPolicy,
    *,
    mode: str,
    seed: int,
    basis: torch.Tensor,
    dense: bool,
    training: dict[str, object] | None,
) -> dict[str, object]:
    lengths = DENSE_LENGTHS if dense else (32, 128, 512, 2048)
    aliases = alias_diagnostics(policy, seed=seed, radius=TEST_RADIUS)
    action_metrics = one_step_metrics(actions, oracle_actions, seed=810_000 + seed)
    subspace = negative_subspace_metrics(actions, oracle_actions, basis)
    evaluation = [
        evaluate_sequences(
            actions,
            oracle_actions,
            policy,
            mode=mode,
            seed=seed,
            length=length,
            batch_size=160 if length <= 16 else 48,
        )
        for length in lengths
    ]
    parity = slot_scan_parity(actions, policy, mode=mode, seed=seed)
    rho = triality_tensor(dtype=torch.float64)
    observed_mse = float(
        F.mse_loss(observed_action(actions), observed_action(oracle_actions))
    )
    triality = triality_residual(actions, rho, seed=820_000 + seed)
    commutator_ratio = vector_commutator(actions) / vector_commutator(oracle_actions)
    passed = (
        observed_mse < 1e-8
        and aliases["write_center_collisions"] == 0
        and aliases["query_center_collisions"] == 0
        and aliases["center_cross_encoder_agreement"] == 1.0
        and aliases["write_alias_agreement"] >= 0.99
        and aliases["query_alias_agreement"] >= 0.99
        and action_metrics["negative"]["mean_cosine"] >= 0.9999
        and subspace["complement_mean_cosine"] >= 0.9999
        and min(row["mean_query_cosine"] for row in evaluation) >= 0.995
        and min(row["minimum_query_cosine"] for row in evaluation) >= 0.98
        and max(row["mean_relative_squared_error"] for row in evaluation) < 1e-3
        and triality < 1e-8
        and commutator_ratio >= 0.90
        and parity["parallel_recurrent_max_error"] < 1e-9
        and parity["streaming_state_scalars"] == 64
    )
    return {
        "name": name,
        "mode": mode,
        "training": training,
        "observed_column_mse": observed_mse,
        "aliases": aliases,
        "one_step_action": action_metrics,
        "negative_subspaces": subspace,
        "triality_equivariance_max_error": triality,
        "commutator_ratio_to_oracle": commutator_ratio,
        "action_scan_parallel_recurrent_max_error": action_scan_parity(
            actions, seed=830_000 + seed
        ),
        "evaluation": evaluation,
        "scan": parity,
        "passed": passed,
    }


def run_seed(
    seed: int,
    *,
    device: torch.device,
    adam_steps_per_stage: int,
    lbfgs_steps: int,
    dense: bool,
) -> dict[str, object]:
    dtype = torch.float64
    generators = torch_triality_generators(dtype=dtype, device=device)
    rho = triality_tensor(dtype=dtype, device=device)
    teacher = sample_teacher(seed=seed, generators=generators)
    basis = negative_calibration_basis(seed, dtype=dtype, device=device)
    design = combined_design_audit(teacher.coefficients, generators, basis)
    if (
        design["minimum_shared_rank"] != 28
        or design["independent_rank_pattern"] != [(25, 25, 13)]
        or design["minimum_independent_slack"] != 21
    ):
        raise RuntimeError(f"invalid prospective design for seed {seed}: {design}")
    joint = train_combined(
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
    oracle_policy = FrozenSlotPolicy("oracle_both", None, None)
    variants = {
        "oracle_triality": variant_report(
            "oracle_triality", oracle_actions, oracle_actions, oracle_policy,
            mode="binding", seed=seed, basis=basis_cpu, dense=dense, training=None
        ),
        "oracle_action_learned_alias": variant_report(
            "oracle_action_learned_alias", oracle_actions, oracle_actions, joint.policy,
            mode="binding", seed=seed, basis=basis_cpu, dense=dense,
            training=joint.report
        ),
        "joint_triality": variant_report(
            "joint_triality", joint.actions, oracle_actions, joint.policy,
            mode="binding", seed=seed, basis=basis_cpu, dense=dense,
            training=joint.report
        ),
        "joint_direct": variant_report(
            "joint_direct", joint.actions, oracle_actions, joint.policy,
            mode="direct", seed=seed, basis=basis_cpu, dense=dense,
            training=joint.report
        ),
        "independent_binding": variant_report(
            "independent_binding", independent.actions, oracle_actions, independent.policy,
            mode="binding", seed=seed, basis=basis_cpu, dense=dense,
            training=independent.report
        ),
        "independent_direct": variant_report(
            "independent_direct", independent.actions, oracle_actions, independent.policy,
            mode="direct", seed=seed, basis=basis_cpu, dense=dense,
            training=independent.report
        ),
        "direct_negative_oracle": variant_report(
            "direct_negative_oracle", oracle_actions, oracle_actions, joint.policy,
            mode="direct", seed=seed, basis=basis_cpu, dense=dense,
            training=joint.report
        ),
        "joint_action_oracle_alias": variant_report(
            "joint_action_oracle_alias", joint.actions, oracle_actions, oracle_policy,
            mode="binding", seed=seed, basis=basis_cpu, dense=dense,
            training=joint.report
        ),
    }
    return {
        "seed": seed,
        "teacher_resamples": teacher.resamples,
        "design": design,
        "joint_coordinate_cosine_with_teacher": float(
            F.cosine_similarity(
                joint.coordinates.flatten(), teacher.coefficients.detach().cpu().flatten(), dim=0
            )
        ),
        "variants": variants,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    names = tuple(rows[0]["variants"])
    passes = {
        name: sum(bool(row["variants"][name]["passed"]) for row in rows)
        for name in names
    }
    complement_wins = 0
    direct_long_wins = 0
    binding_retrieval_parity = 0
    controls_fit = 0
    for row in rows:
        joint = row["variants"]["joint_triality"]
        joint_direct = row["variants"]["joint_direct"]
        binding = row["variants"]["independent_binding"]
        direct = row["variants"]["independent_direct"]
        joint_complement = joint["negative_subspaces"]["complement_mean_cosine"]
        independent_complement = direct["negative_subspaces"][
            "complement_mean_cosine"
        ]
        complement_wins += (
            joint_complement >= 0.9999
            and joint_complement - independent_complement >= 0.05
        )
        joint_long = joint_direct["evaluation"][-1]["mean_query_cosine"]
        direct_long = direct["evaluation"][-1]["mean_query_cosine"]
        direct_long_wins += joint_long >= 0.995 and direct_long <= 0.90
        binding_retrieval_parity += (
            abs(joint_long - binding["evaluation"][-1]["mean_query_cosine"])
            <= 1e-3
        )
        report = binding["training"]
        controls_fit += (
            binding["observed_column_mse"] < 1e-6
            and report["final_binding_endpoint"] < 1e-6
            and report["final_direct_endpoint"] < 1e-6
        )
    gates = {
        "joint_reliability": passes["joint_triality"] >= 8,
        "joint_direct_reliability": passes["joint_direct"] >= 8,
        "oracle_alias_decomposition": passes["oracle_action_learned_alias"] >= 8,
        "oracle_action_decomposition": passes["joint_action_oracle_alias"] >= 8,
        "controls_fit_training": controls_fit == len(rows),
        "joint_complement_advantage": complement_wins == len(rows),
        "joint_direct_length2048_advantage": direct_long_wins >= 8,
    }
    return {
        "seeds": len(rows),
        "passes": passes,
        "controls_fit_training": controls_fit,
        "joint_complement_wins": complement_wins,
        "joint_direct_length2048_wins": direct_long_wins,
        "independent_binding_retrieval_parity": binding_retrieval_parity,
        "gates": gates,
        "passed": all(gates.values()),
    }


def run(
    seeds: list[int],
    *,
    device: torch.device,
    adam_steps_per_stage: int,
    lbfgs_steps: int,
    dense: bool,
) -> dict[str, object]:
    rows = [
        run_seed(
            seed,
            device=device,
            adam_steps_per_stage=adam_steps_per_stage,
            lbfgs_steps=lbfgs_steps,
            dense=dense,
        )
        for seed in seeds
    ]
    return {
        "experiment": "joint blind Spin(8) action and continuous alias completion",
        "device": str(device),
        "adam_steps_per_stage": adam_steps_per_stage,
        "lbfgs_steps": lbfgs_steps,
        "negative_calibration_rank": NEGATIVE_CALIBRATION_RANK,
        "results": rows,
        "summary": summarize(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--adam-steps-per-stage", type=int, default=500)
    parser.add_argument("--lbfgs-steps", type=int, default=150)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/spin8_blind_alias_action_seeds0_9.json"),
    )
    args = parser.parse_args()
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    report = run(
        args.seeds,
        device=torch.device(device_name),
        adam_steps_per_stage=args.adam_steps_per_stage,
        lbfgs_steps=args.lbfgs_steps,
        dense=args.dense,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
