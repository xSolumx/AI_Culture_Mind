"""Train A5 rotor actions with automatic exact joint representation retraction."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import itertools
import json
import math
import os
from pathlib import Path
import time

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn

from changed_generator_transfer import select_changed_generators
from compare_recurrences import GROUPS, FiniteGroup, make_group_batches, parse_input_elements
from joint_a5_rounding import anchor_mechanism_diagnostics, compiled_actions
from latent_group_discovery import (
    RecoveredPermutationGroup,
    TransitionEvidence,
    inverse_cover_partial_evidence,
)
from endpoint_group_discovery import (
    EndpointRecoveryReport,
    GroupEndpointOracle,
    recover_from_endpoint_queries,
)
from endpoint_representation_discovery import (
    EndpointManifoldRecovery,
    recover_endpoint_manifold,
)
from mechanistic_group_actions import (
    PureGroupActionModel,
    _compose_word,
    _rotation_matrix_to_bivector,
    canonical_group_words,
    path_holonomy_objectives,
)
from representation_retraction import (
    CompiledRepresentation,
    RepresentationCandidate,
    compile_nearest_representation,
    local_joint_conjugacy_retraction,
    regular_irrep_candidates,
    token_commutator_max,
)
from robust_channel_gating import final_states


DENSE_LENGTHS = tuple(range(16, 257, 16))
INPUT_LABELS = ("23145", "31245", "23451", "51234")
UNTOUCHED_CLASS_INDEX = 22


def masked_forward(
    model: PureGroupActionModel,
    token_ids: torch.Tensor,
    channel_mask: torch.Tensor,
) -> torch.Tensor:
    actions = model.action_matrices()
    state = model.initial_state(token_ids.shape[0])
    states = []
    for position in range(token_ids.shape[1]):
        selected = actions[token_ids[:, position]]
        state = torch.einsum("bcij,bcj->bci", selected, state)
        states.append(state)
    sequence = torch.stack(states, dim=1)
    return model.decode(sequence * channel_mask[None, None, :, None])


def _set_anchor_actions(
    model: PureGroupActionModel,
    channel: int,
    rotations: np.ndarray,
) -> None:
    parameters = np.stack(
        [
            _rotation_matrix_to_bivector(rotation, model.max_rotor_angle)
            for rotation in rotations
        ]
    )
    values = torch.as_tensor(
        parameters,
        dtype=model.action_parameters.dtype,
        device=model.action_parameters.device,
    )
    with torch.no_grad():
        model.action_parameters[:, channel].copy_(values)


def _channel_vectors(model: PureGroupActionModel) -> np.ndarray:
    return (
        model.action_matrices()
        .detach()[:, :, 1:4, 1:4]
        .cpu()
        .double()
        .numpy()
    )


def discover_representation(
    model: PureGroupActionModel,
    group: FiniteGroup,
    input_elements: tuple[int, ...],
    candidates: tuple[RepresentationCandidate, ...],
    *,
    seed: int,
) -> tuple[int, CompiledRepresentation, float] | None:
    vectors = _channel_vectors(model)
    fits = []
    for channel in range(model.channels):
        compiled = compile_nearest_representation(
            vectors[:, channel],
            group,
            input_elements,
            seed=73_010 + 1_003 * seed + channel,
            candidates=candidates,
        )
        commutator = token_commutator_max(vectors[:, channel])
        fits.append((compiled.alignment_rms, channel, compiled, commutator))
    fits.sort(key=lambda item: item[0])
    rms, channel, compiled, commutator = fits[0]
    if (
        rms <= 0.08
        and compiled.runner_up_rms - rms >= 0.20
        and commutator >= 0.50
    ):
        return channel, compiled, commutator
    return None


def discover_endpoint_manifold_representation(
    model: PureGroupActionModel,
    words: np.ndarray,
    endpoint_labels: np.ndarray,
    *,
    state_count: int,
    seed: int,
    true_group: FiniteGroup | None = None,
) -> tuple[
    tuple[int, CompiledRepresentation, float, EndpointManifoldRecovery] | None,
    list[dict[str, object]],
]:
    """Compile a joint finite representation without a supplied Cayley table."""

    vectors = _channel_vectors(model)
    fits = []
    audit: list[dict[str, object]] = []
    for channel in range(model.channels):
        try:
            recovery = recover_endpoint_manifold(
                vectors[:, channel],
                words,
                endpoint_labels,
                state_count=state_count,
            )
            compiled = compile_nearest_representation(
                vectors[:, channel],
                recovery.group,
                recovery.input_elements,
                seed=83_010 + 1_003 * seed + channel,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            audit.append(
                {
                    "channel": channel,
                    "structural_recovery": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            continue
        commutator = token_commutator_max(vectors[:, channel])
        exact_isomorphism = None
        if true_group is not None:
            mapping = recovery.label_to_element
            exact_isomorphism = bool(
                np.array_equal(
                    recovery.group.table[mapping[:, None], mapping[None, :]],
                    mapping[true_group.table],
                )
            )
        runner_up_gap = compiled.runner_up_rms - compiled.alignment_rms
        passes = {
            "alignment": bool(compiled.alignment_rms <= 0.08),
            "runner_up_gap": bool(runner_up_gap >= 0.20),
            "commutator": bool(commutator >= 0.50),
            "assignment_gap": bool(recovery.minimum_assignment_gap >= 0.10),
            "product_residual": bool(recovery.multiplication_max <= 0.20),
        }
        audit.append(
            {
                "channel": channel,
                "structural_recovery": True,
                "alignment_rms": compiled.alignment_rms,
                "runner_up_rms": compiled.runner_up_rms,
                "runner_up_gap": runner_up_gap,
                "commutator_separation": commutator,
                "class_consistency_rms": recovery.class_consistency_rms,
                "minimum_center_separation": recovery.minimum_center_separation,
                "multiplication_rms": recovery.multiplication_rms,
                "multiplication_max": recovery.multiplication_max,
                "minimum_assignment_gap": recovery.minimum_assignment_gap,
                "exact_isomorphism_to_a5_posthoc": exact_isomorphism,
                "thresholds_passed": passes,
                "all_numerical_thresholds_passed": bool(all(passes.values())),
            }
        )
        fits.append(
            (compiled.alignment_rms, channel, compiled, commutator, recovery)
        )
    if not fits:
        return None, audit
    fits.sort(key=lambda item: item[0])
    rms, channel, compiled, commutator, recovery = fits[0]
    for item in audit:
        item["selected_by_alignment"] = bool(item.get("channel") == channel)
    if (
        rms <= 0.08
        and compiled.runner_up_rms - rms >= 0.20
        and commutator >= 0.50
        and recovery.minimum_assignment_gap >= 0.10
        and recovery.multiplication_max <= 0.20
    ):
        return (channel, compiled, commutator, recovery), audit
    return None, audit


@torch.no_grad()
def evaluate_anchor(
    model: PureGroupActionModel,
    group: FiniteGroup,
    input_elements: tuple[int, ...],
    anchor: int,
    *,
    generator_class: int | None,
    lengths: tuple[int, ...],
    batches: int,
    batch_size: int,
    seed_base: int,
    device: torch.device,
) -> dict[str, object]:
    actions = model.action_matrices()
    eval_elements, eval_actions = compiled_actions(
        actions, group, input_elements, generator_class
    )
    mask = torch.zeros(model.channels, dtype=actions.dtype, device=device)
    mask[anchor] = 1.0
    canonical_words = canonical_group_words(group, input_elements)
    canonical_anchor_actions = torch.stack(
        [_compose_word(actions[:, anchor : anchor + 1], word)[0]
         for word in canonical_words]
    )
    initial_anchor = model.initial_state(1)[0, anchor]
    by_length = {}
    maximum_norm_error = 0.0
    squared_path_drift = 0.0
    path_drift_scalars = 0
    maximum_path_drift = 0.0
    for length in lengths:
        correct = examples = 0
        generated = make_group_batches(
            group,
            batches,
            batch_size,
            length,
            seed_base + length,
            input_elements=eval_elements,
        )
        for tokens, targets in generated:
            tokens = tokens.to(device)
            targets = targets[:, -1].to(device)
            state = final_states(model, eval_actions, tokens)
            predictions = model.decode(state * mask[None, :, None]).argmax(-1)
            correct += int((predictions == targets).sum())
            examples += len(tokens)
            direct = torch.einsum(
                "bij,j->bi", canonical_anchor_actions[targets], initial_anchor
            )
            drift = (state[:, anchor] - direct).norm(dim=-1)
            squared_path_drift += float(drift.square().sum())
            path_drift_scalars += len(drift)
            maximum_path_drift = max(maximum_path_drift, float(drift.max()))
            maximum_norm_error = max(
                maximum_norm_error,
                float((state[:, anchor].norm(dim=-1) - 1.0).abs().max()),
            )
        by_length[str(length)] = correct / examples
    return {
        "by_length": by_length,
        "minimum_accuracy": min(by_length.values()),
        "mean_accuracy": sum(by_length.values()) / len(by_length),
        "gate_pass": min(by_length.values()) >= 0.90,
        "maximum_anchor_state_norm_error": maximum_norm_error,
        "path_vs_canonical_state_drift_rms": math.sqrt(
            squared_path_drift / path_drift_scalars
        ),
        "path_vs_canonical_state_drift_max": maximum_path_drift,
    }


def train_seed(
    seed: int,
    device: torch.device,
    output_directory: Path,
    *,
    table_blind: bool = False,
    endpoint_only: bool = False,
    endpoint_representative_samples: int = 1_024,
    endpoint_length_curriculum: bool = False,
    endpoint_length_mixture: bool = False,
    endpoint_scrambled_blocks: bool = False,
    endpoint_manifold_compiler: bool = False,
    training_steps: int = 2_000,
    inverse_cover_calibration: float | None = None,
    inverse_cover_calibration_pairs_total: int | None = None,
    untouched_class_index: int = UNTOUCHED_CLASS_INDEX,
) -> dict[str, object]:
    if training_steps < 1:
        raise ValueError("training_steps must be positive")
    scheduled_endpoint_training = (
        endpoint_length_curriculum
        or endpoint_length_mixture
        or endpoint_scrambled_blocks
    )
    if scheduled_endpoint_training and training_steps != 2_000:
        raise ValueError("endpoint length schedules require exactly 2,000 steps")
    if sum(
        (
            endpoint_length_curriculum,
            endpoint_length_mixture,
            endpoint_scrambled_blocks,
        )
    ) > 1:
        raise ValueError("choose exactly one endpoint length schedule")
    if endpoint_manifold_compiler and not endpoint_length_curriculum:
        raise ValueError("endpoint manifold compilation requires the frozen curriculum")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    group = GROUPS["a5"]
    input_elements = parse_input_elements(INPUT_LABELS, group)
    latent_compiler = table_blind or endpoint_only
    candidates = (
        None if latent_compiler else regular_irrep_candidates(group, 3, seed=7_301)
    )
    compiler_group = None if latent_compiler else group
    compiler_inputs = None if latent_compiler else input_elements
    source_evidence = (
        TransitionEvidence(group.order, len(input_elements)) if table_blind else None
    )
    evidence = (
        source_evidence
        if table_blind
        and inverse_cover_calibration is None
        and inverse_cover_calibration_pairs_total is None
        else None
    )
    partial_mask_summary: dict[str, object] | None = None
    inferred_inverse_tokens: tuple[int, ...] | None = None
    endpoint_recovery: EndpointRecoveryReport | None = None
    recovered: RecoveredPermutationGroup | None = None
    manifold_recovery: EndpointManifoldRecovery | None = None
    endpoint_label_to_element: np.ndarray | None = None
    recovery_step: int | None = None
    recovery_minimum_edge_count: int | None = None
    if endpoint_only and not endpoint_manifold_compiler:
        oracle = GroupEndpointOracle(group, input_elements)
        recovered, endpoint_recovery = recover_from_endpoint_queries(
            oracle.query,
            state_count=group.order,
            token_count=len(input_elements),
            passive_samples=endpoint_representative_samples,
            passive_word_length=16,
            seed=50_000 + seed,
        )
        compiler_group = recovered.group
        compiler_inputs = recovered.input_elements
        candidates = regular_irrep_candidates(compiler_group, 3, seed=7_301)
        recovery_step = 0
        recovery_minimum_edge_count = 0
    model = PureGroupActionModel(
        len(input_elements),
        group.order,
        family="pure_ga_rotor",
        channels=4,
        max_rotor_angle=2.2,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-3, weight_decay=1e-4
    )
    if scheduled_endpoint_training:
        curriculum = ((250, 1), (250, 2), (250, 4), (250, 8), (1_000, 16))
        curriculum_blocks = []
        for stage, (count, length) in enumerate(curriculum):
            curriculum_blocks.append(
                make_group_batches(
                    group,
                    count,
                    256,
                    length,
                    seed + 1_000 + 10_000 * stage,
                    input_elements=input_elements,
                    held_out_pairs=((0, 2),) if length >= 2 else (),
                )
            )
        train_batches = list(itertools.chain.from_iterable(curriculum_blocks))
        if endpoint_length_mixture:
            order = np.random.default_rng(seed + 8_410_031).permutation(
                len(train_batches)
            )
            train_batches = [train_batches[int(index)] for index in order]
        elif endpoint_scrambled_blocks:
            # Prospective block-order control: preserve every generated batch
            # exactly, changing only the order of the five clean stages.
            block_order = (3, 0, 4, 1, 2)  # L8 -> L1 -> L16 -> L2 -> L4
            train_batches = list(
                itertools.chain.from_iterable(
                    curriculum_blocks[index] for index in block_order
                )
            )
    else:
        frozen_batches = make_group_batches(
            group,
            min(training_steps, 2_000),
            256,
            16,
            seed + 1_000,
            input_elements=input_elements,
            held_out_pairs=((0, 2),),
        )
        # Preserve the preregistered first 2,000 batches bit-for-bit. Longer
        # fixed-length controls stream deterministic continuation chunks so
        # their host-memory cost does not grow with the complete budget.
        continuation = (
            make_group_batches(
                group,
                min(2_000, training_steps - offset),
                256,
                16,
                seed + 101_000 + offset,
                input_elements=input_elements,
                held_out_pairs=((0, 2),),
            )
            for offset in range(2_000, training_steps, 2_000)
        )
        train_batches = itertools.chain.from_iterable(
            itertools.chain((frozen_batches,), continuation)
        )
    anchor: int | None = None
    compiled: CompiledRepresentation | None = None
    exact_actions: np.ndarray | None = None
    trigger_step: int | None = None
    trigger_commutator: float | None = None
    discovery_deadline = (
        1_500 if training_steps == 2_000 else max(1_500, training_steps - 500)
    )
    trajectory: dict[str, object] = {}
    projection_rms_values: list[float] = []
    tangent_norm_values: list[float] = []
    ambient_update_values: list[float] = []
    manifold_word_batches: list[np.ndarray] = []
    manifold_label_batches: list[np.ndarray] = []
    manifold_attempt_audits: list[dict[str, object]] = []
    start = time.perf_counter()
    model.train()
    for step, (tokens, targets) in enumerate(train_batches, start=1):
        if (
            endpoint_manifold_compiler
            and tokens.shape[1] == 8
            and len(manifold_word_batches) < 64
        ):
            # Reuse labels already consumed by the neural curriculum; these
            # are not additional compiler queries.
            manifold_word_batches.append(tokens.numpy().copy())
            manifold_label_batches.append(targets[:, -1].numpy().copy())
        if source_evidence is not None and recovered is None:
            source_evidence.observe(tokens, targets)
            if source_evidence.complete:
                if (
                    inverse_cover_calibration is not None
                    or inverse_cover_calibration_pairs_total is not None
                ) and evidence is None:
                    evidence, partial_mask_summary = inverse_cover_partial_evidence(
                        source_evidence,
                        calibration_fraction=inverse_cover_calibration or 0.0,
                        calibration_pairs_total=(
                            inverse_cover_calibration_pairs_total
                        ),
                        seed=910_001 + seed,
                    )
                    inferred_inverse_tokens = (
                        evidence.infer_inverse_pairs_and_complete()
                    )
                    partial_mask_summary["inferred_inverse_tokens"] = list(
                        inferred_inverse_tokens
                    )
                    partial_mask_summary["completed_edges"] = int(
                        np.sum(evidence.counts == 0)
                    )
                elif evidence is None:
                    evidence = source_evidence
                recovered = evidence.recover(base_state=0)
                recovery_step = step
                recovery_minimum_edge_count = int(evidence.counts.min())
                compiler_group = recovered.group
                compiler_inputs = recovered.input_elements
                candidates = regular_irrep_candidates(
                    compiler_group, 3, seed=7_301
                )
        if endpoint_only:
            # Discard the generator's prefix trace before it reaches the
            # device, task loss, holonomy objective, or any model component.
            targets = targets[:, -1:].clone()
        tokens, targets = tokens.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        if anchor is None:
            logits = model(tokens)
        else:
            mask = torch.zeros(4, dtype=logits_dtype(model), device=device)
            mask[anchor] = 1.0
            logits = masked_forward(model, tokens, mask)
        if endpoint_only:
            task_loss = nn.functional.cross_entropy(
                logits[:, -1], targets[:, -1]
            )
        else:
            task_loss = nn.functional.cross_entropy(
                logits.flatten(0, 1), targets.flatten()
            )
        if (
            anchor is None
            and step > 750
            and compiler_group is not None
            and compiler_inputs is not None
        ):
            multiplier = (2, 3, 4, 5)[(step - 751) % 4]
            if endpoint_label_to_element is not None:
                translation = torch.as_tensor(
                    endpoint_label_to_element,
                    dtype=torch.long,
                    device=targets.device,
                )
                holonomy_targets = translation[targets]
            else:
                holonomy_targets = (
                    recovered.translate_targets(targets)
                    if recovered is not None
                    else targets
                )
            holonomy_loss, margin_loss, _ = path_holonomy_objectives(
                model,
                compiler_group,
                compiler_inputs,
                tokens,
                holonomy_targets,
                word_multiplier=multiplier,
                batch_size=64,
                loss_power=8.0,
                margin_target=0.5,
            )
            ramp = min(1.0, (step - 750) / 500)
        else:
            holonomy_loss = task_loss.new_zeros(())
            margin_loss = task_loss.new_zeros(())
            ramp = 0.0
        loss = task_loss + ramp * (0.01 * holonomy_loss + 0.1 * margin_loss)
        loss.backward()
        total_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        )
        action_gradient_norm = float(
            model.action_parameters.grad.norm()
            if model.action_parameters.grad is not None
            else 0.0
        )
        orbit_gradient_norm = float(
            model.initial_orbit_state.grad.norm()
            if model.initial_orbit_state.grad is not None
            else 0.0
        )
        decoder_gradient_norm = float(
            torch.sqrt(
                sum(
                    parameter.grad.square().sum()
                    for parameter in model.output_head.parameters()
                    if parameter.grad is not None
                )
            )
        )
        previous_exact = exact_actions.copy() if exact_actions is not None else None
        optimizer.step()

        if anchor is not None:
            ambient = _channel_vectors(model)[:, anchor]
            ambient_update_values.append(
                float(np.sqrt(np.mean(np.square(ambient - previous_exact))))
            )
            exact_actions, projection_rms, tangent_norm = (
                local_joint_conjugacy_retraction(ambient, previous_exact)
            )
            projection_rms_values.append(projection_rms)
            tangent_norm_values.append(tangent_norm)
            _set_anchor_actions(model, anchor, exact_actions)
        elif 250 <= step <= discovery_deadline and step % 50 == 0:
            if endpoint_manifold_compiler and step >= 850:
                if manifold_word_batches:
                    discovery, candidate_audit = (
                        discover_endpoint_manifold_representation(
                            model,
                            np.concatenate(manifold_word_batches),
                            np.concatenate(manifold_label_batches),
                            state_count=group.order,
                            seed=seed,
                            true_group=group,
                        )
                    )
                    manifold_attempt_audits.append(
                        {
                            "step": step,
                            "accepted": discovery is not None,
                            "candidates": candidate_audit,
                        }
                    )
                else:
                    discovery = None
            else:
                discovery = (
                    discover_representation(
                        model,
                        compiler_group,
                        compiler_inputs,
                        candidates,
                        seed=seed,
                    )
                    if compiler_group is not None
                    and compiler_inputs is not None
                    and candidates is not None
                    else None
                )
            if discovery is not None:
                if endpoint_manifold_compiler:
                    anchor, compiled, trigger_commutator, manifold_recovery = discovery
                    compiler_group = manifold_recovery.group
                    compiler_inputs = manifold_recovery.input_elements
                    endpoint_label_to_element = manifold_recovery.label_to_element
                    recovery_step = step
                else:
                    anchor, compiled, trigger_commutator = discovery
                trigger_step = step
                _set_anchor_actions(model, anchor, compiled.token_actions)
                # Keep the float64 regular-representation family as the
                # retraction reference. Feeding the float32 rotor conversion
                # back into the next retraction would slowly preserve and
                # accumulate conversion error instead of staying on the exact
                # conjugacy orbit.
                exact_actions = compiled.token_actions.copy()

        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                predictions = logits.argmax(-1)
                final_accuracy = float(
                    (predictions[:, -1] == targets[:, -1]).float().mean()
                )
            trajectory[str(step)] = {
                "loss": float(loss.detach()),
                "task_loss": float(task_loss.detach()),
                "sequence_length": int(tokens.shape[1]),
                "final_position_accuracy": final_accuracy,
                "gradient_norm_before_clip": total_gradient_norm,
                "action_gradient_norm_after_clip": action_gradient_norm,
                "initial_orbit_gradient_norm_after_clip": orbit_gradient_norm,
                "decoder_gradient_norm_after_clip": decoder_gradient_norm,
                "compiled": anchor is not None,
                "anchor_channel": anchor,
                "latest_projection_rms": (
                    projection_rms_values[-1] if projection_rms_values else None
                ),
                "latest_tangent_norm": (
                    tangent_norm_values[-1] if tangent_norm_values else None
                ),
            }
            print(
                f"seed={seed} step={step}/{training_steps} "
                f"loss={float(loss.detach()):.5f} "
                f"final={final_accuracy:.3f} anchor={anchor}"
            )

    elapsed = time.perf_counter() - start
    if anchor is None or compiled is None or exact_actions is None:
        output_directory.mkdir(parents=True, exist_ok=True)
        diagnostic_checkpoint = (
            output_directory / f"uncompiled_retraction_seed{seed}.pt"
        )
        torch.save(
            {
                "family": "pure_ga_rotor",
                "group": "a5",
                "input_elements": input_elements,
                "config": {
                    "seed": seed,
                    "steps": training_steps,
                    "channels": 4,
                    "max_rotor_angle": 2.2,
                    "anchor_channel": None,
                    "endpoint_only": endpoint_only,
                    "endpoint_length_curriculum": endpoint_length_curriculum,
                    "endpoint_length_mixture": endpoint_length_mixture,
                    "endpoint_scrambled_blocks": endpoint_scrambled_blocks,
                    "endpoint_manifold_compiler": endpoint_manifold_compiler,
                },
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
            },
            diagnostic_checkpoint,
        )
        return {
            "training_seed": seed,
            "table_blind": table_blind,
            "endpoint_only": endpoint_only,
            "endpoint_length_curriculum": endpoint_length_curriculum,
            "endpoint_length_mixture": endpoint_length_mixture,
            "endpoint_scrambled_blocks": endpoint_scrambled_blocks,
            "endpoint_manifold_compiler": endpoint_manifold_compiler,
            "training_steps": training_steps,
            "discovery_deadline": discovery_deadline,
            "triggered": False,
            "transition_recovery": transition_recovery_summary(
                evidence,
                recovered,
                recovery_step,
                recovery_minimum_edge_count,
                partial_mask_summary,
            ),
            "endpoint_recovery": (
                asdict(endpoint_recovery) if endpoint_recovery is not None else None
            ),
            "endpoint_manifold_recovery": endpoint_manifold_summary(
                manifold_recovery, group
            ),
            "endpoint_manifold_attempt_audits": manifold_attempt_audits,
            "trajectory": trajectory,
            "diagnostic_checkpoint": str(diagnostic_checkpoint),
            "elapsed_seconds": elapsed,
        }

    model.eval()
    action_matrices = model.action_matrices().detach()
    diagnostics = anchor_mechanism_diagnostics(
        action_matrices[:, anchor], group, input_elements
    )
    dense = {
        "original": evaluate_anchor(
            model,
            group,
            input_elements,
            anchor,
            generator_class=None,
            lengths=DENSE_LENGTHS,
            batches=2,
            batch_size=512,
            seed_base=1_210_000,
            device=device,
        ),
        f"class_{untouched_class_index}_untouched": evaluate_anchor(
            model,
            group,
            input_elements,
            anchor,
            generator_class=untouched_class_index,
            lengths=DENSE_LENGTHS,
            batches=2,
            batch_size=512,
            seed_base=1_310_000,
            device=device,
        ),
    }
    long_stress = {
        "original": evaluate_anchor(
            model,
            group,
            input_elements,
            anchor,
            generator_class=None,
            lengths=(4096,),
            batches=1,
            batch_size=512,
            seed_base=1_410_000,
            device=device,
        ),
        f"class_{untouched_class_index}_untouched": evaluate_anchor(
            model,
            group,
            input_elements,
            anchor,
            generator_class=untouched_class_index,
            lengths=(4096,),
            batches=1,
            batch_size=512,
            seed_base=1_510_000,
            device=device,
        ),
    }
    if table_blind or endpoint_only:
        long_stress[f"class_{untouched_class_index}_L16384"] = evaluate_anchor(
            model,
            group,
            input_elements,
            anchor,
            generator_class=untouched_class_index,
            lengths=(16_384,),
            batches=1,
            batch_size=256,
            seed_base=1_610_000,
            device=device,
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / f"self_compiling_retraction_seed{seed}.pt"
    torch.save(
        {
            "family": "pure_ga_rotor",
            "group": "a5",
            "input_elements": input_elements,
            "config": {
                "seed": seed,
                "steps": training_steps,
                "discovery_deadline": discovery_deadline,
                "channels": 4,
                "max_rotor_angle": 2.2,
                "joint_retraction": True,
                "anchor_channel": anchor,
                "table_blind": table_blind,
                "endpoint_only": endpoint_only,
                "endpoint_representative_samples": endpoint_representative_samples,
                "endpoint_length_curriculum": endpoint_length_curriculum,
                "endpoint_length_mixture": endpoint_length_mixture,
                "endpoint_scrambled_blocks": endpoint_scrambled_blocks,
                "endpoint_manifold_compiler": endpoint_manifold_compiler,
                "inverse_cover_calibration": inverse_cover_calibration,
                "inverse_cover_calibration_pairs_total": (
                    inverse_cover_calibration_pairs_total
                ),
            },
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
        },
        checkpoint_path,
    )
    return {
        "training_seed": seed,
        "table_blind": table_blind,
        "endpoint_only": endpoint_only,
        "endpoint_length_curriculum": endpoint_length_curriculum,
        "endpoint_length_mixture": endpoint_length_mixture,
        "endpoint_scrambled_blocks": endpoint_scrambled_blocks,
        "endpoint_manifold_compiler": endpoint_manifold_compiler,
        "training_steps": training_steps,
        "discovery_deadline": discovery_deadline,
        "transition_recovery": transition_recovery_summary(
            evidence,
            recovered,
            recovery_step,
            recovery_minimum_edge_count,
            partial_mask_summary,
        ),
        "endpoint_recovery": (
            asdict(endpoint_recovery) if endpoint_recovery is not None else None
        ),
        "endpoint_manifold_recovery": endpoint_manifold_summary(
            manifold_recovery, group
        ),
        "endpoint_manifold_attempt_audits": manifold_attempt_audits,
        "triggered": True,
        "trigger_step": trigger_step,
        "anchor_channel": anchor,
        "candidate_index": compiled.candidate_index,
        "trigger_alignment_rms": compiled.alignment_rms,
        "trigger_runner_up_rms": compiled.runner_up_rms,
        "trigger_commutator": trigger_commutator,
        "selected_character_values": sorted(
            {round(float(value), 9) for value in compiled.character}
        ),
        "compiler_invariance_rms": compiled.invariance_rms,
        "compiler_homomorphism_rms": compiled.homomorphism_rms,
        "post_trigger_ambient_update_rms": summarize(ambient_update_values),
        "post_trigger_projection_rms": summarize(projection_rms_values),
        "post_trigger_tangent_norm": summarize(tangent_norm_values),
        "mechanism_diagnostics": diagnostics,
        "dense_evaluation": dense,
        "long_stress": long_stress,
        "trajectory": trajectory,
        "checkpoint": str(checkpoint_path),
        "elapsed_seconds": elapsed,
    }


def endpoint_manifold_summary(
    recovery: EndpointManifoldRecovery | None,
    true_group: FiniteGroup,
) -> dict[str, object] | None:
    if recovery is None:
        return None
    mapping = recovery.label_to_element
    exact_isomorphism = bool(
        np.array_equal(
            recovery.group.table[mapping[:, None], mapping[None, :]],
            mapping[true_group.table],
        )
    )
    return {
        "identity_label": recovery.identity_label,
        "input_elements": list(recovery.input_elements),
        "class_consistency_rms": recovery.class_consistency_rms,
        "minimum_center_separation": recovery.minimum_center_separation,
        "multiplication_rms": recovery.multiplication_rms,
        "multiplication_max": recovery.multiplication_max,
        "minimum_assignment_gap": recovery.minimum_assignment_gap,
        "exact_isomorphism_to_a5_posthoc": exact_isomorphism,
        "extra_endpoint_compiler_labels": 0,
        "reused_training_examples": 64 * 256,
    }


def transition_recovery_summary(
    evidence: TransitionEvidence | None,
    recovered: RecoveredPermutationGroup | None,
    recovery_step: int | None,
    recovery_minimum_edge_count: int | None,
    partial_mask_summary: dict[str, object] | None = None,
) -> dict[str, object] | None:
    if evidence is None or recovered is None:
        return None
    summary = {
        "recovery_step": recovery_step,
        "coverage_at_recovery": (
            partial_mask_summary["observed_fraction"]
            if partial_mask_summary is not None
            else 1.0
        ),
        "minimum_edge_count_at_recovery": recovery_minimum_edge_count,
        "minimum_edge_count_final": int(evidence.counts.min()),
        "generated_group_order": recovered.group.order,
        "input_elements_in_recovered_gauge": list(recovered.input_elements),
        "state_to_element": recovered.state_to_element.tolist(),
        "edge_replay_verified": True,
    }
    if partial_mask_summary is not None:
        summary["partial_inverse_cover"] = partial_mask_summary
    return summary


def logits_dtype(model: PureGroupActionModel) -> torch.dtype:
    return model.initial_orbit_state.dtype


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "mean": float(array.mean()),
        "maximum": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table-blind", action="store_true")
    parser.add_argument("--endpoint-only", action="store_true")
    parser.add_argument("--endpoint-length-curriculum", action="store_true")
    parser.add_argument("--endpoint-length-mixture", action="store_true")
    parser.add_argument("--endpoint-scrambled-blocks", action="store_true")
    parser.add_argument("--endpoint-manifold-compiler", action="store_true")
    parser.add_argument("--training-steps", type=int, default=2_000)
    parser.add_argument(
        "--endpoint-representative-samples", type=int, default=1_024
    )
    parser.add_argument(
        "--inverse-cover-calibration",
        type=float,
        default=None,
        help=(
            "Mask complete transition evidence to a reverse-edge cover with "
            "this calibrated bidirectional fraction, then infer inverse tokens"
        ),
    )
    parser.add_argument(
        "--inverse-cover-calibration-pairs-total",
        type=int,
        default=None,
        help="Use exactly this many globally allocated bidirectional pairs",
    )
    parser.add_argument(
        "--untouched-class-index", type=int, default=UNTOUCHED_CLASS_INDEX
    )
    args = parser.parse_args()
    if args.endpoint_only and args.table_blind:
        parser.error("--endpoint-only and --table-blind are alternative compilers")
    if args.endpoint_length_curriculum and not args.endpoint_only:
        parser.error("--endpoint-length-curriculum requires --endpoint-only")
    if args.endpoint_length_mixture and not args.endpoint_only:
        parser.error("--endpoint-length-mixture requires --endpoint-only")
    if args.endpoint_scrambled_blocks and not args.endpoint_only:
        parser.error("--endpoint-scrambled-blocks requires --endpoint-only")
    if sum(
        (
            args.endpoint_length_curriculum,
            args.endpoint_length_mixture,
            args.endpoint_scrambled_blocks,
        )
    ) > 1:
        parser.error("choose one endpoint length schedule")
    if args.endpoint_manifold_compiler and not args.endpoint_only:
        parser.error("--endpoint-manifold-compiler requires --endpoint-only")
    if args.endpoint_manifold_compiler and not args.endpoint_length_curriculum:
        parser.error("--endpoint-manifold-compiler requires the endpoint curriculum")
    if args.endpoint_only and (
        args.inverse_cover_calibration is not None
        or args.inverse_cover_calibration_pairs_total is not None
    ):
        parser.error("endpoint-only recovery does not use transition-mask flags")
    if args.inverse_cover_calibration is not None and not args.table_blind:
        parser.error("--inverse-cover-calibration requires --table-blind")
    if (
        args.inverse_cover_calibration_pairs_total is not None
        and not args.table_blind
    ):
        parser.error(
            "--inverse-cover-calibration-pairs-total requires --table-blind"
        )
    if (
        args.inverse_cover_calibration is not None
        and args.inverse_cover_calibration_pairs_total is not None
    ):
        parser.error("choose one inverse-cover calibration specification")
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    results = [
        train_seed(
            seed,
            device,
            args.checkpoint_directory,
            table_blind=args.table_blind,
            endpoint_only=args.endpoint_only,
            endpoint_representative_samples=(
                args.endpoint_representative_samples
            ),
            endpoint_length_curriculum=args.endpoint_length_curriculum,
            endpoint_length_mixture=args.endpoint_length_mixture,
            endpoint_scrambled_blocks=args.endpoint_scrambled_blocks,
            endpoint_manifold_compiler=args.endpoint_manifold_compiler,
            training_steps=args.training_steps,
            inverse_cover_calibration=args.inverse_cover_calibration,
            inverse_cover_calibration_pairs_total=(
                args.inverse_cover_calibration_pairs_total
            ),
            untouched_class_index=args.untouched_class_index,
        )
        for seed in args.seeds
    ]
    report = {
        "experiment": (
            "learned endpoint-manifold joint retraction"
            if args.endpoint_manifold_compiler
            else "endpoint-only self-compiling joint retraction"
            if args.endpoint_only
            else
            "partial latent-Cayley inverse-cover joint retraction"
            if args.inverse_cover_calibration is not None
            or args.inverse_cover_calibration_pairs_total is not None
            else "latent-Cayley self-compiling joint retraction"
            if args.table_blind
            else "self-compiling regular-representation joint retraction"
        ),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "torch_version": torch.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "seeds": args.seeds,
        "table_blind": args.table_blind,
        "endpoint_only": args.endpoint_only,
        "endpoint_representative_samples": args.endpoint_representative_samples,
        "endpoint_length_curriculum": args.endpoint_length_curriculum,
        "endpoint_length_mixture": args.endpoint_length_mixture,
        "endpoint_scrambled_blocks": args.endpoint_scrambled_blocks,
        "endpoint_manifold_compiler": args.endpoint_manifold_compiler,
        "training_steps": args.training_steps,
        "inverse_cover_calibration": args.inverse_cover_calibration,
        "inverse_cover_calibration_pairs_total": (
            args.inverse_cover_calibration_pairs_total
        ),
        "untouched_generator_class_index": args.untouched_class_index,
        "dense_lengths": list(DENSE_LENGTHS),
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "seeds": args.seeds,
                "triggered": sum(result["triggered"] for result in results),
            }
        )
    )


if __name__ == "__main__":
    main()
