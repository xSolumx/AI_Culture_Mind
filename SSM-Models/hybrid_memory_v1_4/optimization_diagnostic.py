"""Localize cross-seed Gated Delta optimization failures after G4c."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

if __package__:
    from . import tasks
    from .experiments import gated_delta_association_auxiliary_loss
    from .learnability_screen import CurriculumPhase, _batch
    from .model import HybridMemoryConfig, HybridMemoryLM
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4 import tasks  # type: ignore[no-redef]
    from hybrid_memory_v1_4.experiments import (  # type: ignore[no-redef]
        gated_delta_association_auxiliary_loss,
    )
    from hybrid_memory_v1_4.learnability_screen import (  # type: ignore[no-redef]
        CurriculumPhase,
        _batch,
    )
    from hybrid_memory_v1_4.model import (  # type: ignore[no-redef]
        HybridMemoryConfig,
        HybridMemoryLM,
    )


def _time_gather(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    index = positions.reshape(*positions.shape, *([1] * (values.ndim - 2)))
    index = index.expand(positions.shape[0], positions.shape[1], *values.shape[2:])
    return values.gather(1, index)


def _grad_norms(model: HybridMemoryLM) -> dict[str, float]:
    groups: dict[str, float] = defaultdict(float)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if ".mixer." in name:
            component = name.split(".mixer.", 1)[1].split(".", 1)[0]
        elif name.startswith("blocks.1"):
            component = "attention_block"
        elif name.startswith(("lm_head", "embedding")):
            component = "embedding_or_head"
        else:
            component = "shell"
        groups[component] += float(parameter.grad.float().square().sum())
    return {name: math.sqrt(value) for name, value in sorted(groups.items())}


def _evaluate_mode(
    model: HybridMemoryLM,
    batches: list[tasks.RetrievalBatch],
    *,
    disable_layer: int | None,
) -> float:
    scales = [block.residual_scale.detach().clone() for block in model.blocks]
    if disable_layer is not None:
        model.blocks[disable_layer].residual_scale.data.fill_(-30.0)
    correct = 0
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in batches:
            logits = model(batch.inputs)["logits"]
            predictions = tasks.gather_query_logits(logits, batch).argmax(-1)
            correct += int((predictions == batch.targets).sum())
            count += batch.targets.numel()
    for block, scale in zip(model.blocks, scales, strict=True):
        block.residual_scale.data.copy_(scale)
    return correct / count


def _evaluate_write_thresholds(
    model: HybridMemoryLM,
    batches: list[tasks.RetrievalBatch],
    thresholds: tuple[float, ...],
) -> dict[str, float]:
    mixer = model.blocks[0].mixer
    original = mixer._controls
    reports = {}
    for threshold in thresholds:

        def thresholded(
            inputs: torch.Tensor, _threshold: float = threshold
        ) -> tuple[torch.Tensor, ...]:
            query, key, value, write, retention = original(inputs)
            write = write * (write >= _threshold).to(write.dtype)
            return query, key, value, write, retention

        mixer._controls = thresholded
        reports[str(threshold)] = _evaluate_mode(model, batches, disable_layer=None)
    mixer._controls = original
    return reports


def diagnose(
    checkpoint: Path,
    *,
    seed: int,
    device: torch.device,
    batches: int,
    batch_size: int,
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = HybridMemoryLM(HybridMemoryConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    phase = CurriculumPhase(16, 4, 512, 0)
    cohort = [
        _batch(
            phase,
            batch_size,
            seed=seed * 1_000_000 + index,
            device=device,
        )
        for index in range(batches)
    ]
    correct = 0
    count = 0
    address_correct: torch.Tensor | None = None
    address_count = 0
    write_event_sum: torch.Tensor | None = None
    nonwrite_sum: torch.Tensor | None = None
    retention_write_sum: torch.Tensor | None = None
    retention_nonwrite_sum: torch.Tensor | None = None
    write_count = 0
    nonwrite_count = 0
    state_query_norm = 0.0
    state_final_norm = 0.0
    write_key_off_diagonal = 0.0
    query_key_margin = 0.0
    value_write_norm = 0.0
    value_nonwrite_norm = 0.0
    read_query_norm = 0.0
    target_correct: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    model.eval()
    with torch.no_grad():
        for batch in cohort:
            output = model(batch.inputs, return_diagnostics=True)
            query_logits = tasks.gather_query_logits(output["logits"], batch)
            predictions = query_logits.argmax(-1)
            matches = predictions == batch.targets
            correct += int(matches.sum())
            count += batch.targets.numel()
            for target, matched in zip(
                batch.targets.flatten().tolist(),
                matches.flatten().tolist(),
                strict=True,
            ):
                target_correct[target][0] += int(matched)
                target_correct[target][1] += 1
            diagnostic = output["diagnostics"][0]
            query = diagnostic["query"]
            key = diagnostic["key"]
            value = diagnostic["value"]
            read = diagnostic["read"]
            write = diagnostic["write_strength"]
            retention = diagnostic["retention"]
            state_norm = diagnostic["state_norm"]
            write_positions = batch.metadata["stored_value_positions"]
            write_keys = batch.metadata["stored_keys"]
            query_indices = batch.metadata["query_pair_indices"]
            assert isinstance(write_positions, torch.Tensor)
            assert isinstance(write_keys, torch.Tensor)
            assert isinstance(query_indices, torch.Tensor)
            q = _time_gather(query, batch.query_positions)
            k = _time_gather(key, write_positions)
            similarities = torch.einsum("bqhk,bphk->bhqp", q, k)
            predicted_write = similarities.argmax(-1)
            targets = query_indices[:, None, :].expand_as(predicted_write)
            per_head = (predicted_write == targets).sum(dim=(0, 2))
            address_correct = (
                per_head if address_correct is None else address_correct + per_head
            )
            address_count += batch_size * phase.queries
            key_by_head = k.permute(0, 2, 1, 3)
            gram = key_by_head @ key_by_head.transpose(-1, -2)
            off_diagonal = ~torch.eye(phase.pairs, dtype=torch.bool, device=device)[
                None, None
            ]
            write_key_off_diagonal += float(
                gram.masked_select(off_diagonal).abs().mean()
            )
            matching = similarities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            nonmatching = similarities.masked_fill(
                torch.nn.functional.one_hot(targets, phase.pairs).bool(), -torch.inf
            ).amax(-1)
            query_key_margin += float((matching - nonmatching).mean())
            events = torch.zeros_like(write, dtype=torch.bool)
            events.scatter_(
                1,
                write_positions[:, :, None].expand(-1, -1, write.shape[-1]),
                True,
            )
            event_values = write.masked_select(events).reshape(-1, write.shape[-1])
            nonwrite_values = write.masked_select(~events).reshape(-1, write.shape[-1])
            retention_events = retention.masked_select(events).reshape(
                -1, retention.shape[-1]
            )
            retention_nonwrite = retention.masked_select(~events).reshape(
                -1, retention.shape[-1]
            )
            write_event_sum = (
                event_values.sum(0)
                if write_event_sum is None
                else write_event_sum + event_values.sum(0)
            )
            nonwrite_sum = (
                nonwrite_values.sum(0)
                if nonwrite_sum is None
                else nonwrite_sum + nonwrite_values.sum(0)
            )
            retention_write_sum = (
                retention_events.sum(0)
                if retention_write_sum is None
                else retention_write_sum + retention_events.sum(0)
            )
            retention_nonwrite_sum = (
                retention_nonwrite.sum(0)
                if retention_nonwrite_sum is None
                else retention_nonwrite_sum + retention_nonwrite.sum(0)
            )
            write_count += event_values.shape[0]
            nonwrite_count += nonwrite_values.shape[0]
            value_events = value.masked_select(events[..., None]).reshape(
                -1, value.shape[-1]
            )
            value_elsewhere = value.masked_select((~events)[..., None]).reshape(
                -1, value.shape[-1]
            )
            value_write_norm += float(value_events.float().norm(dim=-1).mean())
            value_nonwrite_norm += float(value_elsewhere.float().norm(dim=-1).mean())
            read_query_norm += float(
                _time_gather(read, batch.query_positions).float().norm(dim=-1).mean()
            )
            state_query_norm += float(
                _time_gather(state_norm.transpose(1, 2), batch.query_positions).mean()
            )
            state_final_norm += float(state_norm[:, :, -1].mean())

    probe = cohort[0]
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(probe.inputs, return_diagnostics=True)
    retrieval = tasks.retrieval_loss(output["logits"], probe)
    retrieval.backward()
    retrieval_gradients = _grad_norms(model)
    model.zero_grad(set_to_none=True)
    output = model(probe.inputs, return_diagnostics=True)
    association = gated_delta_association_auxiliary_loss(output, probe)
    association.backward()
    association_gradients = _grad_norms(model)
    assert address_correct is not None
    assert write_event_sum is not None and nonwrite_sum is not None
    assert retention_write_sum is not None and retention_nonwrite_sum is not None
    weakest_targets = sorted(
        (
            {
                "token": token,
                "correct": values[0],
                "count": values[1],
                "accuracy": values[0] / values[1],
            }
            for token, values in target_correct.items()
        ),
        key=lambda item: (item["accuracy"], item["token"]),
    )[:10]
    return {
        "checkpoint": str(checkpoint),
        "seed": seed,
        "exact_accuracy": correct / count,
        "query_count": count,
        "address_accuracy_per_head": (address_correct / address_count).tolist(),
        "write_strength_at_writes_per_head": (write_event_sum / write_count).tolist(),
        "write_strength_elsewhere_per_head": (nonwrite_sum / nonwrite_count).tolist(),
        "retention_at_writes_per_head": (retention_write_sum / write_count).tolist(),
        "retention_elsewhere_per_head": (
            retention_nonwrite_sum / nonwrite_count
        ).tolist(),
        "mean_state_norm_at_queries": state_query_norm / batches,
        "mean_state_norm_final": state_final_norm / batches,
        "mean_absolute_write_key_off_diagonal": write_key_off_diagonal / batches,
        "mean_query_key_margin": query_key_margin / batches,
        "mean_value_norm_at_writes": value_write_norm / batches,
        "mean_value_norm_elsewhere": value_nonwrite_norm / batches,
        "mean_read_norm_at_queries": read_query_norm / batches,
        "residual_scales": [
            {
                "raw": float(block.residual_scale.detach()),
                "sigmoid": float(block.residual_scale.detach().sigmoid()),
            }
            for block in model.blocks
        ],
        "full_accuracy": _evaluate_mode(model, cohort, disable_layer=None),
        "without_gated_delta_mixer_accuracy": _evaluate_mode(
            model, cohort, disable_layer=0
        ),
        "without_attention_mixer_accuracy": _evaluate_mode(
            model, cohort, disable_layer=1
        ),
        "write_threshold_accuracy": _evaluate_write_thresholds(
            model, cohort, (0.002, 0.005, 0.01, 0.02, 0.05)
        ),
        "retrieval_loss_probe": float(retrieval.detach()),
        "association_loss_probe": float(association.detach()),
        "retrieval_gradient_norms": retrieval_gradients,
        "association_gradient_norms": association_gradients,
        "weakest_target_tokens": weakest_targets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1423, 1427, 1429])
    parser.add_argument(
        "--checkpoint-pattern",
        default="hybrid_v1_4_1_g4c_validation_seed{seed}.pt",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    reports = []
    for seed in args.seeds:
        checkpoint = args.checkpoint_dir / args.checkpoint_pattern.format(seed=seed)
        reports.append(
            diagnose(
                checkpoint,
                seed=seed,
                device=device,
                batches=args.batches,
                batch_size=args.batch_size,
            )
        )
    payload = {
        "schema_version": 1,
        "claim_status": "post-G4c optimization diagnostic",
        "runs": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
