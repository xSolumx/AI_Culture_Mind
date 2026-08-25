"""Prospectively frozen G15A SpinDirac mechanism/observability cohort.

The symmetry arm is deliberately oracle-controlled: exact keys, values, edit
gates, and supplied coordinates hold the memory law fixed while a two-scalar
equivariant calibrator is trained. The separate full-LM no-symmetry arm learns
delayed value recall from token inputs. Keeping these results separate prevents
supplied geometry from being mistaken for generic controller learnability.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from spin8_triality import (
    SPIN8_PAIRS,
    TRIALITY_REPRESENTATIONS,
)
from torch import nn
from torch.nn import functional as F

if __package__:
    from .g15a_tasks import (
        OFF_TORUS_PAIRS,
        SYMMETRY_CLASSES,
        VOCAB_SIZE,
        G15ABatch,
        generate_no_symmetry_batch,
        generate_symmetry_batch,
    )
    from .model import HybridMemoryConfig, HybridMemoryLM, parameter_count
    from .optimizers import HarmonicMuonAdamW
    from .spin_dirac_memory import SpinDiracConfig, SpinDiracMemory
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4.g15a_tasks import (  # type: ignore[no-redef]
        OFF_TORUS_PAIRS,
        SYMMETRY_CLASSES,
        VOCAB_SIZE,
        G15ABatch,
        generate_no_symmetry_batch,
        generate_symmetry_batch,
    )
    from hybrid_memory_v1_4.model import (  # type: ignore[no-redef]
        HybridMemoryConfig,
        HybridMemoryLM,
        parameter_count,
    )
    from hybrid_memory_v1_4.optimizers import (  # type: ignore[no-redef]
        HarmonicMuonAdamW,
    )
    from hybrid_memory_v1_4.spin_dirac_memory import (  # type: ignore[no-redef]
        SpinDiracConfig,
        SpinDiracMemory,
    )

PROTOCOL = Path(__file__).with_name("G15A_EXECUTION_PROTOCOL_2026-08-25.md")
PREREGISTRATION = Path(__file__).with_name("G15_SPIN_DIRAC_PREREGISTRATION.md")
AMENDMENTS = (
    Path(__file__).with_name("G15_SPIN_DIRAC_AMENDMENT_2026-08-25.md"),
    Path(__file__).with_name("G15_SPIN_DIRAC_EDIT_LAW_AMENDMENT_2026-08-25.md"),
)
QUALITY_SEEDS = (2131, 2137, 2141)
EVALUATION_LENGTHS = (64, 256, 1024)
ARM_SPECS = {
    "I": ("identity", "identity"),
    "I+C": ("identity", "clifford"),
    "C": ("commuting_so2", "clifford"),
    "S": ("spin8", "clifford"),
}


@dataclass(frozen=True)
class CohortConfig:
    mode: str
    seeds: tuple[int, ...]
    training_updates: int
    training_batch_size: int
    evaluation_examples: int
    evaluation_microbatch: int
    learning_rate: float
    weight_decay: float
    commissioning_coefficient: float
    calibration_updates: int
    calibration_learning_rate: float
    model_dim: int = 32
    heads: int = 1
    training_length: int = 64
    dtype: str = "float32"


def quality_config() -> CohortConfig:
    return CohortConfig(
        mode="quality",
        seeds=QUALITY_SEEDS,
        training_updates=300,
        training_batch_size=16,
        evaluation_examples=80,
        evaluation_microbatch=8,
        learning_rate=3e-3,
        weight_decay=0.01,
        commissioning_coefficient=0.25,
        calibration_updates=100,
        calibration_learning_rate=5e-2,
    )


def smoke_config() -> CohortConfig:
    return CohortConfig(
        mode="smoke",
        seeds=(17,),
        training_updates=4,
        training_batch_size=4,
        evaluation_examples=10,
        evaluation_microbatch=2,
        learning_rate=3e-3,
        weight_decay=0.01,
        commissioning_coefficient=0.25,
        calibration_updates=4,
        calibration_learning_rate=5e-2,
    )


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (2**63 - 1)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_state() -> tuple[str, list[str]]:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return commit, status


def _atomic_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _model_config(arm: str, config: CohortConfig) -> HybridMemoryConfig:
    transport, readout = ARM_SPECS[arm]
    return HybridMemoryConfig(
        vocab_size=VOCAB_SIZE,
        model_dim=config.model_dim,
        layer_plan=("spin_dirac",),
        spin_dirac_heads=config.heads,
        spin_dirac_transport_mode=transport,  # type: ignore[arg-type]
        spin_dirac_readout_mode=readout,  # type: ignore[arg-type]
        spin_dirac_gate_mode="equivariant_scalar",
        spin_dirac_tie_query_key=True,
        spin_dirac_allow_negative_eigenvalues=False,
        spin_dirac_bound_values=True,
        spin_dirac_minimum_retention=0.999,
        spin_dirac_maximum_retention=0.999999,
        spin_dirac_initial_retention=0.9995,
        spin_dirac_initial_erase_strength=0.10,
        spin_dirac_initial_write_strength=0.10,
        spin_dirac_maximum_coordinate=0.25,
        use_local_conv=False,
        expansion=2,
        dropout=0.0,
    )


def _build_model(
    arm: str, config: CohortConfig, seed: int, device: torch.device
) -> HybridMemoryLM:
    _seed_everything(_stable_seed("g15a-model", seed))
    return HybridMemoryLM(_model_config(arm, config)).to(device)


def _commissioning_loss(
    output: dict[str, Any], batch: G15ABatch
) -> tuple[torch.Tensor, dict[str, float]]:
    diagnostic = output["diagnostics"][0]
    query = diagnostic["query_vector"][:, -1]
    key = diagnostic["key_vector"][:, 0]
    address = (1.0 - (query * key).sum(dim=-1)).mean()
    write = diagnostic["write_strength"]
    write_target = torch.zeros_like(write)
    write_target[:, 0] = 1.0
    epsilon = torch.finfo(write.dtype).eps
    write_loss = F.binary_cross_entropy(
        write.clamp(epsilon, 1.0 - epsilon), write_target
    )
    erase = diagnostic["erase_strength"]
    erase_loss = -torch.log1p(-erase.clamp(max=1.0 - epsilon)).mean()
    retention = diagnostic["retention"]
    retention_loss = (1.0 - retention).square().mean()
    total = address + 0.5 * write_loss + 0.1 * erase_loss + 0.05 * retention_loss
    return total, {
        "address": float(address.detach()),
        "write": float(write_loss.detach()),
        "erase": float(erase_loss.detach()),
        "retention": float(retention_loss.detach()),
    }


@torch.no_grad()
def _evaluate_no_symmetry(
    model: HybridMemoryLM,
    config: CohortConfig,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    model.eval()
    report: dict[str, dict[str, float]] = {}
    for length in EVALUATION_LENGTHS:
        correct = 0
        nll = 0.0
        count = 0
        for offset in range(
            0, config.evaluation_examples, config.evaluation_microbatch
        ):
            size = min(
                config.evaluation_microbatch, config.evaluation_examples - offset
            )
            batch = generate_no_symmetry_batch(
                size,
                length,
                seed=_stable_seed("g15a-eval-no-sym", seed, length, offset),
                heads=config.heads,
            ).to(device)
            output = model(
                batch.token_ids,
                spin_dirac_coordinates=batch.coordinates,
                delta_scan_mode="parallel",
            )
            logits = output["logits"][:, -1]
            correct += int((logits.argmax(-1) == batch.targets).sum())
            nll += float(F.cross_entropy(logits, batch.targets, reduction="sum"))
            count += size
        report[str(length)] = {
            "accuracy": correct / count,
            "bits_per_query": nll / (count * math.log(2.0)),
            "examples": count,
        }
    return report


def _train_no_symmetry(
    arm: str,
    config: CohortConfig,
    *,
    seed: int,
    device: torch.device,
    checkpoint_directory: Path,
) -> dict[str, Any]:
    model = _build_model(arm, config, seed, device)
    optimizer = HarmonicMuonAdamW(
        model,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    loss_samples: dict[str, dict[str, float]] = {}
    schedule_digest = hashlib.sha256()
    model.train()
    for step in range(config.training_updates):
        batch = generate_no_symmetry_batch(
            config.training_batch_size,
            config.training_length,
            seed=_stable_seed("g15a-train-no-sym", seed, step),
            heads=config.heads,
        )
        schedule_digest.update(batch.fingerprint().encode())
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch.token_ids,
            spin_dirac_coordinates=batch.coordinates,
            delta_scan_mode="parallel",
            return_diagnostics=True,
        )
        logits = output["logits"][:, -1]
        retrieval = F.cross_entropy(logits, batch.targets)
        commissioning, components = _commissioning_loss(output, batch)
        loss = retrieval + config.commissioning_coefficient * commissioning
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite G15A loss for {arm} seed {seed}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("non-finite G15A gradient norm")
        optimizer.step()
        if step in {0, config.training_updates // 2, config.training_updates - 1}:
            loss_samples[str(step + 1)] = {
                "total": float(loss.detach()),
                "retrieval": float(retrieval.detach()),
                "commissioning": float(commissioning.detach()),
                "gradient_norm": float(gradient_norm.detach()),
                **components,
            }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    evaluation = _evaluate_no_symmetry(model, config, seed=seed, device=device)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_directory / f"g15a_{arm.replace('+', 'plus')}_seed{seed}.pt"
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(
        {
            "schema_version": 1,
            "arm": arm,
            "seed": seed,
            "config": asdict(model.config),
            "cohort": asdict(config),
            "model_state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "evaluation": evaluation,
            "training_schedule_sha256": schedule_digest.hexdigest(),
        },
        temporary,
    )
    os.replace(temporary, checkpoint)
    result = {
        "parameters": parameter_count(model),
        "state_bytes_per_sequence_fp32": model.state_capacity_bytes(1, torch.float32),
        "training_schedule_sha256": schedule_digest.hexdigest(),
        "loss_samples": loss_samples,
        "training_wall_seconds": training_seconds,
        "mean_synchronized_step_seconds": training_seconds / config.training_updates,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "evaluation": evaluation,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "optimizer_partition": optimizer.partition_report(),
    }
    del optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _oracle_memory(
    arm: str, *, dtype: torch.dtype, device: torch.device
) -> SpinDiracMemory:
    transport, readout = ARM_SPECS[arm]
    return SpinDiracMemory(
        SpinDiracConfig(
            8,
            heads=1,
            transport_mode=transport,  # type: ignore[arg-type]
            readout_mode=readout,  # type: ignore[arg-type]
            gate_mode="equivariant_scalar",
            tie_query_key=True,
            bound_values=True,
            maximum_coordinate=0.25,
        )
    ).to(device=device, dtype=dtype)


def _oracle_controls(
    batch: G15ABatch,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    batch_size, length = batch.token_ids.shape
    carrier = torch.zeros(batch_size, length, 1, 8, dtype=dtype, device=device)
    query = carrier.clone()
    key = carrier.clone()
    value = carrier.clone()
    query[..., 0] = 1.0
    key[..., 0] = 1.0
    if batch.task == "symmetry":
        generator = torch.Generator().manual_seed(_stable_seed("g15a-value", seed))
        initial_value = F.normalize(torch.randn(8, generator=generator), dim=0)
        value[:, 0, 0] = initial_value.to(device=device, dtype=dtype)
    else:
        value[torch.arange(batch_size), 0, 0, batch.labels.to(device)] = 1.0
    erase = torch.zeros(batch_size, length, 1, 1, dtype=dtype, device=device)
    write = torch.zeros_like(erase)
    write[:, 0] = 1.0
    retention = torch.full_like(erase, 0.9999)
    coordinates = batch.coordinates.to(device=device, dtype=dtype)
    return query, key, value, erase, write, retention, coordinates


def _class_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    normalized_predictions = F.normalize(predictions, dim=-1)
    normalized_targets = F.normalize(targets, dim=-1)
    selected = (normalized_predictions @ normalized_targets.T).argmax(-1)
    predicted_labels = labels[selected]
    matches = predicted_labels == labels
    off = labels < len(OFF_TORUS_PAIRS)
    center = labels >= len(OFF_TORUS_PAIRS)
    return {
        "accuracy": float(matches.float().mean()),
        "off_torus_accuracy": float(matches[off].float().mean()),
        "central_pair_accuracy": float(matches[center].float().mean()),
        "normalized_mse": float(F.mse_loss(normalized_predictions, normalized_targets)),
    }


def _fit_calibrator(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    config: CohortConfig,
) -> tuple[torch.Tensor, list[float]]:
    log_scales = nn.Parameter(
        torch.full((2,), math.log(0.5), device=predictions.device)
    )
    optimizer = torch.optim.AdamW(
        [log_scales], lr=config.calibration_learning_rate, weight_decay=0.0
    )
    losses = []
    for _ in range(config.calibration_updates):
        optimizer.zero_grad(set_to_none=True)
        scales = log_scales.exp().repeat_interleave(8)
        calibrated = predictions * scales
        loss = F.mse_loss(calibrated, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return log_scales.detach().exp(), losses


@torch.no_grad()
def _symmetry_predictions(
    memory: SpinDiracMemory,
    batch: G15ABatch,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    supplied_actions: torch.Tensor | None = None,
    controls: list[torch.Tensor] | None = None,
    return_diagnostics: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | str] | None]:
    controls = controls or _oracle_controls(
        batch, seed=seed, device=device, dtype=dtype
    )
    result = memory.forward_controls(
        *controls,
        supplied_actions=supplied_actions,
        scan_mode="parallel",
        return_diagnostics=return_diagnostics,
    )
    if return_diagnostics:
        read, _, diagnostics = result
        return read[:, -1, 0], diagnostics
    read, _ = result
    return read[:, -1, 0], None


def _inner_conjugation_residual(
    arm: str,
    scales: torch.Tensor,
    *,
    seed: int,
) -> float:
    device = torch.device("cpu")
    dtype = torch.float64
    batch = generate_symmetry_batch(
        SYMMETRY_CLASSES, 64, seed=_stable_seed("g15a-conjugation", seed), dtype=dtype
    )
    memory = _oracle_memory(arm, dtype=dtype, device=device)
    controls = _oracle_controls(batch, seed=seed, device=device, dtype=dtype)
    original, diagnostics = _symmetry_predictions(
        memory,
        batch,
        seed=seed,
        device=device,
        dtype=dtype,
        controls=controls,
        return_diagnostics=True,
    )
    assert diagnostics is not None
    actions = diagnostics["transport_actions"]
    assert isinstance(actions, torch.Tensor)
    h_coordinates = torch.zeros(1, 1, 1, 28, dtype=dtype)
    h_coordinates[..., SPIN8_PAIRS.index((2, 4))] = 0.17
    h = memory._transitions(
        controls[1][:1, :1],
        controls[2][:1, :1],
        controls[3][:1, :1],
        controls[4][:1, :1],
        controls[5][:1, :1],
        h_coordinates,
        None,
    )[3][0, 0, 0]
    vector = h[TRIALITY_REPRESENTATIONS.index("vector")]
    positive = h[TRIALITY_REPRESENTATIONS.index("positive")]
    negative = h[TRIALITY_REPRESENTATIONS.index("negative")]
    transformed = list(controls)
    transformed[0] = torch.einsum("ij,bthj->bthi", vector, controls[0])
    transformed[1] = torch.einsum("ij,bthj->bthi", vector, controls[1])
    transformed[2] = torch.einsum("ij,bthj->bthi", positive, controls[2])
    conjugated_actions = torch.einsum(
        "rij,bthrjk,rkl->bthril", h, actions, h.transpose(-1, -2)
    )
    conjugated, _ = _symmetry_predictions(
        memory,
        batch,
        seed=seed,
        device=device,
        dtype=dtype,
        supplied_actions=conjugated_actions,
        controls=transformed,
    )
    scale_vector = scales.detach().to(device=device, dtype=dtype).repeat_interleave(8)
    original = original * scale_vector
    conjugated = conjugated * scale_vector
    expected_positive = torch.einsum("ij,bj->bi", positive, original[:, :8])
    second_action = positive if ARM_SPECS[arm][1] == "identity" else negative
    expected_second = torch.einsum("ij,bj->bi", second_action, original[:, 8:])
    expected = torch.cat((expected_positive, expected_second), dim=-1)
    return float((conjugated - expected).abs().max())


def _run_symmetry_arm(
    arm: str,
    config: CohortConfig,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    dtype = torch.float32
    memory = _oracle_memory(arm, dtype=dtype, device=device)
    teacher = _oracle_memory("S", dtype=dtype, device=device)
    training_batch = generate_symmetry_batch(
        SYMMETRY_CLASSES,
        config.training_length,
        seed=_stable_seed("g15a-sym-train", seed),
    )
    predictions, _ = _symmetry_predictions(
        memory,
        training_batch,
        seed=seed,
        device=device,
        dtype=dtype,
    )
    targets, _ = _symmetry_predictions(
        teacher,
        training_batch,
        seed=seed,
        device=device,
        dtype=dtype,
    )
    scales, calibration_losses = _fit_calibrator(predictions, targets, config)
    evaluations = {}
    for length in EVALUATION_LENGTHS:
        batch = generate_symmetry_batch(
            SYMMETRY_CLASSES,
            length,
            seed=_stable_seed("g15a-sym-eval", seed, length),
        )
        predictions, _ = _symmetry_predictions(
            memory, batch, seed=seed, device=device, dtype=dtype
        )
        targets, _ = _symmetry_predictions(
            teacher, batch, seed=seed, device=device, dtype=dtype
        )
        calibrated = predictions * scales.repeat_interleave(8)
        evaluations[str(length)] = {
            **_class_accuracy(calibrated, targets, batch.labels.to(device)),
            "examples": config.evaluation_examples,
        }
    residual = _inner_conjugation_residual(arm, scales, seed=seed)
    return {
        "oracle_controlled": True,
        "learned_parameters": 2,
        "calibration_scales": scales.tolist(),
        "calibration_loss_first": calibration_losses[0],
        "calibration_loss_last": calibration_losses[-1],
        "evaluation": evaluations,
        "inner_conjugation_max_abs_residual_float64": residual,
    }


def _oracle_semantic_ladder() -> dict[str, float | bool]:
    memory = _oracle_memory("I+C", dtype=torch.float64, device=torch.device("cpu"))
    length = 4
    carrier = torch.zeros(1, length, 1, 8, dtype=torch.float64)
    query = carrier.clone()
    key = carrier.clone()
    value = carrier.clone()
    query[..., 0] = 1.0
    key[..., 0] = 1.0
    value[:, 0, :, 1] = 1.0
    erase = torch.zeros(1, length, 1, 1, dtype=torch.float64)
    write = torch.zeros_like(erase)
    write[:, 0] = 1.0
    retention = torch.full_like(erase, 0.99)
    coordinates = torch.zeros(1, length, 1, 28, dtype=torch.float64)
    read, _ = memory.forward_controls(
        query, key, value, erase, write, retention, coordinates
    )
    one_hot_expected = 0.99 ** (length - 1)
    one_hot_error = abs(float(read[0, -1, 0, 1]) - one_hot_expected)

    value[:, 1, :, 2] = 1.0
    erase[:, 1] = 1.0
    write[:, 1] = 1.0
    read, _ = memory.forward_controls(
        query, key, value, erase, write, retention, coordinates
    )
    overwrite_expected = 0.99 ** (length - 2)
    overwrite_error = abs(float(read[0, -1, 0, 2]) - overwrite_expected)
    collision_old_residual = abs(float(read[0, -1, 0, 1]))
    query[:, -1] = 0.0
    query[:, -1, :, 3] = 1.0
    orthogonal, _ = memory.forward_controls(
        query, key, value, erase, write, retention, coordinates
    )
    orthogonal_norm = float(torch.linalg.vector_norm(orthogonal[:, -1, :, :8]))
    passed = (
        max(
            one_hot_error,
            overwrite_error,
            collision_old_residual,
            orthogonal_norm,
        )
        < 1e-10
    )
    return {
        "one_hot_read_error": one_hot_error,
        "overwrite_error": overwrite_error,
        "repeated_key_old_value_residual": collision_old_residual,
        "orthogonal_query_norm": orthogonal_norm,
        "passed": passed,
    }


def _macro_accuracy(evaluation: dict[str, dict[str, float]]) -> float:
    return sum(row["accuracy"] for row in evaluation.values()) / len(evaluation)


def _adjudicate(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    per_seed = []
    for report in seed_reports:
        arms = report["arms"]
        spin_symmetry = _macro_accuracy(arms["S"]["symmetry"]["evaluation"])
        comparator_symmetry = {
            arm: _macro_accuracy(arms[arm]["symmetry"]["evaluation"])
            for arm in ("I", "I+C", "C")
        }
        spin_no_symmetry = _macro_accuracy(arms["S"]["no_symmetry"]["evaluation"])
        comparator_no_symmetry = {
            arm: _macro_accuracy(arms[arm]["no_symmetry"]["evaluation"])
            for arm in ("I", "I+C", "C")
        }
        symmetry_margins = {
            arm: spin_symmetry - accuracy
            for arm, accuracy in comparator_symmetry.items()
        }
        best_no_symmetry = max(comparator_no_symmetry.values())
        checks = {
            "symmetry_margin_each_comparator_at_least_0_02": all(
                margin + 1e-12 >= 0.02 for margin in symmetry_margins.values()
            ),
            "no_symmetry_noninferiority_within_0_01": (
                spin_no_symmetry + 1e-12 >= best_no_symmetry - 0.01
            ),
            "inner_conjugation_each_arm_at_most_1e_9": all(
                arms[arm]["symmetry"]["inner_conjugation_max_abs_residual_float64"]
                <= 1e-9
                for arm in ARM_SPECS
            ),
        }
        per_seed.append(
            {
                "seed": report["seed"],
                "spin_symmetry_accuracy": spin_symmetry,
                "comparator_symmetry_accuracy": comparator_symmetry,
                "symmetry_margins": symmetry_margins,
                "spin_no_symmetry_accuracy": spin_no_symmetry,
                "comparator_no_symmetry_accuracy": comparator_no_symmetry,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    passed = all(row["passed"] for row in per_seed)
    return {
        "passed": passed,
        "per_seed": per_seed,
        "decision": (
            "G15A passes; conditional S+identity-read and S-broken diagnostics are authorized"
            if passed
            else "G15A fails; Spin transport is not promoted"
        ),
    }


def run(
    config: CohortConfig,
    *,
    device: torch.device,
    checkpoint_directory: Path,
    commit: str,
    status_at_start: list[str],
) -> dict[str, Any]:
    started_at = _now()
    started = time.perf_counter()
    semantic_ladder = _oracle_semantic_ladder()
    seed_reports = []
    reference_parameters: int | None = None
    reference_shapes: dict[str, tuple[int, ...]] | None = None
    for seed in config.seeds:
        arms = {}
        for arm in ARM_SPECS:
            shape_model = _build_model(arm, config, seed, torch.device("cpu"))
            shapes = {
                name: tuple(parameter.shape)
                for name, parameter in shape_model.named_parameters()
            }
            parameters = parameter_count(shape_model)
            if reference_parameters is None:
                reference_parameters = parameters
                reference_shapes = shapes
            elif parameters != reference_parameters or shapes != reference_shapes:
                raise RuntimeError("G15A arm parameter tensors are not exactly matched")
            del shape_model
            symmetry = _run_symmetry_arm(arm, config, seed=seed, device=device)
            no_symmetry = _train_no_symmetry(
                arm,
                config,
                seed=seed,
                device=device,
                checkpoint_directory=checkpoint_directory,
            )
            arms[arm] = {
                "transport_mode": ARM_SPECS[arm][0],
                "readout_mode": ARM_SPECS[arm][1],
                "parameters": parameters,
                "parameter_shapes_sha256": hashlib.sha256(
                    json.dumps(shapes, sort_keys=True).encode()
                ).hexdigest(),
                "symmetry": symmetry,
                "no_symmetry": no_symmetry,
            }
        seed_reports.append({"seed": seed, "arms": arms})
    adjudication = _adjudicate(seed_reports)
    if not bool(semantic_ladder["passed"]):
        adjudication["passed"] = False
        adjudication["decision"] = "G15A stopped: oracle semantic ladder failed"
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("g15a_tasks.py"),
        Path(__file__).with_name("spin_dirac_memory.py"),
        Path(__file__).with_name("model.py"),
        Path(__file__).with_name("optimizers.py"),
        PROTOCOL,
        PREREGISTRATION,
        *AMENDMENTS,
    )
    return {
        "schema_version": 1,
        "experiment": "G15A SpinDirac mechanism and observability cohort",
        "claim_status": (
            "oracle-controlled symmetry mechanism plus learned no-symmetry retrieval; not natural-text evidence"
        ),
        "mode": config.mode,
        "evidentiary": config.mode == "quality" and not status_at_start,
        "started_at": started_at,
        "finished_at": _now(),
        "elapsed_wall_seconds": time.perf_counter() - started,
        "git_commit_at_start": commit,
        "git_status_at_start": status_at_start,
        "protocol": asdict(config),
        "arm_specs": {
            arm: {"transport": modes[0], "readout": modes[1]}
            for arm, modes in ARM_SPECS.items()
        },
        "protocol_files": {
            str(path.relative_to(Path(__file__).parent)): _sha256(path)
            for path in (PROTOCOL, PREREGISTRATION, *AMENDMENTS)
        },
        "source_files": {
            str(path.relative_to(Path(__file__).parent)): _sha256(path)
            for path in source_paths
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "compute_capability": (
                list(torch.cuda.get_device_capability(device))
                if device.type == "cuda"
                else None
            ),
            "dtype": "float32",
        },
        "oracle_semantic_ladder": semantic_ladder,
        "seed_reports": seed_reports,
        "adjudication": adjudication,
        "explicit_nonclaims": [
            "the symmetry result supplies exact coordinates and oracle carrier controls",
            "the learned no-symmetry result does not show Spin necessity",
            "no generic associative-memory, natural-text, long-recall, scaling, or fused-kernel promotion follows",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "quality"), required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = quality_config() if args.mode == "quality" else smoke_config()
    for path in (PROTOCOL, PREREGISTRATION, *AMENDMENTS):
        if not path.is_file():
            raise FileNotFoundError(path)
    commit, status_at_start = _git_state()
    if args.mode == "quality" and status_at_start:
        raise RuntimeError(
            "evidentiary G15A quality mode requires a clean committed worktree"
        )
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if torch.cuda.get_device_capability(device) != (7, 5):
            raise RuntimeError("the frozen local G15A cohort requires exact SM75")
    report = run(
        config,
        device=device,
        checkpoint_directory=args.checkpoint_directory,
        commit=commit,
        status_at_start=status_at_start,
    )
    _atomic_json(args.output, report)
    print(args.output)
    print(json.dumps(report["adjudication"], indent=2, sort_keys=True))
    if args.mode == "quality" and not report["adjudication"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
