"""Frozen G15B commissioned-controller cohort.

This runner intentionally keeps the learned model path ordinary during
training.  Evaluation interventions reconstruct the frozen one-block shell,
form every learned control, replace complete controls, and only then call the
semantic ``forward_controls`` transition.  The distinction is required for a
causal-use claim: partial module hooks can leave an edit or residual bypass.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from .g15b_interleaved_tasks import (
        InterleavedBatch,
        TaskName,
        generate_interleaved_batch,
        oracle_direct_read_accuracy,
    )
    from .model import HybridMemoryConfig, HybridMemoryLM
    from .optimizers import HarmonicMuonAdamW
    from .spin_dirac_memory import SpinDiracConfig, SpinDiracMemory
else:  # pragma: no cover - direct script execution in the bound WSL venv
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hybrid_memory_v1_4.g15b_interleaved_tasks import (  # type: ignore[no-redef]
        InterleavedBatch,
        TaskName,
        generate_interleaved_batch,
        oracle_direct_read_accuracy,
    )
    from hybrid_memory_v1_4.model import (  # type: ignore[no-redef]
        HybridMemoryConfig,
        HybridMemoryLM,
    )
    from hybrid_memory_v1_4.optimizers import (  # type: ignore[no-redef]
        HarmonicMuonAdamW,
    )
    from hybrid_memory_v1_4.spin_dirac_memory import (  # type: ignore[no-redef]
        SpinDiracConfig,
        SpinDiracMemory,
    )


ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "G15B_CONTROL_PROTOCOL_2026-08-25.md"
PREREGISTRATION = ROOT / "G15_SPIN_DIRAC_PREREGISTRATION.md"
QUALITY_SEEDS = (2309, 2311, 2333)
TASK_CYCLE: tuple[TaskName, ...] = (
    "mqar",
    "overwrite",
    "overwrite",
    "selective",
    "needle",
)
EVALUATION_LENGTHS = (128, 512, 1024)
NEEDLE_DISTANCES = {128: 64, 512: 448, 1024: 960}
ARM_SPECS = {
    "I": ("identity", "identity"),
    "C": ("commuting_so2", "clifford"),
    "S": ("spin8", "clifford"),
}
Intervention = Literal[
    "learned", "no_memory", "no_write", "no_erase", "wrong_query", "oracle"
]


@dataclass(frozen=True)
class Phase:
    length: int
    live_keys: int
    max_writes: int
    queries: int
    batch_size: int
    updates: int
    learning_rate: float


QUALITY_PHASES = (
    Phase(64, 2, 4, 4, 32, 800, 0.003),
    Phase(128, 4, 8, 4, 32, 1000, 0.003),
    Phase(256, 8, 16, 8, 16, 1200, 0.003),
    Phase(512, 8, 24, 8, 8, 800, 0.001),
    Phase(1024, 8, 24, 8, 4, 400, 0.001),
)
SMOKE_PHASES = tuple(
    Phase(
        phase.length,
        phase.live_keys,
        phase.max_writes,
        phase.queries,
        min(2, phase.batch_size),
        1,
        phase.learning_rate,
    )
    for phase in QUALITY_PHASES
)


@dataclass(frozen=True)
class CohortConfig:
    mode: Literal["smoke", "quality"]
    seeds: tuple[int, ...]
    phases: tuple[Phase, ...]
    evaluation_decisions: int
    evaluation_batch_cap: int


@dataclass(frozen=True)
class LossWeights:
    retrieval: float = 1.0
    reverse_binding: float = 0.25
    address: float = 0.25
    write: float = 0.25
    erase: float = 0.10
    retention: float = 0.01
    address_temperature: float = 0.10


LOSS_WEIGHTS = LossWeights()


def _stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (2**63 - 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _git(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_provenance() -> tuple[str, list[str]]:
    commit = _git(["rev-parse", "HEAD"])
    status = [line for line in _git(["status", "--short"]).splitlines() if line]
    return commit, status


def frozen_config(mode: Literal["smoke", "quality"]) -> CohortConfig:
    if mode == "quality":
        return CohortConfig(mode, QUALITY_SEEDS, QUALITY_PHASES, 4096, 16)
    if mode == "smoke":
        return CohortConfig(mode, (23,), SMOKE_PHASES, 16, 2)
    raise ValueError("mode must be 'smoke' or 'quality'")


def scored_training_decisions(phases: tuple[Phase, ...]) -> int:
    total = 0
    update = 0
    for phase in phases:
        for _ in range(phase.updates):
            task = TASK_CYCLE[update % len(TASK_CYCLE)]
            total += phase.batch_size * (1 if task == "needle" else phase.queries)
            update += 1
    return total


def build_model(arm: str, seed: int, device: torch.device) -> HybridMemoryLM:
    if arm not in ARM_SPECS:
        raise ValueError(f"unknown G15B arm {arm!r}")
    torch.manual_seed(seed)
    transport, readout = ARM_SPECS[arm]
    model = HybridMemoryLM(
        HybridMemoryConfig(
            vocab_size=69,
            model_dim=64,
            layer_plan=("spin_dirac",),
            expansion=2,
            dropout=0.0,
            tie_embeddings=True,
            use_local_conv=True,
            conv_kernel=4,
            spin_dirac_heads=4,
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
        )
    )
    return model.to(device)


def _gather_time(tensor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    suffix = tensor.shape[2:]
    index = positions.view(*positions.shape, *([1] * len(suffix))).expand(
        *positions.shape, *suffix
    )
    return tensor.gather(1, index)


def _spin_diagnostics(output: dict[str, Any]) -> dict[str, torch.Tensor | str]:
    diagnostics = output.get("diagnostics")
    if not isinstance(diagnostics, tuple) or len(diagnostics) != 1:
        raise RuntimeError("G15B requires exactly one diagnostic Spin-Dirac block")
    row = diagnostics[0]
    if not isinstance(row, dict) or row.get("kind") != "spin_dirac":
        raise RuntimeError("G15B did not receive Spin-Dirac diagnostics")
    return row


def _control_tensor(
    diagnostics: dict[str, torch.Tensor | str], name: str
) -> torch.Tensor:
    tensor = diagnostics.get(name)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"missing tensor diagnostic {name!r}")
    return tensor


def _address_prototypes(
    diagnostics: dict[str, torch.Tensor | str], batch: InterleavedBatch
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = _control_tensor(diagnostics, "key_vector")
    query = _control_tensor(diagnostics, "query_vector")
    write_key_vectors = _gather_time(key, batch.write_positions)
    prototypes = []
    for key_index in range(batch.live_keys.shape[1]):
        selected_key = batch.live_keys[:, key_index : key_index + 1]
        mask = (batch.write_keys == selected_key).to(key.dtype)
        count = mask.sum(dim=1).clamp_min(1.0)
        prototype = torch.einsum("bw,bwhd->bhd", mask, write_key_vectors)
        prototype = F.normalize(prototype / count[:, None, None], dim=-1)
        prototypes.append(prototype)
    prototype_tensor = torch.stack(prototypes, dim=1)
    query_vectors = _gather_time(query, batch.query_positions)
    scores = (
        torch.einsum("bqhd,bkhd->bqk", query_vectors, prototype_tensor)
        / query_vectors.shape[2]
    )
    return prototype_tensor, query_vectors, scores


def _balanced_bce(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = probability.clamp(1e-7, 1.0 - 1e-7)
    target = target.expand_as(probability)
    terms = []
    if bool(target.any()):
        terms.append(-probability[target].log().mean())
    negative = ~target
    if bool(negative.any()):
        terms.append(-(1.0 - probability[negative]).log().mean())
    if not terms:
        raise RuntimeError("balanced BCE received no examples")
    return torch.stack(terms).mean()


def commissioned_losses(
    output: dict[str, Any],
    batch: InterleavedBatch,
    *,
    weights: LossWeights = LOSS_WEIGHTS,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the frozen commissioned loss and its differentiable components."""

    logits = output.get("logits")
    if not isinstance(logits, torch.Tensor):
        raise TypeError("model output is missing logits")
    diagnostics = _spin_diagnostics(output)
    query_logits = _gather_time(logits, batch.query_positions)
    write_logits = _gather_time(logits, batch.write_positions)
    retrieval = F.cross_entropy(query_logits.flatten(0, 1), batch.targets.flatten())
    reverse = F.cross_entropy(write_logits.flatten(0, 1), batch.write_keys.flatten())

    _, _, address_scores = _address_prototypes(diagnostics, batch)
    address = F.cross_entropy(
        (address_scores / weights.address_temperature).flatten(0, 1),
        batch.query_key_indices.flatten(),
    )
    write_strength = _control_tensor(diagnostics, "write_strength")
    erase_strength = _control_tensor(diagnostics, "erase_strength")
    retention = _control_tensor(diagnostics, "retention")
    write = _balanced_bce(write_strength, batch.write_event_mask[..., None, None])
    erase = _balanced_bce(erase_strength, batch.erase_event_mask[..., None, None])
    non_edit = ~batch.write_event_mask
    normalized_retention = (retention - 0.999) / (0.999999 - 0.999)
    retention_loss = (1.0 - normalized_retention[non_edit]).square().mean()
    components = {
        "retrieval": retrieval,
        "reverse_binding": reverse,
        "address": address,
        "write": write,
        "erase": erase,
        "retention": retention_loss,
    }
    total = (
        weights.retrieval * retrieval
        + weights.reverse_binding * reverse
        + weights.address * address
        + weights.write * write
        + weights.erase * erase
        + weights.retention * retention_loss
    )
    return total, components


def _wrong_live_query_override(
    query: torch.Tensor,
    key: torch.Tensor,
    batch: InterleavedBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    overridden = query.clone()
    eligible = torch.zeros_like(batch.query_positions, dtype=torch.bool)
    for row in range(batch.batch_size):
        for query_index in range(batch.queries):
            position = int(batch.query_positions[row, query_index])
            correct = int(batch.query_key_indices[row, query_index])
            candidates: list[tuple[int, int]] = []
            for write_index in range(batch.writes):
                write_position = int(batch.write_positions[row, write_index])
                if write_position >= position:
                    continue
                written_key = int(batch.write_keys[row, write_index])
                live_index = int((batch.live_keys[row] == written_key).long().argmax())
                if live_index != correct:
                    candidates.append((live_index, write_position))
            if not candidates:
                continue
            wrong_index = (correct + 1) % batch.live_keys.shape[1]
            matching = [
                candidate for candidate in candidates if candidate[0] == wrong_index
            ]
            _, source_position = (matching or candidates)[-1]
            overridden[row, position] = F.normalize(key[row, source_position], dim=-1)
            eligible[row, query_index] = True
    return overridden, eligible


def _oracle_control_override(
    controls: list[torch.Tensor], batch: InterleavedBatch
) -> None:
    query, key, _, erase, write, _, _ = controls
    query.zero_()
    key.zero_()
    erase.zero_()
    write.zero_()
    rows = torch.arange(batch.batch_size, device=query.device)
    for write_index in range(batch.writes):
        positions = batch.write_positions[:, write_index]
        key_indices = (
            (batch.live_keys == batch.write_keys[:, write_index : write_index + 1])
            .long()
            .argmax(dim=1)
        )
        for head in range(query.shape[2]):
            key[rows, positions, head, key_indices] = 1.0
        write[rows, positions, :, 0] = 1.0
        erase[rows, positions, :, 0] = batch.overwrite_mask[:, write_index, None].to(
            erase.dtype
        )
    for query_index in range(batch.queries):
        positions = batch.query_positions[:, query_index]
        key_indices = batch.query_key_indices[:, query_index]
        for head in range(query.shape[2]):
            query[rows, positions, head, key_indices] = 1.0


def complete_control_forward(
    model: HybridMemoryLM,
    batch: InterleavedBatch,
    intervention: Intervention,
) -> dict[str, Any]:
    """Run a complete evaluation override immediately before transitions."""

    if intervention not in (
        "learned",
        "no_memory",
        "no_write",
        "no_erase",
        "wrong_query",
        "oracle",
    ):
        raise ValueError(f"unknown G15B intervention {intervention!r}")
    if model.layer_plan != ("spin_dirac",):
        raise ValueError(
            "G15B complete-control evaluation requires one Spin-Dirac block"
        )
    block = model.blocks[0]
    mixer = block.mixer
    if not isinstance(mixer, SpinDiracMemory):
        raise TypeError("G15B block does not contain SpinDiracMemory")
    hidden = model.embedding(batch.token_ids)
    value, outer_gate = block.input_projection(block.mixer_norm(hidden)).chunk(
        2, dim=-1
    )
    if block.local_conv is None:
        mixed = value
    else:
        mixed, _ = block.local_conv(value, None, None)
    mixed = F.silu(mixed)
    controls = [tensor.clone() for tensor in mixer._controls(mixed, None)]
    wrong_query_eligible = torch.ones_like(batch.query_positions, dtype=torch.bool)
    if intervention == "no_write":
        controls[4].zero_()
    elif intervention == "no_erase":
        controls[3].zero_()
    elif intervention == "wrong_query":
        controls[0], wrong_query_eligible = _wrong_live_query_override(
            controls[0], controls[1], batch
        )
    elif intervention == "oracle":
        _oracle_control_override(controls, batch)

    if intervention == "no_memory":
        update = torch.zeros_like(hidden)
    else:
        read, _ = mixer.forward_controls(*controls, scan_mode="parallel")
        output_gate = 1.0 + torch.tanh(mixer.output_gate(mixed).view_as(read))
        update = mixer.output_projection(
            (mixer.output_norm(read) * output_gate).flatten(start_dim=2)
        )
    hidden = hidden + torch.sigmoid(block.residual_scale) * block.dropout(
        update * torch.sigmoid(outer_gate)
    )
    hidden = hidden + block.dropout(block.ffn(block.ffn_norm(hidden)))
    logits = model.lm_head(model.final_norm(hidden))
    names = (
        "query_vector",
        "key_vector",
        "value_positive",
        "erase_strength",
        "write_strength",
        "retention",
        "transport_coordinates",
    )
    diagnostics = {name: tensor for name, tensor in zip(names, controls, strict=True)}
    diagnostics["kind"] = "spin_dirac_complete_control"
    return {
        "logits": logits,
        "diagnostics": diagnostics,
        "wrong_query_eligible": wrong_query_eligible,
    }


def _oracle_memory(arm: str, *, dtype: torch.dtype) -> SpinDiracMemory:
    transport, readout = ARM_SPECS[arm]
    return SpinDiracMemory(
        SpinDiracConfig(
            model_dim=64,
            heads=4,
            transport_mode=transport,  # type: ignore[arg-type]
            readout_mode=readout,  # type: ignore[arg-type]
            gate_mode="equivariant_scalar",
            tie_query_key=True,
            bound_values=True,
            minimum_retention=0.999,
            maximum_retention=0.999999,
        )
    ).to(dtype=dtype)


def _semantic_replay_contract() -> dict[str, Any]:
    memory = _oracle_memory("S", dtype=torch.float64)
    length = 14
    query = torch.zeros(1, length, 4, 8, dtype=torch.float64)
    key = torch.zeros_like(query)
    value = torch.zeros_like(query)
    erase = torch.zeros(1, length, 4, 1, dtype=torch.float64)
    write = torch.zeros_like(erase)
    retention = torch.full_like(erase, 1.0 - 1e-14)
    coordinates = torch.zeros(1, length, 4, 28, dtype=torch.float64)
    for address in range(8):
        key[0, address, 0, address] = 1.0
        value[0, address, 0, (address + 1) % 8] = 1.0
        write[0, address, 0] = 1.0
    query[0, 0, 0, 0] = 1.0
    query[0, 8, 0, 0] = 1.0
    query[0, 9, 0, 1] = 1.0
    key[0, 10, 0, 0] = 1.0
    value[0, 10, 0, 3] = 1.0
    erase[0, 10, 0] = 1.0
    write[0, 10, 0] = 1.0
    query[0, 10, 0, 0] = 1.0
    query[0, 11, 0, 1] = 1.0
    query[0, 12, 0, 0] = 1.0
    query[0, 13, 0, 0] = 1.0
    read, _ = memory.forward_controls(
        query, key, value, erase, write, retention, coordinates
    )
    r = float(retention[0, 0, 0, 0])
    errors = {
        "first_write": abs(float(read[0, 0, 0, 1]) - 1.0),
        "eight_address_survival": abs(float(read[0, 8, 0, 1]) - r**8),
        "unrelated_key_preservation": abs(float(read[0, 9, 0, 2]) - r**8),
        "same_key_overwrite": abs(float(read[0, 10, 0, 3]) - 1.0),
        "unrelated_key_after_overwrite": abs(float(read[0, 11, 0, 2]) - r**10),
        "repeated_query_first": abs(float(read[0, 12, 0, 3]) - r**2),
        "repeated_query_second": abs(float(read[0, 13, 0, 3]) - r**3),
    }
    maximum = max(errors.values())
    return {
        "errors": errors,
        "maximum_absolute_error": maximum,
        "passed": maximum < 1e-10,
    }


def _path_parity(model: HybridMemoryLM, batch: InterleavedBatch) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        parallel = model(batch.token_ids, delta_scan_mode="parallel")["logits"]
        recurrent = model(batch.token_ids, delta_scan_mode="recurrent")["logits"]
        states = None
        chunks = []
        for start in range(0, batch.length, 7):
            result = model(
                batch.token_ids[:, start : start + 7],
                states,
                delta_scan_mode="parallel",
            )
            chunks.append(result["logits"])
            states = result["states"]
        chunked = torch.cat(chunks, dim=1)
        states = None
        steps = []
        for position in range(batch.length):
            logits, states = model.step(batch.token_ids[:, position], states)
            steps.append(logits[:, None])
        stepped = torch.cat(steps, dim=1)
        learned = complete_control_forward(model, batch, "learned")["logits"]
    residuals = {
        "parallel_recurrent": float((parallel - recurrent).abs().max()),
        "parallel_arbitrary_chunk": float((parallel - chunked).abs().max()),
        "parallel_token_step": float((parallel - stepped).abs().max()),
        "ordinary_complete_control": float((parallel - learned).abs().max()),
    }
    return {"residuals": residuals, "passed": max(residuals.values()) <= 5e-5}


def _address_descent_direction(
    diagnostics: dict[str, torch.Tensor | str], batch: InterleavedBatch
) -> dict[str, Any]:
    prototypes, query, _ = _address_prototypes(diagnostics, batch)
    query_leaf = query.detach().clone().requires_grad_(True)
    scores = torch.einsum("bqhd,bkhd->bqk", query_leaf, prototypes.detach())
    scores = scores / query_leaf.shape[2]
    loss = F.cross_entropy(
        (scores / LOSS_WEIGHTS.address_temperature).flatten(0, 1),
        batch.query_key_indices.flatten(),
    )
    (gradient,) = torch.autograd.grad(loss, query_leaf)
    direction = -gradient
    target = prototypes.gather(
        1,
        batch.query_key_indices[..., None, None].expand(
            -1, -1, prototypes.shape[2], prototypes.shape[3]
        ),
    )
    target_gain = (direction * target).sum(dim=(-1, -2))
    wrong_gains = torch.einsum("bqhd,bkhd->bqk", direction, prototypes)
    wrong_gains.scatter_(2, batch.query_key_indices[..., None], -torch.inf)
    margins = target_gain - wrong_gains.max(dim=-1).values
    return {
        "minimum_first_order_target_margin": float(margins.detach().min()),
        "mean_first_order_target_margin": float(margins.detach().mean()),
        "passed": bool((margins > 0).all()),
    }


def _activation_gradient_gate(
    model: HybridMemoryLM, batch: InterleavedBatch
) -> dict[str, Any]:
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(batch.token_ids, return_diagnostics=True)
    diagnostics = _spin_diagnostics(output)
    names = (
        "query_vector",
        "key_vector",
        "value_positive",
        "erase_strength",
        "write_strength",
        "retention",
        "transport_coordinates",
    )
    tensors = {name: _control_tensor(diagnostics, name) for name in names}
    for tensor in tensors.values():
        if tensor.requires_grad:
            tensor.retain_grad()
    loss, _ = commissioned_losses(output, batch)
    loss.backward()
    positions = {
        "query_vector": batch.query_positions,
        "key_vector": batch.write_positions,
        "value_positive": batch.write_positions,
    }
    reports: dict[str, Any] = {}
    for name, tensor in tensors.items():
        gradient = tensor.grad
        if gradient is None:
            norm = 0.0
        elif name in positions:
            norm = float(
                torch.linalg.vector_norm(_gather_time(gradient, positions[name]))
            )
        elif name == "erase_strength":
            norm = float(torch.linalg.vector_norm(gradient[batch.erase_event_mask]))
        elif name == "write_strength":
            norm = float(torch.linalg.vector_norm(gradient[batch.write_event_mask]))
        elif name == "retention":
            norm = float(torch.linalg.vector_norm(gradient[~batch.write_event_mask]))
        else:
            norm = float(torch.linalg.vector_norm(gradient))
        reports[name] = {"gradient_norm": norm, "finite": math.isfinite(norm)}
    transport = model.config.spin_dirac_transport_mode
    required = [name for name in names if name != "transport_coordinates"]
    if transport != "identity":
        required.append("transport_coordinates")
    passed = all(
        reports[name]["finite"] and reports[name]["gradient_norm"] > 0.0
        for name in required
    )
    reports["transport_coordinates"]["required"] = transport != "identity"
    if transport == "identity":
        reports["transport_coordinates"]["structural_reason"] = (
            "identity transport masks the coordinate activation by definition"
        )
    return {"loss": float(loss.detach()), "paths": reports, "passed": passed}


def _optimizer_partition(model: HybridMemoryLM) -> dict[str, Any]:
    optimizer = HarmonicMuonAdamW(model, lr=0.003, weight_decay=0.01)
    trainable = {id(parameter): name for name, parameter in model.named_parameters()}
    assigned = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    counts = {identifier: assigned.count(identifier) for identifier in set(assigned)}
    missing = [
        name for identifier, name in trainable.items() if identifier not in counts
    ]
    duplicate = [
        trainable[identifier] for identifier, count in counts.items() if count != 1
    ]
    report = {
        "trainable_tensors": len(trainable),
        "assigned_tensors": len(assigned),
        "missing": missing,
        "duplicate": duplicate,
        "groups": optimizer.partition_report(),
        "passed": not missing and not duplicate and len(assigned) == len(trainable),
    }
    del optimizer
    return report


def run_preflight(device: torch.device) -> dict[str, Any]:
    semantic = _semantic_replay_contract()
    arms: dict[str, Any] = {}
    reference_shapes: dict[str, tuple[int, ...]] | None = None
    reference_initial: str | None = None
    data_hashes: dict[str, str] = {}
    for arm in ARM_SPECS:
        model = build_model(arm, 23, device)
        shapes = {
            name: tuple(parameter.shape) for name, parameter in model.named_parameters()
        }
        initial = _tensor_sha256(dict(model.named_parameters()))
        if reference_shapes is None:
            reference_shapes = shapes
            reference_initial = initial
        batch = generate_interleaved_batch(
            "overwrite", 2, 128, 4, 8, 4, seed=_stable_seed("g15b-preflight", 23)
        ).to(device)
        data_hashes[arm] = batch.fingerprint()
        parity = _path_parity(model, batch)
        gradients = _activation_gradient_gate(model, batch)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            output = model(batch.token_ids, return_diagnostics=True)
            descent = _address_descent_direction(_spin_diagnostics(output), batch)
        partition = _optimizer_partition(model)
        oracle = {}
        for task in ("mqar", "overwrite", "selective", "needle"):
            oracle_batch = generate_interleaved_batch(
                task,
                2,
                128,
                8,
                24,
                8,
                seed=_stable_seed("g15b-oracle", task),
                needle_distance=64 if task == "needle" else None,
            )
            memory = _oracle_memory(arm, dtype=torch.float64)
            accuracy, residual = oracle_direct_read_accuracy(oracle_batch, memory)
            oracle[task] = {
                "accuracy": accuracy,
                "maximum_absolute_residual": residual,
                "passed": accuracy >= 0.999 and residual < 1e-10,
            }
        paired = shapes == reference_shapes and initial == reference_initial
        checks = {
            "path_parity": parity["passed"],
            "activation_gradients": gradients["passed"],
            "address_descent": descent["passed"],
            "optimizer_partition": partition["passed"],
            "oracle_each_stratum": all(row["passed"] for row in oracle.values()),
            "paired_shapes_and_initial_parameters": paired,
        }
        arms[arm] = {
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "parameter_shapes_sha256": hashlib.sha256(
                json.dumps(shapes, sort_keys=True).encode()
            ).hexdigest(),
            "initial_parameter_sha256": initial,
            "path_parity": parity,
            "activation_gradients": gradients,
            "address_descent": descent,
            "optimizer_partition": partition,
            "oracle": oracle,
            "checks": checks,
            "passed": all(checks.values()),
        }
        del model
    data_identical = len(set(data_hashes.values())) == 1
    passed = (
        semantic["passed"]
        and data_identical
        and all(row["passed"] for row in arms.values())
    )
    return {
        "semantic_replay": semantic,
        "arms": arms,
        "paired_data_hashes": data_hashes,
        "paired_data_identical": data_identical,
        "passed": passed,
    }


def _set_learning_rate(optimizer: HarmonicMuonAdamW, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _batch_for_update(
    phase: Phase, *, seed: int, global_update: int
) -> InterleavedBatch:
    task = TASK_CYCLE[global_update % len(TASK_CYCLE)]
    return generate_interleaved_batch(
        task,
        phase.batch_size,
        phase.length,
        phase.live_keys,
        phase.max_writes,
        phase.queries,
        seed=_stable_seed("g15b-train", seed, global_update, task),
    )


def _binary_counts(probability: torch.Tensor, target: torch.Tensor) -> dict[str, int]:
    prediction = probability >= 0.5
    target = target.expand_as(prediction)
    return {
        "tp": int((prediction & target).sum()),
        "fp": int((prediction & ~target).sum()),
        "fn": int((~prediction & target).sum()),
        "tn": int((~prediction & ~target).sum()),
    }


def _merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for name, value in source.items():
        target[name] = target.get(name, 0) + value


def _classification_report(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn, tn = (counts.get(name, 0) for name in ("tp", "fp", "fn", "tn"))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "false_positive_rate": fp / max(1, fp + tn),
    }


def _evaluation_batch_size(task: TaskName, *, decisions: int, cap: int) -> int:
    queries_per_episode = 1 if task == "needle" else 8
    complete_episodes = decisions // queries_per_episode
    return min(cap, complete_episodes)


@torch.no_grad()
def _evaluate_cell(
    model: HybridMemoryLM,
    task: TaskName,
    length: int,
    *,
    seed: int,
    decisions: int,
    batch_cap: int,
) -> tuple[dict[str, Any], set[str]]:
    model.eval()
    device = model.embedding.weight.device
    batch_size = _evaluation_batch_size(task, decisions=decisions, cap=batch_cap)
    decisions_per_batch = batch_size * (1 if task == "needle" else 8)
    if decisions % decisions_per_batch:
        raise ValueError(
            "G15B evaluation decisions must be divisible by the complete-batch "
            f"decision count ({decisions_per_batch})"
        )
    query_correct = 0
    query_total = 0
    episode_correct = 0
    episode_total = 0
    nll_sum = 0.0
    address_correct = 0
    address_total = 0
    address_margin_sum = 0.0
    address_margin_min = math.inf
    consistency_sum = 0.0
    consistency_count = 0
    write_counts: dict[str, int] = {}
    erase_counts: dict[str, int] = {}
    collision_events = 0
    first_write_fp = first_write_total = 0
    filler_erase_fp = filler_erase_total = 0
    retention_values: list[torch.Tensor] = []
    coordinate_sum_squared = 0.0
    coordinate_count = 0
    coordinate_max = 0.0
    query_quartiles = [{"correct": 0, "total": 0} for _ in range(4)]
    intervention_correct = {
        name: 0
        for name in ("no_memory", "no_write", "no_erase", "wrong_query", "oracle")
    }
    intervention_total = {name: 0 for name in intervention_correct}
    hashes: set[str] = set()
    batch_index = 0
    while query_total < decisions:
        batch = generate_interleaved_batch(
            task,
            batch_size,
            length,
            8,
            24,
            8,
            seed=_stable_seed("g15b-eval", seed, task, length, batch_index),
            needle_distance=NEEDLE_DISTANCES[length] if task == "needle" else None,
        ).to(device)
        hashes.add(batch.fingerprint())
        baseline = complete_control_forward(model, batch, "learned")
        logits = _gather_time(baseline["logits"], batch.query_positions)
        predictions = logits.argmax(dim=-1)
        correct = predictions == batch.targets
        query_correct += int(correct.sum())
        query_total += correct.numel()
        nll_sum += float(
            F.cross_entropy(
                logits.flatten(0, 1), batch.targets.flatten(), reduction="sum"
            )
        )
        episode_correct += int(correct.all(dim=1).sum())
        episode_total += batch.batch_size

        diagnostics = baseline["diagnostics"]
        _, _, address_scores = _address_prototypes(diagnostics, batch)
        address_predictions = address_scores.argmax(dim=-1)
        address_correct += int((address_predictions == batch.query_key_indices).sum())
        address_total += address_predictions.numel()
        target_scores = address_scores.gather(
            2, batch.query_key_indices[..., None]
        ).squeeze(-1)
        wrong_scores = address_scores.clone()
        wrong_scores.scatter_(2, batch.query_key_indices[..., None], -torch.inf)
        if address_scores.shape[-1] > 1:
            margins = target_scores - wrong_scores.max(dim=-1).values
            address_margin_sum += float(margins.sum())
            address_margin_min = min(address_margin_min, float(margins.min()))

        key_vectors = _control_tensor(diagnostics, "key_vector")
        gathered_keys = _gather_time(key_vectors, batch.write_positions)
        for row in range(batch.batch_size):
            for live_key in batch.live_keys[row]:
                mask = batch.write_keys[row] == live_key
                vectors = gathered_keys[row, mask]
                if vectors.shape[0] > 1:
                    base = F.normalize(vectors[0], dim=-1)
                    cosine = (F.normalize(vectors[1:], dim=-1) * base).sum(-1)
                    consistency_sum += float(cosine.sum())
                    consistency_count += cosine.numel()

        write_strength = _control_tensor(diagnostics, "write_strength")
        erase_strength = _control_tensor(diagnostics, "erase_strength")
        _merge_counts(
            write_counts,
            _binary_counts(write_strength, batch.write_event_mask[..., None, None]),
        )
        _merge_counts(
            erase_counts,
            _binary_counts(erase_strength, batch.erase_event_mask[..., None, None]),
        )
        collision_events += int(batch.erase_event_mask.sum())
        erase_prediction = erase_strength.mean(dim=(-1, -2)) >= 0.5
        first_write = batch.write_event_mask & ~batch.erase_event_mask
        filler = batch.roles == 0
        first_write_fp += int((erase_prediction & first_write).sum())
        first_write_total += int(first_write.sum())
        filler_erase_fp += int((erase_prediction & filler).sum())
        filler_erase_total += int(filler.sum())
        retention = _control_tensor(diagnostics, "retention")
        retention_values.append(
            retention[~batch.write_event_mask].detach().cpu().float()
        )
        coordinates = _control_tensor(diagnostics, "transport_coordinates")
        coordinate_sum_squared += float(coordinates.float().square().sum())
        coordinate_count += coordinates.numel()
        coordinate_max = max(coordinate_max, float(coordinates.abs().max()))
        quartiles = (batch.query_positions * 4 // length).clamp_max(3)
        for quartile in range(4):
            mask = quartiles == quartile
            query_quartiles[quartile]["correct"] += int((correct & mask).sum())
            query_quartiles[quartile]["total"] += int(mask.sum())

        for intervention in intervention_correct:
            result = complete_control_forward(model, batch, intervention)  # type: ignore[arg-type]
            selected = _gather_time(result["logits"], batch.query_positions).argmax(-1)
            eligible = (
                result["wrong_query_eligible"]
                if intervention == "wrong_query"
                else torch.ones_like(batch.query_positions, dtype=torch.bool)
            )
            intervention_correct[intervention] += int(
                ((selected == batch.targets) & eligible).sum()
            )
            intervention_total[intervention] += int(eligible.sum())
        batch_index += 1

    retention_flat = torch.cat(retention_values)
    write_report = _classification_report(write_counts)
    erase_report = _classification_report(erase_counts)
    intervention_accuracy = {
        name: intervention_correct[name] / max(1, intervention_total[name])
        for name in intervention_correct
    }
    baseline_accuracy = query_correct / query_total
    return {
        "task": task,
        "length": length,
        "needle_distance": NEEDLE_DISTANCES[length] if task == "needle" else None,
        "live_keys": 1 if task == "needle" else 8,
        "query_decisions": query_total,
        "query_accuracy": baseline_accuracy,
        "exact_episode_accuracy": episode_correct / max(1, episode_total),
        "bits_per_query": nll_sum / query_total / math.log(2.0),
        "address": {
            "top1": address_correct / max(1, address_total),
            "mean_correct_minus_best_wrong_cosine": (
                address_margin_sum / address_total
                if address_scores.shape[-1] > 1
                else None
            ),
            "minimum_correct_minus_best_wrong_cosine": (
                address_margin_min if address_scores.shape[-1] > 1 else None
            ),
            "same_key_overwrite_consistency": consistency_sum
            / max(1, consistency_count),
            "consistency_pairs": consistency_count,
        },
        "write": write_report,
        "erase": {
            **erase_report,
            "first_write_false_positive_rate": first_write_fp
            / max(1, first_write_total),
            "filler_false_positive_rate": filler_erase_fp / max(1, filler_erase_total),
        },
        "retention": {
            "non_edit_p05": float(torch.quantile(retention_flat, 0.05)),
            "non_edit_mean": float(retention_flat.mean()),
            "non_edit_minimum": float(retention_flat.min()),
        },
        "coordinates": {
            "rms": math.sqrt(coordinate_sum_squared / coordinate_count),
            "maximum_absolute": coordinate_max,
        },
        "query_location_quartiles": [
            {
                **row,
                "accuracy": row["correct"] / max(1, row["total"]),
            }
            for row in query_quartiles
        ],
        "interventions": {
            name: {
                "accuracy": accuracy,
                "decisions": intervention_total[name],
                "drop_from_learned": baseline_accuracy - accuracy,
            }
            for name, accuracy in intervention_accuracy.items()
        },
        "collision_writes": collision_events,
        "evaluation_batch_fingerprints": len(hashes),
    }, hashes


def _evaluate(
    model: HybridMemoryLM, config: CohortConfig, *, seed: int
) -> tuple[dict[str, Any], set[str]]:
    cells = {}
    hashes: set[str] = set()
    for task in ("mqar", "overwrite", "selective", "needle"):
        for length in EVALUATION_LENGTHS:
            cell, cell_hashes = _evaluate_cell(
                model,
                task,
                length,
                seed=seed,
                decisions=config.evaluation_decisions,
                batch_cap=config.evaluation_batch_cap,
            )
            cells[f"{task}:L{length}"] = cell
            hashes.update(cell_hashes)
    return {"cells": cells}, hashes


def _train_arm(
    arm: str,
    config: CohortConfig,
    *,
    seed: int,
    device: torch.device,
    checkpoint_directory: Path,
) -> dict[str, Any]:
    model = build_model(arm, seed, device)
    optimizer = HarmonicMuonAdamW(
        model, lr=config.phases[0].learning_rate, weight_decay=0.01
    )
    schedule_hash = hashlib.sha256()
    train_fingerprints: set[str] = set()
    event_counts = {"queries": 0, "writes": 0, "overwrites": 0, "non_edits": 0}
    loss_samples = []
    global_update = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    for phase_index, phase in enumerate(config.phases):
        _set_learning_rate(optimizer, phase.learning_rate)
        for phase_update in range(phase.updates):
            batch = _batch_for_update(phase, seed=seed, global_update=global_update).to(
                device
            )
            fingerprint = batch.fingerprint()
            train_fingerprints.add(fingerprint)
            schedule_hash.update(fingerprint.encode())
            event_counts["queries"] += batch.targets.numel()
            event_counts["writes"] += int(batch.write_event_mask.sum())
            event_counts["overwrites"] += int(batch.erase_event_mask.sum())
            event_counts["non_edits"] += int((~batch.write_event_mask).sum())
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(batch.token_ids, return_diagnostics=True)
            loss, components = commissioned_losses(output, batch)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite G15B loss at update {global_update}")
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError(f"nonfinite G15B gradient at update {global_update}")
            optimizer.step()
            if (
                phase_update == 0
                or phase_update + 1 == phase.updates
                or global_update % 100 == 0
            ):
                loss_samples.append(
                    {
                        "global_update": global_update + 1,
                        "phase": phase_index + 1,
                        "phase_update": phase_update + 1,
                        "task": batch.task,
                        "learning_rate": phase.learning_rate,
                        "total": float(loss.detach()),
                        "gradient_norm": float(gradient_norm.detach()),
                        "components": {
                            name: float(value.detach())
                            for name, value in components.items()
                        },
                    }
                )
            global_update += 1
    _sync(device)
    training_seconds = time.perf_counter() - started
    _sync(device)
    evaluation_started = time.perf_counter()
    evaluation, evaluation_hashes = _evaluate(model, config, seed=seed)
    _sync(device)
    evaluation_seconds = time.perf_counter() - evaluation_started
    intersection = train_fingerprints & evaluation_hashes
    evaluation_schedule_sha256 = hashlib.sha256(
        "|".join(sorted(evaluation_hashes)).encode()
    ).hexdigest()
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_directory / f"g15b_{arm}_seed{seed}.pt"
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(
        {
            "schema_version": 1,
            "experiment": "G15B interleaved commissioned controller",
            "arm": arm,
            "seed": seed,
            "cohort": asdict(config),
            "model_config": asdict(model.config),
            "model_state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "training_schedule_sha256": schedule_hash.hexdigest(),
            "evaluation": evaluation,
        },
        temporary,
    )
    os.replace(temporary, checkpoint)
    result = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "state_bytes_per_sequence_fp32": model.state_capacity_bytes(1, torch.float32),
        "training_updates": global_update,
        "training_schedule_sha256": schedule_hash.hexdigest(),
        "training_batch_fingerprints": len(train_fingerprints),
        "evaluation_batch_fingerprints": len(evaluation_hashes),
        "evaluation_schedule_sha256": evaluation_schedule_sha256,
        "train_evaluation_hash_intersection": sorted(intersection),
        "useful_event_counts": event_counts,
        "loss_samples": loss_samples,
        "training_wall_seconds": training_seconds,
        "mean_synchronized_step_seconds": training_seconds / max(1, global_update),
        "evaluation_wall_seconds": evaluation_seconds,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "optimizer_partition": optimizer.partition_report(),
        "evaluation": evaluation,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
    }
    del optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _arm_absolute_checks(report: dict[str, Any]) -> dict[str, Any]:
    cell_checks = {}
    for name, cell in report["evaluation"]["cells"].items():
        task = cell["task"]
        length = cell["length"]
        checks = {
            "query_accuracy_at_least_0_90": cell["query_accuracy"] >= 0.90,
            "episode_accuracy_at_least_0_50": cell["exact_episode_accuracy"] >= 0.50,
            "write_f1_at_least_0_98": cell["write"]["f1"] >= 0.98,
            "erase_false_positive_at_most_0_02": cell["erase"]["false_positive_rate"]
            <= 0.02,
            "retention_p05_at_least_0_9995": cell["retention"]["non_edit_p05"]
            >= 0.9995,
            "oracle_intervention_at_least_0_999": cell["interventions"]["oracle"][
                "accuracy"
            ]
            >= 0.999,
        }
        if task != "needle":
            checks.update(
                {
                    "address_top1_at_least_0_98": cell["address"]["top1"] >= 0.98,
                    "address_margin_at_least_0_20": cell["address"][
                        "mean_correct_minus_best_wrong_cosine"
                    ]
                    >= 0.20,
                    "wrong_query_drop_at_least_0_50": cell["interventions"][
                        "wrong_query"
                    ]["drop_from_learned"]
                    >= 0.50,
                }
            )
        if task == "overwrite":
            checks["overwrite_erase_recall_at_least_0_95"] = (
                cell["erase"]["recall"] >= 0.95
            )
            if length in (512, 1024):
                checks["no_erase_drop_at_least_0_20"] = (
                    cell["interventions"]["no_erase"]["drop_from_learned"] >= 0.20
                )
        if length in (512, 1024):
            checks["no_memory_drop_at_least_0_50"] = (
                cell["interventions"]["no_memory"]["drop_from_learned"] >= 0.50
            )
            checks["no_write_drop_at_least_0_50"] = (
                cell["interventions"]["no_write"]["drop_from_learned"] >= 0.50
            )
        cell_checks[name] = {"checks": checks, "passed": all(checks.values())}
    for length in (512, 1024):
        mqar = report["evaluation"]["cells"][f"mqar:L{length}"]
        mqar_delta = abs(
            mqar["query_accuracy"] - mqar["interventions"]["no_erase"]["accuracy"]
        )
        cell_checks[f"mqar:L{length}"]["checks"][
            "no_erase_unique_mqar_change_at_most_0_05"
        ] = mqar_delta <= 0.05
        cell_checks[f"mqar:L{length}"]["passed"] = all(
            cell_checks[f"mqar:L{length}"]["checks"].values()
        )
    separation = not report["train_evaluation_hash_intersection"]
    return {
        "cells": cell_checks,
        "train_evaluation_namespace_separation": separation,
        "passed": separation and all(row["passed"] for row in cell_checks.values()),
    }


def _adjudicate(
    seed_reports: list[dict[str, Any]], *, preflight_passed: bool
) -> dict[str, Any]:
    absolute = []
    for seed_report in seed_reports:
        arms = {
            arm: _arm_absolute_checks(report)
            for arm, report in seed_report["arms"].items()
        }
        absolute.append(
            {
                "seed": seed_report["seed"],
                "arms": arms,
                "passed": all(row["passed"] for row in arms.values()),
            }
        )
    transport_cells = {}
    if seed_reports:
        for cell_name in seed_reports[0]["arms"]["S"]["evaluation"]["cells"]:
            means = {
                arm: sum(
                    report["arms"][arm]["evaluation"]["cells"][cell_name][
                        "query_accuracy"
                    ]
                    for report in seed_reports
                )
                / len(seed_reports)
                for arm in ARM_SPECS
            }
            paired_margins = [
                report["arms"]["S"]["evaluation"]["cells"][cell_name]["query_accuracy"]
                - max(
                    report["arms"][arm]["evaluation"]["cells"][cell_name][
                        "query_accuracy"
                    ]
                    for arm in ("I", "C")
                )
                for report in seed_reports
            ]
            checks = {
                "mean_noninferiority_within_0_01": means["S"]
                >= max(means["I"], means["C"]) - 0.01,
                "each_seed_noninferiority_within_0_03": min(paired_margins) >= -0.03,
            }
            transport_cells[cell_name] = {
                "mean_accuracy": means,
                "spin_paired_margins": paired_margins,
                "checks": checks,
                "passed": all(checks.values()),
            }
    passed = (
        preflight_passed
        and bool(seed_reports)
        and all(row["passed"] for row in absolute)
        and all(row["passed"] for row in transport_cells.values())
    )
    return {
        "passed": passed,
        "absolute_controller_gates": absolute,
        "transport_noninferiority": transport_cells,
        "decision": (
            "G15B commissioned controller passes; the conditional external-loss-only lane is authorized"
            if passed
            else "G15B does not pass; G15C and the external-loss-only lane remain blocked"
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
    preflight = run_preflight(device)
    seed_reports = []
    if preflight["passed"]:
        for seed in config.seeds:
            arms = {
                arm: _train_arm(
                    arm,
                    config,
                    seed=seed,
                    device=device,
                    checkpoint_directory=checkpoint_directory,
                )
                for arm in ARM_SPECS
            }
            schedules = {row["training_schedule_sha256"] for row in arms.values()}
            if len(schedules) != 1:
                raise RuntimeError(
                    "paired G15B arms received different training schedules"
                )
            evaluation_schedules = {
                row["evaluation_schedule_sha256"] for row in arms.values()
            }
            if len(evaluation_schedules) != 1:
                raise RuntimeError(
                    "paired G15B arms received different evaluation schedules"
                )
            seed_reports.append({"seed": seed, "arms": arms})
    adjudication = _adjudicate(seed_reports, preflight_passed=preflight["passed"])
    source_paths = (
        Path(__file__),
        ROOT / "g15b_interleaved_tasks.py",
        ROOT / "spin_dirac_memory.py",
        ROOT / "model.py",
        ROOT / "optimizers.py",
        PROTOCOL,
        PREREGISTRATION,
    )
    return {
        "schema_version": 1,
        "experiment": "G15B interleaved commissioned Spin-Dirac controller",
        "claim_status": "commissioned-controller test; not label-free or natural-text evidence",
        "mode": config.mode,
        "evidentiary": config.mode == "quality" and not status_at_start,
        "started_at": started_at,
        "finished_at": _now(),
        "elapsed_wall_seconds": time.perf_counter() - started,
        "git_commit_at_start": commit,
        "git_status_at_start": status_at_start,
        "protocol": asdict(config),
        "loss_weights": asdict(LOSS_WEIGHTS),
        "task_cycle": TASK_CYCLE,
        "quality_scored_training_decisions": scored_training_decisions(QUALITY_PHASES),
        "source_files": {
            str(path.relative_to(ROOT)): _sha256(path) for path in source_paths
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "compute_capability": (
                list(torch.cuda.get_device_capability(device))
                if device.type == "cuda"
                else None
            ),
            "dtype": "float32",
        },
        "preflight": preflight,
        "seed_reports": seed_reports,
        "adjudication": adjudication,
        "explicit_nonclaims": [
            "commissioning uses frozen address, write, erase, and retention labels",
            "a pass would not establish label-free controller discovery",
            "generic tasks cannot establish Spin transport necessity",
            "no natural-text, ordinary next-token, scaling, or fused-efficiency claim follows",
            "all G15A/L/F/R/S results remain separate evidence",
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
    config = frozen_config(args.mode)
    device = torch.device(args.device)
    commit, status = _git_provenance()
    if config.mode == "quality":
        if status:
            raise RuntimeError("G15B quality requires a clean git tree at start")
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("G15B quality requires the bound CUDA runtime")
        if torch.cuda.get_device_capability(device) != (7, 5):
            raise RuntimeError("G15B quality is frozen to SM75")
        if scored_training_decisions(config.phases) != 375_360:
            raise RuntimeError("frozen G15B training decision count drifted")
    report = run(
        config,
        device=device,
        checkpoint_directory=args.checkpoint_directory,
        commit=commit,
        status_at_start=status,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": _sha256(args.output),
                "passed": report["adjudication"]["passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "ARM_SPECS",
    "LOSS_WEIGHTS",
    "QUALITY_PHASES",
    "CohortConfig",
    "Phase",
    "build_model",
    "commissioned_losses",
    "complete_control_forward",
    "frozen_config",
    "run_preflight",
    "scored_training_decisions",
]
