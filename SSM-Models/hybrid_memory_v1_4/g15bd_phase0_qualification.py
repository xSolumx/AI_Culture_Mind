"""Exact G15B-D coupled residual-delta Phase-0 qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from .g15be_phase0_qualification import (
    _config,
    _effective_gates,
    _execution_parity,
    _gradient_reach,
    _maximum_absolute,
    _mixer,
    _model,
    _state_residual,
)
from .model import GatedDeltaState, parameter_count
from .optimizers import partition_optimizer_parameters

ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[1]
PROTOCOL = ROOT / "G15BD_RESIDUAL_DELTA_PROTOCOL_2026-08-26.md"
SOURCE_FILES = (
    ROOT / "transactional_delta.py",
    ROOT / "model.py",
    ROOT / "optimizers.py",
    ROOT / "g15be_phase0_qualification.py",
    Path(__file__).resolve(),
    PROTOCOL,
)
ARMS = {"P": "product", "D": "residual_delta"}
INITIAL_GATE_BOUND = 2e-8
INITIAL_FUNCTION_BOUND = 2e-6
FP64_ALGEBRA_BOUND = 1e-10
SPECTRAL_BOUND = 1.0 + 1e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPOSITORY_ROOT), *arguments],
        text=True,
        encoding="utf-8",
    ).strip()


def _state_dict_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _partition(model: torch.nn.Module) -> dict[str, tuple[str, ...]]:
    partition = partition_optimizer_parameters(model)
    return {
        name: tuple(parameter_name for parameter_name, _ in getattr(partition, name))
        for name in ("muon", "scalar_adamw", "adamw_decay", "adamw_no_decay")
    }


def _matched_initialization(device: torch.device) -> dict[str, Any]:
    models = {
        arm: _model(mode, seed=2581, device=device)
        for arm, mode in ARMS.items()
    }
    product = models["P"]
    residual = models["D"]
    tensor_equality = {
        name: torch.equal(tensor, residual.state_dict()[name])
        for name, tensor in product.state_dict().items()
    }
    torch.manual_seed(2582)
    inputs = torch.randn(3, 17, 32, device=device)
    product_erase, product_write = _effective_gates(_mixer(product), inputs)
    residual_erase, residual_write = _effective_gates(_mixer(residual), inputs)
    gate_residuals = {
        "erase": _maximum_absolute(
            product_erase.expand_as(residual_erase), residual_erase
        ),
        "write": _maximum_absolute(product_write, residual_write),
        "coupled_D": _maximum_absolute(residual_erase, residual_write),
    }
    torch.manual_seed(2583)
    tokens = torch.randint(0, product.config.vocab_size, (2, 31), device=device)
    with torch.no_grad():
        product_output = product(
            tokens, delta_scan_mode="recurrent", return_diagnostics=True
        )
        residual_output = residual(
            tokens, delta_scan_mode="recurrent", return_diagnostics=True
        )
    state_residual = _state_residual(
        product_output["states"], residual_output["states"]
    )
    functional_residuals = {
        "logit": _maximum_absolute(
            product_output["logits"], residual_output["logits"]
        ),
        "memory_state": state_residual["memory"],
        "convolution_state": state_residual["convolution"],
    }
    predictions_equal = torch.equal(
        product_output["logits"].argmax(-1),
        residual_output["logits"].argmax(-1),
    )
    totals = {arm: parameter_count(model) for arm, model in models.items()}
    active = {
        arm: sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        for arm, model in models.items()
    }
    state_bytes = {
        arm: model.state_capacity_bytes(1, torch.float32)
        for arm, model in models.items()
    }
    hashes = {arm: _state_dict_sha256(model) for arm, model in models.items()}
    partitions = {arm: _partition(model) for arm, model in models.items()}
    passed = (
        len(set(totals.values())) == 1
        and len(set(active.values())) == 1
        and len(set(state_bytes.values())) == 1
        and len(set(hashes.values())) == 1
        and partitions["P"] == partitions["D"]
        and all(tensor_equality.values())
        and gate_residuals["erase"] <= INITIAL_GATE_BOUND
        and gate_residuals["write"] <= INITIAL_GATE_BOUND
        and gate_residuals["coupled_D"] == 0.0
        and max(functional_residuals.values()) <= INITIAL_FUNCTION_BOUND
        and predictions_equal
    )
    return {
        "total_parameters": totals,
        "active_parameters": active,
        "state_capacity_bytes_batch1_fp32": state_bytes,
        "initial_parameter_sha256": hashes,
        "optimizer_partition": partitions,
        "state_dict_tensor_equality": tensor_equality,
        "initial_effective_gate_maximum_absolute_residuals": gate_residuals,
        "initial_effective_gate_frozen_bound": INITIAL_GATE_BOUND,
        "initial_function_maximum_absolute_residuals": functional_residuals,
        "initial_function_frozen_bound": INITIAL_FUNCTION_BOUND,
        "initial_predictions_equal": predictions_equal,
        "passed": passed,
    }


def _bounded_and_direct_law(device: torch.device) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for index, (arm, mode) in enumerate(ARMS.items()):
        model = _model(mode, seed=2584 + index, device=device).double()
        mixer = _mixer(model)
        torch.manual_seed(2586 + index)
        inputs = torch.randn(3, 23, 32, device=device, dtype=torch.float64)
        controls = mixer._controls(inputs, inputs)
        erase, write = _effective_gates(mixer, inputs)
        transition, injection = mixer._transitions(*controls[1:], None)
        spectral = float(
            torch.linalg.matrix_norm(transition, ord=2).max().detach().cpu()
        )
        finite = bool(torch.isfinite(transition).all()) and bool(
            torch.isfinite(injection).all()
        )
        interior = bool(((erase > 0.0) & (erase < 1.0)).all()) and bool(
            ((write > 0.0) & (write < 1.0)).all()
        )
        rows[arm] = {
            "mode": mode,
            "transition_spectral_norm_maximum": spectral,
            "transition_and_injection_finite": finite,
            "effective_gates_strictly_interior": interior,
            "passed": finite and interior and spectral <= SPECTRAL_BOUND,
        }

    model = _model("residual_delta", seed=2588, device=device).double()
    mixer = _mixer(model)
    torch.manual_seed(2589)
    inputs = torch.randn(2, 19, 32, device=device, dtype=torch.float64)
    _, key, value, _, edit, write, retention = mixer._controls(inputs, inputs)
    transition, injection = mixer._transitions(
        key, value, edit, edit, write, retention, None
    )
    eye = torch.eye(
        mixer.config.resolved_key_dim, dtype=inputs.dtype, device=device
    )
    projector = key.unsqueeze(-1) * key.unsqueeze(-2)
    expected_transition = (
        eye - edit[..., None, None] * projector.unsqueeze(3)
    ) @ torch.diag_embed(retention).unsqueeze(3)
    expected_injection = key.unsqueeze(3) * (edit * value).unsqueeze(-1)
    batch, length, heads, value_dim, key_dim, _ = expected_transition.shape
    expected_transition = (
        expected_transition.permute(0, 2, 3, 1, 4, 5)
        .contiguous()
        .view(batch, heads * value_dim, length, key_dim, key_dim)
    )
    expected_injection = (
        expected_injection.permute(0, 2, 3, 1, 4)
        .contiguous()
        .view(batch, heads * value_dim, length, key_dim, 1)
    )
    direct = {
        "transition_maximum_absolute_residual": _maximum_absolute(
            transition, expected_transition
        ),
        "injection_maximum_absolute_residual": _maximum_absolute(
            injection, expected_injection
        ),
        "effective_erase_write_maximum_absolute_residual": _maximum_absolute(
            edit, write
        ),
    }
    direct["frozen_bound"] = FP64_ALGEBRA_BOUND
    direct["passed"] = (
        max(
            direct["transition_maximum_absolute_residual"],
            direct["injection_maximum_absolute_residual"],
            direct["effective_erase_write_maximum_absolute_residual"],
        )
        <= FP64_ALGEBRA_BOUND
    )
    return {
        "arms": rows,
        "direct_residual_delta_law": direct,
        "passed": all(row["passed"] for row in rows.values())
        and bool(direct["passed"]),
    }


def qualify(device: torch.device) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    matched = _matched_initialization(device)
    bounded = _bounded_and_direct_law(device)
    execution = {
        arm: {
            "fp64": _execution_parity(
                device, torch.float64, gate_mode=mode, seed=2591
            ),
            "fp32": _execution_parity(
                device, torch.float32, gate_mode=mode, seed=2592
            ),
        }
        for arm, mode in ARMS.items()
    }
    gradients = {
        arm: _gradient_reach(device, mode) for arm, mode in ARMS.items()
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = torch.cuda.max_memory_allocated(device)
    else:
        peak_memory = None
    passed = (
        matched["passed"]
        and bounded["passed"]
        and all(
            row["passed"] for arm in execution.values() for row in arm.values()
        )
        and all(row["passed"] for row in gradients.values())
    )
    return {
        "matched_initialization": matched,
        "bounded_and_direct_law": bounded,
        "execution": execution,
        "gradient_reach": gradients,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": peak_memory,
        "passed": passed,
    }


def _environment(device: torch.device) -> dict[str, Any]:
    cuda = device.type == "cuda"
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if cuda else "cpu",
        "compute_capability": (
            list(torch.cuda.get_device_capability(device)) if cuda else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-commit")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--require-sm75", action="store_true")
    arguments = parser.parse_args()
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain").splitlines()
    if arguments.expected_commit and commit != arguments.expected_commit:
        raise RuntimeError("HEAD does not match --expected-commit")
    if status and not arguments.allow_dirty:
        raise RuntimeError("evidentiary qualification requires a clean checkout")
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if arguments.require_sm75 and (
        device.type != "cuda" or torch.cuda.get_device_capability(device) != (7, 5)
    ):
        raise RuntimeError("qualification requires exact compute capability (7, 5)")
    qualification = qualify(device)
    report = {
        "schema_version": 1,
        "experiment": "G15B-D Phase-0 coupled residual-delta qualification",
        "protocol": str(PROTOCOL.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
        "git_commit_at_start": commit,
        "git_status_at_start": status,
        "evidentiary": not status and arguments.require_sm75,
        "source_files": {
            str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"): _sha256(path)
            for path in SOURCE_FILES
        },
        "environment": _environment(device),
        "qualification": qualification,
        "adjudication": {
            "passed": bool(qualification["passed"]),
            "decision": (
                "authorize prospective G15B-D Phase-1 constructed training"
                if qualification["passed"]
                else "stop G15B-D before training; Phase-0 implementation failed"
            ),
        },
    }
    arguments.artifact.parent.mkdir(parents=True, exist_ok=True)
    arguments.artifact.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["adjudication"], sort_keys=True))
    if not report["adjudication"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = ["qualify"]
