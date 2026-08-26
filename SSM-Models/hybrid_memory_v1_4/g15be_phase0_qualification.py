"""Exact G15B-E effective-edit Phase-0 implementation qualification."""

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

from .model import GatedDeltaState, HybridMemoryConfig, HybridMemoryLM, parameter_count
from .transactional_delta import TransactionalDeltaMemory

ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[1]
PROTOCOL = ROOT / "G15BE_EFFECTIVE_EDIT_PROTOCOL_2026-08-26.md"
SOURCE_FILES = (
    ROOT / "transactional_delta.py",
    ROOT / "model.py",
    Path(__file__).resolve(),
    PROTOCOL,
)
GATE_MODES = ("product", "logit_additive")
FP64_BOUND = 1e-10
INITIAL_GATE_BOUND = 2e-8
SPECTRAL_BOUND = 1.0 + 1e-6
FP32_ROUNDOFF_MULTIPLIER = 128.0


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


def _maximum_absolute(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().detach().cpu())


def _config(gate_mode: str) -> HybridMemoryConfig:
    return HybridMemoryConfig(
        vocab_size=97,
        model_dim=32,
        layer_plan=("transactional_delta",),
        gated_delta_heads=4,
        gated_delta_key_dim=8,
        gated_delta_value_dim=8,
        transactional_controller_mode="full",
        transactional_effective_edit_gate_mode=gate_mode,  # type: ignore[arg-type]
        use_local_conv=True,
        conv_kernel=4,
        expansion=2,
        dropout=0.0,
        tie_embeddings=False,
    )


def _model(gate_mode: str, *, seed: int, device: torch.device) -> HybridMemoryLM:
    torch.manual_seed(seed)
    return HybridMemoryLM(_config(gate_mode)).to(device)


def _state_residual(
    left: tuple[object, ...], right: tuple[object, ...]
) -> dict[str, float]:
    if len(left) != len(right):
        raise ValueError("state tuples have different lengths")
    memory = 0.0
    convolution = 0.0
    for left_state, right_state in zip(left, right, strict=True):
        if not isinstance(left_state, GatedDeltaState) or not isinstance(
            right_state, GatedDeltaState
        ):
            raise TypeError("effective-edit qualification requires GatedDeltaState")
        memory = max(memory, _maximum_absolute(left_state.memory, right_state.memory))
        convolution = max(
            convolution,
            _maximum_absolute(left_state.convolution, right_state.convolution),
        )
    return {"memory": memory, "convolution": convolution}


def _mixer(model: HybridMemoryLM) -> TransactionalDeltaMemory:
    mixer = model.blocks[0].mixer
    if not isinstance(mixer, TransactionalDeltaMemory):
        raise TypeError("G15B-E did not construct TransactionalDeltaMemory")
    return mixer


def _effective_gates(
    mixer: TransactionalDeltaMemory, inputs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, _, event, erase, write, _ = mixer._controls(inputs, inputs)
    if mixer.config.effective_edit_gate_mode == "product":
        return event * erase, event * write
    return erase, write


def _matched_arms(device: torch.device) -> dict[str, Any]:
    product = _model("product", seed=2481, device=device)
    additive = _model("logit_additive", seed=2481, device=device)
    product_state = product.state_dict()
    additive_state = additive.state_dict()
    tensor_equality = {
        name: torch.equal(tensor, additive_state[name])
        for name, tensor in product_state.items()
    }
    torch.manual_seed(2482)
    inputs = torch.randn(3, 17, 32, device=device)
    product_erase, product_write = _effective_gates(_mixer(product), inputs)
    additive_erase, additive_write = _effective_gates(_mixer(additive), inputs)
    erase_residual = _maximum_absolute(product_erase, additive_erase)
    write_residual = _maximum_absolute(product_write, additive_write)
    product_total = parameter_count(product)
    additive_total = parameter_count(additive)
    product_active = sum(p.numel() for p in product.parameters() if p.requires_grad)
    additive_active = sum(p.numel() for p in additive.parameters() if p.requires_grad)
    product_state_bytes = product.state_capacity_bytes(1, torch.float32)
    additive_state_bytes = additive.state_capacity_bytes(1, torch.float32)
    passed = (
        product_total == additive_total
        and product_active == additive_active
        and product_state_bytes == additive_state_bytes
        and all(tensor_equality.values())
        and erase_residual <= INITIAL_GATE_BOUND
        and write_residual <= INITIAL_GATE_BOUND
    )
    return {
        "product_total_parameters": product_total,
        "additive_total_parameters": additive_total,
        "product_active_parameters": product_active,
        "additive_active_parameters": additive_active,
        "product_state_capacity_bytes_batch1_fp32": product_state_bytes,
        "additive_state_capacity_bytes_batch1_fp32": additive_state_bytes,
        "state_capacity_bits_per_single_query_sequence": product_state_bytes * 8,
        "state_dict_tensor_equality": tensor_equality,
        "initial_effective_erase_maximum_absolute_residual": erase_residual,
        "initial_effective_write_maximum_absolute_residual": write_residual,
        "initial_effective_gate_frozen_bound": INITIAL_GATE_BOUND,
        "passed": passed,
    }


def _bounded_update(device: torch.device) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for index, gate_mode in enumerate(GATE_MODES):
        model = _model(gate_mode, seed=2483 + index, device=device).double()
        mixer = _mixer(model)
        torch.manual_seed(2485 + index)
        inputs = torch.randn(3, 29, 32, device=device, dtype=torch.float64)
        controls = mixer._controls(inputs, inputs)
        effective_erase, effective_write = _effective_gates(mixer, inputs)
        transition, injection = mixer._transitions(*controls[1:], None)
        spectral_maximum = float(
            torch.linalg.matrix_norm(transition, ord=2).max().detach().cpu()
        )
        finite = bool(torch.isfinite(transition).all()) and bool(
            torch.isfinite(injection).all()
        )
        gates_interior = bool(
            ((effective_erase > 0.0) & (effective_erase < 1.0)).all()
        ) and bool(((effective_write > 0.0) & (effective_write < 1.0)).all())
        rows[gate_mode] = {
            "transition_spectral_norm_maximum": spectral_maximum,
            "transition_and_injection_finite": finite,
            "effective_gates_strictly_interior": gates_interior,
            "effective_erase_minimum": float(effective_erase.min().detach().cpu()),
            "effective_erase_maximum": float(effective_erase.max().detach().cpu()),
            "effective_write_minimum": float(effective_write.min().detach().cpu()),
            "effective_write_maximum": float(effective_write.max().detach().cpu()),
            "passed": finite
            and gates_interior
            and spectral_maximum <= SPECTRAL_BOUND,
        }
    return {"arms": rows, "passed": all(row["passed"] for row in rows.values())}


def _scaled_fp32_bound(reference: torch.Tensor, length: int) -> float:
    scale = max(1.0, float(reference.detach().abs().max().cpu()))
    return FP32_ROUNDOFF_MULTIPLIER * torch.finfo(torch.float32).eps * length * scale


def _execution_parity(
    device: torch.device, dtype: torch.dtype, *, gate_mode: str, seed: int
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = HybridMemoryLM(_config(gate_mode)).to(device=device, dtype=dtype).eval()
    tokens = torch.randint(0, model.config.vocab_size, (2, 33), device=device)
    with torch.no_grad():
        recurrent = model(
            tokens, delta_scan_mode="recurrent", return_diagnostics=True
        )
        parallel = model(
            tokens, delta_scan_mode="parallel", return_diagnostics=True
        )
        chunks = []
        chunk_reads = []
        chunk_state = None
        for start, stop in ((0, 3), (3, 8), (8, 9), (9, 21), (21, 33)):
            output = model(
                tokens[:, start:stop],
                chunk_state,
                delta_scan_mode="parallel",
                return_diagnostics=True,
            )
            chunks.append(output["logits"])
            chunk_reads.append(output["diagnostics"][0]["read"])
            chunk_state = output["states"]
        chunk_logits = torch.cat(chunks, dim=1)
        chunk_read = torch.cat(chunk_reads, dim=1)

        step_logits = []
        step_state = None
        for position in range(tokens.shape[1]):
            logits, step_state = model.step(tokens[:, position], step_state)
            step_logits.append(logits[:, None])
        token_logits = torch.cat(step_logits, dim=1)

        valid_mask = torch.tensor(
            [
                [(index % 5) != 2 for index in range(17)],
                [(index % 7) not in (1, 4) for index in range(17)],
            ],
            device=device,
            dtype=torch.bool,
        )
        masked = model(
            tokens[:, :17], valid_mask=valid_mask, delta_scan_mode="recurrent"
        )
        masked_step_state = None
        for position in range(17):
            _, masked_step_state = model.step(
                tokens[:, position],
                masked_step_state,
                valid_mask=valid_mask[:, position],
            )

        compact_logit = 0.0
        compact_state = 0.0
        compact_predictions = True
        for row in range(2):
            selected = valid_mask[row]
            compact = model(
                tokens[row : row + 1, :17][:, selected],
                delta_scan_mode="recurrent",
            )
            selected_logits = masked["logits"][row : row + 1, selected]
            compact_logit = max(
                compact_logit,
                _maximum_absolute(selected_logits, compact["logits"]),
            )
            compact_predictions = compact_predictions and torch.equal(
                selected_logits.argmax(-1), compact["logits"].argmax(-1)
            )
            compact_residuals = _state_residual(
                tuple(
                    GatedDeltaState(
                        memory=state.memory[row : row + 1],
                        convolution=state.convolution[row : row + 1],
                    )
                    for state in masked["states"]
                    if isinstance(state, GatedDeltaState)
                ),
                compact["states"],
            )
            compact_state = max(compact_state, *compact_residuals.values())

    if chunk_state is None or step_state is None or masked_step_state is None:
        raise RuntimeError("streaming state was not produced")
    recurrent_read = recurrent["diagnostics"][0]["read"]
    parallel_read = parallel["diagnostics"][0]["read"]
    logit_residuals = {
        "parallel_vs_recurrent": _maximum_absolute(
            parallel["logits"], recurrent["logits"]
        ),
        "chunk_vs_recurrent": _maximum_absolute(chunk_logits, recurrent["logits"]),
        "step_vs_recurrent": _maximum_absolute(token_logits, recurrent["logits"]),
        "masked_compact": compact_logit,
    }
    state_rows = {
        "parallel_vs_recurrent": _state_residual(
            parallel["states"], recurrent["states"]
        ),
        "chunk_vs_recurrent": _state_residual(chunk_state, recurrent["states"]),
        "step_vs_recurrent": _state_residual(step_state, recurrent["states"]),
        "masked_step_vs_full": _state_residual(masked_step_state, masked["states"]),
        "masked_compact": {"maximum": compact_state},
    }
    read_residuals = {
        "parallel_vs_recurrent": _maximum_absolute(parallel_read, recurrent_read),
        "chunk_vs_recurrent": _maximum_absolute(chunk_read, recurrent_read),
    }
    predictions_exact = {
        "parallel_vs_recurrent": torch.equal(
            parallel["logits"].argmax(-1), recurrent["logits"].argmax(-1)
        ),
        "chunk_vs_recurrent": torch.equal(
            chunk_logits.argmax(-1), recurrent["logits"].argmax(-1)
        ),
        "step_vs_recurrent": torch.equal(
            token_logits.argmax(-1), recurrent["logits"].argmax(-1)
        ),
        "masked_compact": compact_predictions,
    }
    maximum_logit = max(logit_residuals.values())
    maximum_state = max(
        residual for row in state_rows.values() for residual in row.values()
    )
    maximum_read = max(read_residuals.values())
    if dtype == torch.float64:
        bounds = {"logit": FP64_BOUND, "state": FP64_BOUND, "read": FP64_BOUND}
    else:
        reference_state = recurrent["states"][0]
        if not isinstance(reference_state, GatedDeltaState):
            raise TypeError("transactional model returned the wrong state type")
        bounds = {
            "logit": _scaled_fp32_bound(recurrent["logits"], tokens.shape[1]),
            "state": _scaled_fp32_bound(reference_state.memory, tokens.shape[1]),
            "read": _scaled_fp32_bound(recurrent_read, tokens.shape[1]),
        }
    passed = (
        maximum_logit <= bounds["logit"]
        and maximum_state <= bounds["state"]
        and maximum_read <= bounds["read"]
        and all(predictions_exact.values())
    )
    return {
        "dtype": str(dtype),
        "gate_mode": gate_mode,
        "logit_maximum_absolute_residuals": logit_residuals,
        "state_maximum_absolute_residuals": state_rows,
        "read_maximum_absolute_residuals": read_residuals,
        "predictions_exact": predictions_exact,
        "reference_maximum_absolute_logit": float(
            recurrent["logits"].abs().max().detach().cpu()
        ),
        "reference_maximum_absolute_state": float(
            recurrent["states"][0].memory.abs().max().detach().cpu()
        ),
        "reference_maximum_absolute_read": float(
            recurrent_read.abs().max().detach().cpu()
        ),
        "bounds": bounds,
        "fp32_bound_formula": (
            None
            if dtype == torch.float64
            else "128 * finfo(float32).eps * sequence_length * max(1, reference_absmax)"
        ),
        "passed": passed,
    }


def _gradient_reach(device: torch.device, gate_mode: str) -> dict[str, Any]:
    model = _model(gate_mode, seed=2491, device=device).train()
    torch.manual_seed(2492)
    tokens = torch.randint(0, model.config.vocab_size, (3, 29), device=device)
    output = model(tokens, delta_scan_mode="parallel")
    memory_state = output["states"][0]
    if not isinstance(memory_state, GatedDeltaState):
        raise TypeError("transactional model returned the wrong state type")
    loss = output["logits"].square().mean() + memory_state.memory.square().mean()
    loss.backward()
    parameters = dict(model.named_parameters())
    required_suffixes = (
        "embedding.weight",
        "local_conv.conv.weight",
        "mixer.query_projection.weight",
        "mixer.key_projection.weight",
        "mixer.value_projection.weight",
        "mixer.commit_projection.weight",
        "mixer.erase_projection.weight",
        "mixer.write_projection.weight",
        "mixer.decay_projection.weight",
        "mixer.output_gate.weight",
        "mixer.output_projection.weight",
        "lm_head.weight",
    )
    rows: dict[str, Any] = {}
    for suffix in required_suffixes:
        matching = [name for name in parameters if name.endswith(suffix)]
        parameter_rows = []
        for name in matching:
            gradient = parameters[name].grad
            parameter_rows.append(
                {
                    "name": name,
                    "finite": gradient is not None
                    and bool(torch.isfinite(gradient).all()),
                    "nonzero": gradient is not None
                    and bool(torch.count_nonzero(gradient)),
                    "maximum_absolute_gradient": (
                        None
                        if gradient is None
                        else float(gradient.abs().max().detach().cpu())
                    ),
                }
            )
        rows[suffix] = {
            "matching_parameters": parameter_rows,
            "passed": bool(parameter_rows)
            and all(row["finite"] and row["nonzero"] for row in parameter_rows),
        }
    return {
        "gate_mode": gate_mode,
        "loss": float(loss.detach().cpu()),
        "paths": rows,
        "passed": all(row["passed"] for row in rows.values()),
    }


def qualify(device: torch.device) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    matched = _matched_arms(device)
    bounded = _bounded_update(device)
    execution = {
        gate_mode: {
            "fp64": _execution_parity(
                device, torch.float64, gate_mode=gate_mode, seed=2493
            ),
            "fp32": _execution_parity(
                device, torch.float32, gate_mode=gate_mode, seed=2494
            ),
        }
        for gate_mode in GATE_MODES
    }
    gradients = {
        gate_mode: _gradient_reach(device, gate_mode) for gate_mode in GATE_MODES
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
            row["passed"]
            for arm in execution.values()
            for row in arm.values()
        )
        and all(row["passed"] for row in gradients.values())
    )
    return {
        "matched_arms": matched,
        "bounded_update": bounded,
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

    report = {
        "schema_version": 1,
        "experiment": "G15B-E Phase-0 effective-edit implementation qualification",
        "protocol": str(PROTOCOL.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
        "git_commit_at_start": commit,
        "git_status_at_start": status,
        "evidentiary": not status and arguments.require_sm75,
        "source_files": {
            str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"): _sha256(path)
            for path in SOURCE_FILES
        },
        "environment": _environment(device),
        "qualification": qualify(device),
    }
    report["adjudication"] = {
        "passed": bool(report["qualification"]["passed"]),
        "decision": (
            "authorize prospective G15B-E Phase-1 constructed training"
            if report["qualification"]["passed"]
            else "stop G15B-E before training; Phase-0 implementation failed"
        ),
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
