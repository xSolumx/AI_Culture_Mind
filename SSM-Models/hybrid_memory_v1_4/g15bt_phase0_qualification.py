"""Reproducible G15B-T Phase-0 implementation qualification.

The evidentiary path requires a clean Git checkout and an exact SM75 CUDA
device.  It checks the matched full/history arms, structural strict causality,
affine contraction, FP64 scan/chunk/step parity, FP32 logit parity and exact
predictions, masked streaming state, and finite nonzero gradients.
"""

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
import torch.nn.functional as F

from .model import (
    CausalDepthwiseConv1d,
    GatedDeltaState,
    HybridMemoryConfig,
    HybridMemoryLM,
    parameter_count,
)
from .transactional_delta import TransactionalDeltaMemory

ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[1]
PROTOCOL = ROOT / "G15BT_TRANSACTIONAL_DELTA_PROTOCOL_2026-08-26.md"
SOURCE_FILES = (
    ROOT / "transactional_delta.py",
    ROOT / "model.py",
    Path(__file__).resolve(),
    PROTOCOL,
)
FP64_BOUND = 1e-10
FP32_LOGIT_BOUND = 5e-4
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


def _maximum_absolute(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().detach().cpu())


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
            raise TypeError("transactional qualification requires GatedDeltaState")
        memory = max(memory, _maximum_absolute(left_state.memory, right_state.memory))
        convolution = max(
            convolution,
            _maximum_absolute(left_state.convolution, right_state.convolution),
        )
    return {"memory": memory, "convolution": convolution}


def _config(controller_mode: str) -> HybridMemoryConfig:
    return HybridMemoryConfig(
        vocab_size=97,
        model_dim=32,
        layer_plan=("transactional_delta", "transactional_delta"),
        gated_delta_heads=4,
        gated_delta_key_dim=8,
        gated_delta_value_dim=8,
        transactional_controller_mode=controller_mode,  # type: ignore[arg-type]
        use_local_conv=True,
        conv_kernel=4,
        expansion=2,
        dropout=0.0,
        tie_embeddings=False,
    )


def _matched_arms(device: torch.device) -> tuple[dict[str, Any], HybridMemoryLM]:
    torch.manual_seed(2381)
    full = HybridMemoryLM(_config("full")).to(device)
    torch.manual_seed(2381)
    history = HybridMemoryLM(_config("history")).to(device)
    full_shapes = {
        name: list(parameter.shape) for name, parameter in full.named_parameters()
    }
    history_shapes = {
        name: list(parameter.shape) for name, parameter in history.named_parameters()
    }
    full_total = parameter_count(full)
    history_total = parameter_count(history)
    full_active = sum(p.numel() for p in full.parameters() if p.requires_grad)
    history_active = sum(p.numel() for p in history.parameters() if p.requires_grad)
    full_state = full.state_capacity_bytes(2, torch.float32)
    history_state = history.state_capacity_bytes(2, torch.float32)
    passed = (
        full_total == history_total
        and full_active == history_active
        and full_shapes == history_shapes
        and full_state == history_state
        and full.state_dict().keys() == history.state_dict().keys()
    )
    return (
        {
            "full_total_parameters": full_total,
            "history_total_parameters": history_total,
            "full_active_parameters": full_active,
            "history_active_parameters": history_active,
            "full_state_capacity_bytes_batch2_fp32": full_state,
            "history_state_capacity_bytes_batch2_fp32": history_state,
            "named_parameter_shapes_equal": full_shapes == history_shapes,
            "state_dict_keys_equal": full.state_dict().keys()
            == history.state_dict().keys(),
            "both_compute_full_and_history_views": True,
            "passed": passed,
        },
        history,
    )


def _causality_and_contraction(
    model: HybridMemoryLM, device: torch.device
) -> dict[str, Any]:
    torch.manual_seed(2382)
    convolution = CausalDepthwiseConv1d(32, 4).to(device=device, dtype=torch.float64)
    inputs = torch.randn(2, 8, 32, device=device, dtype=torch.float64)
    full, history, _ = convolution.full_and_strict_history(inputs)
    current_changed = inputs.clone()
    current_changed[:, 5] += (
        torch.randn(2, 32, device=device, dtype=torch.float64) * 7.0
    )
    changed_full, changed_history, _ = convolution.full_and_strict_history(
        current_changed
    )
    prior_changed = inputs.clone()
    prior_changed[:, 4] += torch.randn(2, 32, device=device, dtype=torch.float64) * 7.0
    _, prior_history, _ = convolution.full_and_strict_history(prior_changed)

    mixer = model.blocks[0].mixer
    if not isinstance(mixer, TransactionalDeltaMemory):
        raise TypeError("history arm did not construct TransactionalDeltaMemory")
    mixer = mixer.double()
    current_controls = mixer._controls(F.silu(full[:, 5:6]), F.silu(history[:, 5:6]))
    changed_controls = mixer._controls(
        F.silu(changed_full[:, 5:6]), F.silu(changed_history[:, 5:6])
    )
    edit_controls_equal = all(
        torch.equal(left, right)
        for left, right in zip(current_controls[1:], changed_controls[1:], strict=True)
    )
    transition, injection = mixer._transitions(*current_controls[1:], None)
    changed_transition, changed_injection = mixer._transitions(
        *changed_controls[1:], None
    )
    initial = torch.randn(2, 4, 8, 8, device=device, dtype=torch.float64)
    _, final = mixer._recurrent_states(transition, injection, initial)
    _, changed_final = mixer._recurrent_states(
        changed_transition, changed_injection, initial
    )
    spectral_maximum = float(
        torch.linalg.matrix_norm(transition, ord=2).max().detach().cpu()
    )
    current_full_effect = _maximum_absolute(full[:, 5], changed_full[:, 5])
    current_history_residual = _maximum_absolute(history[:, 5], changed_history[:, 5])
    prior_history_effect = _maximum_absolute(history[:, 5], prior_history[:, 5])
    transition_equal = torch.equal(transition, changed_transition)
    injection_equal = torch.equal(injection, changed_injection)
    state_equal = torch.equal(final, changed_final)
    passed = (
        current_full_effect > 0.0
        and current_history_residual == 0.0
        and prior_history_effect > 0.0
        and edit_controls_equal
        and transition_equal
        and injection_equal
        and state_equal
        and bool(torch.isfinite(transition).all())
        and bool(torch.isfinite(injection).all())
        and spectral_maximum <= SPECTRAL_BOUND
    )
    return {
        "current_full_effect": current_full_effect,
        "current_history_maximum_absolute_residual": current_history_residual,
        "prior_history_effect": prior_history_effect,
        "edit_controls_bit_identical": edit_controls_equal,
        "transition_bit_identical": transition_equal,
        "injection_bit_identical": injection_equal,
        "post_update_state_bit_identical": state_equal,
        "transition_spectral_norm_maximum": spectral_maximum,
        "transition_finite": bool(torch.isfinite(transition).all()),
        "injection_finite": bool(torch.isfinite(injection).all()),
        "passed": passed,
    }


def _execution_parity(
    device: torch.device, dtype: torch.dtype, *, seed: int
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = HybridMemoryLM(_config("history")).to(device=device, dtype=dtype).eval()
    tokens = torch.randint(0, model.config.vocab_size, (2, 33), device=device)
    with torch.no_grad():
        recurrent = model(tokens, delta_scan_mode="recurrent")
        parallel = model(tokens, delta_scan_mode="parallel")
        chunks = []
        chunk_state = None
        for start, stop in ((0, 3), (3, 8), (8, 9), (9, 21), (21, 33)):
            output = model(
                tokens[:, start:stop], chunk_state, delta_scan_mode="parallel"
            )
            chunks.append(output["logits"])
            chunk_state = output["states"]
        step_logits = []
        step_state = None
        for position in range(tokens.shape[1]):
            logits, step_state = model.step(tokens[:, position], step_state)
            step_logits.append(logits[:, None])
        chunk_logits = torch.cat(chunks, dim=1)
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

    if chunk_state is None or step_state is None or masked_step_state is None:
        raise RuntimeError("streaming state was not produced")
    logit_residuals = {
        "parallel_vs_recurrent": _maximum_absolute(
            parallel["logits"], recurrent["logits"]
        ),
        "chunk_vs_recurrent": _maximum_absolute(chunk_logits, recurrent["logits"]),
        "step_vs_recurrent": _maximum_absolute(token_logits, recurrent["logits"]),
    }
    state_residuals = {
        "parallel_vs_recurrent": _state_residual(
            parallel["states"], recurrent["states"]
        ),
        "chunk_vs_recurrent": _state_residual(chunk_state, recurrent["states"]),
        "step_vs_recurrent": _state_residual(step_state, recurrent["states"]),
        "masked_step_vs_full": _state_residual(masked_step_state, masked["states"]),
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
    }
    maximum_logit = max(logit_residuals.values())
    maximum_state = max(
        residual
        for comparison in state_residuals.values()
        for residual in comparison.values()
    )
    bound = FP64_BOUND if dtype == torch.float64 else FP32_LOGIT_BOUND
    passed = (
        maximum_logit <= bound
        and maximum_state <= bound
        and all(predictions_exact.values())
    )
    return {
        "dtype": str(dtype),
        "logit_maximum_absolute_residuals": logit_residuals,
        "state_maximum_absolute_residuals": state_residuals,
        "predictions_exact": predictions_exact,
        "frozen_bound": bound,
        "passed": passed,
    }


def _gradient_reach(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(2387)
    model = HybridMemoryLM(_config("history")).to(device).train()
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
        if not matching:
            rows[suffix] = {"matching_parameters": [], "passed": False}
            continue
        parameter_rows = []
        for name in matching:
            gradient = parameters[name].grad
            finite = gradient is not None and bool(torch.isfinite(gradient).all())
            nonzero = gradient is not None and bool(torch.count_nonzero(gradient))
            parameter_rows.append(
                {
                    "name": name,
                    "finite": finite,
                    "nonzero": nonzero,
                    "maximum_absolute_gradient": (
                        None
                        if gradient is None
                        else float(gradient.abs().max().detach().cpu())
                    ),
                }
            )
        rows[suffix] = {
            "matching_parameters": parameter_rows,
            "passed": all(row["finite"] and row["nonzero"] for row in parameter_rows),
        }
    return {
        "loss": float(loss.detach().cpu()),
        "paths": rows,
        "passed": all(row["passed"] for row in rows.values()),
    }


def qualify(device: torch.device) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    matched, history_model = _matched_arms(device)
    causality = _causality_and_contraction(history_model, device)
    fp64 = _execution_parity(device, torch.float64, seed=2385)
    fp32 = _execution_parity(device, torch.float32, seed=2386)
    gradients = _gradient_reach(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = torch.cuda.max_memory_allocated(device)
    else:
        peak_memory = None
    elapsed = time.perf_counter() - started
    passed = all(
        section["passed"] for section in (matched, causality, fp64, fp32, gradients)
    )
    return {
        "matched_arms": matched,
        "causality_and_contraction": causality,
        "fp64_execution": fp64,
        "fp32_execution": fp32,
        "gradient_reach": gradients,
        "elapsed_seconds": elapsed,
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
        "compute_capability": list(torch.cuda.get_device_capability(device))
        if cuda
        else None,
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
        "experiment": "G15B-T Phase-0 implementation qualification",
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
            "authorize prospective G15B-T Phase-1 constructed training"
            if report["qualification"]["passed"]
            else "stop G15B-T before training; Phase-0 implementation failed"
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
